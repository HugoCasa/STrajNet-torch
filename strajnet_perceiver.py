import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from TrajNet import TrajNetCrossAttention
from FG_MSA import FGMSA
from Pyramid3DDecoder import Pyramid3DDecoder
from einops import rearrange, repeat

from typing import Optional

from torchinfo import summary

from perceiver.model.core import PerceiverEncoder, InputAdapter, FourierPositionEncoding

from typing import Tuple


class PatchEmbed(torch.nn.Module):
    def __init__(
        self,
        img_size=(256, 256),
        patch_size=(4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
    ):
        super(PatchEmbed, self).__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(normalized_shape=embed_dim, eps=1e-5)
        else:
            self.norm = None

    def forward(self, x: Tensor):
        B, H, W, C = x.size()
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = x.permute(0, 3, 1, 2)  # B C H W
        x = self.proj(x)

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = torch.reshape(
            x,
            shape=[
                -1,
                (H // self.patch_size[0]) * (W // self.patch_size[0]),
                self.embed_dim,
            ],
        )
        if self.norm is not None:
            x = self.norm(x)
        return x


class OccFlowInputPatcher(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        embed_dim: int,
    ):
        self.embed_dim = embed_dim
        super().__init__()

        self.patch_embed_vecicle = PatchEmbed(
            img_size=image_shape,
            in_chans=11,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.patch_embed_map = PatchEmbed(
            img_size=image_shape,
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.patch_embed_flow = PatchEmbed(
            img_size=image_shape,
            in_chans=2,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )

        self.all_patch_norm = nn.LayerNorm(eps=1e-5, normalized_shape=embed_dim)

    def forward(self, x, map_img, flow):
        vec = x[:, :, :, :, 0]
        x = self.patch_embed_vecicle(vec)

        # have to pad the map image to match the size of the vecicle occupancy / flow
        maps = self.patch_embed_map(map_img)
        maps = torch.reshape(maps, [-1, 64, 64, self.embed_dim])
        maps = F.pad(maps, [0, 0, 32, 32, 32, 32, 0, 0])
        maps = torch.reshape(maps, [-1, 128 * 128, self.embed_dim])
        x = torch.cat([x, maps], dim=1)

        x = torch.cat([x, self.patch_embed_flow(flow)], dim=1)

        x = self.all_patch_norm(x)  # (B, 16384 * 3, 96)

        return x  # (B, 16384 * 3, 96)


class OccFlowInputAdapter(InputAdapter):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        embed_dim: int,
    ):
        position_encoding = FourierPositionEncoding(
            input_shape=input_shape, num_frequency_bands=24
        )
        super().__init__(embed_dim + position_encoding.num_position_encoding_channels())

        self.position_encoding = position_encoding

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        ogm, map_img, flow = torch.chunk(x, 3, dim=1)
        pos_enc = self.position_encoding(b)  # (B, 16384, 98)
        ogm = torch.cat([ogm, pos_enc], dim=-1)  # (B, 16384, 194)
        map_img = torch.cat([map_img, pos_enc], dim=-1)  # (B, 16384, 194)
        flow = torch.cat([flow, pos_enc], dim=-1)  # (B, 16384, 194)
        # trajs (B, 64, 96  + 98 pos_enc)
        return torch.cat([ogm, map_img, flow], dim=1)  # (B, 16384 * 3 + 64, 194)


# decoder
# query (B, 256 * 256, 194) => (B, 256 * 256, 8 * 3)


class STrajNetPerceiver(torch.nn.Module):
    def __init__(
        self,
        cfg,
        actor_only=True,
        sep_actors=False,
        fg_msa=True,
        fg=True,
    ):
        super().__init__()

        self.input_patcher = OccFlowInputPatcher(
            image_shape=cfg["input_size"], embed_dim=cfg["embed_dim"]
        )
        input_adapter = OccFlowInputAdapter(
            input_shape=[s // 4 for s in cfg["input_size"]], embed_dim=cfg["embed_dim"]
        )
        self.encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=256,  # 16 * 16
            num_latent_channels=cfg["embed_dim"] * 4,  # 384
            num_cross_attention_layers=1,
            num_self_attention_blocks=4,  # = number of swinT blocks
        )

        if sep_actors:
            traj_cfg = dict(traj_heads=4, att_heads=6, out_dim=384, no_attn=True)
        else:
            traj_cfg = dict(traj_heads=4, att_heads=6, out_dim=384, no_attn=False)

        resolution = [8, 16, 32]
        hw = resolution[4 - len(cfg["depths"][:])]
        self.trajnet_attn = TrajNetCrossAttention(
            traj_cfg,
            actor_only=actor_only,
            pic_size=(hw, hw),
            pic_dim=768 // (2 ** (4 - len(cfg["depths"][:]))),
            multi_modal=True,
            sep_actors=sep_actors,
        )
        self.fg_msa = fg_msa
        self.fg = fg
        if fg_msa:
            self.fg_msa_layer = FGMSA(
                q_size=(16, 16),
                kv_size=(16, 16),
                n_heads=8,
                n_head_channels=48,
                n_groups=8,
                out_dim=384,
                use_last_ref=False,
                fg=fg,
            )
        self.decoder = Pyramid3DDecoder(
            config=None,
            img_size=cfg["input_size"],
            pic_dim=768 // (2 ** (4 - len(cfg["depths"][:]))),
            use_pyramid=False,
            timestep_split=True,
            shallow_decode=(4 - len(cfg["depths"][:])),
            flow_sep_decode=False,
            conv_cnn=False,
        )

        dummy_ogm = torch.zeros(
            (1,)
            + cfg["input_size"]
            + (
                11,
                2,
            )
        )
        dummy_map = torch.zeros((1,) + (256, 256) + (3,))
        dummy_obs_actors = torch.zeros([1, 48, 11, 8])
        dummy_occ_actors = torch.zeros([1, 16, 11, 8])
        dummy_ccl = torch.zeros([1, 256, 10, 7])
        dummy_flow = torch.zeros((1,) + cfg["input_size"] + (2,))
        self.ref_res = None
        self(
            dummy_ogm,
            dummy_map,
            obs=dummy_obs_actors,
            occ=dummy_occ_actors,
            mapt=dummy_ccl,
            flow=dummy_flow,
        )
        summary(self)

    def forward(
        self,
        ogm,
        map_img,
        training=True,
        obs=None,
        occ=None,
        mapt=None,
        flow=None,
    ):
        # visual encoder:
        x = self.encoder(self.input_patcher(ogm, map_img, flow))
        q = x

        if self.fg_msa:
            q = torch.reshape(q, [-1, 16, 16, 384])
            # fg-msa:
            res, pos, ref = self.fg_msa_layer(q, training=training)
            q = res + q
            q = torch.reshape(q, [-1, 16 * 16, 384])
        query = torch.repeat_interleave(torch.unsqueeze(q, dim=1), repeats=8, axis=1)
        if self.fg:
            # added Projected flow-features to each timestep
            ref = torch.reshape(ref, [-1, 8, 256, 384])
            query = ref + query

        # time-sep-cross attention and vector encoders:
        obs_value = self.trajnet_attn(query, obs, occ, mapt, training)

        # fpn decoding:
        y = self.decoder(obs_value, training)
        y = torch.reshape(y.permute([0, 2, 3, 1, 4]), [-1, 256, 256, 32])
        return y


if __name__ == "__main__":
    cfg = dict(
        input_size=(512, 512),
        window_size=8,
        embed_dim=96,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
    )
    model = STrajNetPerceiver(cfg)
