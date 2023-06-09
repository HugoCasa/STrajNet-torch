import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F

from torchinfo import summary

from perceiver.model.core import (
    PerceiverEncoder,
    InputAdapter,
    FourierPositionEncoding,
    PerceiverDecoder,
    OutputAdapter,
    TrainableQueryProvider,
)

from typing import Tuple, Sequence
from functools import partial

from flexivit import FlexiPatchEmbed
from flexivit.utils import resize_abs_pos_embed

from MultiHeadAttention import MultiHeadAttention
import numpy as np


class TrajEncoder(nn.Module):
    def __init__(self,num_heads=4,out_dim=384):
        super(TrajEncoder, self).__init__()
        self.node_feature = nn.Sequential(nn.Conv1d(5, 64, kernel_size=1), nn.ELU())
        self.node_attention = MultiHeadAttention(input_channels=(64,64,64), num_heads=num_heads, head_size=64, dropout=0.1, output_size=64*5)
        self.vector_feature = nn.Linear(3, 64, bias=False)
        self.sublayer = nn.Sequential(nn.Linear(384, out_dim), nn.ELU())

    def forward(self, inputs, mask):
        mask = mask.to(torch.float32)
        mask = torch.matmul(mask[:, :, np.newaxis], mask[:, np.newaxis, :])
        nodes = self.node_feature(inputs[:, :, :5].permute(0,2,1))
        nodes = nodes.permute(0,2,1) # (B, 11, 5)
        nodes = self.node_attention(inputs=[nodes, nodes, nodes], mask=mask) # (B, 11, 54*5)
        nodes, _ = torch.max(nodes, 1) # (B, 64*5)
        vector = self.vector_feature(inputs[:, 0, 5:])
        out = torch.concat([nodes, vector], dim=1) # (B, 384)
        polyline_feature = self.sublayer(out) # (B, 384)

        return polyline_feature



class FlexiInputPatcher(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        embed_dim: int,
        patch_size: Tuple[int, int],
        patch_size_seq: Sequence[int],
        base_pos_embed_size: int,
    ):
        self.embed_dim = embed_dim
        super().__init__()

        kwargs = {
            "norm_layer": nn.LayerNorm,
            "embed_dim": embed_dim,
            "patch_size": patch_size,  # base patch size
            "patch_size_seq": patch_size_seq,
            "grid_size": base_pos_embed_size,
        }

        self.patch_embed_vehicle = FlexiPatchEmbed(
            in_chans=11,
            img_size=image_shape,
            **kwargs,
        )

        self.patch_embed_map = FlexiPatchEmbed(
            in_chans=3,
            img_size= (image_shape[0] // 2, image_shape[1] // 2), # map is half the width/height of the vehicle occupancy and flow
            **kwargs,
        )

        self.patch_embed_flow = FlexiPatchEmbed(
            in_chans=2,
            img_size=image_shape,
            **kwargs,
        )

        self.all_patch_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, ogm, map_img, flow, patch_size=None):
        """
        Args:
            ogm: (B, 512, 512, 11, 2)
            map_img: (B, 256, 256, 3)
            flow: (B, 256, 256, 2)
            patch_size: only used when evaluating
        """
        ogm = ogm[:, :, :, :, 0]
        ogm = self.patch_embed_vehicle(
            ogm.permute([0, 3, 1, 2]), patch_size=patch_size
        )  # (B, N1, 384)

        # have to pad the map image to match the size of the vecicle occupancy / flow
        map_img = self.patch_embed_map(
            map_img.permute([0, 3, 1, 2]),
            patch_size=patch_size, 
        )  # (B, N2, 384)

        flow = self.patch_embed_flow(
            flow.permute([0, 3, 1, 2]),
            patch_size=patch_size, 
        ) # (B, N3, 384)

        x = self.all_patch_norm(torch.cat([ogm, map_img, flow], dim=1))
        ogm, map_img, flow = torch.split(x, [ogm.shape[1], map_img.shape[1], flow.shape[1]], dim=1) # (B, N1, 384), (B, N2, 384), (B, N3, 384)

        return ogm, map_img, flow


class FlexiPositionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        img_size: Tuple[int, int],
        num_patches: int,
        base_pos_embed_size: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.resize_pos_embed = partial(
            resize_abs_pos_embed,
            old_size=base_pos_embed_size,
            num_prefix_tokens=0,
        )
        self.ogm_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.ogm_pos_embed, std=0.02)
        self.map_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.map_pos_embed, std=0.02)
        self.flow_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.flow_pos_embed, std=0.02)

    def forward(self, ogm, map_img, flow):
        new_ogm_size = int(math.sqrt(ogm.shape[1]))
        ogm = ogm + self.resize_pos_embed(self.ogm_pos_embed, new_ogm_size)  # (B, N1, 384)

        new_map_size = int(math.sqrt(map_img.shape[1]))
        map_img = map_img + self.resize_pos_embed(
            self.map_pos_embed, new_map_size
        )  # (B, N2, 384)

        new_flow_size = int(math.sqrt(flow.shape[1]))
        flow = flow + self.resize_pos_embed(
            self.flow_pos_embed, new_flow_size
        )  # (B, N3, 384)
        return ogm, map_img, flow


class FlexiInputAdapter(InputAdapter):
    def forward(self, x: Tensor) -> Tensor:
        return x




class OccFlowOutputAdapter(OutputAdapter):
    def __init__(
        self,
        output_shape: Tuple[int, int],
        num_query_channels: int,
        num_output_channels: int,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.num_output_channels = num_output_channels
        self.output_layer = nn.Sequential(
            nn.ELU(), nn.Linear(num_query_channels, num_output_channels)
        )

    def forward(self, x):
        x = self.output_layer(x)  # (B, 256 * 256, 8 * 4)
        x = torch.reshape(
            x,
            [-1, self.output_shape[0], self.output_shape[1], self.num_output_channels],
        )
        return x


class FlexiPerceiver(torch.nn.Module):
    def __init__(
        self,
        cfg,
        base_patch_size=(16, 16),
        base_pos_embed_size=32,  # pos embed size = grid size given base patch size
        patch_size_seq=(4, 8, 12, 16, 20, 24, 30, 40, 48),
    ):
        super().__init__()

        # trajectories encoder
        self.traj_encoder = TrajEncoder(out_dim=cfg["embed_dim"] * 4)
        self.bi_embed = torch.tensor([[1,0],[0,1]], dtype=torch.float32).repeat_interleave(torch.tensor([48, 16]), dim=0)
        self.seg_embed = nn.Linear(2, cfg["embed_dim"] * 4, bias=False)
        self.obs_norm = nn.LayerNorm(eps=1e-3, normalized_shape=cfg["embed_dim"] * 4)
        self.occ_norm = nn.LayerNorm(eps=1e-3, normalized_shape=cfg["embed_dim"] * 4)

        self.input_patcher = FlexiInputPatcher(
            image_shape=cfg["input_size"],
            embed_dim=cfg["embed_dim"] * 4, # 384
            patch_size=base_patch_size,  # base patch size
            patch_size_seq=patch_size_seq,
            base_pos_embed_size=base_pos_embed_size,
        )
        self.position_encoder = FlexiPositionEncoder(
            embed_dim=cfg["embed_dim"] * 4,
            img_size=cfg["input_size"],
            num_patches=base_pos_embed_size * base_pos_embed_size,
            base_pos_embed_size=base_pos_embed_size,
        )
        input_adapter = FlexiInputAdapter(
            num_input_channels=cfg["embed_dim"] * 4,
        )
        self.encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=512, # increased from 384, 256
            num_latent_channels=cfg["embed_dim"] * 4,  # 384
            num_cross_attention_layers=1,
            num_self_attention_blocks=6,
            init_scale=0.1,
        )

        self.decoder = PerceiverDecoder(
            num_latent_channels=cfg["embed_dim"] * 4,  # 384
            output_query_provider=TrainableQueryProvider(
                num_queries=256 * 256,
                num_query_channels=384, # increased from 256, 128
            ),
            output_adapter=OccFlowOutputAdapter(
                output_shape=(256, 256),
                num_query_channels=384, # increased from 256, 128
                num_output_channels=8 * 4,
            ),
            init_scale=0.1,
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
        dummy_flow = torch.zeros((1,) + cfg["input_size"] + (2,))
        self.ref_res = None
        self(
            dummy_ogm,
            dummy_map,
            obs=dummy_obs_actors,
            occ=dummy_occ_actors,
            flow=dummy_flow,
        )
        summary(self)

    def forward(
        self,
        ogm,
        map_img,
        obs=None,
        occ=None,
        flow=None,
        patch_size=None,
    ):
        # trajectories encoder:
        obs_mask = torch.not_equal(obs, 0)[:,:,:,0]
        obs = [self.traj_encoder(obs[:, i],obs_mask[:,i]) for i in range(48)]
        obs = torch.stack(obs,dim=1) # (B, 48, 384)

        occ_mask = torch.not_equal(occ, 0)[:,:,:,0]
        occ = [self.traj_encoder(occ[:, i],occ_mask[:,i]) for i in range(16)]
        occ = torch.stack(occ,dim=1) # (B, 16, 384)

        embed = self.bi_embed[np.newaxis, :, :].repeat_interleave(occ.size()[0], dim=0).to(obs.device)
        embed = self.seg_embed(embed)

        obs = self.obs_norm(obs + embed[:,:48,:])
        occ = self.occ_norm(occ + embed[:,48:,:])
        trajs = torch.cat([obs, occ], dim=1)  # (B, 64, 384)

        # visual features patching:
        ogm, map_img, flow = self.input_patcher(
            ogm, map_img, flow, patch_size=patch_size
        )
        # positional encoding
        ogm, map_img, flow = self.position_encoder(ogm, map_img, flow) 
        # concatenate trajs with visual features:
        x = torch.cat([ogm, map_img, flow, trajs], dim=1)  # (B, N1 + N2 + N3 + 64, 384)

        # encoder
        x = self.encoder(x)  # (B, 256, 384)

        # decoder
        x = self.decoder(x)  # (B, 256, 256, 32)

        return x


if __name__ == "__main__":
    cfg = dict(
        input_size=(512, 512),
        window_size=8,
        embed_dim=96,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
    )
    model = FlexiPerceiver(cfg)
