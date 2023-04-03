import os
import logging
from typing import Dict

import submitit
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from google.protobuf import text_format
from torchmetrics import MeanMetric

from filesDataset import FilesDataset
from strajNet import STrajNet
from loss import OGMFlow_loss
from metrics import OGMFlowMetrics, print_metrics
import occu_metric as occupancy_flow_metrics
import occupancy_flow_grids


LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# torch.autograd.set_detect_anomaly(True)

config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
config_text = """
num_past_steps: 10
num_future_steps: 80
num_waypoints: 8
cumulative_waypoints: false
normalize_sdc_yaw: true
grid_height_cells: 256
grid_width_cells: 256
sdc_y_in_grid: 192
sdc_x_in_grid: 128
pixels_per_meter: 3.2
agent_points_per_side_length: 48
agent_points_per_side_width: 16
"""
text_format.Parse(config_text, config)

NUM_PRED_CHANNELS = 4
ogm_weight = 1000.0
occ_weight = 1000.0
flow_origin_weight = 1000.0
flow_weight = 1.0


class Trainer:
    def __init__(
        self,
        lr: float,
        batch_size: int,
        epochs: int,
        files_dir: str,
        save_dir: str,
        checkpoint_path: str = None,
        local=False,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.files_dir = files_dir
        self.save_dir = save_dir
        self.checkpoint_path = checkpoint_path
        self.local = local

    def __call__(self):
        if self.local:
            self._setup_local()
        else:
            self._setup_slurm()
        self._train()

    def _setup_slurm(self):
        self.dist_env = (
            submitit.helpers.TorchDistributedEnvironment().export()
        )  # export the variables for distributed training

        LOG.info(
            f"Process group: {self.dist_env.world_size} tasks, rank: {self.dist_env.rank}"
        )
        dist.init_process_group("nccl")

    def _setup_local(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=0, world_size=1)

        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        self.dist_env = dotdict(
            {
                "world_size": 1,
                "rank": 0,
            }
        )

    def _parse_record(sefl, features: Dict[str, Tensor]):
        """Convert features to the right types"""

        features["centerlines"] = features["centerlines"].to(torch.float32)

        features["actors"] = features["actors"].to(torch.float32)
        features["occl_actors"] = features["occl_actors"].to(torch.float32)

        features["ogm"] = features["ogm"].to(torch.float32)

        features["map_image"] = features["map_image"].to(torch.float32) / 256
        features["vec_flow"] = features["vec_flow"]

        features["gt_flow"] = features["gt_flow"][
            :, 128 : 128 + 256, 128 : 128 + 256, :
        ]
        features["origin_flow"] = features["origin_flow"][
            :, 128 : 128 + 256, 128 : 128 + 256, :
        ]
        features["gt_obs_ogm"] = features["gt_obs_ogm"].to(torch.float32)[
            :, 128 : 128 + 256, 128 : 128 + 256, :
        ]
        features["gt_occ_ogm"] = features["gt_occ_ogm"].to(torch.float32)[
            :, 128 : 128 + 256, 128 : 128 + 256, :
        ]

        return features

    def _get_dataloader(self):
        """Get training and validation dataloaders"""

        dataset = FilesDataset(
            path=self.files_dir + "/train_numpy", transform=self._parse_record
        )

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # done by the sampler
            pin_memory=True,
            num_workers=0,
            sampler=DistributedSampler(dataset),
        )

        val_dataset = FilesDataset(
            path=self.files_dir + "/val_numpy", transform=self._parse_record
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # done by the sampler
            pin_memory=True,
            num_workers=0,
            sampler=DistributedSampler(val_dataset),
        )

        return train_loader, val_loader

    def _train(self):
        local_rank = 0  # as one task per gpu, device is always 0

        cfg = dict(
            input_size=(512, 512),
            window_size=8,
            embed_dim=96,
            depths=[2, 2, 2],
            num_heads=[3, 6, 12],
        )
        model = STrajNet(
            cfg, actor_only=True, sep_actors=False, fg_msa=True, fg=True
        ).to(local_rank)
        # wrap model for distributed training
        model = DDP(model, device_ids=[local_rank])
        loss_fn = OGMFlow_loss(
            config,
            no_use_warp=False,
            use_pred=False,
            use_gt=True,
            ogm_weight=ogm_weight,
            occ_weight=occ_weight,
            flow_origin_weight=flow_origin_weight,
            flow_weight=flow_weight,
            use_focal_loss=False,
        )

        train_loader, val_loader = self._get_dataloader()

        # optimizer = torch.optim.NAdam(model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(train_loader)
        )

        if self.checkpoint_path is not None:
            # if checkpoint path given, load weights
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            last_epoch = checkpoint["epoch"]
            LOG.info(f"Loaded checkpoint ({last_epoch+1} epochs)")
        else:
            last_epoch = -1

        dist.barrier()
        LOG.info("Initialization passed successfully.")

        for epoch in range(self.epochs):
            if epoch <= last_epoch:
                LOG.info(f"Epoch {epoch+1} already trained")
                continue

            LOG.info(f"Epoch {epoch+1}\n-------------------------------")
            LOG.info(f"Learning rate: {scheduler.get_lr()}")
            train_loss = MeanMetric().to(local_rank)
            train_loss_occ = MeanMetric().to(local_rank)
            train_loss_flow = MeanMetric().to(local_rank)
            train_loss_warp = MeanMetric().to(local_rank)

            model.train()

            train_loader.sampler.set_epoch(epoch)

            for batch_idx, data in enumerate(train_loader):
                # inputs: will automatically be put on right device when passed to model
                map_img = data["map_image"]
                centerlines = data["centerlines"]
                actors = data["actors"]
                occl_actors = data["occl_actors"]
                ogm = data["ogm"]
                flow = data["vec_flow"]

                # ground truths directly passed to device for loss / metrics
                gt_obs_ogm = data["gt_obs_ogm"].to(local_rank)
                gt_occ_ogm = data["gt_occ_ogm"].to(local_rank)
                gt_flow = data["gt_flow"].to(local_rank)
                origin_flow = data["origin_flow"].to(local_rank)

                # forward pass
                outputs = model(
                    ogm,
                    map_img,
                    obs=actors,
                    occ=occl_actors,
                    mapt=centerlines,
                    flow=flow,
                )

                # compute loss
                true_waypoints = self._warpped_gt(
                    gt_ogm=gt_obs_ogm,
                    gt_occ=gt_occ_ogm,
                    gt_flow=gt_flow,
                    origin_flow=origin_flow,
                )
                logits = self._get_pred_waypoint_logits(outputs)
                loss_dict = loss_fn(
                    true_waypoints=true_waypoints,
                    pred_waypoint_logits=logits,
                    curr_ogm=ogm[:, :, :, -1, 0],
                )
                loss_value = torch.sum(sum(loss_dict.values()))

                # backward pass
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                scheduler.step()

                # update losses
                train_loss.update(loss_dict["observed_xe"])
                train_loss_occ.update(loss_dict["occluded_xe"])
                train_loss_flow.update(loss_dict["flow"])
                train_loss_warp.update(loss_dict["flow_warp_xe"])

                obs_loss = train_loss.compute() / ogm_weight
                occ_loss = train_loss_occ.compute() / occ_weight
                flow_loss = train_loss_flow.compute() / flow_weight
                warp_loss = train_loss_warp.compute() / flow_origin_weight

                # log loss
                current = self.batch_size * (batch_idx + 1) * self.dist_env.world_size
                print(
                    f"\nobs. loss: {obs_loss:>7f}, occl. loss:  {occ_loss:>7f}, flow loss: {flow_loss:>7f}, warp loss: {warp_loss:>7f}  [{current:>5d}]",
                    flush=True,
                )


            LOG.info("Validation\n-------------------------------")

            valid_loss = MeanMetric().to(local_rank)
            valid_loss_occ = MeanMetric().to(local_rank)
            valid_loss_flow = MeanMetric().to(local_rank)
            valid_loss_warp = MeanMetric().to(local_rank)

            valid_metrics = OGMFlowMetrics(local_rank, no_warp=False)

            model.eval()

            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    # inputs: will automatically be put on right device when passed to model
                    map_img = data["map_image"]
                    centerlines = data["centerlines"]
                    actors = data["actors"]
                    occl_actors = data["occl_actors"]
                    ogm = data["ogm"]
                    flow = data["vec_flow"]

                    # ground truths directly put on device for loss / metrics
                    gt_obs_ogm = data["gt_obs_ogm"].to(local_rank)
                    gt_occ_ogm = data["gt_occ_ogm"].to(local_rank)
                    gt_flow = data["gt_flow"].to(local_rank)
                    origin_flow = data["origin_flow"].to(local_rank)

                    # forward pass
                    outputs = model(
                        ogm,
                        map_img,
                        obs=actors,
                        occ=occl_actors,
                        mapt=centerlines,
                        flow=flow,
                    )

                    # compute losses
                    true_waypoints = self._warpped_gt(
                        gt_ogm=gt_obs_ogm,
                        gt_occ=gt_occ_ogm,
                        gt_flow=gt_flow,
                        origin_flow=origin_flow,
                    )
                    logits = self._get_pred_waypoint_logits(outputs)
                    loss_dict = loss_fn(
                        true_waypoints=true_waypoints,
                        pred_waypoint_logits=logits,
                        curr_ogm=ogm[:, :, :, -1, 0],
                    )
                    loss_value = torch.sum(sum(loss_dict.values()))

                    # update losses
                    valid_loss.update(loss_dict["observed_xe"])
                    valid_loss_occ.update(loss_dict["occluded_xe"])
                    valid_loss_flow.update(loss_dict["flow"])
                    valid_loss_warp.update(loss_dict["flow_warp_xe"])

                    pred_waypoints = self._apply_sigmoid_to_occupancy_logits(logits)
                    metrics = self.val_metric_func(
                        config, true_waypoints, pred_waypoints
                    )
                    valid_metrics.update(metrics)

                    obs_loss = valid_loss.compute() / ogm_weight
                    occ_loss = valid_loss_occ.compute() / occ_weight
                    flow_loss = valid_loss_flow.compute() / flow_weight
                    warp_loss = valid_loss_warp.compute() / flow_origin_weight

                    # log loss
                    current = (
                        self.batch_size * (batch_idx + 1) * self.dist_env.world_size
                    )
                    print(
                        f"\nobs. loss: {obs_loss:>7f}, occl. loss:  {occ_loss:>7f}, flow loss: {flow_loss:>7f}, warp loss: {warp_loss:>7f}  [{current:>5d}]",
                        flush=True,
                    )

            val_res_dict = valid_metrics.compute()
            print_metrics(val_res_dict, no_warp=False)

            if self.dist_env.rank == 0:  # global rank = 0
                LOG.info("Saving model\n-------------------------------")
                # save model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss_value,
                    },
                    f"{self.save_dir}/model_{epoch+1}.pt",
                )

    def _warpped_gt(
        self,
        gt_ogm: torch.Tensor,
        gt_occ: torch.Tensor,
        gt_flow: torch.Tensor,
        origin_flow: torch.Tensor,
    ) -> occupancy_flow_grids.WaypointGrids:
        true_waypoints = occupancy_flow_grids.WaypointGrids()

        for k in range(8):
            true_waypoints.vehicles.observed_occupancy.append(gt_ogm[:, k])
            true_waypoints.vehicles.occluded_occupancy.append(gt_occ[:, k])
            true_waypoints.vehicles.flow.append(gt_flow[:, k])
            true_waypoints.vehicles.flow_origin_occupancy.append(origin_flow[:, k])

        return true_waypoints

    def _get_pred_waypoint_logits(
        self,
        model_outputs: torch.Tensor,
    ) -> occupancy_flow_grids.WaypointGrids:
        """Slices model predictions into occupancy and flow grids."""
        pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

        # Slice channels into output predictions.
        for k in range(config.num_waypoints):
            index = k * NUM_PRED_CHANNELS
            waypoint_channels = model_outputs[
                :, :, :, index : index + NUM_PRED_CHANNELS
            ]
            pred_observed_occupancy = waypoint_channels[:, :, :, :1]
            pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
            pred_flow = waypoint_channels[:, :, :, 2:]
            pred_waypoint_logits.vehicles.observed_occupancy.append(
                pred_observed_occupancy
            )
            pred_waypoint_logits.vehicles.occluded_occupancy.append(
                pred_occluded_occupancy
            )
            pred_waypoint_logits.vehicles.flow.append(pred_flow)

        return pred_waypoint_logits

    def _apply_sigmoid_to_occupancy_logits(
        self,
        pred_waypoint_logits: occupancy_flow_grids.WaypointGrids,
    ) -> occupancy_flow_grids.WaypointGrids:
        """Converts occupancy logits with probabilities."""
        pred_waypoints = occupancy_flow_grids.WaypointGrids()
        pred_waypoints.vehicles.observed_occupancy = [
            torch.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
        ]
        pred_waypoints.vehicles.occluded_occupancy = [
            torch.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
        ]
        pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
        return pred_waypoints

    def val_metric_func(self, config, true_waypoints, pred_waypoints):
        return occupancy_flow_metrics.compute_occupancy_flow_metrics(
            config=config,
            true_waypoints=true_waypoints,
            pred_waypoints=pred_waypoints,
            no_warp=False,
        )
