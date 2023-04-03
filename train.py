
import os
import torch
import torch.nn as nn
import math
import copy
import numpy as np
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from google.protobuf import text_format
import occupancy_flow_grids

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from loss import OGMFlow_loss

from strajNet import STrajNet

from torchmetrics import MeanMetric
import occu_metric as occupancy_flow_metrics
from metrics import OGMFlowMetrics, print_metrics

from filesDataset import FilesDataset

from time import time
from tqdm import tqdm
import sys

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

# Parameters
SAVE_DIR = "./weights"
FILES_DIR = "./preprocessed_data"
CHECKPOINT_PATH = None

# Hyper parameters
NUM_PRED_CHANNELS = 4
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

# loss weights
ogm_weight = 1000.0
occ_weight = 1000.0
flow_origin_weight = 1000.0
flow_weight = 1.0

# torch.autograd.set_detect_anomaly(True)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _warpped_gt(
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
    model_outputs: torch.Tensor,
) -> occupancy_flow_grids.WaypointGrids:
    """Slices model predictions into occupancy and flow grids."""
    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

    # Slice channels into output predictions.
    for k in range(config.num_waypoints):
        index = k * NUM_PRED_CHANNELS
        waypoint_channels = model_outputs[:, :, :, index : index + NUM_PRED_CHANNELS]
        pred_observed_occupancy = waypoint_channels[:, :, :, :1]
        pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
        pred_flow = waypoint_channels[:, :, :, 2:]
        pred_waypoint_logits.vehicles.observed_occupancy.append(pred_observed_occupancy)
        pred_waypoint_logits.vehicles.occluded_occupancy.append(pred_occluded_occupancy)
        pred_waypoint_logits.vehicles.flow.append(pred_flow)

    return pred_waypoint_logits


def _apply_sigmoid_to_occupancy_logits(
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


def val_metric_func(config, true_waypoints, pred_waypoints):
    return occupancy_flow_metrics.compute_occupancy_flow_metrics(
        config=config,
        true_waypoints=true_waypoints,
        pred_waypoints=pred_waypoints,
        no_warp=False,
    )


def parse_record(features):
    """
    Convert features to the right types
    """

    features["centerlines"] = features["centerlines"].to(torch.float32)

    features["actors"] = features["actors"].to(torch.float32)
    features["occl_actors"] = features["occl_actors"].to(torch.float32)

    features["ogm"] = features["ogm"].to(torch.float32)

    features["map_image"] = features["map_image"].to(torch.float32) / 256
    features["vec_flow"] = features["vec_flow"]

    features["gt_flow"] = features["gt_flow"][:, 128 : 128 + 256, 128 : 128 + 256, :]
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


def setup(gpu_id):
    """
    Setup model, DDP, loss, optimizer and scheduler
    """
    cfg = dict(
        input_size=(512, 512),
        window_size=8,
        embed_dim=96,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
    )
    model = STrajNet(cfg, actor_only=True, sep_actors=False, fg_msa=True, fg=True).to(
        gpu_id
    )
    model = DDP(model, device_ids=[gpu_id])
    loss_fn = OGMFlow_loss(
        config,
        no_use_warp=False,
        use_pred=False,
        use_gt=True,
        ogm_weight=ogm_weight,
        occ_weight=occ_weight,
        flow_origin_weight=flow_origin_weight,
        flow_weight=flow_weight,
        use_focal_loss=True,
    )
    optimizer = torch.optim.NAdam(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(30438*1.5), T_mult=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
    return model, loss_fn, optimizer, scheduler


def get_dataloader(gpu_id, world_size):
    """
    Get training and validation dataloaders
    """

    dataset = FilesDataset(path=FILES_DIR + "/train_numpy", transform=parse_record)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # done by the sampler
        pin_memory=True,
        num_workers=4,
        sampler=DistributedSampler(dataset),
    )

    val_dataset = FilesDataset(path=FILES_DIR + "/val_numpy", transform=parse_record)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # done by the sampler
        pin_memory=True,
        num_workers=4,
        sampler=DistributedSampler(val_dataset),
    )

    return train_loader, val_loader


def model_training(gpu_id, world_size):
    """
    Model training and validation
    """

    ddp_setup(gpu_id, world_size)

    model, loss_fn, optimizer, scheduler = setup(gpu_id)
    train_loader, val_loader = get_dataloader(gpu_id, world_size)

    if CHECKPOINT_PATH is not None:
        # if checkpoint path given, load weights
        map_location = {"cuda:%d" % 0: "cuda:%d" % gpu_id}
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        continue_ep = checkpoint["epoch"] + 1
        if gpu_id == 0:
            print(f"Continue_training...ep:{continue_ep+1}")
    else:
        continue_ep = 0

    train_size = 0
    val_size = 0
    for epoch in range(EPOCHS):
        if epoch < continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch + 1, EPOCHS))
            continue

        # TRAINING
        if gpu_id == 0:
            print(f"Epoch {epoch+1}\n-------------------------------")
        size = train_size or 0
        train_loss = MeanMetric().to(gpu_id)
        train_loss_occ = MeanMetric().to(gpu_id)
        train_loss_flow = MeanMetric().to(gpu_id)
        train_loss_warp = MeanMetric().to(gpu_id)

        model.train()

        train_loader.sampler.set_epoch(epoch)

        loop = (
            tqdm(
                enumerate(train_loader),
                total=math.ceil(size / (BATCH_SIZE * world_size)),
            )
            if gpu_id == 0
            else enumerate(train_loader)
        )
        for batch, data in loop:
            # inputs: will automatically be put on right device when passed to model
            map_img = data["map_image"]
            centerlines = data["centerlines"]
            actors = data["actors"]
            occl_actors = data["occl_actors"]
            ogm = data["ogm"]
            flow = data["vec_flow"]

            # ground truths directly passed to device for loss / metrics
            gt_obs_ogm = data["gt_obs_ogm"].to(gpu_id)
            gt_occ_ogm = data["gt_occ_ogm"].to(gpu_id)
            gt_flow = data["gt_flow"].to(gpu_id)
            origin_flow = data["origin_flow"].to(gpu_id)

            # forward pass
            outputs = model(
                ogm, map_img, obs=actors, occ=occl_actors, mapt=centerlines, flow=flow
            )

            # compute loss
            true_waypoints = _warpped_gt(
                gt_ogm=gt_obs_ogm,
                gt_occ=gt_occ_ogm,
                gt_flow=gt_flow,
                origin_flow=origin_flow,
            )
            logits = _get_pred_waypoint_logits(outputs)
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

            # update losses
            train_loss.update(loss_dict["observed_xe"])
            train_loss_occ.update(loss_dict["occluded_xe"])
            train_loss_flow.update(loss_dict["flow"])
            train_loss_warp.update(loss_dict["flow_warp_xe"])

            obs_loss = train_loss.compute() / ogm_weight
            occ_loss = train_loss_occ.compute() / occ_weight
            flow_loss = train_loss_flow.compute() / flow_weight
            warp_loss = train_loss_warp.compute() / flow_origin_weight

            if gpu_id == 0:
                # print training losses
                batch_size = data["ogm"].size(dim=0)
                current = (batch * BATCH_SIZE + batch_size) * world_size
                print(
                    f"\nobs. loss: {obs_loss:>7f}, occl. loss:  {occ_loss:>7f}, flow loss: {flow_loss:>7f}, warp loss: {warp_loss:>7f}  [{current:>5d}/{size:>5d}]",
                    flush=True,
                )

        scheduler.step()

        # VALIDATION
        if gpu_id == 0:
            train_size = current
            print(f"Validation\n-------------------------------")
        size = val_size or 0
        valid_loss = MeanMetric().to(gpu_id)
        valid_loss_occ = MeanMetric().to(gpu_id)
        valid_loss_flow = MeanMetric().to(gpu_id)
        valid_loss_warp = MeanMetric().to(gpu_id)

        valid_metrics = OGMFlowMetrics(gpu_id, no_warp=False)

        model.eval()
        with torch.no_grad():
            loop = (
                tqdm(
                    enumerate(val_loader),
                    total=math.ceil(size / (BATCH_SIZE * world_size)),
                )
                if gpu_id == 0
                else enumerate(val_loader)
            )
            for batch, data in loop:
                # inputs: will automatically be put on right device when passed to model
                map_img = data["map_image"]
                centerlines = data["centerlines"]
                actors = data["actors"]
                occl_actors = data["occl_actors"]
                ogm = data["ogm"]
                flow = data["vec_flow"]

                # ground truths directly put on device for loss / metrics
                gt_obs_ogm = data["gt_obs_ogm"].to(gpu_id)
                gt_occ_ogm = data["gt_occ_ogm"].to(gpu_id)
                gt_flow = data["gt_flow"].to(gpu_id)
                origin_flow = data["origin_flow"].to(gpu_id)

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
                true_waypoints = _warpped_gt(
                    gt_ogm=gt_obs_ogm,
                    gt_occ=gt_occ_ogm,
                    gt_flow=gt_flow,
                    origin_flow=origin_flow,
                )
                logits = _get_pred_waypoint_logits(outputs)
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

                pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)
                metrics = val_metric_func(config, true_waypoints, pred_waypoints)
                valid_metrics.update(metrics)

                obs_loss = valid_loss.compute() / ogm_weight
                occ_loss = valid_loss_occ.compute() / occ_weight
                flow_loss = valid_loss_flow.compute() / flow_weight
                warp_loss = valid_loss_warp.compute() / flow_origin_weight

                if gpu_id == 0:
                    # print validation losses
                    batch_size = data["ogm"].size(dim=0)
                    current = (batch * BATCH_SIZE + batch_size) * world_size
                    print(
                        f"\nobs. loss: {obs_loss:>7f}, occl. loss:  {occ_loss:>7f}, flow loss: {flow_loss:>7f}, warp loss: {warp_loss:>7f}  [{current:>5d}/{size:>5d}]",
                        flush=True,
                    )

        val_res_dict = valid_metrics.compute()

        if gpu_id == 0:
            val_size = current

            # print validation metrics
            print_metrics(val_res_dict, no_warp=False)

            # save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss_value,
                },
                f"{SAVE_DIR}/model_{epoch+1}.pt",
            )

    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    mp.spawn(model_training, args=[world_size], nprocs=world_size)
