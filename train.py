
import os
import torch
import torch.nn as nn
import copy
import numpy as np
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from google.protobuf import text_format
from tfrecord.torch.dataset import MultiTFRecordDataset
import occupancy_flow_grids

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from loss import OGMFlow_loss

from strajNet import STrajNet
from dataset import DistributedMultiTFRecordDataset

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

REPLICA = 1

# Hyper parameters
NUM_PRED_CHANNELS = 4
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
SAVE_DIR = "/work/vita/casademo/weights/torch"

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
    origin_flow: torch.Tensor,) -> occupancy_flow_grids.WaypointGrids:

    true_waypoints = occupancy_flow_grids.WaypointGrids()

    for k in range(8):
        true_waypoints.vehicles.observed_occupancy.append(gt_ogm[:,k])
        true_waypoints.vehicles.occluded_occupancy.append(gt_occ[:,k])
        true_waypoints.vehicles.flow.append(gt_flow[:,k])
        true_waypoints.vehicles.flow_origin_occupancy.append(origin_flow[:,k])
    
    return true_waypoints

def _get_pred_waypoint_logits(
    model_outputs: torch.Tensor) -> occupancy_flow_grids.WaypointGrids:
    """Slices model predictions into occupancy and flow grids."""
    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

    # Slice channels into output predictions.
    for k in range(config.num_waypoints):
        index = k * NUM_PRED_CHANNELS
        waypoint_channels = model_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
        pred_observed_occupancy = waypoint_channels[:, :, :, :1]
        pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
        pred_flow = waypoint_channels[:, :, :, 2:]
        pred_waypoint_logits.vehicles.observed_occupancy.append(
            pred_observed_occupancy)
        pred_waypoint_logits.vehicles.occluded_occupancy.append(
            pred_occluded_occupancy)
        pred_waypoint_logits.vehicles.flow.append(pred_flow)

    return pred_waypoint_logits

def parse_record(record):
    features = copy.deepcopy(record)
    new_dict = {}
    # print(features['actors'].tostring())
    new_dict['centerlines'] = torch.reshape(torch.frombuffer(features['centerlines'],dtype=torch.float64),[256,10,7]).to(torch.float32)

    new_dict['actors'] = torch.reshape(torch.frombuffer(features['actors'], dtype=torch.float64), [48,11,8]).to(torch.float32)
    new_dict['occl_actors'] = torch.reshape(torch.frombuffer(features['occl_actors'], dtype=torch.float64), [16,11,8]).to(torch.float32)

    new_dict['gt_flow'] = torch.reshape(torch.frombuffer(features['gt_flow'], dtype=torch.float32), [8,512,512,2])[:,128:128+256,128:128+256,:]
    new_dict['origin_flow'] = torch.reshape(torch.frombuffer(features['origin_flow'], dtype=torch.float32), [8,512,512,1])[:,128:128+256,128:128+256,:]

    new_dict['ogm'] = torch.reshape(torch.frombuffer(features['ogm'], dtype=torch.bool), [512,512,11,2]).to(torch.float32)

    new_dict['gt_obs_ogm'] = torch.reshape(torch.frombuffer(features['gt_obs_ogm'], dtype=torch.bool), [8,512,512,1]).to(torch.float32)[:,128:128+256,128:128+256,:]
    new_dict['gt_occ_ogm'] = torch.reshape(torch.frombuffer(features['gt_occ_ogm'], dtype=torch.bool), [8,512,512,1]).to(torch.float32)[:,128:128+256,128:128+256,:]
    
    new_dict['map_image'] = (torch.reshape(torch.frombuffer(features['map_image'], dtype=torch.int8), [256,256,3]).to(torch.float32) / 256)
    new_dict['vec_flow'] = torch.reshape(torch.frombuffer(features['vec_flow'], dtype=torch.float32), [512,512,2])

    return new_dict



def setup(gpu_id):
    cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
    model = STrajNet(cfg,actor_only=True,sep_actors=False, fg_msa=True, fg=True).to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
    loss_fn = OGMFlow_loss(config, replica=REPLICA,no_use_warp=False,use_pred=False,use_gt=True,
    ogm_weight=ogm_weight, occ_weight=occ_weight,flow_origin_weight=flow_origin_weight,flow_weight=flow_weight,use_focal_loss=False)
    optimizer = torch.optim.NAdam(model.parameters(), lr=LR) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(30438*1.5), T_mult=1)
    return model, loss_fn, optimizer, scheduler

def get_dataloader(gpu_id, world_size):
    tfrecord_pattern = "/work/vita/datasets/waymo110/preprocessed_data/train/{}.tfrecords"
    files = os.listdir('/work/vita/datasets/waymo110/preprocessed_data/train')

    splits = {file.split(".")[0]:1 for file in files}

    dataset = DistributedMultiTFRecordDataset(
                            tfrecord_pattern,
                            index_pattern=None,
                            splits=splits,
                            compression_type="gzip",
                            transform=parse_record,
                            shuffle_queue_size=64,
                            infinite=False,
                            gpu_id=gpu_id,
                            world_size=world_size)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    return train_loader

def model_training(gpu_id, world_size):

    ddp_setup(gpu_id, world_size)

    model, loss_fn, optimizer, scheduler = setup(gpu_id)
    train_loader = get_dataloader(gpu_id, world_size)

    for epoch in range(EPOCHS):
        print(f"[GPU{gpu_id}] Epoch {epoch+1}\n-------------------------------")
        size = 487008 // world_size
        train_loss = []
        train_loss_occ = []
        train_loss_flow = []
        train_loss_warp = []
        for batch, data in enumerate(train_loader):
            map_img = data['map_image']
            centerlines = data['centerlines']
            actors = data['actors']
            occl_actors = data['occl_actors']

            ogm = data['ogm']
            gt_obs_ogm = data['gt_obs_ogm']
            gt_occ_ogm = data['gt_occ_ogm']
            gt_flow = data['gt_flow']
            origin_flow = data['origin_flow']

            flow = data['vec_flow']

            true_waypoints = _warpped_gt(gt_ogm=gt_obs_ogm,gt_occ=gt_occ_ogm,gt_flow=gt_flow,origin_flow=origin_flow)

            outputs = model(ogm,map_img,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)
            logits = _get_pred_waypoint_logits(outputs)
            loss_dict = loss_fn(true_waypoints=true_waypoints,pred_waypoint_logits=logits,curr_ogm=ogm[:,:,:,-1,0])
            loss_value = torch.sum(sum(loss_dict.values()))
       
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(torch.mean(loss_dict['observed_xe']*REPLICA))
            train_loss_occ.append(torch.mean(loss_dict['occluded_xe']*REPLICA))
            train_loss_flow.append(torch.mean(loss_dict['flow']*REPLICA))
            train_loss_warp.append(torch.mean(loss_dict['flow_warp_xe']*REPLICA))

            obs_loss = torch.mean(torch.stack(train_loss))/ogm_weight
            occ_loss = torch.mean(torch.stack(train_loss_occ))/occ_weight
            flow_loss = torch.mean(torch.stack(train_loss_flow))/flow_weight
            warp_loss = torch.mean(torch.stack(train_loss_warp))/flow_origin_weight
            
            current = batch * BATCH_SIZE
            print(f"[GPU{gpu_id}] obs. loss: {obs_loss:>7f}, occl. loss:  {occ_loss:>7f}, flow loss: {flow_loss:>7f}, warp loss: {warp_loss:>7f}  [{current:>5d}/{size:>5d}]")

            

        if (gpu_id == 0):
            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_value,
            }, f'{SAVE_DIR}/model_{epoch+1}.pt')
    
    destroy_process_group()

if __name__ == "__main__":

    

    world_size = torch.cuda.device_count()

    mp.spawn(model_training, args=[world_size], nprocs=world_size)



    # model_training(train_loader)
    # data = next(iter(loader))
    # print(data)