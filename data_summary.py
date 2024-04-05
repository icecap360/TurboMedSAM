import torch
import argparse
import copy
import os
from torch import distributed as dist
from framework import import_module, setup_multi_processes, get_dist_info, logger, read_cfg_str, get_device, init_random_seed, set_random_seed, create_dataloader, set_visible_devices, EpochBasedRunner, IterBasedRunner, create_optimizer, create_object_from_params
import framework
import traceback
import sys 
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local_world_size', type=int, default = 1)
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args



def main(args):
    abs_config_path = os.path.join("configs", args.config)
    cfg = import_module(os.path.basename(args.config), 
                        abs_config_path)
    
    # set multi-process settings
    setup_multi_processes(cfg.compute)

    # set cudnn_benchmark
    if cfg.compute.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # init distributed env first, since logger depends on the dist info.
    if not (cfg.compute['job_launcher']['type'] == 'none' and not torch.distributed.is_available()) and not torch.distributed.is_initialized():
        framework.init_dist_custom(cfg.compute['job_launcher']['dist_params']['backend'], args.local_rank, args.local_world_size )
    rank, world_size = get_dist_info()
    distributed = (world_size != 1)

    # select the gpu device.
    if not distributed:
        gpu_id = cfg.compute["gpu_ids"][0]
        set_visible_devices(gpu_id)
    else:
        assert world_size == len(cfg.compute["gpu_ids"])
        gpu_id = cfg.compute["gpu_ids"][rank % world_size]
        set_visible_devices(gpu_id)

    device = get_device(cfg.compute['use_cpu'])
    logger.info(f'Rank{rank}: GPU assigned: {gpu_id}')
    logger.info(f'Rank{rank}: Device type: {device}')
    
    train_loader = create_dataloader(cfg.data,
                                        cfg.compute, 
                                        cfg.seed, distributed, "train")
    sum_means = np.array([0.,0.,0.])
    sum_stds = np.array([0.,0.,0.])
    log_freq = 50
    n_samples = 0
    for i, data_batch in enumerate(train_loader):
        image = data_batch[0]['image']
        n_samples += 1
        sum_means += torch.mean(image, dim=(0, 2,3)).detach().numpy()
        sum_stds += torch.std(image, dim=(0, 2,3)).detach().numpy()
        if i % log_freq == 0:
            print(str(i)+'/'+str(len(train_loader)), sum_means/n_samples, sum_stds/n_samples)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
