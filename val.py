import torch
import argparse
import copy
import os
from torch import distributed as dist
from framework import import_module, setup_multi_processes, get_dist_info, logger, read_cfg_str, get_device, init_random_seed, set_random_seed, create_dataloader, set_visible_devices, EpochBasedRunner, IterBasedRunner, create_optimizer, create_object_from_params
import framework
import traceback
import sys 

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
    
    work_dir = os.path.join(cfg.work_dir, cfg.exp_name)
    os.makedirs(work_dir, exist_ok=True)

    # Set random seeds
    seed = init_random_seed(cfg.seed, device=device)
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed
    
    # Create the logger
    if rank == 0:
        logger.init_message(work_dir, cfg.exp_name, read_cfg_str(abs_config_path))
        logger.info_and_print(f'Distributed training: {distributed}')
        logger.info_and_print(f'Available GPUs: {cfg.compute["gpu_ids"]}')
    logger.info(f'Rank{rank}: GPU assigned: {gpu_id}')
    logger.info(f'Rank{rank}: Device type: {device}')
    logger.info(f'Rank{rank}: Set random seed to {seed}')

    model = cfg.model
    model.init_weights()
    
    model = model.to(device)
        
    cfg.optimizer['optimizer'] = create_optimizer(cfg.optimizer, model)
    cfg.lr_scheduler = create_object_from_params(cfg.lr_scheduler, optimizer=cfg.optimizer['optimizer'])
    
    if cfg.runner['type'].lower() == 'epoch':
        runner = EpochBasedRunner(
            model=model,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
            metric=cfg.metric,
            lr_scheduler=cfg.lr_scheduler,
            work_dir=work_dir,
            logger=logger, 
            distributed=distributed,
            device=device,
            compute=cfg.compute,
            runner=cfg.runner,
            save_optimizer=True,
            )
    else:
        runner = IterBasedRunner(
            model=model,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
            metric=cfg.metric,
            lr_scheduler=cfg.lr_scheduler,
            work_dir=work_dir,
            logger=logger, 
            distributed=distributed,
            device=device,
            compute=cfg.compute,
            runner=cfg.runner,
            save_optimizer=True,
            )
    
    if not os.path.isabs(runner.checkpoint_path):
        runner.checkpoint_path = os.path.join(runner.work_dir, runner.checkpoint_path) 
    assert os.path.exists(runner.checkpoint_path), 'Checkpoint must be a valid path'
    runner.resume(runner.checkpoint_path, runner.distributed, map_location=runner.device, resume_lr_scheduler=True)
    
    val_loader = create_dataloader(cfg.data,
                                cfg.compute, 
                                cfg.seed, distributed,  "val")
    
    for loader in val_loader:
        runner.val(loader, runner._epoch)

if __name__ == '__main__':
    args = parse_args()
    main(args)
