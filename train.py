import torch
import argparse
import copy
import os
from torch import distributed as dist
from framework import import_module, setup_multi_processes, get_dist_info, logger, read_cfg_str, get_device, init_random_seed, set_random_seed, create_dataloader, set_visible_devices, EpochBasedRunner, IterBasedRunner

import framework


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

    if cfg.compute["use_cpu"]:
        raise Exception('Training for CPU is not supported')
    
    # init distributed env first, since logger depends on the dist info.
    if not (cfg.compute['job_launcher']['type'] == 'none' and not torch.distributed.is_available()):
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
    
    train_loader = create_dataloader(cfg.data,
                                            cfg.compute, 
                                            cfg.seed, distributed, "train")
    val_loader = create_dataloader(cfg.data,
                                          cfg.compute, 
                                          cfg.seed, distributed,  "val")
    
    model = model.to(device)
    if distributed:
        # find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,
            broadcast_buffers=False,
            find_unused_parameters=False
            )
    
    if cfg.runner_type.lower() == 'epoch':
        runner = EpochBasedRunner(
            model=model,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
            metric=cfg.metric,
            lr_scheduler=cfg.lr_scheduler,
            device=device,
            work_dir=work_dir,
            logger=logger, 
            grad_clip=cfg.grad_clip,
            distributed=distributed,
            use_cpu=cfg.compute['use_cpu'],
            val_freq_epoch=cfg.val_freq_epoch,
            save_freq=cfg.save_freq_epoch,
            save_optimizer=True,
            max_epochs=cfg.max_epochs,
            broadcast_bn_buffer=cfg.compute['broadcast_bn_buffer'],
            batch_size=cfg.compute['samples_per_gpu']
            )
    else:
        runner = IterBasedRunner(
            model=model,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
            metric=cfg.metric,
            lr_scheduler=cfg.lr_scheduler,
            device=device,
            work_dir=work_dir,
            logger=logger, 
            grad_clip=cfg.grad_clip,
            distributed=distributed,
            use_cpu=cfg.compute['use_cpu'],
            val_freq_iter=cfg.val_freq_iter,
            save_freq=cfg.save_freq_iter,
            save_optimizer=True,
            max_iters=cfg.max_iters,
            broadcast_bn_buffer=cfg.compute['broadcast_bn_buffer'],
            batch_size=cfg.compute['samples_per_gpu']
            )
    
    runner.run(train_loader, 
               val_loader)
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
