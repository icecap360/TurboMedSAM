import torch
import argparse
import copy
import os
from torch import distributed as dist
from framework import import_module, setup_multi_processes, init_dist, get_dist_info, logger, read_cfg_str, get_device, init_random_seed, set_random_seed, CreatePytorchDataloaders, set_visible_devices, EpochBasedRunner

# from framework import 


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
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
    parser.add_argument(
        '--gpu-id',
        type=list,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    return args



def main(local_rank, local_world_size):
    
    args = parse_args()

    cfg = import_module(args.config)
    
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.compute.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.gpu_id is not None:
        cfg.compute["gpu_id"] = args.gpu_id
    
    if cfg.compute["use_cpu"]:
        raise Exception('Training for CPU is not supported')
    
    # init distributed env first, since logger depends on the dist info.
    if cfg.compute['job_launcher']['type'] is None:
        distributed = False
    else:
        distributed = True
        init_dist(cfg.dist_params['backend'], local_rank, local_world_size )
        
    # init distributed env first, since logger depends on the dist info.
    if not distributed:
        set_visible_devices(cfg.compute["gpu_id"])
        gpu_number = cfg.compute["gpu_id"]
    else:
        rank, world_size = get_dist_info()
        assert local_world_size == len(cfg.compute["available_gpus"])
        gpu_number = cfg.compute["available_gpus"][rank % world_size]
        set_visible_devices(gpu_number)

    device = get_device()
        
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Create the logger
    logger.init(cfg.work_dir, cfg.exp_name, read_cfg_str(args.config))
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'GPU assigned: {gpu_number}')
    logger.info(f'Available GPUs: {cfg.compute["available_gpus"]}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(f'Device type: {cfg.compute["device"]}')

    # Set random seeds
    seed = init_random_seed(cfg.seed, device=device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed

    model = cfg.model
    model.init_weights()
    
    train_loader = CreatePytorchDataloaders(cfg.data,
                                            cfg.compute, 
                                            cfg.seed, distributed, "train")
    val_loader = CreatePytorchDataloaders(cfg.data,
                                          cfg.compute, 
                                          cfg.seed, distributed,  "val")
    
    model = model.to(device)
    if distributed:
        # find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device,
            broadcast_buffers=False,
            find_unused_parameters=False
            )
    
    runner = EpochBasedRunner(
        model,
        cfg.optimizer,
        cfg.work_dir,
        logger, 
        'train',
        max_epochs=cfg.max_epochs,
        val_freq=1,
        save_freq=1,
        save_optimizer=True,
        )
    
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
