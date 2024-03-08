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

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if cfg.compute["use_cpu"]:
        raise Exception('Training for CPU is not supported')
    
    # init distributed env first, since logger depends on the dist info.
    if cfg.compute['job_launcher']['type'] is None:
        distributed = False
    else:
        distributed = True
        init_dist(cfg.compute['job_launcher']['dist_params']['backend'], args.local_rank, args.local_world_size )
        
    # init distributed env first, since logger depends on the dist info.
    if not distributed:
        gpu_id = cfg.compute["gpu_ids"][0]
        set_visible_devices(gpu_id)
    else:
        rank, world_size = get_dist_info()
        assert world_size == len(cfg.compute["gpu_ids"])
        gpu_id = cfg.compute["gpu_ids"][rank % world_size]
        set_visible_devices(gpu_id)

    device = get_device(cfg.compute['use_cpu'])
        
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Create the logger
    logger.init(cfg.work_dir, cfg.exp_name, read_cfg_str(abs_config_path))
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'GPU assigned: {gpu_id}')
    logger.info(f'Available GPUs: {cfg.compute["gpu_ids"]}')
    logger.info(f'Device type: {device}')

    # Set random seeds
    seed = init_random_seed(cfg.seed, device=device)
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed, deterministic=False)
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
            device_ids=[gpu_id],
            broadcast_buffers=False,
            find_unused_parameters=False
            )
    
    runner = EpochBasedRunner(
        model=model,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
        metric=cfg.metric,
        lr_scheduler=cfg.lr_scheduler,
        device=device,
        work_dir=cfg.work_dir,
        logger=logger, 
        grad_clip=cfg.grad_clip,
        distributed=distributed,
        use_cpu=cfg.compute['use_cpu'],
        val_freq=cfg.val_freq,
        save_freq=cfg.save_freq,
        save_optimizer=True,
        max_epochs=cfg.max_epochs,
        broadcast_bn_buffer=cfg.compute['broadcast_bn_buffer']
        )
    
    runner.run(train_loader, 
               val_loader)
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
