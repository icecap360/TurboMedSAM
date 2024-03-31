import torch
import argparse
import copy
import os
import time
from torch import distributed as dist
import torch.nn as nn
from tqdm import tqdm
from progress.bar import Bar
from framework import import_module, setup_multi_processes, init_dist_custom, get_dist_info, logger, read_cfg_str, get_device, init_random_seed, set_random_seed, create_dataloader, set_visible_devices, dict_to_device
from copy import deepcopy
import sys 

# from framework import 

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--benchmark', type=bool, default=True)
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

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

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
    if not (cfg.compute['job_launcher']['type'] == 'none' and not torch.distributed.is_available()):
        init_dist_custom(cfg.compute['job_launcher']['dist_params']['backend'], args.local_rank, args.local_world_size )
    rank, world_size = get_dist_info()
    distributed = (world_size != 1)

    # select the device.
    if not cfg.compute['use_cpu'] and not distributed:
        gpu_id = cfg.compute["gpu_ids"][0]
        set_visible_devices(gpu_id)
    elif not cfg.compute['use_cpu']:
        assert world_size % len(cfg.compute["gpu_ids"]) == 0
        gpu_id = cfg.compute["gpu_ids"][rank % len(cfg.compute["gpu_ids"])]
        set_visible_devices(gpu_id)

    device = get_device(cfg.compute['use_cpu'])
    
    query_yes_no('Are you sure you want to procede with inference (type y for each process)?')
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
    logger.info(f'Rank{rank}: Device type: {device}')
    logger.info(f'Rank{rank}: Set random seed to {seed}')
    if not cfg.compute['use_cpu']:
        logger.info(f'Rank{rank}: GPU assigned: {gpu_id}')

    model = cfg.model
    model.init_weights()
    
    saver_cfg = deepcopy(cfg.saver)
    saver_type = saver_cfg.pop('type')
    saver = saver_type(work_dir, cfg.data['inference'], saver_cfg)

    inference_loader = create_dataloader(cfg.data,
                                        cfg.compute, 
                                        cfg.seed, distributed, "inference")
    
    model = model.to(device)
    # if distributed:
    #     # find_unused_parameters = cfg.get('find_unused_parameters', False)
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=None,
    #         broadcast_buffers=False,
    #         find_unused_parameters=False
    #         )
    #  # Synchronization of BatchNorm's buffer (running_mean and running_var) is not supported in the DDP of pytorch, which may cause the inconsistent performance of models in different ranks, so we broadcast BatchNorm's buffers of rank 0 to other ranks to avoid this.
    # if distributed and cfg.compute['broadcast_bn_buffer']:
    #     for name, module in model.named_modules():
    #         if isinstance(module,
    #                         nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
    #             dist.broadcast(module.running_var, 0)
    #             dist.broadcast(module.running_mean, 0)

    model.eval()
    time.sleep(2)  # This line can prevent deadlock problem in some multi-gpu cases
    for data in tqdm(inference_loader, position=rank, total=len(inference_loader)):
        with torch.no_grad():
            inputs = data[0]
            inputs = dict_to_device(inputs, device)
            preds = model(inputs)
            if args.save_results:
                saver.save(inputs, preds)

if __name__ == '__main__':
    args = parse_args()
    main(args)
