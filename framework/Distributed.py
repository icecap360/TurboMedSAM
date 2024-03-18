import torch
import os
import platform
import warnings
import cv2
import torch.multiprocessing as mp
import functools
import subprocess
from collections import OrderedDict
import numpy as np
from torch import distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,_unflatten_dense_tensors)
import random
from getpass import getuser
from socket import gethostname

def init_dist(backend, rank, world_size):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # if launcher == 'pytorch':
    #     _init_dist_pytorch(backend, **kwargs)
    # elif launcher == 'mpi':
    #     _init_dist_mpi(backend, **kwargs)
    # elif launcher == 'slurm':
    #     _init_dist_slurm(backend, **kwargs)
    # else:
    #     raise ValueError(f'Invalid launcher type: {launcher}')
    if os.environ.get('MASTER_ADDR') is None:
        # raise Exception('Missing MASTER_ADDR')
        os.environ["MASTER_ADDR"] = "localhost"
    if os.environ.get('MASTER_PORT') is None:
        # raise Exception('Missing MASTER_PORT')
        os.environ["MASTER_PORT"] = "29501"
    if not os.environ.get('RANK') is None:
        rank = int(os.environ.get('RANK'))
    if not os.environ.get('WORLD_SIZE') is None:
        world_size = int(os.environ.get('WORLD_SIZE'))
    dist.init_process_group(backend=backend, 
                            rank=rank,
                            world_size=world_size)

def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def set_visible_devices(device=None):
    torch.cuda.set_device(device)
        
def get_device(use_cpu = False):
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {
        'npu': hasattr(torch, 'npu') and torch.npu.is_available(),
        'cuda': torch.cuda.is_available(),
        'mlu': hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    if use_cpu:
        return 'cpu'
    else:
        return device_list[0] if len(device_list) >= 1 else 'cpu'

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def get_dist_info():
    # if TORCH_VERSION < '1.0':
    #     initialized = dist._initialized
    # else:
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def get_host_info() -> str:
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host

def setup_multi_processes(compute_cfg):
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = compute_cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can change '
                f'this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = compute_cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    workers_per_gpu = compute_cfg.get('workers_per_gpu', 1)

    if 'OMP_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)
