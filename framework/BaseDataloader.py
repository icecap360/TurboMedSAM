import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
from abc import ABC, abstractmethod 
from .BaseDataSamplers import ClassBalancedSampler
from .Distributed import get_dist_info
from functools import partial
from .Logger import logger
from packaging.version import parse
import warnings
import random
from copy import deepcopy
from torchvision import transforms

def create_dataloader(data_settings, compute_settings, seed, is_distributed, split_type):
    
    if not (torch.__version__ != 'parrots'
            and digit_version(torch.__version__) >= digit_version('1.7.0')):
        logger.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')
    
    if is_distributed:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = compute_settings["samples_per_gpu"]
        num_workers = compute_settings["workers_per_gpu"]
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        # num_gpus = len(compute_settings["gpu_ids"])
        # batch_size = num_gpus * compute_settings["samples_per_gpu"]
        # num_workers = num_gpus * compute_settings["workers_per_gpu"]
        # I don't support DataParallel, I use 1 GPU in the nonparallel case so num_gpus is always 1
        batch_size = compute_settings["samples_per_gpu"]
        num_workers = compute_settings["workers_per_gpu"]
    
    dataloader_settings = data_settings[split_type]['dataloader_creator']
    dataloader_creator = dataloader_settings.pop('type')
    return dataloader_creator(data_settings = data_settings,
                           compute_settings=compute_settings,
                           seed = seed,
                           is_distributed = is_distributed,
                           split_type = split_type,
                           batch_size = batch_size,
                           **dataloader_settings)

class BaseDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, sampler, pin_memory, workers_per_gpu, prefetch_factor, persistent_workers, init_fn, name='Basic', drop_last=False):
        self.name = name
        super().__init__(dataset, 
                            batch_size=batch_size,
                            sampler=sampler,
                            pin_memory=pin_memory,
                            drop_last=drop_last,
                            num_workers=workers_per_gpu, # I don't support DataParallel, I use 1 GPU in the nonparallel case so num_gpus is always 1
                            prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers,
                            worker_init_fn=init_fn,
                        #  collate_fn=partial(collate, samples_per_gpu=compute_settings["samples_per_gpu"],)
        )
    def get_name(self):
        return self.name
    
def basic_dataloader_creator(data_settings, compute_settings, seed, is_distributed, split_type, batch_size, drop_last=False,name='Basic'):
    if is_distributed:
        rank, world_size = get_dist_info()
    else:
        rank, world_size = 0, 1
    init_fn = partial(
        worker_init_fn, 
        num_workers=compute_settings['workers_per_gpu'], 
        rank=rank,
        seed=seed) if seed is not None else None
    
    dataset_settings = deepcopy(data_settings[split_type]['dataset'])
    dataset_type = dataset_settings.pop("type")
    dataset = dataset_type( split_type = split_type, 
                        **dataset_settings
            )
    
    sampler_settings = deepcopy(data_settings[split_type]["sampler"])
    sampler_type = sampler_settings.pop('type')
    sampler = sampler_type(dataset, 
                        num_replicas=world_size, 
                        rank=rank,
                        seed=seed,
                        **sampler_settings)

    return BaseDataLoader(dataset, 
                        batch_size=batch_size,
                        sampler=sampler,
                        pin_memory=compute_settings["pin_memory"],
                        workers_per_gpu=compute_settings["workers_per_gpu"], # I don't support DataParallel, I use 1 GPU in the nonparallel case so num_gpus is always 1
                        prefetch_factor=compute_settings["prefetch_factor"],
                        persistent_workers=compute_settings["persistent_workers"],
                        init_fn=init_fn,
                        name=name,
                        drop_last=drop_last
                    #  collate_fn=partial(collate, samples_per_gpu=compute_settings["samples_per_gpu"],)
    )

class ListDataLoader():
    def __init__(self, loaders):
        self.loaders = loaders
        self.current_index = 0
    def __iter__(self):
        self.current_index = 0
        return self
    def __len__(self):
        return len(self.loaders)
    def __next__(self):
        if self.current_index < len(self.loaders):
            loader = self.loaders[self.current_index]
            self.current_index += 1
            return loader
        else:
            raise StopIteration

def basic_val_dataloader_creator(data_settings, compute_settings, seed, is_distributed, split_type, batch_size, name='Basic', drop_last=False):
        loaders = [basic_dataloader_creator(data_settings, compute_settings, seed, is_distributed, split_type, batch_size, name=name, drop_last=drop_last )]
        return ListDataLoader(loaders)

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    
def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                logger.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)