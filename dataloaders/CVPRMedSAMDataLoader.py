import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
from abc import ABC, abstractmethod 
from functools import partial
from packaging.version import parse
import warnings
import random
from copy import deepcopy
from torchvision import transforms
from framework import get_dist_info, worker_init_fn, digit_version, logger, ListDataLoader, BaseDataLoader
from datasets import CVPRMedSAMDataset, get_MedSAM_classes
from copy import deepcopy

def CVPRMedSAM_val_dataloader_creator(data_settings, compute_settings, seed, is_distributed,   split_type, batch_size, drop_last=False, start_idx=0):
    if is_distributed:
        rank, world_size = get_dist_info()
    else:
        rank, world_size = 0, 1
    init_fn = partial(
        worker_init_fn, 
        num_workers=compute_settings['workers_per_gpu'], 
        rank=rank,
        seed=seed) if seed is not None else None
    
    dataset_settings = data_settings[split_type]['dataset']
    dataset_type = dataset_settings.pop("type")
    sampler_settings = data_settings[split_type]["sampler"]
    sampler_type = sampler_settings.pop('type')
    
    classes = get_MedSAM_classes(dataset_settings['root_dir'],
                                        split_type)
    if dataset_settings.get('subset_classes'):
        classes = dataset_settings.pop('subset_classes')
    classes.sort(reverse=True)
    
    loaders = []
    for cat in classes:
        dataset_cat = dataset_type(
            split_type=split_type,
            subset_classes=[cat],
            **dataset_settings
        )
        sampler = sampler_type(dataset_cat, 
                            num_replicas=world_size, 
                            rank=rank,
                            seed=seed,
                            **sampler_settings)

        loader = BaseDataLoader(dataset_cat,
                        batch_size=batch_size,
                        sampler=sampler,
                        pin_memory=compute_settings["pin_memory"],
                        drop_last=drop_last,
                        workers_per_gpu=compute_settings["workers_per_gpu"], # I don't support DataParallel, I use 1 GPU in the nonparallel case so num_gpus is always 1
                        prefetch_factor=compute_settings["prefetch_factor"],
                        persistent_workers=compute_settings["persistent_workers"],
                        init_fn=init_fn,
                        name=cat
            )
        loaders.append(loader)
    return ListDataLoader(loaders)
