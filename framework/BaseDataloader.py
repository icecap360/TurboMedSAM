import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
from abc import ABC, abstractmethod 
from BaseDataSamplers import ClassBalancedSampler
from Registry import Registry, build_from_cfg


def CreatePytorchDataloaders(data_settings, split_type, sampler=None):
    data_dir = data_settings["data_root"]
    dataset_type = data_settings["dataset_class"]
    dataset = dataset_type(data_dir, split_type, 
                                 transform=data_settings[split_type+"_pipeline"], 
                                 transform_target=data_settings[split_type+"_target_pipeline"])
    data_loader = DataLoader(dataset, 
                             batch_size=data_settings["batch_size"],
                             shuffle=True,
                             sampler=sampler,
                             pin_memory=data_settings["pin_memory"],
                             drop_last=data_settings["drop_last"],
                             num_workers=data_settings["num_workers"],
                             prefetch_factor=data_settings["prefetch_factor"],
                             persistent_workers=data_settings["persistent_workers"]
                             )


