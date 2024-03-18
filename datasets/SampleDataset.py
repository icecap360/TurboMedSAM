import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
import torch.distributed as dist
from framework import BaseDataset

class SampleDataset(BaseDataset):
    """Face Landmarks dataset."""

    def __init__(self, split_type, pipeline=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(split_type, pipeline, input_transform, target_transform)
        
        if split_type is 'train':
            self.data = np.linspace(0, 999, 1000) + np.random.normal(scale=0.1)
            self.label = np.linspace(0, 999, 1000) * 10
        elif split_type is 'val':
            self.data = np.linspace(1000, 1199, 200) + np.random.normal(scale=0.1)
            self.label = np.linspace(1000, 1199, 200) * 10
        elif split_type is 'test':
            self.data = np.linspace(2000, 2199, 200) + np.random.normal(scale=0.1)
            self.label = np.linspace(2000, 2199, 200) * 10
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform_databatch({'input': np.float32(self.data[idx])},
                                        {'output': np.float32(self.label[idx])}, {'idx': idx})
    
    def get_cat2indices(self):
        raise NotImplementedError()

    def get_categories(self):
        raise NotImplementedError()
    

def sample_pipeline(inputs, meta):
    new_inputs = {}
    for key, value in inputs.items():
        if type(value) in (np.array, float):
            new_inputs[key] = torch.tensor(value)
        else:
            # If the value is not a tensor, keep it as is
            new_inputs[key] = value
    new_inputs['meta'] = meta
    return new_inputs