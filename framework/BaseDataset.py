import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
from abc import ABC, abstractmethod 

class BaseDataset(ABC, Dataset):

    @abstractmethod
    def __init__(self, split_type, pipeline=None, input_transform=None, target_transform=None):
        super(BaseDataset, self).__init__()
        self.split_type = split_type
        self.input_transform = WrapperFunc(input_transform) if input_transform else None
        self.target_transform = WrapperFunc(target_transform) if target_transform else None
        self.pipeline = WrapperFunc(pipeline) if pipeline else None
        assert not((self.pipeline is None) and (self.input_transform is None) and (self.target_transform is None)), 'Either pipeline or input_transform, transform_output must be specified'

    def transform_databatch(self, inputs:dict, outputs:dict, meta:dict):
        if self.pipeline != None:
            return self.pipeline(inputs, outputs, meta)
        else:
            return self.input_transform(inputs, meta), self.target_transform(outputs, meta)

    def get_cat2indices(self):
        """Get a dict with class as key and indices as values, which will be
        used in :class:`ClassBalancedSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """ 
        pass
    
    def get_categories(self):
        pass

class WrapperFunc:
    def __init__(self, func):
        self.func = func

    def wrapper_func(self, *args):
        # Call the stored function with the additional 'self' argument
        return self.func(*args)
    
    def __call__(self, *args):
        return self.wrapper_func(*args)
