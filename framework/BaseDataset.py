import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
from abc import ABC, abstractmethod 
from BaseDataSamplers import ClassBalancedSampler

class BaseDataset(ABC, Dataset):

    @abstractmethod
    def __init__(self, root_dir, split_type, transform=None, transform_target=None):
        self.root_dir = root_dir
        self.split_type = split_type
        self.transform = transform
        self.transform_target = transform_target


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
