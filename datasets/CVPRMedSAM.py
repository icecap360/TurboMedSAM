import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
import torch.distributed as dist
from framework import BaseDataset

class CVPRMedSAMDataset(BaseDataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split_type, transform=None, transform_target=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, split_type, transform, transform_target)
        self.npz_dir = os.path.join(self.root_dir, self.split_type)
        self.npz_paths = glob.glob(os.path.join(self.npz_dir, "**/*.npz"), recursive=True)
        self.classes = os.listdir(self.npz_dir)

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # npz_paths = [self.npz_paths[i] for i in idx]
        # # classes = [path.split_type('/')[0] for path in npz_paths]
        # npzs = [np.load(n, allow_pickle=True, mmap_mode="r") for n in npz_paths]
        # imgs = [npz['imgs'] for npz in npzs]
        # gts = [npz['gts'] for npz in npzs]
        
        npz_path = self.npz_paths[idx]
        # classes = npz_path.split_type('/')[0]
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        image = npz['imgs'] 
        mask = npz['gts']
        if self.transform:
            image = self.transform(image)
        if self.transform_target:
            mask = self.transform_target(mask)
        inputs = dict(
            image=image,
        )
        outputs = dict(
            mask=mask,
        )
        return inputs, outputs
    
    def get_cat2indices(self):
        if self.classes is None:
            raise ValueError('self.classes can not be None')
        # sort the label index
        cat2imgs = dict()
        for i in range(len(self.classes)):
            class_i = self.classes[i]
            search = '/' + class_i+ '/'
            cat2imgs[class_i] = [i for i,path in enumerate(self.npz_paths) if search in path]
        return cat2imgs
    def get_categories(self):
        pass
    

