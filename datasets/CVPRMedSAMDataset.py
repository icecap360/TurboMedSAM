import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
import torch.distributed as dist
from framework import BaseDataset
import cv2
import random
import json  

class CVPRMedSAMDataset(BaseDataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split_type, subset_classes=None, pipeline=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(split_type, pipeline, input_transform, target_transform)
        self.root_dir = root_dir
        self.npz_dir = os.path.join(self.root_dir, self.split_type)
        self.classes = os.listdir(self.npz_dir)
        self.subset_classes = subset_classes

        if self.subset_classes is None:
            self.npz_paths = glob.glob(os.path.join(self.npz_dir, "**/*.npz"), recursive=True)
        else:
            self.npz_paths = []
            for cat in self.subset_classes:
                assert type(cat) is str
                self.npz_paths += glob.glob(os.path.join(self.npz_dir, cat, "**/*.npz"), recursive=True)
        
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
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        image = npz['imgs'] 
        target = npz['gts']
        meta = dict(
            idx=idx,
            npz_path = npz_path,
            original_shape = image.shape,
            modality = self.get_modality(npz_path)
        )

        if min(image.shape) > 3:
            meta['image_type'] = '3D'
            meta['spacing'] = npz['spacing']
        else:
            meta['image_type'] = '2D'

        return self.transform_databatch({'image': image},
                                        {'mask': target}, 
                                        meta)    
    
    def get_modality(self, path:str):
        path = path.replace(self.npz_dir, '')
        return os.path.normpath(path).split(os.sep)[0]

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
        return self.classes

class CVPRMedSAMInferenceDataset(CVPRMedSAMDataset):

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # npz_paths = [self.npz_paths[i] for i in idx]
        # # classes = [path.split_type('/')[0] for path in npz_paths]
        # npzs = [np.load(n, allow_pickle=True, mmap_mode="r") for n in npz_paths]
        # imgs = [npz['imgs'] for npz in npzs]
        # gts = [npz['gts'] for npz in npzs]
        
        npz_path = self.npz_paths[idx]
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        image = npz['imgs'] 
        bbox = npz['bbox'] 
        meta = dict(
            idx=idx,
            npz_path = npz_path,
            original_shape = image.shape,
            modality = self.get_modality(npz_path)
        )
        if min(image.shape) > 3:
            meta['image_type'] = '3D'
            meta['spacing'] = npz['spacing']
        else:
            meta['image_type'] = '2D'

        return self.transform_databatch({'image': image,
                                         "bbox": bbox},
                                        meta)    


class CVPRMedSAMEncoderDataset(CVPRMedSAMDataset):
    def __init__(self, root_dir, split_type, subset_classes=None, pipeline=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(CVPRMedSAMDataset, self).__init__(split_type, pipeline, input_transform, target_transform)
        self.root_dir = root_dir
        self.npz_dir = os.path.join(self.root_dir, self.split_type)
        self.classes = os.listdir(self.npz_dir)
        self.subset_classes = subset_classes
        if self.subset_classes is None:
            self.npz_paths = glob.glob(os.path.join(self.npz_dir, "**/*.npz"), recursive=True)
        else:
            self.npz_paths = []
            for cat in self.subset_classes:
                assert type(cat) is str
                self.npz_paths += glob.glob(os.path.join(self.npz_dir, cat, "**/*.npz"), recursive=True)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # npz_paths = [self.npz_paths[i] for i in idx]
        # # classes = [path.split_type('/')[0] for path in npz_paths]
        # npzs = [np.load(n, allow_pickle=True, mmap_mode="r") for n in npz_paths]
        # imgs = [npz['imgs'] for npz in npzs]
        # gts = [npz['gts'] for npz in npzs]
        
        npz_path = self.npz_paths[idx]
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        image = npz['imgs'] 
        meta = dict(
            idx=idx,
            npz_path = npz_path,
            original_shape = image.shape,
            modality = self.get_modality(npz_path)
        )
        if min(image.shape) > 3:
            meta['image_type'] = '3D'
            meta['spacing'] = npz['spacing']
        else:
            meta['image_type'] = '2D'

        return self.transform_databatch({'image': image}, {},
                                        meta)    

class CVPRMedSAMEncoderPreComputed(BaseDataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, teacher_root, split_type, subset_classes=None, pipeline=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(split_type, pipeline, input_transform, target_transform)
        self.root_dir = root_dir
        self.npz_dir = os.path.join(self.root_dir, self.split_type)
        self.classes = os.listdir(self.npz_dir)
        self.teacher_root = teacher_root
        self.subset_classes = subset_classes

        if self.subset_classes is None:
            self.npz_paths = glob.glob(os.path.join(self.npz_dir, "**/*.npz"), recursive=True)
        else:
            self.npz_paths = []
            for cat in self.subset_classes:
                assert type(cat) is str
                self.npz_paths += glob.glob(os.path.join(self.npz_dir, cat, "**/*.npz"), recursive=True)
        
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
        encoder_result_path = os.path.join(
            self.teacher_root,
            os.path.relpath(npz_path, self.root_dir)
        )
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        image = npz['imgs'] 
        meta = dict(
            idx=idx,
            npz_path = npz_path,
            original_shape = image.shape,
            modality = self.get_modality(npz_path)
        )

        if min(image.shape) > 3:
            meta['image_type'] = '3D'
            meta['spacing'] = npz['spacing']
        else:
            meta['image_type'] = '2D'
        
        npz = np.load(encoder_result_path, allow_pickle=True, mmap_mode="r")

        return self.transform_databatch({'image': image},
                                        {'teacher_embeddings': npz['embeddings'].astype(np.float32)},
                                        meta)    
    
    def get_modality(self, path:str):
        path = path.replace(self.npz_dir, '')
        return os.path.normpath(path).split(os.sep)[0]

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
        return self.classes
    
class CVPRMedSAMDatasetFFCVWrite(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split_type):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split_type = split_type
        self.root_dir = root_dir
        self.npz_dir = os.path.join(self.root_dir, self.split_type)
        self.classes = os.listdir(self.npz_dir)
        self.npz_paths = glob.glob(os.path.join(self.npz_dir, "**/*.npz"), recursive=True)
        
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
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        
        image = npz['imgs'] 
        assert image.shape[2] == 3
        
        target = npz['gts']
        if len(target.shape) != 3:
            target = np.stack((target,target,target), 2)
            
        meta = dict(
            idx=idx,
            npz_path = npz_path,
            original_shape = image.shape,
            modality = self.get_modality(npz_path)
        )

        if min(image.shape) > 3:
            meta['image_type'] = '3D'
            meta['spacing'] = npz['spacing']
        else:
            meta['image_type'] = '2D'

        return image, target, json.dumps(meta, default=int)

    def get_modality(self, path:str):
        path = path.replace(self.npz_dir, '')
        return os.path.normpath(path).split(os.sep)[0]
    
def get_MedSAM_classes(root_dir, split_type):
    return os.listdir(os.path.join(root_dir, split_type))
