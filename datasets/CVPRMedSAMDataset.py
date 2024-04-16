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
from torchvision import tv_tensors
from torchvision.transforms import v2

class CVPRMedSAMDataset(BaseDataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split_type, bbox_shift=5, subset_classes=None, pipeline=None, transform=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(split_type, pipeline, transform, input_transform, target_transform)
        self.root_dir = root_dir
        self.bbox_shift = bbox_shift
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
        image = np.moveaxis(npz['imgs'],2,0)
        target = npz['gts']
        meta = dict(
            idx=idx,
            npz_path = npz_path,
            original_shape = image.shape,
            modality = self.get_modality(npz_path)
        )
        
        label_ids = np.unique(target)[1:]
        try:
            target = np.uint8(target == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(npz_path, 'label_ids.tolist()', label_ids.tolist())
            target = np.uint8(target == np.max(target)) # only one label, (256, 256)
        gt2D = np.uint8(target > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = tv_tensors.BoundingBoxes(
            np.float32(np.array([x_min, y_min, x_max, y_max])),
            canvas_size=(H,W),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            requires_grad=False
            )
        bboxes = v2.Resize((image.shape[1:]))(bboxes)

        if min(image.shape) > 3:
            meta['image_type'] = '3D'
            meta['spacing'] = npz['spacing']
        else:
            meta['image_type'] = '2D'

        return self.transform_databatch({'image': tv_tensors.Image(image), 'bbox': bboxes},
                                        {'mask': tv_tensors.Mask(target)}, 
                                        meta) 
    
    def get_modality(self, path:str):
        path = path.replace(self.npz_dir, '')
        if path[0] == '/':
            path = path[1:]
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
    def __init__(self, root_dir, split_type, bbox_shift=5, pipeline=None, transform=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(CVPRMedSAMDataset, self).__init__(split_type, pipeline, transform, input_transform, target_transform)
        self.root_dir = root_dir
        self.bbox_shift = bbox_shift
        self.npz_dir = self.root_dir
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
        image = np.moveaxis(npz['imgs'],2,0)
        bbox = tv_tensors.BoundingBoxes(npz['boxes'], 
                                        canvas_size=image.shape[1:],
                                        format=tv_tensors.BoundingBoxFormat.XYXY,
                                        requires_grad=False)
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

        return self.transform_databatch({'image': tv_tensors.Image(image),
                                         "bbox": bbox},
                                         {},
                                        meta)    

class CVPRMedSAMEncoderDataset(CVPRMedSAMDataset):
    def __init__(self, root_dir, split_type, subset_classes=None, pipeline=None, transform=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(CVPRMedSAMDataset, self).__init__(split_type, pipeline, transform, input_transform, target_transform)
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
        image = np.moveaxis(npz['imgs'],2,0)
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

        return self.transform_databatch({'image': tv_tensors.Image(image)}, {},
                                        meta)    

class CVPRMedSAMEncoderPreCompDataset(BaseDataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, teacher_root, split_type, subset_classes=None, pipeline=None, transform=None, input_transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(split_type, pipeline, transform, input_transform, target_transform)
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
        image = np.moveaxis(npz['imgs'],2,0)
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
        teacher_embeddings = tv_tensors.Mask(
            npz['embeddings'].astype(np.float32),
            dtype=torch.float32
        )

        return self.transform_databatch({'image': tv_tensors.Image(image)},
                                        {'teacher_embeddings': teacher_embeddings},
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
    
class CVPRMedSAMDatasetFFCVWrite(CVPRMedSAMDataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split_type):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(Dataset, self).__init__()
        self.split_type = split_type
        self.root_dir = root_dir
        self.npz_dir = os.path.join(self.root_dir, self.split_type)
        self.classes = os.listdir(self.npz_dir)
        self.npz_paths = glob.glob(os.path.join(self.npz_dir, "**/*.npz"), recursive=True)
        
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
        
        image = np.moveaxis(npz['imgs'],2,0)
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
