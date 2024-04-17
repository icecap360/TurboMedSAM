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
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
class CVPRMedSAMPipeline:

    def __init__(self, img_shape, target_mask_shape,
                 normalize, means, stds):
        self.img_shape = img_shape
        self.resize_img_transform = v2.Resize((img_shape, img_shape))
        self.target_mask_shape = target_mask_shape
        self.resize_mask_transform = v2.Resize((target_mask_shape, target_mask_shape), interpolation=InterpolationMode.NEAREST)
        if normalize:
            self.normalize_transform = v2.Normalize(mean=means, std=stds)
        else:
            self.normalize_transform = None
    def pipeline(self, inputs, outputs, meta):
        if meta['image_type'] == '2D':
            return self.preprocess_2D(
            inputs['image'],
            inputs['bbox'],
            outputs['mask'],
            meta
            )
        elif meta['image_type'] == '3D':
            return self.preprocess_3D(
            inputs['image'],
            inputs['bbox'],
            outputs['mask'],
            meta
        )
        else:
            raise Exception(meta['image_type']+'is unknown')
        
    def pipeline_2D(self, inputs, outputs, meta):
        return self.preprocess_2D(
            inputs['image'],
            inputs['bbox'],
            outputs['mask'],
            meta
        )
    
    def pipeline_inference(self, inputs, outputs, meta):
        img_padded, _ = self.img_transform(
            inputs['image'],
            self.resize_img_transform, 
            self.normalize_transform)
        bboxes = self.resize_img_transform(inputs["bbox"])
        return  {"image" : np.float32(img_padded),
                'bbox' : np.float32(bboxes[None, None, ...]),
                "meta"  : meta}, outputs
    
    def pipeline_encoder(self, inputs, outputs, meta):
        img_padded, _ = self.img_transform(
            inputs['image'],
            self.resize_img_transform, 
            self.normalize_transform)
        return  {"image" : np.float32(img_padded),
                "meta"  : meta}, outputs
        
    def preprocess_3D(self, voxels, bbox, gt, meta):
        inputs = dict(
            images = [],
            bboxes= [],
            meta=meta)
        outputs = dict(
            mask = gt)
        for i in range(voxels.shape[0]):
            inputs2D, _ = self.preprocess_2D(voxels[i, :, :], bbox, gt[i, :, :], meta)
            inputs["images"].append(inputs2D["image"])
            inputs["bboxes"].append(inputs2D["bbox"])
        return inputs, outputs

    def preprocess_2D_FFCV(self, img, gt, meta, input_img_shape, target_mask_shape, normalize):
        img_resize = self._resize_longest_side(img, input_img_shape) # Resizing
        img_padded = self._pad_image(img_resize, input_img_shape) # (256, 256, 3)
        if normalize:
            img_padded = self.normalize_img(img_padded, self.means, self.stds)
        gt2D = self.target_transform(gt, meta['npz_path'], target_mask_shape)

        # add data augmentation: random fliplr and random flipud
        # if self.data_aug:
        #     if random.random() > 0.5:
        #         img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
        #         gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
        #         # print('DA with flip left right')
        #     if random.random() > 0.5:
        #         img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
        #         gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        #         # print('DA with flip upside down')
            
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, bbox_shift))
        x_max = min(W, x_max + random.randint(0, bbox_shift))
        y_min = max(0, y_min - random.randint(0, bbox_shift))
        y_max = min(H, y_max + random.randint(0, bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": np.float32(img_padded),
            "bbox": np.float32(bboxes[None, None, ...]), # (B, 1, 4)
            "meta": meta
        }, {
            "mask": np.float32(gt2D[None, :,:]),
            # "original_mask": np.int64(gt[None, :,:]>0), # problem because this varies with instances!
            "meta": meta,
        }

    def preprocess_2D(self, img, bboxes, gt, meta):
        img_padded, img_resize_shape = self.img_transform(img, 
            self.resize_img_transform, 
            self.normalize_transform)
        bboxes = self.resize_img_transform(bboxes)
        gt2D = self.target_transform(gt, meta['npz_path'])

        # add data augmentation: random fliplr and random flipud
        # if self.data_aug:
        #     if random.random() > 0.5:
        #         img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
        #         gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
        #         # print('DA with flip left right')
        #     if random.random() > 0.5:
        #         img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
        #         gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        #         # print('DA with flip upside down')
        
        return {
            "image": np.float32(img_padded),
            "bbox": np.float32(bboxes[None, None, ...]), # (B, 1, 4)
            "meta": meta
        }, {
            "mask": np.float32(gt2D[None, :,:]),
            # "original_mask": np.int64(gt[None, :,:]>0), # problem because this varies with instances!
            "meta": meta,
        }

    def img_transform(self, img_3c, resize_img_transform, normalize_transform):
        # img_resize = self._resize_longest_side(img_3c, target_length=target_length) # Resizing
        # img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        # img_padded = self._pad_image(img_resize, target_length) # (256, 256, 3)
        # # convert the shape to (3, H, W)
        # img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        # assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        # if normalize:
            # if isinstance(means, np.ndarray):
            #     return (img - means)/stds
            # return (img - np.array(means).reshape(3,1,1))/np.array(stds).reshape(3,1,1)
        #     img_padded = self.normalize_img(img_padded, means, stds)
        # return img_padded, img_resize.shape
    
        img_resize = resize_img_transform(img_3c) # Resizing
        img_resize = (img_resize - img_resize.min()) / torch.clip(img_resize.max() - img_resize.min(), min=1e-8, max=None)  # normalize to [0, 1], (H, W, 3
        # convert the shape to (3, H, W)
        if normalize_transform:
            img_resize = normalize_transform(img_resize)
        return img_resize, img_resize.shape

    def normalize_img(self, img, means, stds):
        if isinstance(means, np.ndarray):
            return (img - means)/stds
        return (img - np.array(means).reshape(3,1,1))/np.array(stds).reshape(3,1,1)

    def _resize_longest_side(self, image, target_length, interpolation=cv2.INTER_AREA):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=interpolation)

    def _pad_image(self, image, target_length):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = target_length - h
        padw = target_length - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))
        return image_padded

    def target_transform(self, gt, npz_path):
        gt_resize = self.resize_mask_transform(gt)
        label_ids = torch.unique(gt_resize)[1:]
        try:
            gt2D = (gt_resize == random.choice(label_ids.tolist())).type(torch.uint8)
        except:
            print(npz_path, 'label_ids.tolist()', label_ids)
            gt2D = (gt_resize == torch.max(gt_resize)).type(torch.uint8)
        return gt2D
    
    

class CVPRMedSAMDistillationPipeline(CVPRMedSAMPipeline):

    def __init__(self, student_image_shape, teacher_image_shape, 
                 student_normalize, teacher_normalize, stds, means):
        self.student_resize_transform = v2.Resize((student_image_shape,student_image_shape))
        self.teacher_resize_transform = v2.Resize((teacher_image_shape,student_image_shape))
        if student_normalize:
            self.student_normalize_transform = v2.Normalize(
                mean=means, std=stds
            )
        else:
            self.student_normalize_transform = None
        if teacher_normalize:
            self.teacher_normalize_transform = v2.Normalize(
                mean=means, std=stds
            )
        else:
            self.teacher_normalize_transform = None

    def pipeline_teacher_student(self, inputs, outputs, meta):
        img_student_padded, _ = self.img_transform(
            inputs['image'],
            self.student_resize_transform, 
            self.student_normalize_transform)
        img_teacher_padded, _ = self.img_transform(
            inputs['image'],
            self.teacher_resize_transform, 
            self.teacher_normalize_transform)
        return  {"student_image" : np.float32(img_student_padded), "teacher_image" : np.float32(img_teacher_padded), "meta"  : meta}, outputs
