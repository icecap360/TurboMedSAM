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

class CVPRMedSAMPipeline:

    def __init__(self, input_img_shape, target_mask_shape, bbox_shift):
        self.input_img_shape = input_img_shape
        self.target_mask_shape = target_mask_shape
        self.bbox_shift = bbox_shift

    def pipeline(self, inputs, outputs, meta):
        if meta['image_type'] == '2D':
            return self.preprocess_2D(
            inputs['image'],
            outputs['mask'],
            meta,
            self.input_img_shape,
            self.target_mask_shape,
            self.bbox_shift
            )
        elif meta['image_type'] == '3D':
            return self.preprocess_3D(
            inputs['image'],
            outputs['mask'],
            meta,
            self.input_img_shape,
            self.target_mask_shape,
            self.bbox_shift
        )
        else:
            raise Exception(meta['image_type']+'is unknown')
        
    def pipeline_2D(self, inputs, outputs, meta):
        return self.preprocess_2D(
            inputs['image'],
            outputs['mask'],
            meta,
            self.input_img_shape,
            self.target_mask_shape,
            self.bbox_shift
        )
    
    def pipeline_inference(self, inputs, meta):
        img_padded, _ = self.img_transform(
            inputs['image'],
            self.input_img_shape)
        return  {"image" : np.float32(img_padded),
                'bbox' : np.float32(inputs["bbox"][None, None, ...]),
                "meta"  : meta}
    
    def pipeline_encoder(self, inputs, outputs, meta):
        img_padded, _ = self.img_transform(
            inputs['image'],
            self.input_img_shape)
        return  {"image" : np.float32(img_padded),
                "meta"  : meta}, outputs
        
    def preprocess_3D(self, voxels, gt, meta, input_img_shape, target_mask_shape, bbox_shift):
        inputs = dict(
            images = [],
            bboxes= [],
            meta=meta)
        outputs = dict(
            mask = gt,
            meta=meta)
        inputs, outputs = dict(), dict()
        for i in range(voxels.shape[0]):
            inputs2D, _ = self.preprocess_2D(voxels[i, :, :], gt[i, :, :], meta, bbox_shift)
            inputs["images"].append(inputs2D["image"])
            inputs["bboxes"].append(inputs2D["bbox"])
        return inputs, outputs

    def preprocess_2D_FFCV(self, img, gt, meta, input_img_shape, target_mask_shape, bbox_shift):
        img_resize = self._resize_longest_side(img, input_img_shape) # Resizing
        img_padded = self._pad_image(img_resize, input_img_shape) # (256, 256, 3)
        
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

    def preprocess_2D(self, img, gt, meta, input_img_shape, target_mask_shape, bbox_shift):
        img_padded, img_resize_shape = self.img_transform(img, input_img_shape)
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

    def img_transform(self, img_3c, target_length):
        img_resize = self._resize_longest_side(img_3c, target_length=target_length) # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = self._pad_image(img_resize, target_length) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        return img_padded, img_resize.shape

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

    def target_transform(self, gt, npz_path, target_length):
        gt_resize = self._resize_longest_side( gt, target_length, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        gt_resize = self._pad_image(gt_resize, target_length) # (256, 256)
        label_ids = np.unique(gt_resize)[1:]
        try:
            gt2D = np.uint8(gt_resize == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(npz_path, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt_resize == np.max(gt_resize)) # only one label, (256, 256)
        return gt2D
    
    

class CVPRMedSAMDistillationPipeline(CVPRMedSAMPipeline):

    def __init__(self, student_image_shape, teacher_image_shape, target_mask_shape, bbox_shift):
        self.student_image_shape = student_image_shape
        self.teacher_image_shape = teacher_image_shape
        self.target_mask_shape = target_mask_shape
        self.bbox_shift = bbox_shift

    def pipeline_teacher_student(self, inputs, outputs, meta):
        img_student_padded, _ = self.img_transform(
            inputs['image'],
            self.student_image_shape)
        img_teacher_padded, _ = self.img_transform(
            inputs['image'],
            self.teacher_image_shape)
        return  {"student_image" : np.float32(img_student_padded), "teacher_image" : np.float32(img_teacher_padded), "meta"  : meta}, outputs
