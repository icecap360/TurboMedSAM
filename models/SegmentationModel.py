import os
import random
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from .projects.LiteMedSAM.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from .projects.LiteMedSAM.tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse
from framework import BaseDetector, extract_idx

__all__ = ['SegmentationModel']

class SegmentationModel(BaseDetector):
    def __init__(self,
                model,
                init_cfg=None
                ):
        
        super().__init__(init_cfg)
        
        self.model = model
                
    def forward(self, input_params):
        result = {'masks':[]}
        batch_size = len(input_params['image'])
        device = input_params['image'].device
        for i in range(batch_size):
            input_i = extract_idx(input_params, i)
            boxes_i = input_i['bbox']
            n_boxes = boxes_i.shape[3]
            target_size = (input_i['meta']['original_shape']['L'], input_i['meta']['original_shape']['W'])
            mask = torch.zeros(target_size, device=device)
            for j in range(n_boxes):
                input_i['bbox'] = boxes_i[:, :, :, j, :]
                logits = self.model(input_i)['logits']
                mask_j = self.postprocess_logits(logits, target_size)[0]
                label = torch.tensor(j+1, device=device)
                mask[mask_j>0] = j+1
            result['masks'].append(mask)

        return result

    def init_weights(self, state_dict=None, strict=True):
        self.model.init_weights(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

    @torch.no_grad()
    def postprocess_logits(self, logits, original_size):
        """
        Do cropping and resizing
        """
        pred = torch.sigmoid(logits)
        # Resize
        pred = F.interpolate(pred,
                             size=(original_size[0], original_size[1]), 
                             mode='bilinear', 
                             align_corners=False)
        pred = (pred>0.5).type(torch.uint8).squeeze(1)

        return pred