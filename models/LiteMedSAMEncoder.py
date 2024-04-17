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
from framework import BaseDetector

__all__ = ['LiteMedSAMEncoder']

class LiteMedSAMEncoder(BaseDetector):
    def __init__(self,
                settings,
                encoder = None,
                init_cfg=None,
                ):
        
        super().__init__(init_cfg)
        
        if encoder:
            self.image_encoder = encoder
        else:
            encoder_cfg = settings
            self.image_encoder = TinyViT(
                img_size=encoder_cfg['img_size'],
                in_chans=encoder_cfg['in_chans'],
                embed_dims=encoder_cfg['embed_dims'],
                depths=encoder_cfg['depths'],
                num_heads=encoder_cfg['num_heads'],
                window_sizes=encoder_cfg['window_sizes'],
                mlp_ratio=encoder_cfg['mlp_ratio'],
                drop_rate=encoder_cfg['drop_rate'],
                drop_path_rate=encoder_cfg['drop_path_rate'],
                use_checkpoint=encoder_cfg['use_checkpoint'],
                mbconv_expand_ratio=encoder_cfg['mbconv_expand_ratio'],
                local_conv_size=encoder_cfg['local_conv_size'],
                layer_lr_decay=encoder_cfg['layer_lr_decay']
            )
                
    def forward(self, input_params):
        image = input_params['image']
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        return {
            "embeddings": image_embedding, 
        }
        
    def init_weights(self, state_dict=None, strict=True):
        if self.init_cfg == None and state_dict == None:
            return
        elif state_dict:
            self.load_state_dict(state_dict,  strict= strict)
        elif 'pretrained' in self.init_cfg['type'].lower():
            if not 'checkpoint' in self.init_cfg.keys():
                raise Exception('Missing checkpoint')   
            if 'medsam' in self.init_cfg['checkpoint'].lower():
                medsam_state_dict = torch.load(self.init_cfg['checkpoint'])
                image_encoder_params = [k for k in medsam_state_dict.keys() if k.startswith('image_encoder')]
                img_encoder_dict = { k.replace('image_encoder.', ''):medsam_state_dict[k] for k in image_encoder_params}
                self.load_state_dict(img_encoder_dict, 
                                 strict=self.init_cfg.get('strict') or True)
            else:
                self.load_state_dict(torch.load(self.init_cfg['checkpoint']), 
                                 strict=self.init_cfg.get('strict') or True)
        else:
            raise Exception('init_cfg is formatted incorrectly')