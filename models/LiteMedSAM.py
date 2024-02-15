import os
import random
import monai
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

from projects.LiteMedSAM.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from projects.LiteMedSAM.tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse
from framework import DETECTORS, BaseDetector


@DETECTORS.register()
class LiteMedSAM(BaseDetector):
    def __init__(self,
                cfg
                ):
        
        super().__init__()
        
        encoder_cfg = cfg['image_encoder']
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
        
        decoder_cfg = cfg['mask_decoder']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=decoder_cfg['num_multimask_outputs'],
            transformer=TwoWayTransformer(
                depth=decoder_cfg['transformer_depth'],
                embedding_dim=decoder_cfg['transformer_embeddim'],
                mlp_dim=decoder_cfg['mlp_dim'],
                num_heads=decoder_cfg['num_heads'],
            ),
            transformer_dim=decoder_cfg['transformer_dim'],
            iou_head_depth=decoder_cfg['iou_head_depth'],
            iou_head_hidden_dim=decoder_cfg['iou_head_hidden_dim'],
        )
        
        prompt_encoder_cfg = cfg['prompt_encoder']
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_encoder_cfg['embed_dim'],
            image_embedding_size=prompt_encoder_cfg['image_embedding_size'],
            input_image_size=prompt_encoder_cfg['input_image_size'],
            mask_in_chans=prompt_encoder_cfg['mask_in_chans']
            )
        
    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

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