import torch
import os
from framework import BaseLoss, get_dist_info
import torch.nn as nn
from copy import deepcopy
import monai 

class MedSAMLoss(BaseLoss):
    def __init__(self, 
                 loss_weight: dict):
        super().__init__(loss_weight)
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    def forward_loss(self, pred, target):
        '''
        Calculates the loss_dict, it should contain the loss for each process.
        '''

        return {'loss_seg': self.seg_loss(pred['mask'], target['mask']), 
                'loss_ce': self.ce_loss(pred['mask'], target['mask'])}
