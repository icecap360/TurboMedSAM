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
        self.iou_loss = nn.MSELoss(reduction='mean')
    
    def forward_loss(self, pred, target):
        '''
        Calculates the loss_dict, it should contain the loss for each process.
        '''
        iou_gt = self.cal_iou(torch.sigmoid(pred['logits']) > 0.5, target['mask'].bool())
        return {'loss_dice': self.seg_loss(pred['logits'], target['mask']), 
                'loss_ce': self.ce_loss(pred['logits'], target['mask'].float()),
                'loss_iou': self.iou_loss(pred['iou'], iou_gt)
}

    def cal_iou(self, result, reference):
        intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
        union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
        
        iou = intersection.float() / union.float()
        
        return iou.unsqueeze(1)