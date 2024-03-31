import torch
import os
from framework import BaseLoss, get_dist_info
import torch.nn as nn
from copy import deepcopy

class SampleLoss(BaseLoss):
    def __init__(self, 
                 loss_weight: dict):
        super().__init__(loss_weight)
        self.base_loss = torch.nn.L1Loss()

    def forward_loss(self, pred, ground_truth):
        '''
        Calculates the loss_dict, it should contain the loss for each process.
        '''
        # preds = [p['pred'] for p in pred]
        # preds = torch.stack(preds)
        # targets = ground_truth['output'].reshape(preds.shape[0], -1)
        return {'loss_pred': self.base_loss(pred['pred'], ground_truth['output'])}
