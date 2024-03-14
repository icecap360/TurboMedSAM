import torch
import os
from framework import BaseLoss, get_dist_info
import torch.nn as nn
from copy import deepcopy

class SimpleMedSAMLoss(BaseLoss):
    def __init__(self, 
                 loss_weight: dict):
        super().__init__(loss_weight)
        self.base_loss = torch.nn.MSELoss()

    def forward_loss(self, pred, ground_truth):
        '''
        Calculates the loss_dict, it should contain the loss for each process.
        '''
        with open('drawn_sample_paths.txt', 'a') as writer:
            for sample in ground_truth['path']:
                writer.write(sample + '\n')
        return {'loss_pred': torch.tensor([[0.0]], device='cuda')}
