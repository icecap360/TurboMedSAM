import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod
from .BaseLoss import add_loss, convert_loss_float, BaseLoss

class BaseMetric():

    def __init__(self, loss: BaseLoss):
        self.loss = loss
        super(BaseMetric, self).__init__()

        
    @abstractmethod
    def get_metrics(self, pred, target, device) -> dict:
        pass
    
    def sum_metrics(self, metrics_dicts, device)-> dict:
        '''
        Losses in the loss_dicts should be unweighted.
        '''
        sum_dict = dict()
        for k in metrics_dicts[0].keys():
            sum_dict[k] = torch.sum(torch.stack([
                    metrics_dicts[j][k].to(device) for j in range(len(metrics_dicts))
                ]))
        return sum_dict