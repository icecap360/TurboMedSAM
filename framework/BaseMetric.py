import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod
from .BaseLoss import add_loss, convert_loss_float, BaseLoss

class BaseMetric():
        
    @abstractmethod
    def get_metrics(self, pred, target, device) -> dict:
        pass
    
    def average_metrics(self, metrics_dicts, device)-> dict:
        '''
        Losses in the loss_dicts should be unweighted.
        '''
        if len(metrics_dicts) == 0:
            return dict()
        sum_dict = dict()
        for k in metrics_dicts[0].keys():
            sum_dict[k] = torch.mean(torch.cat([
                    metrics_dicts[j][k].to(device).reshape(1) for j in range(len(metrics_dicts))
                ]))
        return sum_dict