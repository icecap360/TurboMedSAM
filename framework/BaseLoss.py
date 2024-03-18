import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from .BaseModules import BaseModule
from torch.optim.lr_scheduler import _LRScheduler
from abc import abstractmethod, ABC
import torch.distributed as dist
from .Distributed import get_dist_info
from copy import deepcopy, copy

class BaseLoss( ABC):
    def __init__(self, 
                 loss_weight: dict,
                 ):
        self.loss_weight = loss_weight
        super(BaseLoss, self).__init__()
        assert np.all([type(v) in (float, int) for v in loss_weight.values()]) 
        
    @abstractmethod
    def forward_loss(self, pred, ground_truth):
        '''
        Calculates the loss_dict
        '''
        pass
        
    def calc_weighted_loss(self, 
                      loss_dict: dict, 
                      loss_weight: dict,
                      requires_grad, 
                      device):
        
        loss_dict_keys = sorted(loss_dict.keys())
        loss_weight_keys = sorted(loss_weight.keys())
        assert len(loss_dict_keys) == len(loss_weight_keys)
        
        if requires_grad:
            weighted_sum_loss = torch.tensor(0., device=device, requires_grad=True )
        else:
            weighted_sum_loss = torch.tensor(0., device=device, requires_grad=False )
            
        for i in range(len(loss_dict_keys)):
            var_key = loss_dict_keys[i]
            weight_key = var_key
            weighted_sum_loss = weighted_sum_loss + loss_dict[var_key] * torch.tensor(loss_weight[weight_key], dtype=torch.float32,  device=device, requires_grad=False)
        return  weighted_sum_loss
    
        
    def average_loss(self, loss_dicts, device):
        '''
        Calculates the total_loss across all processes
        This should be executed after forward function.
        Losses in the loss_dicts should be unweighted.
        '''
        sum_loss_dict = dict()
        for k in loss_dicts[0].keys():
            sum_loss_dict[k] = torch.mean(torch.cat([
                    loss_dicts[j][k].reshape(1).to(device) for j in range(len(loss_dicts))
                ]))
        return sum_loss_dict
    
    # def forward(self, pred, target, device):
    #     self.loss_dict = self.forward_loss(pred, target)
    #     self.weighted_sum_loss = self.calc_weighted_loss(self.loss_dict, self.loss_weight, device)
    #     return self.loss_dict, self.weighted_sum_loss
    
    def backward(self):
        return self.weighted_sum_loss.backward()
    
def convert_loss_float(loss):
    return {k:v.item() for (k,v) in loss.items()}

def add_loss(loss1dict, loss2dict):
    new_loss = loss1dict.copy()
    for k in loss1dict.keys():
        new_loss[k] = loss1dict[k] + loss2dict[k]
    return new_loss
