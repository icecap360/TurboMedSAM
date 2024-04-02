import torch
import os
from copy import deepcopy

def create_optimizer(optimizer):
    if isinstance(optimizer['optimizer'], torch.optim.Optimizer):
        return optimizer
    params_dict = deepcopy(optimizer['optimizer'])
    optimizer_type = params_dict.pop('type')
    optimizer['optimizer'] = optimizer_type(**params_dict)
    return optimizer