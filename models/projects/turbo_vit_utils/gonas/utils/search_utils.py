import math

import numpy as np
import torch
from thop import profile


# Utility functions
def get_params(model):
    """Calculates the total parameters in the model"""
    total_parameters = 0
    for param in model.parameters():
        total_parameters += np.prod(param.size())
    return total_parameters


def get_flops(model, input_shape):
    """Calculates the flops in the model"""
    dummy_inputs = torch.zeros(*input_shape)
    flops = profile(model, inputs=(dummy_inputs,), verbose=False)[0]
    return math.floor(flops)


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
