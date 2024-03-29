import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod
from framework import add_loss, convert_loss_float, BaseMetric

class SampleMetric(BaseMetric):
    def get_metrics(self, pred, target, device) -> dict:
        return {
            'metric1': torch.tensor(1.0, device=device)
        }

