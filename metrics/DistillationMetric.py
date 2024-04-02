import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod
from framework import add_loss, convert_loss_float, BaseMetric

class DistillationMetric(BaseMetric):
    def __init__(self, precomputed_teacher):
        self.mse_loss = torch.nn.MSELoss()
        self.precomputed_teacher = precomputed_teacher
    def get_metrics(self, pred, target, device) -> dict:
        if self.precomputed_teacher:
            student_outputs = pred['embeddings']
            teacher_outputs = target['teacher_embeddings']
        else:
            student_outputs = pred['student_embeddings']
            teacher_outputs = pred['teacher_embeddings']
        return {
            'mse': self.mse_loss(student_outputs, teacher_outputs)
        }

