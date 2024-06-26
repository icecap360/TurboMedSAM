"""
Implements the knowledge distillation loss, proposed in deit and used in RepViT
"""
import torch
from torch.nn import functional as F
import torch.nn as nn
from framework import BaseLoss

class DistillationLoss(BaseLoss):
    def __init__(self, 
                 distillation_type, 
                 tau,
                 precomputed_teacher,
                 loss_weight: dict):
        super().__init__(loss_weight)
        assert distillation_type in ['none', 'soft', 'hard', 'mse']
        self.distillation_type = distillation_type
        self.tau = tau
        self.precomputed_teacher = precomputed_teacher
        self.mse_loss = torch.nn.MSELoss()
    def forward_loss(self, pred, target):
        if self.precomputed_teacher:
            student_outputs = pred['embeddings']
            teacher_outputs = target['teacher_embeddings']
        else:
            student_outputs = pred['student_embeddings']
            teacher_outputs = pred['teacher_embeddings']
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(student_outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / student_outputs.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                student_outputs, teacher_outputs.argmax(dim=1))
        elif self.distillation_type == 'mse':
            distillation_loss = self.mse_loss(student_outputs, teacher_outputs)
        return {'loss_distillation': distillation_loss}

# class DistillationLoss(torch.nn.Module):
#     """
#     This module wraps a standard criterion and adds an extra knowledge distillation loss by
#     taking a teacher model prediction and using it as additional supervision.
#     """

#     def __init__(self, base_criterion: torch.nn.Module,
#                  distillation_type: str, alpha: float, tau: float, teacher_model: torch.nn.Module):
#         super().__init__()
#         self.base_criterion = base_criterion
#         self.teacher_model = teacher_model
#         assert distillation_type in ['none', 'soft', 'hard']
#         self.distillation_type = distillation_type
#         self.alpha = alpha
#         self.tau = tau

#     def forward(self, inputs, outputs, labels):
#         """
#         Args:
#             inputs: The original inputs that are feed to the teacher model
#             outputs: the outputs of the model to be trained. It is expected to be
#                 either a Tensor, or a Tuple[Tensor, Tensor], with the original output
#                 in the first position and the distillation predictions as the second output
#             labels: the labels for the base criterion
#         """
#         outputs_kd = None
#         if not isinstance(outputs, torch.Tensor):
#             # assume that the model outputs a tuple of [outputs, outputs_kd]
#             outputs, outputs_kd = outputs
#         base_loss = self.base_criterion(outputs, labels)
#         if self.distillation_type == 'none':
#             return base_loss

#         if outputs_kd is None:
#             raise ValueError("When knowledge distillation is enabled, the model is "
#                              "expected to return a Tuple[Tensor, Tensor] with the output of the "
#                              "class_token and the dist_token")
#         # don't backprop throught the teacher
#         with torch.no_grad():
#             teacher_outputs = self.teacher_model(inputs)

#         if self.distillation_type == 'soft':
#             T = self.tau
#             # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
#             # with slight modifications
#             distillation_loss = F.kl_div(
#                 F.log_softmax(outputs_kd / T, dim=1),
#                 F.log_softmax(teacher_outputs / T, dim=1),
#                 reduction='sum',
#                 log_target=True
#             ) * (T * T) / outputs_kd.numel()
#         elif self.distillation_type == 'hard':
#             distillation_loss = F.cross_entropy(
#                 outputs_kd, teacher_outputs.argmax(dim=1))

#         loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
#         return loss