"""
Implements the knowledge distillation loss, proposed in deit and used in RepViT
"""
import torch
from torch.nn import functional as F
import torch.nn as nn
from framework import BaseLoss
import monai 

class MedSAMTeacherLoss(BaseLoss):
    def __init__(self, 
                 distillation_type, 
                 tau,
                 loss_weight: dict):
        super().__init__(loss_weight)
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.iou_loss = nn.MSELoss(reduction='mean')
        assert distillation_type in ['none', 'soft', 'hard', 'bce']
        self.distillation_type = distillation_type
        self.tau = tau
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
    def forward_loss(self, pred, target):
        student_outputs = pred['student_logits']
        teacher_outputs = pred['teacher_logits']
        if self.distillation_type == 'soft' or self.distillation_type == 'hard':
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            p = torch.sigmoid(student_outputs).view(-1,1)
            p = torch.stack((p, 1-p), dim=1).squeeze(-1)
            q = torch.sigmoid(teacher_outputs).view(-1,1)
            q = torch.stack((q, 1-q), dim=1).squeeze(-1)
            distillation_loss = F.kl_div(
                p,
                q,
                reduction='batchmean',
                log_target=True
            )
        elif self.distillation_type == 'bce':
            distillation_loss = self.bce_loss(student_outputs, torch.sigmoid(teacher_outputs))
        
        target_mask = target['student_mask']
        if len(target_mask.shape)==3:
            target_mask = target_mask[:, None, :, :]
            target_mask = F.interpolate(target_mask, 
                (student_outputs.shape[-2], student_outputs.shape[-1]),
                mode="nearest",
            )
        
        iou_gt = self.cal_iou(torch.sigmoid(student_outputs) > 0.5, target_mask.bool())
        return {
            'loss_distillation': distillation_loss, 
            'loss_dice': self.seg_loss(student_outputs, target_mask),     
            'loss_ce': self.ce_loss(student_outputs, target_mask.float()),
            'loss_iou': self.iou_loss(pred['student_iou'], iou_gt)
            }

    def cal_iou(self, result, reference):
        intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
        union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
        
        iou = intersection.float() / union.float()
        
        return iou.unsqueeze(1)