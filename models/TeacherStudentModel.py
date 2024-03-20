import torch
import torch.nn as nn
import torch.nn.functional as F
from framework import BaseModule
from typing import Optional, Tuple, Type

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class TeacherStudentModel(BaseModule):
    def __init__(
        self,
        student: BaseModule,
        teacher: BaseModule,
        init_cfg = None
    ) -> None:
        super().__init__(init_cfg)
        self.student = student
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, data_batch):
        with torch.no_grad():
            teacher_pred = self.teacher(data_batch)
        student_pred = self.student(data_batch)
        res = {}
        for key in student_pred.keys():
            res['teacher_'+key] = teacher_pred[key]
            res['student_'+key] = student_pred[key]
        return res
    
    def state_dict(self):
        return self.student.state_dict()
    
    def init_weights(self):
        self.student.init_weights()
        self.teacher.init_weights()
        return super().init_weights()