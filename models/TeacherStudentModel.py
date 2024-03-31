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
        student_image_size = None,
        teacher_image_size = None,
        init_cfg = None
    ) -> None:
        super().__init__(init_cfg)
        self.student = student
        self.teacher = teacher
        self.student_image_size = student_image_size
        self.teacher_image_size = teacher_image_size
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, data_batch):
        with torch.no_grad():
            if 'teacher_image' in data_batch.keys():
                assert not 'image' in data_batch.keys()
                data_batch['image'] = data_batch['teacher_image']
                teacher_pred = self.teacher(data_batch)
                data_batch.pop('image')
            else:
                teacher_pred = self.teacher(data_batch)

        if 'student_image' in data_batch.keys():
            assert not 'image' in data_batch.keys()
            data_batch['image'] = data_batch['student_image']
            student_pred = self.student(data_batch)
            data_batch.pop('image')
        else:
            student_pred = self.student(data_batch)

        res = {}
        for key in student_pred.keys():
            res['teacher_'+key] = teacher_pred[key]
            res['student_'+key] = student_pred[key]
        return res
    
    def state_dict(self, destination=None, prefix=None, keep_vars=None):
        return self.student.state_dict(destination, prefix, keep_vars)
    
    def init_weights(self):
        self.student.init_weights()
        self.teacher.init_weights()
        return super().init_weights()