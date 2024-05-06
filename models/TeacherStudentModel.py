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
        apply_cutmix = False,
        student_image_size = None,
        teacher_image_size = None,
        init_cfg = None
    ) -> None:
        super().__init__(init_cfg)
        self.student = student
        self.teacher = teacher
        self.applycutmix = apply_cutmix
        self.student_image_size = student_image_size
        self.teacher_image_size = teacher_image_size
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, data_batch):
        with torch.no_grad():
            if 'teacher_image' in data_batch.keys():
                assert not 'image' in data_batch.keys()
                data_batch['image'] = data_batch['teacher_image']
                # data_batch['bbox'] = data_batch['teacher_bbox']
                teacher_pred = self.teacher(data_batch)
                data_batch.pop('image')
                # data_batch.pop('bbox')
            else:
                teacher_pred = self.teacher(data_batch)

        if 'student_image' in data_batch.keys():
            assert not 'image' in data_batch.keys()
            data_batch['image'] = data_batch['student_image']
            # data_batch['bbox'] = data_batch['student_bbox']
            student_pred = self.student(data_batch)
            data_batch.pop('image')
            # data_batch.pop('bbox')
        else:
            student_pred = self.student(data_batch)

        res = {}
        for key in student_pred.keys():
            res['teacher_'+key] = teacher_pred[key]
            res['student_'+key] = student_pred[key]
        return res
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.student.state_dict(destination, prefix, keep_vars)
    
    def init_weights(self, state_dict=None, strict=True):
        if state_dict:
            self.student.init_weights(state_dict, strict)
        else:
            self.student.init_weights(None, strict)
        self.teacher.init_weights(None, strict)
        # return super().init_weights(state_dict, strict)