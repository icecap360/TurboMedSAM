import os
from framework import BasePipeline
from torchvision.transforms import v2
from torch.utils.data import default_collate
import random 

class TeacherStudentPipeline(BasePipeline):
    def __init__(self, transform, student_transform=None, teacher_transform=None, target_transform=None, collate_transforms= None, collate_functionals=None):
        self.transform = transform
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        self.collate_functionals = collate_functionals
        self.collate_transforms = collate_transforms
        self.target_transform = target_transform
        if collate_transforms:
            assert len(self.collate_functionals) == len(self.collate_transforms)
        
    def pipeline(self, inputs, targets, meta):
        inputs, targets = self.transform(inputs, targets)
        if self.student_transform:
            if not self.target_transform:
                student_inputs, student_targets = self.student_transform(inputs, targets)
                
        else:
            inputs, targets = inputs, targets
        if self.teacher_transform:
            teacher_inputs, teacher_targets = self.teacher_transform(inputs, targets)
        else:
            teacher_inputs, teacher_targets = inputs, targets
        inputs, targets = {}, {}
        inputs['meta'] = meta
        for k in student_inputs:
            inputs['student_'+k] = student_inputs[k]
        for k in student_targets:
            targets['student_'+k] = student_targets[k]
        for k in teacher_inputs:
            inputs['teacher_'+k] = teacher_inputs[k]
        for k in teacher_targets:
            targets['teacher_'+k] = teacher_targets[k]
        return inputs, targets
    
    def collate_fn(self, batch):
        batch = default_collate(batch)
        index = random.choice(range(len(self.collate_transforms)))
        transform = self.collate_transforms[index]
        functional = self.collate_functionals[index]
        params = transform._get_params(batch[0]['student_image'])
        batch[0]['student_image'] = functional(
            batch[0]['student_image'],
            params
        )
        batch[0]['teacher_image'] = functional(
            batch[0]['teacher_image'],
            params
        )
        return batch