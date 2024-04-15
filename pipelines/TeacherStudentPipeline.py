import os

class TeacherStudentPipeline:
    def __init__(self, student_transform, teacher_transform):
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
    def pipeline(self, inputs, targets, meta):
        student_inputs, student_targets = self.student_transform(inputs, targets)
        teacher_inputs, teacher_targets = self.teacher_transform(inputs, targets)
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