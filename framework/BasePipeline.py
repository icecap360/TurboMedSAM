import os

class BasePipeline:
    def __init__(self, transform):
        self.transform = transform
    def pipeline(self, inputs, targets, meta):
        inputs, targets = self.transform(inputs, targets)
        inputs['meta'] = meta
        return inputs, targets