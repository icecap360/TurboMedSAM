import os
from torch.utils.data import default_collate

class BasePipeline:
    def __init__(self, transform, collate_transform=None):
        self.transform = transform
        self.collate_transform = collate_transform
    def pipeline(self, inputs, targets, meta):
        inputs, targets = self.transform(inputs, targets)
        inputs['meta'] = meta
        return inputs, targets
    def collate_fn(self, batch):
        assert self.collate_transform, 'Collate_transform must be specified'
        return self.collate_transform(*default_collate(batch))