import torch
from abc import abstractmethod, ABCMeta
import os
class BaseSaver(metaclass=ABCMeta):

    def __init__(self, work_dir, data_settings, saver_settings):
        self.data_settings = data_settings
        self.work_dir = work_dir
        self.subdirectory = saver_settings['subdirectory']
        self.keys = saver_settings['keys']
        self.result_dir = os.path.join(work_dir, self.subdirectory)

    @abstractmethod
    def save(self, inputs, preds):
        pass