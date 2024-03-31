import torch
from abc import abstractmethod, ABCMeta
import os
class BaseSaver(metaclass=ABCMeta):

    def __init__(self, work_dir, data_settings, saver_settings):
        self.data_settings = data_settings
        self.work_dir = work_dir
        self.directory = saver_settings['directory']
        self.keys = saver_settings['keys']
        self.dtype = saver_settings.get('dtype')
        self.result_dir = self.directory
    @abstractmethod
    def save(self, inputs, preds):
        pass