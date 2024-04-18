import torch
from abc import abstractmethod, ABCMeta
import os
import shutil

class BaseSaver(metaclass=ABCMeta):

    def __init__(self, work_dir, data_settings, saver_settings):
        self.data_settings = data_settings
        self.work_dir = work_dir
        assert os.path.exists(saver_settings['directory'])
        self.directory = saver_settings['directory']
        self.keys = saver_settings['keys']
        self.dtype = saver_settings.get('dtype')
        self.result_dir = os.path.join(saver_settings['directory'], 'results')
        shutil.rmtree(os.path.abspath(self.result_dir), ignore_errors=True)
        os.makedirs(self.result_dir, exist_ok=True)
        

    @abstractmethod
    def save(self, inputs, preds):
        pass