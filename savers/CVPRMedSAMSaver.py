import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
import torch.distributed as dist
from framework import BaseDataset, BaseSaver
import cv2
import random
import shutil
import multiprocessing as mp

__all__ = ['CVPRMedSAMSaver']

class CVPRMedSAMSaver(BaseSaver):
    """Face Landmarks dataset."""

    def __init__(self, work_dir, data_settings, saver_settings):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(work_dir, data_settings, saver_settings)
        self.root_data_dir = data_settings['dataset']['root_dir']
        shutil.rmtree(os.path.abspath(self.result_dir), ignore_errors=True)
        self.copy_dir_structure(self.root_data_dir, 
                                self.result_dir)
            
    def copy_dir_structure(self, root_data_dir, result_dir):
        for f in os.listdir(root_data_dir):
            has_npz = np.any(list(map(lambda x: '.npz' in x, os.listdir(os.path.join(root_data_dir, f)) )))
            if has_npz:
                os.makedirs(os.path.join(result_dir, f), exist_ok=True)
            else:
                self.copy_dir_structure(os.path.join(root_data_dir, f), os.path.join(result_dir, f))
        
    def save(self, inputs, preds):
        paths = inputs['meta']['npz_path']
        for i in range(len(paths)):
            path = paths[i]
            new_pred = {}
            pred = preds['embeddings'][i]
            new_pred['embeddings'] = pred.half().cpu().detach().numpy()
            new_path = os.path.join(self.result_dir, 
                                    path.replace(self.root_data_dir, ''))
            np.savez_compressed(new_path, **new_pred)
        # for i in range(len(paths)):
        #     path = paths[i]
        #     new_pred = {}
        #     for key in self.keys:
        #         assert key in preds.keys()
        #         pred = preds[key][i]
        #         if torch.is_tensor(pred):
        #             new_pred[key] = pred.cpu().detach().numpy()
        #         elif isinstance(pred, np.ndarray):
        #             new_pred[key] = pred
        #         else:
        #             new_pred[key] = np.array(pred)
        #     new_path = os.path.join(self.result_dir, 
        #                             path.replace(self.root_data_dir, ''))
        #     np.savez_compressed(new_path, **new_pred)

    def save_single(self, path, pred):
        new_pred = {}
        new_pred['embeddings'] = pred.half().cpu().detach().numpy()
        new_path = os.path.join(self.result_dir, 
                                path.replace(self.root_data_dir, ''))
        np.savez_compressed(new_path, **new_pred)
