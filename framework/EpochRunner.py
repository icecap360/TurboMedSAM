import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
from abc import ABC, abstractmethod, ABCMeta 
from .BaseDataSamplers import ClassBalancedSampler
import time 
from .Distributed import get_dist_info, get_host_info
from .Hook import Hook
from collections import OrderedDict
import re
import shutil
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union, no_type_check)
from .BaseModules import BaseModule
from .Logger import Logger
from .BaseScheduler import BaseScheduler
from .BaseLoss import BaseLoss
from .BaseMetric import BaseMetric
from tqdm import tqdm
import torch.distributed as dist
from .utils import CustomDict, multi_gpu_test, single_gpu_test, dict_to_device, len_dict, collect_results_cpu, collect_results_gpu
from copy import deepcopy
import tempfile
from .BaseRunner import BaseRunner

class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self,
                 model: BaseModule,
                 optimizer: torch.optim,
                 loss: BaseLoss,
                 metric: BaseMetric,
                 lr_scheduler: BaseScheduler,
                 device,
                 work_dir,
                 logger: Logger,
                 grad_clip : dict,
                 distributed,
                 use_cpu=False,
                 broadcast_bn_buffer = True,
                 val_freq_epoch=1,
                 save_freq=1,
                 save_optimizer=False,
                 max_epochs=None,
                 batch_size=None):
        self.val_freq_epoch = val_freq_epoch
        super().__init__(model=model,
                 optimizer=optimizer,
                 loss=loss,
                 metric=metric,
                 lr_scheduler=lr_scheduler,
                 device=device,
                 work_dir=work_dir,
                 logger=logger,
                 grad_clip=grad_clip,
                 distributed=distributed,
                 use_cpu=use_cpu,
                 broadcast_bn_buffer=broadcast_bn_buffer,
                 save_freq=save_freq,
                 save_optimizer=save_optimizer,
                 max_iters=None,
                 max_epochs=max_epochs,
                 batch_size=batch_size)

    def train_epoch(self, data_loader: DataLoader):
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.model.train()
        
        samples_processed = 0
        for i, data_batch in enumerate(data_loader):
            inputs, targets = data_batch[0], data_batch[1]
            batch_size = len_dict(inputs)
            targets = dict_to_device(targets, self.device)
            inputs = dict_to_device(inputs, self.device)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            
            loss_dict = self.loss.forward_loss(preds, targets)
            
            weighted_sum_loss = self.loss.calc_weighted_loss(loss_dict, self.loss.loss_weight, requires_grad=True, device=self.device)
            
            float_loss_dict = {key: value.item() for key, value in loss_dict.items()}
            float_weighted_sum_loss = weighted_sum_loss.item()
            
            samples_processed += batch_size
            
            train_message = self.logger.train_epoch_message(
                    epoch = self._epoch+1,
                    rank = self._rank,
                    max_epoch = self._max_epochs,
                    samples_processed = samples_processed,
                    total_samples = len(data_loader.sampler),
                    lr = self.current_lr(),
                    loss_dict = float_loss_dict,
                    total_loss = float_weighted_sum_loss
                )
            self.log_train(train_message)
            
            weighted_sum_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm = self.grad_clip['max_norm'], 
                norm_type = self.grad_clip['norm_type'])
            self.optimizer.step()
            self.lr_scheduler.step_iter()
            
            self.call_hook('after_train_iter')
            del data_batch, inputs, preds, batch_size
            self._iter += 1
        
        self.lr_scheduler.step()
        if self._epoch % self.save_freq == 0:
            if self.distributed:
                dist.barrier()
                if self._rank != 0:
                    dist.barrier()
                else:
                    self.save_checkpoint(
                        out_dir = self.work_dir,
                        save_optimizer = self.save_optimizer,
                        save_scheduler = True,
                        filename = 'epoch_{epoch}.pth'.format(
                            epoch=self._epoch
                        ))
                    dist.barrier()
            else:
                self.save_checkpoint(
                    out_dir = self.work_dir,
                    save_optimizer = self.save_optimizer,
                    save_scheduler = True,
                    filename = 'epoch_{epoch}.pth'.format(
                            epoch=self._epoch
                        ))
        self.call_hook('after_train_epoch')
    
    # @torch.no_grad()
    def val(self, data_loader):
        if self.distributed:
            dist.barrier()
        self.call_hook('before_val_epoch')
        self.log_info_and_print('\nVALIDATING\n')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.model.eval()
        
        loss_dict, metrics_dict = self.get_results(data_loader, gpu_collect=True)
            
        if self.distributed and self._rank != 0:
            dist.barrier()
        else:
            weighted_sum_loss = self.loss.calc_weighted_loss(loss_dict, self.loss.loss_weight, requires_grad=False, device=self.device)
        
            float_loss_dict = {key: value.item() for key, value in loss_dict.items()}
            float_metrics_dict = {key: value.item() for key, value in metrics_dict.items()}
            float_weighted_sum_loss = weighted_sum_loss.item()
            
            val_message = self.logger.val_epoch_message(
                        loader_name = data_loader.get_name(),
                        epoch = self._epoch + 1,
                        max_epoch = self._max_epochs,
                        loss_dict = float_loss_dict,
                        total_loss = float_weighted_sum_loss,
                        metrics_dict = float_metrics_dict
                    )
            self.log_info_and_print(val_message)
        
        if self.distributed and self._rank == 0:
            dist.barrier()
        
        self.call_hook('after_val_epoch')

    def run(self,
            data_loader_train: DataLoader,
            data_loader_val: DataLoader,
            ) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        work_dir = self.work_dir 
        
        self.log_info_and_print('Start running, host: {}, work_dir: {}'.format(get_host_info(), work_dir))
        self.log_info_and_print('Hooks will be executed in the following order:\n{}'.format(self.get_hook_info()))
        self.log_info_and_print('max: {} epochs'.format(self._max_epochs))
        self.call_hook('before_run')

        while self._epoch < self._max_epochs:
            data_loader_train.sampler.set_epoch(self._epoch)
            self.train_epoch(data_loader=data_loader_train)
            for loader in data_loader_val:
                loader.sampler.set_epoch(self._epoch)
            if self._epoch % self.val_freq_epoch == 0:
                for loader in data_loader_val:
                    self.val(data_loader=loader,)
            
            self._epoch += 1

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
                
    # def qasim_result_collect(self, data_loader):
    #     predictions = CustomDict()
    #     labels = CustomDict()
    #     rank, _ = get_dist_info()

    #     for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
    #         self.call_hook('before_val_iter')
    #         batch_size = len_dict(data_batch[0])
    #         assert len(data_batch[0]) == 2
    #         inputs, label = data_batch[:, 0], data_batch[:, 1]
                            
    #         with torch.no_grad():
    #             self.preds = self.model(inputs)
    #             predictions.add_dictlist(self.preds)
    #             labels.add_dictlist(label)
    #         self.call_hook('after_val_iter')

    #     if self.distributed:
    #         dist.barrier()
    #         if rank != 0:
    #             dist.barrier()
    #         else:
    #             predictions_all = [deepcopy(p) for p in predictions]
    #             labels_all = [deepcopy(p) for p in labels]
    #             dist.all_gather_object(predictions_all, predictions)
    #             dist.all_gather_object(predictions_all, predictions)
                
    #             predictions = CustomDict()
    #             labels = CustomDict()
    #             for p in predictions_all:
    #                 predictions.add_dictlist(p.data)
    #             for l in labels_all:
    #                 labels.add_dictlist(label.data)
    #         dist.barrier()
    #     return predictions, labels
    # def mmcv_result_collect(self, data_loader: DataLoader):
    #     # Synchronization of BatchNorm's buffer (running_mean
    #     # and running_var) is not supported in the DDP of pytorch,
    #     # which may cause the inconsistent performance of models in
    #     # different ranks, so we broadcast BatchNorm's buffers
    #     # of rank 0 to other ranks to avoid this.
    #     if self.broadcast_bn_buffer:
    #         model = self.model
    #         for name, module in model.named_modules():
    #             if isinstance(module,
    #                           nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
    #                 dist.broadcast(module.running_var, 0)
    #                 dist.broadcast(module.running_mean, 0)
                    
    #     tmpdir = tempfile.TemporaryDirectory()
    #     if tmpdir is None:
    #         tmpdir = os.path.join(self.work_dir, '.eval_hook')

    #     # Changed results to self.results so that MMDetWandbHook can access
    #     # the evaluation results and log them to wandb.
    #     if self.distributed:
    #         results = multi_gpu_test(
    #             self.model,
    #             data_loader,
    #             tmpdir=tmpdir,
    #             gpu_collect=(not self.use_cpu))
    #     else:
    #         results = single_gpu_test(self.model, 
    #                                   data_loader)
        
    #     tmpdir.cleanup()

    #     return results