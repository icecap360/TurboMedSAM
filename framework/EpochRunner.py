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
from .Profiling import AverageMeter, ProgressMeter
from torch.distributed.optim import ZeroRedundancyOptimizer

class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self,
                 model: BaseModule,
                 optimizer: dict,
                 loss: BaseLoss,
                 metric: BaseMetric,
                 lr_scheduler: BaseScheduler,
                 device,
                 work_dir,
                 logger: Logger,
                 distributed,
                 compute: dict,
                 runner: dict,
                 save_optimizer=False):
        self.val_freq_epoch = runner['val_freq_epoch']
        super().__init__(model=model,
                 optimizer=optimizer['optimizer'],
                 loss=loss,
                 metric=metric,
                 lr_scheduler=lr_scheduler,
                 device=device,
                 work_dir=work_dir,
                 logger=logger,
                 distributed=distributed,
                 use_cpu=compute['use_cpu'],
                 broadcast_bn_buffer=compute['broadcast_bn_buffer'],
                 save_freq=runner['save_freq_iter'],
                 save_optimizer=save_optimizer,
                 max_iters=None,
                 max_epochs=runner['max_epochs'],
                 batch_size=compute['batch_size'],
                 samples_per_gpu=compute['samples_per_gpu'],
                 log_freq = runner['log_freq'],
                 use_amp=compute['use_amp'],
                 grad_clip=optimizer.get('grad_clip'),
                 resume_train=runner['resume_train'],
                 checkpoint_path=runner.get('checkpoint_path'),
                 resume_dataloader=runner.get('resume_dataloader')
                 )

    def train_epoch(self, data_loader: DataLoader):
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.model.train()
        
        batch_time = AverageMeter('Model', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        debug_time = AverageMeter('Debug', ':6.3f')
        progress = ProgressMeter(
            len(data_loader.sampler),
            [batch_time, data_time])
        end = time.time()

        samples_processed = 0

        for i, data_batch in enumerate(data_loader):
            inputs, targets = data_batch[0], data_batch[1]
            batch_size = len(inputs['meta']['idx'])
            targets = dict_to_device(targets, self.device)
            inputs = dict_to_device(inputs, self.device)
            data_time.update(time.time() - end)
            end = time.time()
            
            self._inner_iter = i
            self.call_hook('before_train_iter')
            if self.distributed:
                dist.barrier()
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                preds = self.model(inputs)    
                loss_dict = self.loss.forward_loss(preds, targets)
                weighted_sum_loss = self.loss.calc_weighted_loss(loss_dict, self.loss.loss_weight, requires_grad=True, device=self.device)

            scaled_loss = self.scaler.scale(weighted_sum_loss)
            scaled_loss.backward(retain_graph=True)

            if self.grad_clip != None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm = self.grad_clip['max_norm'], 
                    norm_type = self.grad_clip['norm_type'])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.lr_scheduler.step_iter()

            batch_time.update(time.time() - end)
            samples_processed += batch_size

            if i % self.log_freq == 0:
                float_loss_dict = {key: value.item() for key, value in loss_dict.items()}
                float_weighted_sum_loss = weighted_sum_loss.item()
                train_message = self.logger.train_epoch_message(
                        rank = self._rank,
                        epoch = self._epoch,
                        max_epoch = self._max_epochs,
                        samples_processed = samples_processed,
                        total_samples = len(data_loader.sampler),
                        lr = self.current_lr(),
                        loss_dict = float_loss_dict,
                        total_loss = float_weighted_sum_loss,
                        progress = progress
                    )
                self.log_train(train_message)
                
            if i % self.save_freq == 0:
                if isinstance(self.optimizer, ZeroRedundancyOptimizer) :
                    self.optimizer.consolidate_state_dict(0)
                if self._rank ==0:
                    filename = 'epoch_{epoch}_{iter}.pth'.format(
                                    epoch=self._epoch,
                                    iter=i
                                )
                    self.logger.info_and_print('Saving to checkpoint '+filename+' ...')
                    self.save_checkpoint(
                                out_dir = self.work_dir,
                                save_optimizer = self.save_optimizer,
                                save_scheduler = True,
                                filename = filename)
                
            self.call_hook('after_train_iter')
            del data_batch, inputs, preds, batch_size
            self._iter += 1
            end = time.time()

        self.lr_scheduler.step()
        self.call_hook('after_train_epoch')
    
    # @torch.no_grad()
    def val(self, data_loader, epoch):
        super().val(epoch, self._max_epochs, data_loader)
    
    def run(self,
            data_loader_train: DataLoader,
            data_loader_val: DataLoader,
            remaining_train_loader: DataLoader = None,
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
        self.logger.reset_start_time_eta()
        
        while self._epoch < self._max_epochs:

            if not (remaining_train_loader==None) and self.resume_dataloader:
                remaining_train_loader.sampler.set_epoch(self._epoch)
                self.train_epoch(data_loader=remaining_train_loader)
                del remaining_train_loader
                remaining_train_loader = None
            else:
                data_loader_train.sampler.set_epoch(self._epoch)
                self.train_epoch(data_loader=data_loader_train)

            for loader in data_loader_val:
                loader.sampler.set_epoch(self._epoch)
            
            self._epoch += 1
            if isinstance(self.optimizer, ZeroRedundancyOptimizer) :
                    self.optimizer.consolidate_state_dict(0)
            if self._rank ==0:
                filename = 'epoch_{epoch}.pth'.format(
                                epoch=self._epoch,
                            )
                self.logger.info_and_print('Saving to checkpoint '+filename+' ...')
                self.save_checkpoint(
                            out_dir = self.work_dir,
                            save_optimizer = self.save_optimizer,
                            save_scheduler = True,
                            filename =filename)
            if self._epoch % self.val_freq_epoch == 0:
                for loader in data_loader_val:
                    self.val(loader, self._epoch)

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
        
def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape, torch.max(grad), torch.min(grad), torch.max(torch.abs(grad)), torch.std(grad), torch.mean(grad))
        except AttributeError: 
            print ("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:  
        try:
            print(grad.shape, torch.max(grad), torch.min(grad), torch.max(torch.abs(grad)), torch.std(grad), torch.mean(grad))
        except AttributeError: 
            print ("None found for Gradient")
    print("\n")