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
    def train_epoch(self, data_loader: DataLoader):
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        rank, _ = get_dist_info()
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.model.train()
        data_loader.sampler.set_epoch(self._epoch) 
        
        samples_processed = 0
        for i, data_batch in enumerate(data_loader):
            
            inputs, targets = data_batch[0], data_batch[1]
            batch_size = len_dict(inputs)
            targets = dict_to_device(targets, self.device)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            
            loss_dict = self.loss.forward_loss(preds, targets)
            
            weighted_sum_loss = self.loss.calc_weighted_loss(loss_dict, self.loss.loss_weight, requires_grad=True, device=self.device)
            
            float_loss_dict = {key: value.item() for key, value in loss_dict.items()}
            float_weighted_sum_loss = weighted_sum_loss.item()
            
            samples_processed += batch_size
            
            train_message = self.logger.train_message(
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
                if rank != 0:
                    dist.barrier()
                else:
                    self.save_checkpoint(
                        out_dir = self.work_dir,
                        save_optimizer = self.save_optimizer,
                        save_scheduler = True)
                    dist.barrier()
            else:
                self.save_checkpoint(
                    out_dir = self.work_dir,
                    save_optimizer = self.save_optimizer,
                    save_scheduler = True)
        self.call_hook('after_train_epoch')
        self._epoch += 1

    def var_collect(self, variable, tmpdir_path_str, size: int, gpu_collect: bool = True): 
        tmpdir = tempfile.TemporaryDirectory()
        if tmpdir is None:
            tmpdir = tmpdir_path_str

        # collect results from all ranks
        if gpu_collect:
            result_from_ranks = collect_results_gpu(variable, size)
        else:
            result_from_ranks = collect_results_cpu(variable, size, tmpdir)
        
        tmpdir.cleanup()
        return result_from_ranks

    def get_results(self, data_loader: DataLoader, gpu_collect = False):
        # Synchronization of BatchNorm's buffer (running_mean and running_var) is not supported in the DDP of pytorch, which may cause the inconsistent performance of models in different ranks, so we broadcast BatchNorm's buffers of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = self.model
            for name, module in model.named_modules():
                if isinstance(module,
                              nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = tempfile.TemporaryDirectory()
        if tmpdir is None:
            tmpdir = os.path.join(self.work_dir, '.eval_hook')

        model.eval()
        results = []
        loss_dicts = []
        metrics_dicts = []
        time.sleep(2)  # This line can prevent deadlock problem in some multi-gpu cases
        for data in tqdm(data_loader, total=len(data_loader)):
            with torch.no_grad():
                inputs, targets = data[0], data[1]
                targets = dict_to_device(targets, self.device)
                preds = model(inputs)
                batch_loss_dict = self.loss.forward_loss(preds, targets)
                batch_metrics_dict = self.metrics.get_metrics(preds, targets, self.device)

            loss_dicts.append(batch_loss_dict)
            metrics_dicts.append(batch_metrics_dict)
            del batch_loss_dict, batch_metrics_dict
        
        summed_loss_dict = self.loss.sum_loss(loss_dicts, self.device)
        summed_metrics_dict = self.metrics.sum_metrics(metrics_dicts, self.device)  
        
        if self.distributed:
            # collect results from all ranks
            if gpu_collect:
                losses_from_ranks = collect_results_gpu(summed_loss_dict, self._world_size)
                metrics_from_ranks = collect_results_gpu(summed_metrics_dict, self._world_size)
            else:
                losses_from_ranks = collect_results_cpu(summed_loss_dict, self._world_size, tmpdir)
                metrics_from_ranks = collect_results_cpu(summed_metrics_dict, self._world_size, tmpdir)
            if self._rank == 0:
                return self.loss.sum_loss(losses_from_ranks, self.device), self.metrics.sum_metrics(metrics_from_ranks, self.device)  
            else:
                return None, None
        else:
            return summed_loss_dict, summed_metrics_dict
        
    
    # @torch.no_grad()
    def val(self, data_loader):
        self.call_hook('before_val_epoch')
        self.log_info('\nVALIDATING\n')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        rank, _ = get_dist_info()
        self.model.eval()
        # predictions, labels = self.qasim_result_collect(data_loader)
        
        loss_dict, metrics_dict = self.get_results(data_loader, gpu_collect=True)
        
        if self.distributed and rank != 0:
            dist.barrier()
        else:
            weighted_sum_loss = self.loss.calc_weighted_loss(loss_dict, self.loss.loss_weight, requires_grad=False, device=self.device)
        
            float_loss_dict = {key: value.item() for key, value in loss_dict.items()}
            float_metrics_dict = {key: value.item() for key, value in metrics_dict.items()}
            float_weighted_sum_loss = weighted_sum_loss.item()
            
            val_message = self.logger.val_message(
                        epoch = self._epoch,
                        max_epoch = self._max_epochs,
                        loss_dict = float_loss_dict,
                        total_loss = float_weighted_sum_loss,
                        metrics_dict = float_metrics_dict
                    )
            self.log_info(val_message)
        
        if self.distributed and rank == 0:
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

        # if self.mode == 'train':
        self._max_iters = self._max_epochs * len(data_loader_train)

        work_dir = self.work_dir 
        
        self.log_info('Start running, host: {}, work_dir: {}'.format(get_host_info(), work_dir))
        self.log_info('Hooks will be executed in the following order:\n{}'.format(self.get_hook_info()))
        self.log_info('max: {} epochs'.format(self._max_epochs))
        self.call_hook('before_run')

        while self._epoch < self._max_epochs:
            data_loader_train.sampler.set_epoch(self._epoch)
            self.train_epoch(data_loader=data_loader_train)
            
            data_loader_val.sampler.set_epoch(self._epoch)
            if self._epoch % self.val_freq == 0:
                self.val(data_loader=data_loader_val,)
            
            self._epoch += 1

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        save_scheduler: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        
        # if self.meta is not None:
            # meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self._epoch + 1, iter=self._iter)

        filename = filename_tmpl.format(self._epoch + 1)
        filepath = os.path.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        
        if hasattr(self.model, 'CLASSES') and self.model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=self.model.CLASSES)
        
        checkpoint = {
            'meta': meta,
            'state_dict': self.model.state_dict(),  # type: ignore
        }
        
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        if save_scheduler:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, filepath)
        
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = os.path.join(out_dir, 'latest.pth')
            if os.path.exists(dst_file):
                os.remove(dst_file)
            try:
                os.symlink(filepath, dst_file)
            except:
                shutil.copy(filepath, dst_file, )
                
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