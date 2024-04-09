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

class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """

    def __init__(self,
                 model: BaseModule,
                 optimizer: torch.optim.Optimizer,
                 loss: BaseLoss,
                 metric: BaseMetric,
                 lr_scheduler: BaseScheduler,
                 device,
                 work_dir,
                 logger: Logger,
                 distributed,
                 batch_size: int,
                 samples_per_gpu: int,
                 use_cpu=False,
                 broadcast_bn_buffer = True,
                 save_freq=1,
                 log_freq = 1,
                 save_optimizer=False,
                 max_iters=None,
                 max_epochs=None,
                 use_amp = False,
                 grad_clip = None,
                 resume_train = False,
                 resume_checkpoint = None):
        
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.metrics = metric
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        self.log_freq = log_freq
        self.grad_clip = grad_clip 
        self.distributed = distributed
        self.use_cpu = use_cpu
        self.device = device
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.batch_size = batch_size
        self.samples_per_gpu = samples_per_gpu
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.resume_train = resume_train
        self.resume_checkpoint = resume_checkpoint

        # create work_dir
        if isinstance(work_dir, str):
            self.work_dir = os.path.abspath(work_dir)
            os.makedirs(self.work_dir, exist_ok=True)

        self._rank, self._world_size = get_dist_info()
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        
        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters

        self.loss.initialize(self.device)

        assert (self.resume_train and self.resume_checkpoint) or not self.resume_train
        if self.resume_train:
            if not os.path.isabs(self.resume_checkpoint):
                self.resume_checkpoint = os.path.join(self.work_dir, self.resume_checkpoint) 
            assert os.path.exists(self.resume_checkpoint), 'Checkpoint must be a valid path'
            self.resume(self.resume_checkpoint, self.distributed, map_location=self.device, 
                        resume_lr_scheduler=True)

    @property
    def model_name(self) -> str:
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self) -> int:
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self) -> int:
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self) -> int:
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters


    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(self, data_loaders, **kwargs):
        pass
    
    def save_checkpoint(self,
                        out_dir: str,
                        filename: str,
                        save_optimizer: bool = True,
                        save_scheduler: bool = True,
                        save_scaler: bool = True,
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
        meta.update(epoch=self._epoch, iter=self._iter, 
                    logger=self.logger.path, batch_size=self.batch_size,
                    samples_per_gpup=self.samples_per_gpu)

        filepath = os.path.join(out_dir, filename)
        
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
        if save_scaler:
            checkpoint['scaler'] = self.scaler.state_dict()
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

    def current_lr(self):
        lr = [group['lr'] for group in self.optimizer.param_groups]
        return lr

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        momentums = []
        for group in self.optimizer.param_groups:
            if 'momentum' in group.keys():
                momentums.append(group['momentum'])
            elif 'betas' in group.keys():
                momentums.append(group['betas'][0])
            else:
                momentums.append(0)
        return momentums

    def register_hook(self,
                      hook) -> None:
        self._hooks.append(hook)

    def call_hook(self, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(
        self,
        filename: str,
        distributed,
        map_location = torch.device('cpu'),
        strict: bool = True,
        load_optimizer: bool = True,
        load_lr_scheduler: bool = True,
        load_scaler: bool = True,
        revise_keys = [],
    ):
        checkpoint = torch.load(filename, 
                                map_location=map_location,
                                )
        
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {filename}')
            
        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
            if 'optimizer' in checkpoint and load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint and load_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'scaler' in checkpoint and load_scaler:
                self.scaler.load_state_dict(checkpoint['scaler'])
            self._epoch = checkpoint['meta']['epoch']
            self._iter = checkpoint['meta']['iter']
        else:
            model_state_dict = checkpoint 
        
        if distributed:
            self.model.module.load_checkpoint(
                model_state_dict,
                strict = strict,
                revise_keys = revise_keys)
        else:
            self.model.load_checkpoint(
                model_state_dict,
                strict = strict,
                revise_keys = revise_keys)
        
        return checkpoint
        # state_dict = torch.load(filename, 
        #                         map_location=map_location,
        #                         )
        # if not isinstance(state_dict, dict):
        #     raise RuntimeError(
        #         f'No state_dict found in checkpoint file {filename}')
        # # get state_dict from checkpoint
        # if 'state_dict' in state_dict:
        #     state_dict = state_dict['state_dict']
        
        # # strip prefix of state_dict
        # metadata = getattr(state_dict, '_metadata', OrderedDict())
        # for p, r in revise_keys:
        #     state_dict = OrderedDict(
        #         {re.sub(p, r, k): v
        #         for k, v in state_dict.items()})
        # # Keep metadata in state_dict
        # state_dict._metadata = metadata
        
        # self.model.load_state_dict(state_dict, strict= strict)
        # load_state_dict(model, state_dict, strict, logger)
        # return checkpoint
        
    def log_info_and_print(self, message):
        if not self.distributed or (self.distributed and self._rank == 0):
            self.logger.info_and_print(message)
    def log_train(self, message):
        if not self.distributed or (self.distributed and self._rank == 0):
            self.logger.info_and_print(message)
        else:
            self.logger.info(message)
            
    @no_type_check
    def resume(self,
               checkpoint: str,
               distributed, 
               resume_optimizer: bool = True,
               resume_lr_scheduler: bool = True,
               resume_scaler: bool = True,
               map_location: Union[str, Callable] = 'cpu'):
        self.load_checkpoint(checkpoint, distributed, map_location, strict=True, 
                             load_optimizer=resume_optimizer, 
                             load_lr_scheduler=resume_lr_scheduler, 
                             load_scaler=resume_scaler, 
                             revise_keys= [(r'^module.', ''), (r'^_orig_mod.', '')])
        self.log_info_and_print('resumed epoch {}, iter {}'.format(self._epoch, self._iter))
        
    def get_hook_info(self) -> str:
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)
    
    def get_results(self, data_loader: DataLoader, gpu_collect = False):
        model = self.model

        # Synchronization of BatchNorm's buffer (running_mean and running_var) is not supported in the DDP of pytorch, which may cause the inconsistent performance of models in different ranks, so we broadcast BatchNorm's buffers of rank 0 to other ranks to avoid this.
        if self.distributed and self.broadcast_bn_buffer:
            for name, module in model.named_modules():
                if isinstance(module,
                              nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)


        model.eval()
        loss_dicts = []
        metrics_dicts = []
        time.sleep(2)  # This line can prevent deadlock problem in some multi-gpu cases
        for data in tqdm(data_loader, position=self._rank, total=len(data_loader)):
            with torch.no_grad():
                inputs, targets = data[0], data[1]
                targets = dict_to_device(targets, self.device)
                inputs = dict_to_device(inputs, self.device)
                preds = model(inputs)
                batch_loss_dict = self.loss.forward_loss(preds, targets)
                batch_metrics_dict = self.metrics.get_metrics(preds, targets, self.device)
                loss_dicts.append({k:v.item() for k,v in batch_loss_dict.items()})
                metrics_dicts.append({k:v.item() for k,v in batch_metrics_dict.items()})
                del batch_loss_dict, batch_metrics_dict, inputs, preds
        
        avg_loss_dict = self.loss.average_loss_float(loss_dicts, self.device)
        avg_metrics_dict = self.metrics.average_metrics_float(metrics_dicts, self.device)  
        
        if self.distributed:
            # collect results from all ranks
            if gpu_collect:
                losses_from_ranks = collect_results_gpu(avg_loss_dict, self._world_size)
                metrics_from_ranks = collect_results_gpu(avg_metrics_dict, self._world_size)
            else:
                tmpdir = tempfile.TemporaryDirectory()
                if tmpdir is None:
                    tmpdir = os.path.join(self.work_dir, '.eval_hook')
                losses_from_ranks = collect_results_cpu(avg_loss_dict, self._world_size, tmpdir)
                metrics_from_ranks = collect_results_cpu(avg_metrics_dict, self._world_size, tmpdir)
            if self._rank == 0:
                return self.loss.average_loss_float(losses_from_ranks, self.device), self.metrics.average_metrics_float(metrics_from_ranks, self.device)  
            else:
                return None, None
        else:
            return avg_loss_dict, avg_metrics_dict

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
    
    # @torch.no_grad()
    def val(self, iters, max_iters, data_loader):
        if self.distributed:
            dist.barrier()
        self.call_hook('before_val_epoch')
        self.log_info_and_print('\n\n-------VALIDATING-------')
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
            
            val_message = self.logger.val_message(
                        loader_name = data_loader.get_name(),
                        iters= iters,
                        max_iters = max_iters,
                        loss_dict = float_loss_dict,
                        total_loss = float_weighted_sum_loss,
                        metrics_dict = float_metrics_dict
                    )
            self.log_info_and_print(val_message)
        
        if self.distributed and self._rank == 0:
            dist.barrier()