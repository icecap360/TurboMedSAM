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
                 val_freq=1,
                 save_freq=1,
                 save_optimizer=False,
                 max_iters=None,
                 max_epochs=None):
        
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.metrics = metric
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        self.grad_clip = grad_clip 
        self.distributed = distributed
        self.use_cpu = use_cpu
        self.device = device
        self.broadcast_bn_buffer = broadcast_bn_buffer
        
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
    def train_epoch(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(self, data_loaders, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str,
                        save_optimizer: bool = True,
                        create_symlink: bool = True) -> None:
        pass

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
        map_location = torch.device('cpu'),
        strict: bool = True,
        revise_keys = [] #[(r'^module.', '')],
    ):
        checkpoint = torch.load(filename, 
                                map_location=map_location,
                                )
        
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {filename}')
            
        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint 
             
        self.model.load_checkpoint(
            model_state_dict,
            logger = self.logger,
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
        
    def log_info(self, message):
        if not self.distributed or (self.distributed and self._rank == 0):
            self.logger.log(message)
    def log_train(self, message):
        if self._rank == 0:
            self.logger.info(message)
        else:
            self.logger.print_screen(message)
            
    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               resume_lr_scheduler: bool = True,
               map_location: Union[str, Callable] = 'default'):
        if map_location == 'default':
            checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if self.meta is None:
            self.meta = {}
        # self.meta.setdefault('hook_msgs', {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        # self.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

        # Re-calculate the number of iterations when resuming
        # models with different number of GPUs
        # if 'config' in checkpoint['meta']:
        #     config = mmcv.Config.fromstring(
        #         checkpoint['meta']['config'], file_format='.py')
        #     previous_gpu_ids = config.get('gpu_ids', None)
        #     if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
        #             previous_gpu_ids) != self.world_size:
        #         self._iter = int(self._iter * len(previous_gpu_ids) /
        #                          self.world_size)
        #         self.logger.info('the iteration number is changed due to '
        #                          'change of GPU number')

        # resume meta information meta
        self.meta = checkpoint['meta']

        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint and resume_lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
        self.log_info('resumed epoch {}, iter {}'.format(self._epoch, self._iter))
        
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
    

    