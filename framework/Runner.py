import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
from abc import ABC, abstractmethod, ABCMeta 
from BaseDataSamplers import ClassBalancedSampler
from Registry import Registry, build_from_cfg
import time 
from Distributed import get_dist_info
from Hook import Hook
from collections import OrderedDict
import re
import shutil
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union, no_type_check)
from BaseModules import BaseModule
from Logger import Logger

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
                 loss,
                 metrics,
                 work_dir,
                 logger: Logger,
                 val_freq=1,
                 save_freq=1,
                 save_optimizer=False,
                 max_iters=None,
                 max_epochs=None):
        
        assert hasattr(model, 'train_step')

        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.loss = loss
        self.metrics = metrics
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        
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
    def train(self):
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
                      hook,
                      priority) -> None:
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
        return self.model.load_checkpoint(
            filename,
            logger = self.logger,
            map_location = map_location,
            strict = strict,
            revise_keys = revise_keys)
        
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

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
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
            
        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
        
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


class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if train_mode:
            self.model.train()
            outputs = self.model(data_batch,
                                **kwargs)
        else:
            self.model.eval()
            outputs = self.model(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"model forward must return a dict')
        self.outputs = outputs

    def train(self, data_loader):
        
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            
            self.model.train()
            self.preds = self.model(data_batch)
            loss_dict, total_loss = self.loss.forward(self.preds)
            self.logger.train_step(
                epoch=self.epoch,
                max_epoch=self._max_epochs,
                batch_index=i*len(data_batch),
                total_batches=len(self.data_loader),
                lr=self.current_lr(),
                loss_dict=loss_dict,
                total_loss = total_loss
            )
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            
            
            
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader):
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            # run_iter valvulates the loss
            self.run_iter(data_batch, train_mode=False, **kwargs)
            # now we calculate the metrics
            
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')

    def run(self,
            data_loader_train: DataLoader,
            data_loader_val: DataLoader,
            max_epochs: Optional[int] = None,
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
        if max_epochs is not None:
            self.logger.warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        if self.mode == 'train':
            self._max_iters = self._max_epochs * len(data_loader_train)

        work_dir = self.work_dir 
        
        self.logger.info('Start running, host: %s, work_dir: %s',
                         (), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('max: %d epochs',
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            data_loader_train.sampler.set_epoch(self.epoch)
            self.train(data_loader=data_loader_train)
            
            data_loader_val.sampler.set_epoch(self.epoch)
            if self.epoch % self.val_freq == 0:
                self.val(data_loader=data_loader_val,)
            
            if self.epoch % self.save_freq == 0:
                self.save_checkpoint(self, self.work_dir)
            self.epoch += 1

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
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
        
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = os.path.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        self.save_checkpoint(self, filepath,save_optimizer, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = os.path.join(out_dir, 'latest.pth')
            try:
                os.symlink(filepath, dst_file)
            except:
                shutil.copy(filepath, dst_file)
                

    def save_checkpoint(self, filepath, save_optimizer, meta=None):
        if hasattr(self.model, 'CLASSES') and self.model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=self.model.CLASSES)
        
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(get_state_dict(model))  # type: ignore
        }
        
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        