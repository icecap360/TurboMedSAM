import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from .BaseModules import BaseModule
from torch.optim.lr_scheduler import _LRScheduler

class BaseScheduler(_LRScheduler):
    '''
    Implements iter based warmup for the scheduler class
    '''
    def __init__(self, 
                 regular_scheduler,
                 optimizer, 
                 last_epoch=-1, 
                 verbose=False, 
                 warmup=None,
                 warmup_by_epoch=False,
                 warmup_iters=0,
                 warmup_epochs=0,
                 warmup_ratio=0.1,
                 warmup_value = None):
        
        assert optimizer == regular_scheduler.optimizer
        
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant_ratio', 'linear', 'exp', 'constant_value']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant_ratio" and "linear"')
            assert (warmup_by_epoch and warmup_epochs>0) or (not warmup_by_epoch and warmup_iters>0), 'warmup epochs or warmup iters must be specified'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'
        if warmup == 'constant_value':
            assert not (warmup_value == None)            

        self.regular_scheduler = regular_scheduler
        self.regular_lr = regular_scheduler.get_last_lr()
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_value = warmup_value
        self.warmup_epochs = warmup_epochs
        self.warmup_by_epoch = warmup_by_epoch
        self.iter_cnt = 0

        self._last_lr = self.regular_scheduler.get_last_lr()
        
        super(BaseScheduler, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        self.update_lr()
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        # for group in self.optimizer.param_groups:
        #         group.setdefault('initial_lr', group['lr'])
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'base_scheduler')}
        state_dict['base_scheduler'] = {
            "base_scheduler": self.regular_scheduler.state_dict()
            }
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        base_scheduler = state_dict.pop('base_scheduler')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['base_scheduler'] = base_scheduler

        self.regular_scheduler.load_state_dict(base_scheduler)

    def step_iter(self):
        self.iter_cnt += 1
        self.update_lr()

    def step(self, epoch=None):
        self.regular_scheduler.step(epoch=epoch)
        self.regular_lr = self.regular_scheduler.get_last_lr()
        self._last_lr = self.regular_lr
        self.update_lr()
        self.last_epoch += 1

    def update_lr(self):
        if self.warmup is None or \
                (not self.warmup_by_epoch and self.iter_cnt > self.warmup_iters) or \
                (self.warmup_by_epoch and self.last_epoch > self.warmup_epochs):
            return
        elif (not self.warmup_by_epoch and self.iter_cnt == self.warmup_iters) or \
            (self.warmup_by_epoch and self.last_epoch == self.warmup_epochs):
            self._last_lr = self.regular_lr
        else:
            if self.warmup_by_epoch:
                self._last_lr = self.get_warmup_lr_epochs()
            else:
                self._last_lr = self.get_warmup_lr_iters()

    def get_warmup_lr_iters(self):        
        return self._get_warmup_lr(self.regular_lr, self.iter_cnt, self.warmup_iters)
    def get_warmup_lr_epochs(self):
        return self._get_warmup_lr(self.regular_lr, self.last_epoch, self.warmup_epochs)

    def _get_warmup_lr(self, regular_lr, iters, total ):
        if self.warmup == 'constant_ratio':
            warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
        elif self.warmup == 'linear':
            k = (1 - iters / total) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - iters / total)
            warmup_lr = [_lr * k for _lr in regular_lr]
        elif self.warmup == 'constant_value':
            warmup_lr = [self.warmup_value for _ in regular_lr]
        return warmup_lr