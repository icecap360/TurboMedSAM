import os
import torch.nn as nn
from typing import Iterable, Optional
import copy
import torch
from abc import ABCMeta

class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg: Optional[dict] = None):
        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)
        
        self.initialize()
        
    def initialize(self):
        self.init_weights()
        
    def init_weights(self):
        if self.init_cfg is None:
            return
        elif 'pretrained' in self.init_cfg['type'].lower():
            if not 'checkpoint' in self.init_cfg.keys():
                raise Exception('Missing checkpoint')    
            self.load_state_dict(torch.load(self.init_cfg['checkpoint']))
        else:
            raise Exception('init_cfg is formatted incorrectly')
        
    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s
    
    

class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 modules: Optional[Iterable] = None,
                 init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict in openmmlab.

    Args:
        modules (dict, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 modules: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
