import os
import torch.nn as nn
from typing import Iterable, Optional
import copy
import torch
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import re 
from typing import Callable, Dict, List, Optional, Tuple, Union
from .Distributed import get_dist_info


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg: Optional[dict] = None):
        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)
        
        self.initialize()
    
    # @abstractmethod
    # def forward_loss(self, data_batch) -> Dict:
    #     pass
    
    # @abstractmethod
    # def forward_pred(self, data_batch) -> Dict:
    #     pass
    
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
    
    @abstractmethod
    def forward(self, data_batch):
        """
        This function simply returns the predictions in s list format (i.e. each prediction sample should be its own disctionary ) 
        """
        pass
    
    def load_checkpoint(
        self,
        state_dict: dict,
        logger,
        strict: bool = True,
        revise_keys = [] #[(r'^module.', '')],
    ):
        # strip prefix of state_dict
        metadata = getattr(state_dict, '_metadata', ())
        for p, r in revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                for k, v in state_dict.items()})
        # Keep metadata in state_dict
        state_dict._metadata = metadata
        
        self.load_state_dict(self.model, state_dict, strict, logger)
        return state_dict
    
    def load_state_dict(self, state_dict, logger, strict: bool = True):
        unexpected_keys: List[str] = []
        all_missing_keys: List[str] = []
        err_msg: List[str] = []

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()  # type: ignore
        if metadata is not None:
            state_dict._metadata = metadata  # type: ignore

        # use _load_from_state_dict to enable checkpoint version control
        def load_module(module, prefix=''):
            # recursively check parallel module in case that the model has a
            # complicated structure, e.g., nn.Module(nn.Module(DDP))
            # if is_module_wrapper(module):
            #     module = module.module
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, 
                                        local_metadata, 
                                        True,
                                        all_missing_keys, unexpected_keys,
                                        err_msg)
            # for name, child in module._modules.items():
            #     if child is not None:
            #         load_module(child, prefix + name + '.')

        load_module(self.model)
        # ignore "num_batches_tracked" of BN layers
        missing_keys = [
            key for key in all_missing_keys if 'num_batches_tracked' not in key
        ]

        if unexpected_keys:
            err_msg.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
        if missing_keys:
            err_msg.append(
                f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

        rank, _ = get_dist_info()
        if len(err_msg) > 0 and rank == 0:
            err_msg.insert(
                0, 'The model and loaded state dict do not match exactly\n')
            err_msg = '\n'.join(err_msg)  # type: ignore
            if strict:
                raise RuntimeError(err_msg)
            else:
                logger.warning(err_msg)            
            
            return super().load_state_dict(state_dict, strict)

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
