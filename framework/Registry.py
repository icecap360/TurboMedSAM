import os
import inspect
from typing import Any, Dict, Optional

class Registry:
    
    def __init__(self, identity, modules = dict()) -> None:
        assert type(modules) is dict
        self.identity = identity
        self.modules = modules
    
    def add_members(self, new_members):
        self.modules.update(new_members)  
    
    def register(self, module, module_name=None, force=False):
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError('module must be a class or a function, '
                            f'but got {type(module)}')
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self.modules:
                raise KeyError(f'{name} is already registered in {self.name}')
            self.modules[name] = module
        return module
    
    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        if not isinstance(key, str):
            raise Exception('key for registry search {self.identity} must be string')
        if key in self.modules.keys():
            return self.modules[key]
        else:
            raise Exception(key + ' missing in registry ' + self.identity)
        
def build_from_cfg( registry: Registry, cfg, default_args: Optional[Dict] = None):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.identity} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')


DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
MODELS = Registry('models')
BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
PRIOR_GENERATORS = Registry('Generator for anchors and points')
