import torch
import os
from copy import deepcopy
from .utils import create_object_from_params
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor


def create_optimizer(params_dict, model):
    use_galore = params_dict.get('use_galore')

    optimizer_dict = deepcopy(params_dict['optimizer'])

    if use_galore:
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': 128, 'update_proj_gap': 50, 'scale': 1.0, 'proj_type': "std"}]
        if optimizer_dict['type'] == torch.optim.AdamW:
            optimizer_dict['type'] = GaLoreAdamW
            return create_object_from_params(optimizer_dict, params=param_groups)
        elif optimizer_dict['type']==GaLoreAdamW8bit or optimizer_dict['type']==GaLoreAdamW :
            return create_object_from_params(optimizer_dict, params=param_groups)
        else:
            raise Exception('Optimizer type is not supported')
    else:
        if isinstance(optimizer_dict, torch.optim.Optimizer):
            return optimizer_dict
        else:
            return create_object_from_params(optimizer_dict, params=model.parameters())
    
