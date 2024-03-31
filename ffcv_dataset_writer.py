import torch
import argparse
import copy
import os
from torch import distributed as dist
from framework import import_module, setup_multi_processes, get_dist_info, logger, read_cfg_str, get_device, init_random_seed, set_random_seed, create_dataloader, set_visible_devices, EpochBasedRunner, IterBasedRunner
from copy import deepcopy
import framework
import ffcv
import numpy as np

def dataset_creator(dataset_settings, split_type):
    dataset_settings = deepcopy(dataset_settings)
    dataset_type = dataset_settings.pop("type")
    return dataset_type(
            split_type=split_type,
            **dataset_settings
        )

def parse_args():
    parser = argparse.ArgumentParser(description='Write a FFCV dataset')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args



def main(args):
    abs_config_path = os.path.join("configs", args.config)
    cfg = import_module(os.path.basename(args.config), 
                        abs_config_path)
    
    split_types = ['train', 'val', 'test']
    datasets = []
    writers = []
    for split in split_types:
        datasets.append(dataset_creator(cfg.ffcv_writer[split]['dataset'], split))
        writer = ffcv.writer.DatasetWriter(
            os.path.join(cfg.ffcv_writer['root_dir'], split+'.ffcv'), 
            {'input_image':
                ffcv.fields.RGBImageField(),
            'target_mask':
                ffcv.fields.RGBImageField(),
            'meta': ffcv.fields.JSONField()},
            num_workers=cfg.ffcv_writer['num_workers'],
            page_size=ffcv.writer.MIN_PAGE_SIZE*4
            )
    
    for dataset in datasets:
        writer.from_indexed_dataset(dataset)

if __name__ == '__main__':
    args = parse_args()
    main(args)
