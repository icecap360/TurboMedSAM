import torch
import models
import datasets
import os 
from framework import ClassBalancedSampler, BaseScheduler, basic_dataloader_creator, DistributedSampler
import losses
import metrics
import pipelines
from torchvision import transforms
import dataloaders
from functools import partial
import savers
batch_size = 64
compute = dict(
    gpu_ids = [0, 1,2],
    use_cpu = True,
    use_amp = True,
    mp_start_method = 'fork',
    opencv_num_threads=0,
    cudnn_benchmark=False,
    workers_per_gpu=4,
    samples_per_gpu=batch_size,
    batch_size=batch_size,
    pin_memory=False,
    prefetch_factor=2,
    broadcast_bn_buffer=True,
    persistent_workers=False,
    job_launcher = dict(
        type='pytorch', #['none', 'pytorch', 'slurm', 'mpi'],
        dist_params = dict(backend='nccl', port=29514)
        )
    )

image_size = 1024
data_root = '/pub4/qasim/MedSAM/split_npzs_3chnl/'
pipeline_type = pipelines.CVPRMedSAMPipeline(
    input_img_shape = image_size, 
    target_mask_shape = image_size,
    bbox_shift=5,
    normalize=False,
    means = None,
    stds = None,
)
seed = 0

data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict(
            type = ClassBalancedSampler,
            num_sample_class =  1,
            subset_classes = None,
            ),
        dataloader_creator = dict( type= basic_dataloader_creator)
        ),
    val=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
    ),
    inference=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir='/pub4/qasim/MedSAM/split_npzs_3chnl/',
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)
