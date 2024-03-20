import torch
import models
import datasets
from torch.utils.data import DistributedSampler
import os 
from framework import ClassBalancedSampler, BaseScheduler, basic_dataloader_creator
import losses
import metrics
import pipelines
from torchvision import transforms
import dataloaders
from functools import partial
import savers

batch_size = 1
image_size = 1024

model = models.repvit_model(
            init_cfg={
                "type": "pretrained",
                "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/repvit_sam.pt",
                "strict": False
            },
            distillation=True,
            num_classes=0
        )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05) # as per paper - lr=5e-4*(batch_size//512), batch_size=1024
grad_clip = dict(max_norm=35, norm_type=2)
lr_scheduler = BaseScheduler(
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1516396//batch_size,
            eta_min=1e-5,
            verbose=True
        ),
    optimizer = optimizer,
    warmup_by_epoch = True,
    warmup_epochs = 5,
    warmup = 'constant_value',
    warmup_iters = 10000,
    warmup_value = 1e-6
)

compute = dict(
    gpu_ids = [0,1,2],
    use_cpu = False,
    mp_start_method = 'fork',
    opencv_num_threads=0,
    cudnn_benchmark=False,
    workers_per_gpu=4,
    samples_per_gpu=batch_size,
    pin_memory=False,
    prefetch_factor=2,
    broadcast_bn_buffer=True,
    persistent_workers=False,
    job_launcher = dict(
        type='pytorch', #['none', 'pytorch', 'slurm', 'mpi'],
        dist_params = dict(backend='nccl', port=29515)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
runner_type= 'epoch'
max_epochs = 300
max_iters = 10000
val_freq_epoch = 1
save_freq_epoch = 1
val_freq_iter = 100
save_freq_iter = 100

loss = losses.DistillationLoss(
    distillation_type = 'mse',
    tau = 1.0,
    precomputed_teacher = True,
    loss_weight = {
        'loss_distillation': 1.0,
        }
    )
metric = metrics.NoMetric()
checkpoint = ''
resume = False
custom_hooks = []
seed = 0

data_root = '/pub4/qasim/MedSAM/split_npzs_3chnl/'
# teacher_root = '/pub2/data/qasim/MedSAM1024/results'
teacher_root = '/home/qasim/Projects/TurboMedSAM/work_dir/MedSAMEncoder/results'
pipeline_type = pipelines.CVPRMedSAMPipeline(
    target_length=image_size,
    bbox_shift=5
)
data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderPreComputed,
            # classes=classes,
            root_dir=data_root,
            teacher_root=teacher_root,
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
            type = datasets.CVPRMedSAMEncoderPreComputed,
            # classes=classes,
            root_dir=data_root,
            teacher_root=teacher_root,
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderPreComputed,
            # classes=classes,
            root_dir=data_root,
            teacher_root=teacher_root,
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
    ),
    inference=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderPreComputed,
            # classes=classes,
            root_dir='/pub4/qasim/MedSAM/split_npzs_3chnl/',
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)

saver = dict(
    type = savers.CVPRMedSAMSaver,
    directory = os.path.join(work_dir, exp_name, 'results'),
    keys = ['embeddings']
)