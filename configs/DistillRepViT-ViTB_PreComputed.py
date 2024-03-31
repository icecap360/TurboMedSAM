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

batch_size = 24
image_size = 1024

model = models.repvit_model_m1_5(
            init_cfg=None, #{
            #     "type": "pretrained",
            #     "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/repvit_sam.pt",
            #     "strict": False
            # },
            distillation=True,
            num_classes=0
        )

compute = dict(
    gpu_ids = [0,1,2],
    use_cpu = False,
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
        dist_params = dict(backend='nccl', port=29515)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
runner = dict(
    type = 'epoch',
    max_epochs = 4, #300
    max_iters = 10000,
    val_freq_epoch = 1,
    val_freq_iter = 1,
    save_freq_epoch = 1,
    save_freq_iter = 1000,
    log_freq = 5,
    resume_train = False,
    resume_checkpoint = 'epoch_3.pth',
)

optimizer = dict(
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3*(batch_size*3/4096), weight_decay=0.01), # TinyViT pretraining settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3*(batch_size*3/256), weight_decay=0.025), # RepViT default settings
    # grad_clip = dict(max_norm=5, norm_type=2)
)
lr_scheduler = BaseScheduler(
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer['optimizer'],
            T_max=3,
            eta_min=1e-6,
            verbose=True
        ),
    optimizer = optimizer['optimizer'],
    warmup = 'constant_value',
    warmup_by_epoch = True,
    warmup_epochs = 1, # should be 5 if 300 epochs used
    warmup_iters = 10000,
    warmup_value = 2e-6
)
loss = losses.DistillationLoss(
    distillation_type = 'mse',
    tau = 1.0,
    precomputed_teacher = True,
    loss_weight = {
        'loss_distillation': 1.0,
        }
    )
metric = metrics.NoMetric()
custom_hooks = []
seed = 0

data_root = '/pub4/qasim/MedSAM/split_npzs_3chnl/'
# teacher_root = '/pub2/data/qasim/MedSAM1024/results'
teacher_root = '/pub5/qasim/MedSAM/MedSAM1024/results'
pipeline_type = pipelines.CVPRMedSAMPipeline(
    input_img_shape=image_size,
    target_mask_shape=256,
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