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
from torch.distributed.optim import ZeroRedundancyOptimizer

batch_size = 8
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

optimizer = dict(
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3*(batch_size*3/256), weight_decay=0.025), # RepViT default settings
    optimizer = dict(type = ZeroRedundancyOptimizer,
                     optimizer_class = torch.optim.AdamW, 
                     lr=1e-4*(batch_size*3/256), 
                     eps= 1e-07,
                     weight_decay=0.025),
    grad_clip = dict(max_norm=0.2, norm_type=2)
)

lr_scheduler = dict(
    type = BaseScheduler,
    regular_scheduler = dict(
            type=torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=4,
            eta_min=5e-6,
            verbose=True
        ),
    warmup_by_epoch = False,
    warmup_epochs = 1,
    warmup = 'constant_value',
    warmup_iters = 75000,
    warmup_value = 1e-4
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
runner = dict(
    type = 'iter',
    max_epochs = 10000, #300
    max_iters = 100000,
    val_freq_epoch = 500,
    val_freq_iter = 10000,
    save_freq_iter = 99999999,
    log_freq = 5,
    resume_train = False,
    # resume_checkpoint = 'epoch_1_2e-5lr_01042024.pth',
    # resume_checkpoint = 'exception_02042024.pth',
    resume_checkpoint = '/home/qasim/Projects/TurboMedSAM/work_dir/DistillRepViT-ViTB_PreComputed/epoch_2_20000.pth',
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
    bbox_shift=5,
    normalize=True,
    means = [0.2482501, 0.21106622, 0.20026337], 
    stds = [0.3038128, 0.27170245, 0.26680432]
    )
data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderPreCompDataset,
            # classes=classes,
            root_dir=data_root,
            teacher_root=teacher_root,
            pipeline=pipeline_type.pipeline_encoder_train),
        sampler = dict(
            type = ClassBalancedSampler,
            num_sample_class =  1,
            subset_classes = None,
            ),
        dataloader_creator = dict( type= basic_dataloader_creator)
        ),
    val=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderPreCompDataset,
            # classes=classes,
            root_dir=data_root,
            teacher_root=teacher_root,
            pipeline=pipeline_type.pipeline_encoder_train),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderPreCompDataset,
            # classes=classes,
            root_dir=data_root,
            teacher_root=teacher_root,
            pipeline=pipeline_type.pipeline_encoder_train),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
    ),
    inference=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderPreCompDataset,
            # classes=classes,
            root_dir='/pub4/qasim/MedSAM/split_npzs_3chnl/',
            pipeline=pipeline_type.pipeline_encoder_train),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)

saver = dict(
    type = savers.CVPRMedSAMSaver,
    directory = os.path.join(work_dir, exp_name, 'results'),
    keys = ['embeddings']
)

compute = dict(
    gpu_ids = [0,1,2],
    use_cpu = False,
    use_amp = False,
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