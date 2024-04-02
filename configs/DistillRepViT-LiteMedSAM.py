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
from torch.distributed.optim import ZeroRedundancyOptimizer

batch_size = 5
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
vit_patch_size = 16

model = models.TeacherStudentModel(
    student=models.repvit_model_m1_5(
            # init_cfg={
            #     "type": "pretrained",
            #     # "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/repvit_sam.pt",
            #     "strict": True
            # },
            distillation=True,
            num_classes=0
        ),
    teacher=models.LiteMedSAMEncoder(
        settings=dict(
            img_size=256,
            in_chans=3,
            embed_dims=[
                64, ## (64, 256, 256)
                128, ## (128, 128, 128)
                160, ## (160, 64, 64)
                320 ## (320, 64, 64) 
            ],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
            init_cfg = {
                "type": "pretrained",
                "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/lite_medsam.pth"
            }
            )
    )
)

optimizer = dict(
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3*(batch_size*3/256), weight_decay=0.025), # RepViT default settings
    optimizer = dict(type = torch.optim.AdamW, # ZeroRedundancyOptimizer,
                    #  optimizer_class = torch.optim.AdamW, 
                     lr=1e-3*(batch_size*3/256), 
                     weight_decay=0.025),
    grad_clip = dict(max_norm=0.2, norm_type=2)
)

lr_scheduler = dict(
    type = BaseScheduler,
    regular_scheduler = dict(
            type=torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=3,
            eta_min=3e-5,
            verbose=True
        ),
    warmup_by_epoch = True,
    warmup_epochs = 1,
    warmup = 'constant_value',
    warmup_iters = 10000,
    warmup_value = 1e-4
    )

compute = dict(
    gpu_ids = [0,1,2,3],
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
        dist_params = dict(backend='nccl', port=29514)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
runner= dict(
    type= 'epoch',
    max_epochs = 3, #300
    max_iters = 10000,
    val_freq_epoch = 1,
    val_freq_iter = 10000,
    save_freq_iter = 15000,
    log_freq=5,
    resume_train = False,
    resume_checkpoint = None,
)
loss = losses.DistillationLoss(
    distillation_type = 'mse',
    tau = 1.0,
    precomputed_teacher = False,
    loss_weight = {
        'loss_distillation': 1.0,
        }
    )
metric = metrics.DistillationMetric(precomputed_teacher=False)
custom_hooks = []
seed = 0

data_root = '/data/qasim/MedSAM/split_npzs_3chnl/'
pipeline_type = pipelines.CVPRMedSAMDistillationPipeline(
    student_image_shape = 1024,
    teacher_image_shape = 256,
    target_mask_shape = 256,
    bbox_shift=5
)
data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_teacher_student),
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
            pipeline=pipeline_type.pipeline_teacher_student),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_teacher_student),
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

saver = dict(
    type = savers.CVPRMedSAMSaver,
    directory = os.path.join(work_dir, exp_name, 'results'),
    keys = ['embeddings']
)
ffcv_writer = dict(
    root_dir = '/pub4/qasim/MedSAM/split_npzs_3chnl',
    num_workers = 16,
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDatasetFFCVWrite,
            root_dir=data_root)
        ),
    val=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDatasetFFCVWrite,
            root_dir=data_root)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDatasetFFCVWrite,
            root_dir=data_root)
    ),
)