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
from torchvision.transforms import v2

image_size = 1024
batch_size = 2
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
vit_patch_size = 16

model = models.TeacherStudentModel(
    student=models.repvit_model_m2_3(
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
    optimizer = dict(type = torch.optim.AdamW, # type = ZeroRedundancyOptimizer,
                    #  optimizer_class = torch.optim.AdamW, 
                     lr=1e-3*(batch_size*3/256), 
                     eps=1e-7,
                     weight_decay=0.025),
    grad_clip = dict(max_norm=0.2, norm_type=2),
    use_galore=True,
)

lr_scheduler = dict(
    type = BaseScheduler,
    regular_scheduler = dict(
            type=torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=3,
            eta_min=3e-5,
            verbose=True
        ),
    warmup_by_epoch = False,
    warmup_epochs = 1,
    warmup = 'constant_value',
    warmup_iters = 10000,
    warmup_value = 5e-5
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
    save_freq_iter = 10000,
    log_freq=5,
    resume_train = False,
    checkpoint_path = None,
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
test_transform_student = v2.Compose(
    [
        v2.Normalize(mean = [0.2482501, 0.21106622, 0.20026337],     
                     std = [0.3038128, 0.27170245, 0.26680432])
    ])
test_transform = v2.Compose(
    [
        # v2.ToImage(),
        v2.Resize(size=(image_size, image_size), antialias=True),
        # v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ])
train_transform_student = v2.Compose(
    [v2.Normalize(mean = [0.2482501, 0.21106622, 0.20026337],     
                std = [0.3038128, 0.27170245, 0.26680432])
    ])
train_transform = v2.Compose(
    [
        v2.Resize(size=(image_size+256, image_size+256), antialias=True),
        v2.RandomResizedCrop(size=(image_size, image_size), 
                             scale=(0.5, 1.0), 
                             ratio=(0.75, 1.3333),
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # v2.RandAugment(num_ops=2,
        #                magnitude=9),
        # v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        v2.ToDtype(torch.float32, scale=True)
    ])

train_collate_transforms = [transforms.MixPatch(1.0, 256),
    transforms.NoLabelCutMix(alpha=1.0)]
train_collate_functionals = [transforms.MixPatchFunctional,
    transforms.NoLabelCutMixFunctional]

train_pipeline = pipelines.TeacherStudentPipeline(
    train_transform, 
    student_transform=train_transform_student, 
    collate_transforms=train_collate_transforms,
    collate_functionals=train_collate_functionals)
test_pipeline = pipelines.TeacherStudentPipeline(test_transform, student_transform=test_transform_student)

data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=train_pipeline.pipeline),
        sampler = dict(
            type = ClassBalancedSampler,
            num_sample_class =  1,
            subset_classes = None,
            ),
        collate_fn = train_pipeline.collate_fn,
        dataloader_creator = dict( type= basic_dataloader_creator)
        ),
    val=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=test_pipeline.pipeline.pipeline_encoder_teacher_student),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=test_pipeline.pipeline),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
    ),
    inference=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir='/pub4/qasim/MedSAM/split_npzs_3chnl/',
            pipeline=test_pipeline.pipeline),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)

saver = dict(
    type = savers.CVPRMedSAMEmbeddingSaver,
    directory = work_dir,
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