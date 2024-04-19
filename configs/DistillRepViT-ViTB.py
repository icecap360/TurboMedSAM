import torch
import models
import datasets
import os 
from framework import ClassBalancedSampler, BaseScheduler, basic_dataloader_creator, DistributedSampler, BasePipeline
import losses
import metrics
import pipelines
import custom_transforms
from torchvision.transforms import v2
import dataloaders
from functools import partial
import savers

batch_size = 11
image_size = 1024
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
vit_patch_size = 16

model = models.TeacherStudentModel(
    student=models.repvit_model_m1_1(
            init_cfg=None, #{
            #     "type": "pretrained",
            #     "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/repvit_sam.pt",
            #     "strict": True
            # },
            distillation=True,
            num_classes=0
        ),
    teacher=models.ViT(
        distillation=True,
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        init_cfg = {
            "type": "pretrained",
            "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/medsam_image_encoder_1024.pth"
        }
        )
)

optimizer = dict(
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3*(batch_size*3/256), weight_decay=0.025), # RepViT default settings
    optimizer = dict(type=torch.optim.AdamW,#type = ZeroRedundancyOptimizer,
                    #  optimizer_class = torch.optim.AdamW, 
                     lr=1e-3*(batch_size*3/256), 
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
    warmup_by_epoch = True,
    warmup_epochs = 4,
    warmup = 'constant_value',
    warmup_iters = 40000,
    warmup_value = 4e-5
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
    pin_memory=True,
    prefetch_factor=2,
    broadcast_bn_buffer=True,
    persistent_workers=True,
    job_launcher = dict(
        type='pytorch', #['none', 'pytorch', 'slurm', 'mpi'],
        dist_params = dict(backend='nccl', port=29514)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
runner = dict(
    type = 'epoch',
    max_epochs = 4, #300
    max_iters = 10000,
    val_freq_epoch = 2,
    val_freq_iter = 50,
    save_freq_iter = 10000,
    log_freq = 5,
    resume_train = False,
    # checkpoint_path = 'epoch_1_2e-5lr_01042024.pth',
    # checkpoint_path = 'exception_02042024.pth',
    checkpoint_path = '/home/qasim/Projects/TurboMedSAM/work_dir/DistillRepViT-ViTB_PreComputed/epoch_2_20000.pth',
)

loss = losses.DistillationLoss(
    distillation_type = 'mse',
    tau = 1.0,
    precomputed_teacher = False,
    loss_weight = {
        'loss_distillation': 1.0,
        }
    )
metric = metrics.NoMetric()
custom_hooks = []
seed = 0

data_root = '/pub4/qasim/MedSAM/split_npzs_3chnl/'
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

train_collate_transforms = [custom_transforms.MixPatch(1.0, 256),
    custom_transforms.NoLabelCutMix(alpha=1.0)]
train_collate_functionals = [custom_transforms.MixPatchFunctional,
    custom_transforms.NoLabelCutMixFunctional]

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
            pipeline=test_pipeline.pipeline),
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