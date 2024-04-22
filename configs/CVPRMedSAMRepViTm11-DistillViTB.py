import torch
import models
import datasets
import os 
from framework import ClassBalancedSampler, BaseScheduler, basic_dataloader_creator, DistributedSampler, BasePipeline
import losses
import metrics
import pipelines
from torchvision import transforms
import dataloaders
from torchvision.transforms import v2
import savers 
from functools import partial
from custom_transforms import RandomRotateDiscrete

img_size = 1024
batch_size = 10
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
vit_patch_size = 16

model = models.TeacherStudentModel(
    student=models.LiteMedSAM(
        image_encoder = models.repvit_model_m1_1(
            init_cfg={
                "type": "pretrained",
                "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/DistillRepViTm11-ViTB_aggressive_aug_epoch_4.pth",
                "strict": True
            },
            distillation=False,
            num_classes=0
        ),
        settings=dict(
            prompt_encoder=dict(
                embed_dim=256,
                image_embedding_size=(64, 64),
                input_image_size=(img_size, img_size),
                mask_in_chans=16
                ),
            mask_decoder=dict(
                num_multimask_outputs=3,
                transformer_depth=2,
                transformer_embedding_dim=256,
                transformer_mlp_dim=2048,
                transformer_num_heads=8,
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                )
            ),
            init_cfg={
                "type": "pretrained",
                "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/medsam_vit_b.pth",
                "no_image_encoder": True
            },
        ),
    teacher=models.LiteMedSAM(
        image_encoder=models.ViT(
                distillation=False,
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=img_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
                ),
        settings=dict(
            prompt_encoder=dict(
                embed_dim=256,
                image_embedding_size=(img_size//vit_patch_size, img_size // vit_patch_size),
                input_image_size=(img_size, img_size),
                mask_in_chans=16
                ),
            mask_decoder=dict(
                num_multimask_outputs=3,
                transformer_depth=2,
                transformer_embedding_dim=256,
                transformer_mlp_dim=2048,
                transformer_num_heads=8,
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                )
            ),
        init_cfg={
            "type": "pretrained",
            "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/medsam_vit_b.pth",
            "no_image_encoder": False
        },
        ),
        
)

optimizer = dict(
    optimizer = dict(
        type = torch.optim.AdamW,
        lr=1e-3*(batch_size*3/256), weight_decay=0.01),
    grad_clip = dict(max_norm=0.2, norm_type=2)
)
lr_scheduler = dict(
    type = BaseScheduler,
    regular_scheduler = dict(
            type=torch.optim.lr_scheduler.MultiStepLR,
            gamma=0.1,
            milestones=[2],
            verbose=True
        ),
    warmup_by_epoch = False,
    warmup_epochs = 1,
    warmup = 'constant_value',
    warmup_iters = 40000,
    warmup_value = 1e-4
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
    find_unused_parameters = True,
    job_launcher = dict(
        type='pytorch', #['none', 'pytorch', 'slurm', 'mpi'],
        dist_params = dict(backend='nccl', port=29515)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
runner = dict(
    type= 'epoch',
    max_epochs = 4, #300
    max_iters = 10000,
    val_freq_epoch = 2,
    val_freq_iter = 1000,
    save_freq_iter = 10000,
    log_freq=5,
    resume_train = False,
    checkpoint_path = '/home/qasim/Projects/TurboMedSAM/work_dir/CVPRMedSAMRepViTm11-DistillViTB/epoch_2.pth',
)

loss = losses.MedSAMTeacherLoss(
    distillation_type='bce',
    tau=3,
    loss_weight={
    'loss_dice': 1.0,
    'loss_ce': 1.0,
    'loss_iou': 1.0,
    'loss_distillation': 1.0,
    })
metric = metrics.MedSAMTeacherMetrics(class_thresholds=[5])
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
        v2.Resize(size=(img_size, img_size), antialias=True),
        # v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ])
train_transform_student = v2.Compose(
    [v2.Normalize(mean = [0.2482501, 0.21106622, 0.20026337],     
                std = [0.3038128, 0.27170245, 0.26680432])
    ])
train_transform = v2.Compose(
    [
        # v2.Resize(size=(img_size+256, img_size+256), antialias=True),
        # v2.RandomResizedCrop(size=(img_size, img_size), 
        #                      scale=(0.5, 1.0), 
        #                      ratio=(0.75, 1.3333),
        #                      antialias=True), -- this causes divergent training
        v2.Resize(size=(img_size, img_size), antialias=True),
        RandomRotateDiscrete(angles=[-90,90,0,180]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # v2.RandAugment(num_ops=2,
        #                magnitude=9),
        # v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        v2.ToDtype(torch.float32, scale=True)
    ])

# train_collate_transforms = [transforms.NoLabelCutMix(alpha=1.0)]
# train_collate_functionals = [transforms.NoLabelCutMixFunctional]

train_pipeline = pipelines.TeacherStudentPipeline(
    train_transform, 
    student_transform=train_transform_student, 
    # collate_transforms=train_collate_transforms,
    # collate_functionals=train_collate_functionals
    )
test_pipeline = pipelines.TeacherStudentPipeline(test_transform, student_transform=test_transform_student)

pipeline_type = pipelines.CVPRMedSAMPipelineDistillation(
    student_image_shape=1024,
    teacher_image_shape=1024,
    student_normalize=True,
    teacher_normalize=False,
    target_mask_shape=256,
    means = [0.2482501, 0.21106622, 0.20026337],     
    stds = [0.3038128, 0.27170245, 0.26680432])

data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=train_pipeline.pipeline),
        sampler = dict(
            type = ClassBalancedSampler,
            num_sample_class =  1,
            subset_classes = None,
            ),
        # collate_fn = train_pipeline.collate_fn,
        dataloader_creator = dict( type= basic_dataloader_creator)
        ),
    val=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
    ),
    inference=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMInferenceDataset,
            # classes=classes,
            root_dir='/data/qasim/MedSAM/official_val',
            pipeline=pipeline_type.pipeline),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)
saver = dict(
    type = savers.CVPRMedSAMSaver,
    directory = '/home/qasim/Projects/TurboMedSAM/work_dir/CVPRMedSAMRepViTm11',
    checkpoint = '/home/qasim/Projects/TurboMedSAM/checkpoints/RepViTm11_epoch4-Distill_ViTB_AggressiveAugmentation_epoch_2.pth',
    keys = ['logits'],
    dtype= ['uint8']
)
