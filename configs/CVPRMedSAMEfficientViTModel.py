import models.EfficientViTModel
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
from models.projects.turbo_vit_utils.gonas.darwinai.blockspecs import BlockSpec

img_size = 1024
batch_size = 12

model = models.LiteMedSAM(
        image_encoder = models.efficientvit_backbone_b1(
            init_cfg=None, 
            distillation=False,
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
                "checkpoint" :  "./checkpoints/medsam_vit_b.pth",
                "no_image_encoder": True,
            },
        )

optimizer = dict(
    optimizer = dict(
        type = torch.optim.AdamW,
        lr=1e-4, 
        weight_decay=0.01),
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
    warmup_iters = 10000,
    warmup_value = 4e-5
    )

compute = dict(
    gpu_ids = [0],
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
    max_epochs = 8, #300
    max_iters = 10000,
    val_freq_epoch = 8,
    val_freq_iter = 1000,
    save_freq_iter = 10000,
    log_freq=5,
    resume_train = True,
    checkpoint_path = '/pub0/qasim/TurboMedSAM/checkpoints/RepViTm11_epoch4-FrozenBB_FullDataset-Distill_ViTB_AggressiveAugmentation_epoch_4.pth',
)

loss = losses.MedSAMLoss({
    'loss_dice': 1.0,
    'loss_ce': 1.0,
    'loss_iou': 1.0,
})
metric = metrics.MedSAMMetrics(class_thresholds=[5])
custom_hooks = []
seed = 0

data_root = '/data/qasim/MedSAM/full_dataset_3_chnl/'
pipeline_type = pipelines.CVPRMedSAMPipeline(
    img_shape=img_size,
    target_mask_shape=256,
    normalize=True,
    means = [0.2482501, 0.21106622, 0.20026337],     
    stds = [0.3038128, 0.27170245, 0.26680432],
    apply_random_flip=True,
    apply_random_rotate=True)

data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_official),
        sampler = dict(
            type = ClassBalancedSampler,
            num_sample_class =  1,
            subset_classes = None,
            ),
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
            pipeline=pipeline_type.pipeline_inference),
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
