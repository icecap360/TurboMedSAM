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

batch_size = 3
image_size = 1024
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
vit_patch_size = 16
model = models.LiteMedSAM(
        encoder=models.ViTMedSAM(
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
                out_chans=prompt_embed_dim
                ),
        settings=dict(
            prompt_encoder=dict(
                embed_dim=256,
                image_embedding_size=(64, 64),
                input_image_size=(256, 256),
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
            )
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
    warmup_iters = 10000,
    warmup_value = 1e-5
    )

compute = dict(
    gpu_ids = [0,1,2,3],
    use_cpu = False,
    use_amp = False,
    mp_start_method = 'fork',
    opencv_num_threads=0,
    cudnn_benchmark=False,
    workers_per_gpu=2,
    samples_per_gpu=batch_size,
    batch_size=batch_size,
    pin_memory=False,
    prefetch_factor=2,
    broadcast_bn_buffer=True,
    persistent_workers=False,
    job_launcher = dict(
        type='none', #['none', 'pytorch', 'slurm', 'mpi'],
        dist_params = dict(backend='nccl', port=29515)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
runner = dict(
    type= 'epoch',
    max_epochs = 1, #300
    max_iters = 10000,
    val_freq_epoch = 1,
    val_freq_iter = 1000,
    save_freq_iter = 1000,
    log_freq=5,
    resume_train = True,
    checkpoint_path = '/home/qasim/Projects/TurboMedSAM/checkpoints/medsam_vit_b.pth',
)

loss = losses.MedSAMLoss({
    'loss_dice': 1.0,
    'loss_ce': 1.0,
    'loss_iou': 1.0,
})
metric = metrics.MedSAMMetrics(class_thresholds=[5])
custom_hooks = []
seed = 0

data_root = '/data/qasim/MedSAM/split_npzs_3chnl/'
pipeline_type = pipelines.CVPRMedSAMPipeline(
    img_shape=image_size,
    target_mask_shape=256,
    normalize=False,
    means = [0.2482501, 0.21106622, 0.20026337],     
    stds = [0.3038128, 0.27170245, 0.26680432])
data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_2D),
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
            root_dir='/data/qasim/MedSAM/split_npzs_3chnl/',
            pipeline=pipeline_type.pipeline_inference),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)

