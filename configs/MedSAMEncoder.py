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

batch_size = 1
image_size = 1024
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
vit_patch_size = 16
model = models.ViT(
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
            "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/medsam_image_encoder_256.pth"
        }
        )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
grad_clip = dict(max_norm=35, norm_type=2)
lr_scheduler = BaseScheduler(
    regular_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            gamma=0.1,
            milestones=[48,96],
            verbose=True
        ),
    optimizer=optimizer,
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.1
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
    checkpoint_path = None
    )

loss = losses.MedSAMLoss({
    'loss_dice': 1.0,
    'loss_ce': 1.0,
    'loss_iou': 1.0,
})
metric = metrics.MedSAMMetrics(class_thresholds=[5])
checkpoint = ''
custom_hooks = []
seed = 0

data_root = '/pub4/qasim/MedSAM/split_npzs_3chnl/'
pipeline_type = pipelines.CVPRMedSAMPipeline(
    img_shape=image_size,
    target_mask_shape=256,
    normalize=False,
    means = [0.2482501, 0.21106622, 0.20026337],     stds = [0.3038128, 0.27170245, 0.26680432])
data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_2D_train),
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
            pipeline=pipeline_type.pipeline_train),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_train),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
    ),
    inference=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir='/pub4/qasim/MedSAM/split_npzs_3chnl/',
            pipeline=pipeline_type.pipeline_encoder_train),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)

saver = dict(
    type = savers.CVPRMedSAMEmbeddingSaver,
    directory = '/pub2/data/qasim/MedSAM1024A/resultsA',
    keys = ['embeddings']
)