import torch
import models
import datasets
import os 
from framework import ClassBalancedSampler, BaseScheduler

model = models.LiteMedSAM(
        settings=dict(
            image_encoder=dict(
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
                layer_lr_decay=0.8
                ),
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

optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
grad_clip = dict(max_norm=35, norm_type=2)
base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)
lr_scheduler = BaseScheduler(
    regular_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,
            patience=5,
            cooldown=0
        ),
    optimizer=optimizer,
    warmup=True,
    warmup_iters=500,
    warmup_ratio=0.1
    )

compute = dict(
    gpu_id = [0],
    pin_memory=True,
    use_cpu = False,
    mp_start_method = 'fork',
    opencv_num_threads=0,
    cudnn_benchmark=False,
    workers_per_gpu=4,
    samples_per_gpu=4,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
    job_launcher = dict(
        type='pytorch', #['none', 'pytorch', 'slurm', 'mpi'],
        dist_params = dict(backend='nccl', port=29515)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)
resume_from = None
eval_metrics = []
val_freq = 1
save_freq = 1
loss = 
checkpoint = ''
resume = False
custom_hooks = []
seed = 0
max_epochs = 48

dataset_type = datasets.CVPRMedSAMDataset
data_root = '/pub4/qasim/MedSAM/split_npzs/'
data = dict(
    drop_last = False,
    sampler = {
        "type" : ClassBalancedSampler,
        "num_sample_class": 1,
        "subset_classes": None,
    },
    train=dict(
        type=dataset_type,
        # classes=classes,
        data_root=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # classes=classes,
        data_root=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # classes=classes,
        data_root=data_root,
        ann_file='test.json',
        pipeline=test_pipeline))

