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
    gpu_ids = [0,1],
    use_cpu = False,
    use_amp = False,
    mp_start_method = 'fork',
    opencv_num_threads=0,
    cudnn_benchmark=False,
    workers_per_gpu=2,
    samples_per_gpu=8,
    batch_size=8,
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
    resume_checkpoint = None,
)

loss = losses.MedSAMLoss({
    'loss_dice': 1.0,
    'loss_ce': 1.0,
    'loss_iou': 1.0,
})
metric = metrics.MedSAMMetrics(class_thresholds=[5])
custom_hooks = []
seed = 0

data_root = '/pub4/qasim/MedSAM/split_npzs_3chnl/'
pipeline_type = pipelines.CVPRMedSAMPipeline(
    input_img_shape=256,
    target_mask_shape=256,
    bbox_shift=5,
    normalize=False,
    means = [0.2482501, 0.21106622, 0.20026337],     stds = [0.3038128, 0.27170245, 0.26680432])
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
            root_dir='/pub4/qasim/MedSAM/split_npzs_3chnl/',
            pipeline=pipeline_type.pipeline_inference),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type = basic_dataloader_creator)
    ),
)

