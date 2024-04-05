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

batch_size = 5
image_size = 1024
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
vit_patch_size = 16

model = models.TeacherStudentModel(
    student=models.repvit_model(
            init_cfg={
                "type": "pretrained",
                "checkpoint" :  "/home/qasim/Projects/TurboMedSAM/checkpoints/repvit_sam.pt",
                "strict": True
            },
            distillation=True,
            num_classes=0
        ),
    teacher=models.ViTMedSAM(
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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05) # as per paper - lr=5e-4*(batch_size//512), batch_size=1024
grad_clip = dict(max_norm=35, norm_type=2)
lr_scheduler = BaseScheduler(
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1516396//batch_size,
            eta_min=1e-5,
            verbose=True
        ),
    optimizer = optimizer,
    warmup_by_epoch = False,
    warmup_epochs = 5,
    warmup = 'constant_value',
    warmup_iters = 20,
    warmup_value = 1e-6
)

compute = dict(
    gpu_ids = [1,2],
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
runner = dict(
    type= 'iter',
    max_epochs = 1, #300
    max_iters = 10000,
    val_freq_epoch = 1,
    val_freq_iter = 1000,
    save_freq_iter = 10,
    log_freq=5,
    resume_train = True,
    resume_checkpoint = None,)

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
pipeline_type = pipelines.CVPRMedSAMPipeline(
    input_img_shape = image_size, 
    target_mask_shape = image_size,
    bbox_shift=5,
    normalize=False,
    means = [0.2482501, 0.21106622, 0.20026337],     stds = [0.3038128, 0.27170245, 0.26680432])
data = dict(
    train=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_encoder),
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
            pipeline=pipeline_type.pipeline_encoder),
        sampler = dict( type = DistributedSampler),
        dataloader_creator = dict( type= dataloaders.CVPRMedSAM_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(       
            type = datasets.CVPRMedSAMEncoderDataset,
            # classes=classes,
            root_dir=data_root,
            pipeline=pipeline_type.pipeline_encoder),
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