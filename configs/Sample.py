import torch
from torch.utils.data import DistributedSampler
import models
import datasets
import os 
from framework import ClassBalancedSampler, BaseScheduler
from torchvision import transforms
from metrics import SampleMetric
from losses import SampleLoss

model = models.SampleModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
grad_clip = dict(max_norm=35, norm_type=2)
lr_scheduler = BaseScheduler(
    regular_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[2,4],
            gamma=0.1
        ),
    optimizer=optimizer,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1
    )

compute = dict(
    gpu_ids = [0,1,2],
    use_cpu = False,
    mp_start_method = 'fork',
    opencv_num_threads=0,
    cudnn_benchmark=False,
    workers_per_gpu=4,
    samples_per_gpu=4,
    pin_memory=False,
    broadcast_bn_buffer = True,
    prefetch_factor=2,
    persistent_workers=False,
    job_launcher = dict(
        type='pytorch', #['none', 'pytorch', 'slurm', 'mpi'],
        dist_params = dict(backend='nccl', port=29515)
        )
    )

work_dir = 'work_dir'
exp_name = os.path.basename(__file__)[:-3]
resume_from = None
eval_metrics = []
val_freq = 1
save_freq = 1
loss = SampleLoss(
    loss_weight = {
        'loss_pred': 1.0
    }
    )
metric = SampleMetric(loss)
checkpoint = ''
resume = False
custom_hooks = []
seed = 0
max_epochs = 48

dataset_type = datasets.SampleDataset
data_root = '/pub4/qasim/MedSAM/split_npzs/'
pipeline = transforms.Compose(
    transforms.ToTensor()
)
data = dict(
    drop_last = False,
    sampler = {
        "type" : DistributedSampler,
        # "num_sample_class": 1,
        # "subset_classes": None,
    },
    train=dict(
        type=dataset_type,
        # classes=classes,
        data_root=data_root,
        input_pipeline=pipeline,
        target_pipeline=pipeline),
    val=dict(
        type=dataset_type,
        # classes=classes,
        data_root=data_root,
        input_pipeline=pipeline,
        target_pipeline=pipeline),
    test=dict(
        type=dataset_type,
        # classes=classes,
        data_root=data_root,
        ann_file='test.json',
        input_pipeline=pipeline,
        target_pipeline=pipeline))

