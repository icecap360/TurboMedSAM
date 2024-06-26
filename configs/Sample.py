import torch
import models
import datasets
import os 
from framework import basic_dataloader_creator, basic_val_dataloader_creator, BaseScheduler, DistributedSampler
from torchvision import transforms
from metrics import SampleMetric
from losses import SampleLoss

model = models.SampleModel()

optimizer = {
    "optimizer": torch.optim.SGD(model.parameters(), lr=0.001)
}
grad_clip = dict(max_norm=35, norm_type=2)
lr_scheduler = BaseScheduler(
    regular_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer['optimizer'],
            milestones=[2,4],
            gamma=0.1
        ),
    optimizer=optimizer['optimizer'],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1
    )

compute = dict(
    gpu_ids = [0],
    use_cpu = False,
    use_amp = False,
    mp_start_method = 'fork',
    opencv_num_threads=0,
    cudnn_benchmark=False,
    workers_per_gpu=4,
    samples_per_gpu=2,
    batch_size=2,
    pin_memory=False,
    broadcast_bn_buffer = True,
    prefetch_factor=2,
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
    max_epochs = 5, #300
    max_iters = 10000,
    val_freq_epoch = 1,
    val_freq_iter = 1000,
    save_freq_iter = 1000,
    log_freq=5,
    resume_train = True,
    checkpoint_path = 'epoch_2.pth',
)
loss = SampleLoss(
    loss_weight = {
        'loss_pred': 1.0
    }
    )
metric = SampleMetric()
custom_hooks = []
seed = 0

dataset_type = datasets.SampleDataset
data_root = '/pub4/qasim/MedSAM/split_npzs_trash1/'
data = dict(
    drop_last = False,
    train=dict(
        dataset = dict(
            type=dataset_type,
            input_transform=datasets.sample_pipeline,
            target_transform=datasets.sample_pipeline),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= basic_dataloader_creator)
        ),
    val=dict(
        dataset = dict(
            type=dataset_type,
            input_transform=datasets.sample_pipeline,
            target_transform=datasets.sample_pipeline),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= basic_val_dataloader_creator)
        ),
    test=dict(
        dataset = dict(
            type=dataset_type,
            input_transform=datasets.sample_pipeline,
            target_transform=datasets.sample_pipeline),
        sampler = dict(type = DistributedSampler),
        dataloader_creator = dict( type= basic_val_dataloader_creator)
        )
    )
# saver = dict(
#     type = savers.CVPRMedSAMSaver,
#     subdirectory = 'results',
#     keys = ['embeddings']
# )