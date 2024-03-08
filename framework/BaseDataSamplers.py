import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import os
import torch.distributed as dist
import math 
from .BaseDataset import BaseDataset

class ClassBalancedSampler(sampler.Sampler):
    r"""Makes sure each class is sampled from equally.
    Also restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~ClassBalancedSampler` instance as a
    :class:`DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    .. see: https://mmdetection.readthedocs.io/en/v2.24.1/_modules/mmdet/datasets/samplers/class_aware_sampler.html
    
    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = ClassBalancedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """
    def __init__(self,
                 dataset: BaseDataset,
                 samples_per_gpu=1,
                 workers_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 subset_classes=None,
                 shuffle=True,
                 num_sample_class=1,
                 **kwargs):
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        
        self.dataset = dataset
        self.n_process = num_replicas
        self.rank = rank
        self.epoch = 0
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.subset_classes = subset_classes
        self.num_sample_class = num_sample_class
        self.shuffle = shuffle
        self.seed = seed

        self.n_samples_per_process = int(
            math.ceil(
                len(self.dataset) * 1.0 / self.n_process /
                self.samples_per_gpu)) * self.samples_per_gpu
        # total_size = the number of indices ~ length of dataset
        self.total_size = self.n_samples_per_process * self.n_process

        
        categories = self.dataset.get_categories()
        assert hasattr(dataset, 'get_cat2indices'), \
            'dataset must have `get_cat2indices` function'
        self.cat_dict = dataset.get_cat2indices()
        # get number of images containing each category
        self.num_cat = [len(x) for x in self.cat_dict.values()]
        # filter categories without images
        self.valid_cat = [
            categories[i] for i, length in enumerate(self.num_cat) if length != 0
        ]
        # filter categories if user requests a subset
        if not (self.subset_classes is None):
            self.valid_cat = [
                cat for cat in self.valid_cat if cat in self.subset_classes
            ]
        self.num_classes = len(self.valid_cat)

        # generate the order of indices
        self.prev_generator_seed = self.seed
        self.indices = []
        self.generate_indices(self.seed)
        
        super(ClassBalancedSampler, self).__init__()
    
    def gen_cat_img_inds(self, cls_list, data_dict, num_sample_cls):
            """Traverse the categories and extract `num_sample_cls` image
            indexes of the corresponding categories one by one."""
            id_indices = []
            for _ in range(len(cls_list)):
                cls_idx = next(cls_list)
                for _ in range(num_sample_cls):
                    id = next(data_dict[cls_idx])
                    id_indices.append(id)
            return id_indices
    
    def generate_indices(self, new_seed):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(new_seed)

        # initialize label list
        cat_iter_list = RandomCycleIter(self.valid_cat, generator=g)
        # initialize each per-label image list
        data_iter_dict = dict()
        for cat in self.valid_cat:
            data_iter_dict[cat] = RandomCycleIter(self.cat_dict[cat], generator=g)
        
        # num_bins represents the number of calls to gen_cat_img_inds to get total_size data samples
        # Math uses fact that gen_cat_img_inds generates num_classes*num_sample_class indices together
        num_bins = int(
            math.ceil(self.total_size * 1.0 / self.num_classes /
                      self.num_sample_class))
        indices = []
        for _ in range(num_bins):
            indices += self.gen_cat_img_inds(cat_iter_list, data_iter_dict,
                                        self.num_sample_class)
        
        # add or remove data samples to make it evenly divisible
        if len(indices) >= self.total_size:
            indices = indices[:self.total_size]
        else:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        self.indices = indices

    def __iter__(self):
        generator_seed = self.epoch+self.seed
        if self.prev_generator_seed != generator_seed:
            self.generate_indices(generator_seed)
            self.prev_generator_seed = generator_seed

        # subsample
        offset = self.n_samples_per_process * self.rank
        indices = self.indices[offset:offset + self.n_samples_per_process]
        assert len(indices) == self.n_samples_per_process

        return iter(indices)
    
    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RandomCycleIter:
    """Shuffle the list and do it again after the list have traversed.

    The implementation logic is referred to
    https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/datasets/loader/sampler.py

    Example:
        >>> label_list = [0, 1, 2, 4, 5]
        >>> g = torch.Generator()
        >>> g.manual_seed(0)
        >>> label_iter_list = RandomCycleIter(label_list, generator=g)
        >>> index = next(label_iter_list)
    Args:
        data (list or ndarray): The data that needs to be shuffled.
        generator: An torch.Generator object, which is used in setting the seed
            for generating random numbers.
    """  # noqa: W605

    def __init__(self, data, generator=None):
        self.data = data
        self.length = len(data)
        self.index = torch.randperm(self.length, generator=generator).numpy()
        self.i = 0
        self.generator = generator

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.i == self.length:
            self.index = torch.randperm(
                self.length, generator=self.generator).numpy()
            self.i = 0
        idx = self.data[self.index[self.i]]
        self.i += 1
        return idx
