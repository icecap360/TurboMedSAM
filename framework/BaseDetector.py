import torch
import torch.nn as nn
from .BaseModules import BaseModule
from abc import ABCMeta, abstractmethod

class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False
        
    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # @abstractmethod
    # def extract_feat(self, imgs):
    #     """Extract features from images."""
    #     pass
    
    # def extract_feats(self, imgs):
    #     """Extract features from multiple images.

    #     Args:
    #         imgs (list[torch.Tensor]): A list of images. The images are
    #             augmented from the same image but in different ways.

    #     Returns:
    #         list[torch.Tensor]: Features of different images
    #     """
    #     assert isinstance(imgs, list)
    #     return [self.extract_feat(img) for img in imgs]
    
    # @abstractmethod
    # def forward_train(self, imgs, img_metas):
    #     """
    #     Args:
    #         img (Tensor): of shape (N, C, H, W) encoding input images.
    #             Typically these should be mean centered and std scaled.
    #         img_metas (list[dict]): List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys, see
    #             :class:`mmdet.datasets.pipelines.Collect`.
    #         kwargs (keyword arguments): Specific to concrete implementation.
    #     """
    #     pass
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # batch_input_shape = tuple(imgs[0].size()[-2:])
        # for img_meta in img_metas:
        #     img_meta['batch_input_shape'] = batch_input_shape

    # @abstractmethod
    # def forward_test(self, imgs, img_metas, **kwargs):
    #     """
    #     Args:
    #         imgs (List[Tensor]): the outer list indicates test-time
    #             augmentations and inner Tensor should have a shape NxCxHxW,
    #             which contains all images in the batch.
    #         img_metas (List[List[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch.
    #     """
        # for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
        #     if not isinstance(var, list):
        #         raise TypeError(f'{name} must be a list, but got {type(var)}')

        # num_augs = len(imgs)
        # if num_augs != len(img_metas):
        #     raise ValueError(f'num of augmentations ({len(imgs)}) '
        #                      f'!= num of image meta ({len(img_metas)})')

        # # NOTE the batched image size information may be useful, e.g.
        # # in DETR, this is needed for the construction of masks, which is
        # # then used for the transformer_head.
        # for img, img_meta in zip(imgs, img_metas):
        #     batch_size = len(img_meta)
        #     for img_id in range(batch_size):
        #         img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        