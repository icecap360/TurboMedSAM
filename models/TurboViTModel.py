# Copyright (c) Vision and Image Processing Lab Waterloo. All rights reserved.
import warnings

import torch.nn as nn

from framework import BaseModule

# gen_vit imports
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from hiera.hiera import Head, HieraBlock, PatchEmbed
from hiera.hiera_utils import (
    Reroll,
    Unroll,
    conv_nd,
    do_masked_conv,
    do_pool,
    pretrained_model
)
from timm.models.layers import DropPath, Mlp
from .projects.turbo_vit_utils.gonas.darwinai.blockspecs import BlockSpec
from .projects.turbo_vit_utils.gonas.search.blocks import BlockSpecs
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .projects.turbo_vit_utils.models.helpers import build_model_with_cfg, checkpoint_seq
from .projects.turbo_vit_utils.models.registry import register_model
from datetime import datetime
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np 
from framework import BaseModule, keep_keys

def _cfg(**kwargs):
    return {
            'input_size': (3, 1216, 1216), 
            # fields below are currenly unused
            'pool_size': None,
            'crop_pct': .9, 
            'interpolation': 'bicubic', 
            'fixed_input_size': False,
            'mean': IMAGENET_DEFAULT_MEAN, 
            'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'patch_embed.proj', 
            'classifier': 'head',
            'q_pool':3,
            'frozen_stages': -1
        }
default_cfgs = {
    'GenViT_2427': _cfg(),
}

def _create_genvit(variant, pretrained, **kwargs):
    return build_model_with_cfg(CompressibleHiera,
                                variant,
                                pretrained,
                                **kwargs)


 

default_cfgs = {

    'GenViT_2427': _cfg(),

}
@register_model
def GenViT_2427(pretrained, **kwargs):
    block_specs = [
        BlockSpec(channels=64, depth=1),
        BlockSpec(channels=128, depth=2),
        BlockSpec(channels=256, depth=8),
        BlockSpec(channels=512, depth=2),
    ]
    blockspecs = HieraBlockSpecs(*block_specs)
    model_args = dict(blockspecs=blockspecs, num_heads=1)
    return _create_genvit('GenViT_2427', pretrained=pretrained, **model_args)

def make_hiera(blockspecs: Sequence[BlockSpec]):
    blockspecs = HieraBlockSpecs(*blockspecs)
    model = CompressibleHiera(blockspecs, num_heads=1)
    return model

class HieraBlockSpecs(BlockSpecs):

    """A list of BlockSpecs for the stages in Hiera.

    This class can be directly passed to `darwinai.torch.builder.build_model()` as

    it satisfies the requirement of being a `Sequence[BlockSpec]`.

 

    Args:

        stage1: First stage of Hiera.

        stage2: Second stage of Hiera.

        stage3: Third stage of Hiera.

        stage4: Fourth stage of Hiera.

 

    Raises:

        ValueError: If any of the BlockSpecs are None.

    """

 

    def __init__(

        self, stage1: BlockSpec, stage2: BlockSpec, stage3: BlockSpec, stage4: BlockSpec

    ):

        blockspecs = [stage1, stage2, stage3, stage4]

        if None in blockspecs:

            raise ValueError("BlockSpec for stage 1, 2, 3, or 4 cannot be None.")

        super().__init__([stage1, stage2, stage3, stage4])

 

    @property

    def stage1(self):

        return self[0]

 

    @property

    def stage2(self):

        return self[1]

 

    @property

    def stage3(self):

        return self[2]

 

    @property

    def stage4(self):

        return self[3]

class CompressibleHiera(nn.Module):

    def __init__(

        self,

        blockspecs: HieraBlockSpecs,

        input_size: Tuple[int, ...] = (224, 224),

        in_chans: int = 3,

        num_heads: int = 1,  # initial number of heads

        num_classes: int = 1000,

        q_pool: int = 3,  # number of q_pool stages

        q_stride: Tuple[int, ...] = (2, 2),

        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)

        # mask_unit_attn: which stages use mask unit attention?

        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),

        head_mul: float = 2.0,

        dim_mul: float = 2.0,

        patch_kernel: Tuple[int, ...] = (7, 7),

        patch_stride: Tuple[int, ...] = (4, 4),

        patch_padding: Tuple[int, ...] = (3, 3),

        mlp_ratio: float = 4.0,

        drop_path_rate: float = 0.0,

        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),

        head_dropout: float = 0.0,

        head_init_scale: float = 0.001,

        sep_pos_embed: bool = False,

        frozen_stages: int = -1

    ):

        super().__init__()

        self.blockspecs = blockspecs
        stages = [bs.depth for bs in blockspecs]
        embed_dim = blockspecs.stage1.channels
        depth = sum(stages)
        
        self.tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)
        
        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        # self.patch_embed = PatchEmbed(
        #     in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        # )
        self.patch_embed = PatchEmbed(
            in_chans, 96, patch_kernel, patch_stride, patch_padding
        )
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            # self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, 96))
 
        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )

        self.reroll = Reroll(
            input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )

        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

 
        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]
            if i - 1 in self.stage_ends:
                cur_stage += 1
                dim_out = int(embed_dim * dim_mul)
                # dim_out = blockspecs[cur_stage].channels
                num_heads = int(num_heads * head_mul)
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride
            if i == 0:
                embed_dim = 96

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)

        # Initialize everything
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(partial(self._init_weights))

        # self.head.projection.weight.data.mul_(head_init_scale)

        # self.head.projection.bias.data.mul_(head_init_scale)

        # print(self.blocks_repr)
        
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, frozen_stages):
        assert frozen_stages >=0 and frozen_stages <= len(self.blocks)
        self.frozen_stages = frozen_stages
        
        if self.frozen_stages > 0:
            self.get_pos_embed().requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            stage = 1
            for i, blk in enumerate(self.blocks):
                self.blocks[i].eval()
                for param in self.blocks[i].parameters():
                    param.requires_grad = False
                if i in self.stage_ends:
                    stage+=1
                    if stage > self.frozen_stages:
                        break

    def _init_weights(self, m, init_bias=0.02):

        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):

            nn.init.trunc_normal_(m.weight, std=0.02)

            if isinstance(m, nn.Linear) and m.bias is not None:

                nn.init.constant_(m.bias, init_bias)

        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(m.bias, init_bias)

            nn.init.constant_(m.weight, 1.0)

 

    @torch.jit.ignore

    def no_weight_decay(self):

        if self.sep_pos_embed:

            return ["pos_embed_spatial", "pos_embed_temporal"]

        else:

            return ["pos_embed"]

 

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:

        """

        Generates a random mask, mask_ratio fraction are dropped.

        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.

        """

        B = x.shape[0]

        # Tokens selected for masking at mask unit level

        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units

        len_keep = int(num_windows * (1 - mask_ratio))

        noise = torch.rand(B, num_windows, device=x.device)

 

        # Sort noise for each sample

        ids_shuffle = torch.argsort(

            noise, dim=1

        )  # ascend: small is keep, large is remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

 

        # Generate the binary mask: 1 is *keep*, 0 is *remove*

        # Note this is opposite to original MAE

        mask = torch.zeros([B, num_windows], device=x.device)

        mask[:, :len_keep] = 1

        # Unshuffle to get the binary mask

        mask = torch.gather(mask, dim=1, index=ids_restore)

 

        return mask.bool()

 

    def get_pos_embed(self) -> torch.Tensor:

        if self.sep_pos_embed:

            return self.pos_embed_spatial.repeat(

                1, self.tokens_spatial_shape[0], 1

            ) + torch.repeat_interleave(

                self.pos_embed_temporal,

                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],

                dim=1,

            )

        else:

            return self.pos_embed

 

    def forward(

        self,

        x: torch.Tensor,

        mask: torch.Tensor = None,

        return_intermediates: bool = False,

    ) -> torch.Tensor:

        """

        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.

        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.

        """

        # Slowfast training passes in a list

        if isinstance(x, list):

            x = x[0]

        intermediates = []

        x = self.patch_embed(

            x,

            mask=mask.view(

                x.shape[0], 1, *self.mask_spatial_shape

            )  # B, C, *mask_spatial_shape

            if mask is not None

            else None,

        )

        x = x + self.get_pos_embed()

        x = self.unroll(x)

 

        # Discard masked tokens

        if mask is not None:

            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(

                x.shape[0], -1, x.shape[-1]

            )

 

        for i, blk in enumerate(self.blocks):

            # print(x.shape)

            x =  blk(x)

            if return_intermediates and i in self.stage_ends:

                intermediates.append(self.reroll(x, i, mask=mask))

 

        # if mask is None:

        #     x = x.mean(dim=1)

        #     x = self.norm(x)

        #     x = self.head(x)

 

        # x may not always be in spatial order here.

        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and

        # q_stride = (2, 2), not all unrolls were consumed,

        # intermediates[-1] is x in spatial order

        if return_intermediates:

            return x, intermediates

 
        return torch.moveaxis(self.reroll(x, self.stage_ends[-1], mask=mask),3,1)
        return x

    

    @property

    def blocks_repr(self):

        stages_channels = []

        stages_channels.append(self.blocks[0].dim_out)

        stages_depths = [1]

 

        for i, blk in enumerate(self.blocks[1:]):

            if blk.dim_out != stages_channels[-1]:

                stages_channels.append(blk.dim_out)

                stages_depths.append(1)

            else:

                stages_depths[-1] += 1

        return [(c, d) for c, d in zip(stages_channels, stages_depths)]

counter = 0

class TurboViTModel(BaseModule):
    def __init__(self, 
                input_size,
                num_heads,
                compile,
                block_specs,
                frozen_stages = -1,
                input_chans = 3,
                head_mul=2.0,
                init_cfg=None,
                pool_size= None,
                crop_pct= .9, 
                interpolation= 'bicubic', 
                fixed_input_size= False,
                mean= IMAGENET_DEFAULT_MEAN, 
                std= IMAGENET_DEFAULT_STD,
                first_conv= 'patch_embed.proj', 
                classifier= 'head',
                q_pool = 3,
                blockspecs=None,
                in_channels=None,
                distillation = False,
        ):

        # self.fixConvs = [
        #     ConvModule( 48, 48, kernel_size=1, act_cfg=dict(type='ReLU')),
        #     ConvModule( 96, 96, kernel_size=1, act_cfg=dict(type='ReLU')),
        #     ConvModule( 192, 192, kernel_size=1, act_cfg=dict(type='ReLU')),
        #     ConvModule( 384, 384, kernel_size=1, act_cfg=dict(type='ReLU')),
        # ]
        # for i in range(len(self.fixConvs)):
        #     self.fixConvs[i].to('cuda')

        super(TurboViTModel, self).__init__(init_cfg)
        self.relu = nn.ReLU()
        self.distillation = distillation
        self.block_specs = block_specs
        self.blockspecs = HieraBlockSpecs(*self.block_specs)
        self.model_args = dict(blockspecs=self.blockspecs, 
                               input_size=(input_size, input_size),
                               frozen_stages= frozen_stages,
                               in_chans=input_chans,
                               num_heads=num_heads, 
                               q_pool=q_pool,
                               head_mul=head_mul)
        default_cfgs['GenViT_2427'] =  {
                    'input_size': (input_chans, input_size, input_size), 
                    # fields below are currenly unused
                    'pool_size': pool_size,
                    'crop_pct': crop_pct, 
                    'interpolation': interpolation, 
                    'fixed_input_size': fixed_input_size,
                    'mean': mean, 
                    'std': std,
                    'q_pool': q_pool,
                    'first_conv': first_conv, 
                    'classifier': classifier,
                }
        self.backbone = build_model_with_cfg(CompressibleHiera,
                                'GenViT_2427',
                                pretrained=False,
                                **self.model_args)
        
        self.compile = compile
        if compile:
            self.backbone = torch.compile(self.backbone)
        
        # self.logSummary()

    def logSummary(self):
        import torchsummary
        with open('/home/m45ali/Qasim/temp_debug/mmdetection_turbovit_summary.txt', 'w') as f:
            #report, _ =.summary_string(self.backbone, ...)
            torchsummary.summary(self.backbone.to('cuda'), (3, 512, 512))
    
    def init_weights(self, state_dict=None, strict=True):
        if self.init_cfg == None and state_dict==None:
            return
        elif state_dict:
            keys = [k for k in state_dict.keys()]
            if keys[0].startswith('module.'):
                state_dict = keep_keys(state_dict,"module.", [(r"^module.", "")])
            self.load_state_dict(
                        state_dict, 
                        strict= strict)
        elif 'pretrained' in self.init_cfg['type'].lower():
            if not 'checkpoint' in self.init_cfg.keys():
                raise Exception('Missing checkpoint')   
            self.load_state_dict(
                        keep_keys(torch.load(self.init_cfg['checkpoint'])['state_dict'],"module.",
                                  [(r"^module.", "")]), 
                        strict=self.init_cfg.get('strict') or True)
        else:
            raise Exception('init_cfg is formatted incorrectly')

    def forward(self, data_batch):  # should return a tuple/list
        if self.distillation:
            x = data_batch['image']
        else:
            x = data_batch
        # x, intermediates =  self.backbone.forward(x, return_intermediates=True)
        x =  self.backbone.forward(x, return_intermediates=False)
        if self.distillation:
            return {'embeddings': x}
        else:
            return x
