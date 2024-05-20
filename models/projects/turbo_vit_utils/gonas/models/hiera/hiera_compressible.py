"""Defines the HieraCompressible class."""
import math
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from hiera.hiera import Head, Hiera, HieraBlock, PatchEmbed
from hiera.hiera_utils import Reroll, Unroll

from gonas.models.hiera.hiera_styles import HieraBlockSpecs
from gonas.search.profiler import ModelProfiler

INPUT_SHAPE = [1, 3, 224, 224]


class HieraCompressible(Hiera):
    """Generates a compressible version of Hiera based on the provided blockspecs."""

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
    ):
        """Initialize a HieraCompressible model.

        This constructor is designed to function identically to Hiera's constructor,
        except that it generates the architecture based on the provided blockspecs.
        """
        # No need to call Hiera's constructor since this function does everything it
        # does. Instead, call the constructor of Hiera's parent class.
        nn.Module.__init__(self)

        self.blockspecs = blockspecs
        self.num_classes = num_classes
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

        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
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
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

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
        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

        # Compute the actual architecture representation.
        self.arch_repr = self.blocks_repr(self)
        self.profiler = ModelProfiler(
            model=self,
            input_shape=INPUT_SHAPE,
            attributes_to_serialize=["blockspecs", "arch_repr"],
        )

    @property
    def loggable_dict(self) -> dict:
        """Return the serialized dict of the model."""
        return self.profiler.loggable_dict

    @staticmethod
    def blocks_repr(hiera: Union["HieraCompressible", Hiera]) -> List[Tuple[int, int]]:
        """Return a list of tuples of (channels, depth) for each stage in Hiera.

        This corresponds to a simplified representation of the Hiera architecture, but in the
        language of DarwinAI's BlockSpecs.

        Args:
            hiera: Hiera model (can be vanilla Hiera or compressible Hiera).

        Returns:
            A list of tuples of (channels, depth) for each stage in Hiera.
        """
        stages_channels = []
        stages_channels.append(hiera.blocks[0].dim_out)
        stages_depths = [1]

        for blk in hiera.blocks[1:]:
            if blk.dim_out != stages_channels[-1]:
                stages_channels.append(blk.dim_out)
                stages_depths.append(1)
            else:
                stages_depths[-1] += 1

        return [(c, d) for c, d in zip(stages_channels, stages_depths)]
