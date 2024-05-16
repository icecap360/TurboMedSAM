"""Module defining HieraBlocksSearcherConfig."""

from dataclasses import dataclass, field
from typing import Type

from gonas.models.hiera.hiera_compressible import HieraCompressible
from gonas.models.hiera.hiera_styles import (
    HieraArchStyle,
    HieraBlockSpecs,
    HieraBlockSpecsConfig,
)
from gonas.search.blocks import BlockSpecs, BlockSpecsConfig
from gonas.search.profiler import ModelProfiler
from gonas.search.searcher import BlocksSearcher, BlocksSearcherConfig


@dataclass
class HieraBlocksSearcherConfig(BlocksSearcherConfig):
    """Config for HieraBlocksSearcher."""

    _target: Type = field(default_factory=lambda: BlocksSearcher[HieraCompressible])
    """Target class to instantiate."""

    _model_type: Type[HieraCompressible] = field(
        default_factory=lambda: HieraCompressible
    )
    """Model type to use for search."""

    _blockspecs_type: Type[HieraBlockSpecs] = field(
        default_factory=lambda: HieraBlockSpecs
    )
    """Blockspecs type to use for search."""

    _blockspecs_config_type: Type[HieraBlockSpecsConfig] = field(
        default_factory=lambda: HieraBlockSpecsConfig
    )
    """Blockspecs config type to use for search."""

    arch_style: HieraArchStyle = HieraArchStyle.BASE
    """Macro architecture style to use for initial blockspecs."""

    blocks_name: str = "hiera"
    """Name of the blocks to search for."""
