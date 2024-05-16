"""Definitions for Hiera related BlockSpecs, ArchStyles, and YAML handlers."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Type, Union

import yaml
from hiera import hiera_base_224, hiera_tiny_224

from gonas.darwinai.blockspecs import BlockSpec, BlockSpecConfig
from gonas.search.arch_styles import ArchStyle
from gonas.search.blocks import BlockSpecs, BlockSpecsConfig


class HieraBlockSpecs(BlockSpecs):
    """A list of BlockSpecs for the stages in Hiera.

    This class can be directly passed to `darwinai.torch.builder.build_model()` as it satisfies the requirement of being a `Sequence[BlockSpec]`.

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
        """Initialize BlockSpecs list with a BlockSpec for each stage of Hiera."""
        blockspecs = [stage1, stage2, stage3, stage4]
        if None in blockspecs:
            raise ValueError("BlockSpec for stage 1, 2, 3, or 4 cannot be None.")
        super().__init__(blockspecs)

    @property
    def stage1(self):
        """Return the BlockSpec corresponding to stage 1 of Hiera."""
        return self[0]

    @property
    def stage2(self):
        """Return the BlockSpec corresponding to stage 2 of Hiera."""
        return self[1]

    @property
    def stage3(self):
        """Return the BlockSpec corresponding to stage 3 of Hiera."""
        return self[2]

    @property
    def stage4(self):
        """Return the BlockSpec corresponding to stage 4 of Hiera."""
        return self[3]

    def to_config(self) -> "HieraBlockSpecsConfig":
        """Convert this HieraBlockSpecs to a HieraBlockSpecsConfig representation.

        This is handy for easier serialization.
        """
        return HieraBlockSpecsConfig(
            stage1=self.stage1.to_config(),
            stage2=self.stage2.to_config(),
            stage3=self.stage3.to_config(),
            stage4=self.stage4.to_config(),
        )


@dataclass
class HieraBlockSpecsConfig(BlockSpecsConfig[HieraBlockSpecs]):
    """Configuration for instantiating the BlockSpecs for a Hiera model.

    Default configs correspond to Hiera-B (base).
    """

    _target: Type[HieraBlockSpecs] = field(default_factory=lambda: HieraBlockSpecs)
    """Target class to instantiate."""
    _version: str = "0.1.0"

    stage1: BlockSpecConfig = field(
        default_factory=lambda: BlockSpecConfig(
            channels=96, depth=2, freeze_channel=False, freeze_depth=False
        )
    )
    """Specifies the block config for the first stage."""

    stage2: BlockSpecConfig = field(
        default_factory=lambda: BlockSpecConfig(
            channels=192, depth=3, freeze_channel=True, freeze_depth=False
        )
    )
    """Specifies the block config for the second stage."""

    stage3: BlockSpecConfig = field(
        default_factory=lambda: BlockSpecConfig(
            channels=384, depth=16, freeze_channel=True, freeze_depth=False
        )
    )
    """Specifies the block config for the third stage."""

    stage4: BlockSpecConfig = field(
        default_factory=lambda: BlockSpecConfig(
            channels=768, depth=3, freeze_channel=True, freeze_depth=False
        )
    )
    """Specifies the block config for the fourth stage."""

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Dict[str, Union[int, bool]]],
        target: Type = HieraBlockSpecs,
    ) -> "HieraBlockSpecsConfig":
        """Create a `HieraBlockSpecsConfig` object from a dictionary."""
        return super().from_dict(data=data, target=target)  # type: ignore

    @classmethod
    def deserialize(
        cls, yaml_path: Path, target: Type = HieraBlockSpecs
    ) -> "HieraBlockSpecsConfig":
        """Create a `HieraBlockSpecsConfig` object from a YAML file."""
        return super().deserialize(yaml_path, target=target)  # type: ignore


class HieraArchStyle(ArchStyle):
    """Enumeration of the different macro architecture styles for Hiera."""

    BASE = ArchStyle.auto()
    TINY = ArchStyle.auto()
    BASE_CHANNEL_UNFROZEN_DEPTH_FROZEN = ArchStyle.auto()
    TINY_CHANNEL_UNFROZEN_DEPTH_FROZEN = ArchStyle.auto()
    BASE_CHANNEL_UNFROZEN_DEPTH1_UNFROZEN = ArchStyle.auto()
    TINY_CHANNEL_UNFROZEN_DEPTH1_UNFROZEN = ArchStyle.auto()
    BASE_CHANNEL_UNFROZEN_DEPTH_UNFROZEN = ArchStyle.auto()
    TINY_CHANNEL_UNFROZEN_DEPTH_UNFROZEN = ArchStyle.auto()

    @property
    def baseline_style(self):
        """Return the baseline style for this architecture style.

        Examples:
            >>> HieraArchStyle.BASE.baseline_style
            HieraArchStyle.BASE
            >>> HieraArchStyle.TINY_CHANNEL_UNFROZEN_DEPTH_UNFROZEN.baseline_style
            HieraArchStyle.TINY
            >>> HieraArchStyle.BASE_CHANNEL_UNFROZEN_DEPTH1_UNFROZEN.baseline_style
            HieraArchStyle.BASE
        """
        baseline_style_name = self.name.split("_")[0]
        return getattr(HieraArchStyle, baseline_style_name)

    @property
    def pretrained_model(self) -> Callable:
        """Return a function for building the corresponding pretrained Hiera model."""
        baseline_style = self.baseline_style
        if baseline_style == HieraArchStyle.BASE:
            return hiera_base_224
        elif baseline_style == HieraArchStyle.TINY:
            return hiera_tiny_224
        else:
            raise ValueError(f"Invalid baseline style: {baseline_style}")

    @property
    def blockspecs_config(self) -> HieraBlockSpecsConfig:
        """Return the HieraBlockSpecsConfig object corresponding to the arch style."""
        return hiera_blockspecs_config_factory(self)


def hiera_blockspecs_config_factory(
    arch_style: HieraArchStyle,
) -> HieraBlockSpecsConfig:
    """Instantiate HieraBlockSpecsConfig objects for different arch styles.

    Args:
        arch_style: The architecture style to instantiate.

    Returns:
        A HieraBlockSpecsConfig object.
    """
    base_config = HieraBlockSpecsConfig()

    if "TINY" in arch_style.name:
        base_config.stage1.depth = 1
        base_config.stage2.depth = 2
        base_config.stage3.depth = 7
        base_config.stage4.depth = 2

    if "CHANNEL_UNFROZEN_DEPTH_FROZEN" in arch_style.name:
        base_config.stage1.freeze_depth = True
        base_config.stage2.freeze_depth = True
        base_config.stage3.freeze_depth = True
        base_config.stage4.freeze_depth = True

    if "CHANNEL_UNFROZEN_DEPTH1_UNFROZEN" in arch_style.name:
        base_config.stage1.freeze_depth = False
        base_config.stage2.freeze_depth = True
        base_config.stage3.freeze_depth = True
        base_config.stage4.freeze_depth = True

    if "CHANNEL_UNFROZEN_DEPTH_UNFROZEN" in arch_style.name:
        base_config.stage1.freeze_depth = False
        base_config.stage2.freeze_depth = False
        base_config.stage3.freeze_depth = False
        base_config.stage4.freeze_depth = False

    return base_config
