import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Dict, Type, Union

from strenum import StrEnum

from ..utils import misc_utils as mu

if mu.is_package_available("darwinai"):
    from darwinai.torch.builder import BlockSpec as DarwinaiBlockSpec
    from darwinai.torch.builder import BuildMetrics

    class BlockSpec(DarwinaiBlockSpec):
        """Simple wrapper around Darwinai's BlockSpec to make it compatible with the
        BlockSpec class from gonas."""

        def to_config(self) -> "BlockSpecConfig":
            """Returns the BlockSpec config for this BlockSpec."""
            return BlockSpecConfig(
                channels=self.channels,
                depth=self.depth,
                freeze_channel=self.freeze_channel,
                freeze_depth=self.freeze_depth,
            )

        def __repr__(self):
            return f"BlockSpec(channels={self.channels}, depth={self.depth}, freeze_channel={self.freeze_channel}, freeze_depth={self.freeze_depth})"

else:

    @dataclass
    class BlockSpec:  # type: ignore
        """This class is used to describe a block in the neural network architecture.

        The number of channels and depth for this block will be altered when exploring a
        suitable architecture by the build_model function.
        """

        channels: int
        depth: int
        freeze_channel: bool = False
        freeze_depth: bool = False

        def to_config(self) -> "BlockSpecConfig":
            """Returns the BlockSpec config for this BlockSpec."""
            return BlockSpecConfig(
                channels=self.channels,
                depth=self.depth,
                freeze_channel=self.freeze_channel,
                freeze_depth=self.freeze_depth,
            )

        def __repr__(self):
            return f"BlockSpec(channels={self.channels}, depth={self.depth}, freeze_channel={self.freeze_channel}, freeze_depth={self.freeze_depth})"

    class BuildMetrics(StrEnum):  # type: ignore
        """Defines the search criteria for build_model."""

        FLOPS = enum.auto()
        PARAMETERS = enum.auto()


@dataclass
class BlockSpecConfig:
    _target: Type = field(default_factory=lambda: BlockSpec)
    _version: str = "0.1.0"
    channels: int = 256
    depth: int = 8
    freeze_channel: bool = False
    freeze_depth: bool = False

    def setup(self, **kwargs) -> BlockSpec:
        """Return the instantiated object using the config."""
        return self._target(
            self.channels, self.depth, self.freeze_channel, self.freeze_depth
        )

    def to_dict(self) -> Dict[str, Union[int, bool]]:
        self_as_dict = dataclasses.asdict(self)
        del self_as_dict["_target"]
        self_as_dict["_class"] = self.__class__.__name__
        return self_as_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Union[int, bool]]) -> "BlockSpecConfig":
        _target = BlockSpec
        _version: str = data.pop("_version", "0.1.0")  # type: ignore
        class_name = data.pop("_class")
        assert (
            class_name == cls.__name__
        ), f"Class {class_name} does not match expected {cls.__name__}"
        return cls(_target=_target, _version=_version, **data)  # type: ignore
