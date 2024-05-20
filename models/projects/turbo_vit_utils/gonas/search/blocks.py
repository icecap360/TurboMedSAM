"""Definitions for BlockSpecs, BlockSpecsConfig, and YAML handler."""
from collections import UserList
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Generic, List, Sequence, Type, TypeVar, Union

import torch
import torch.nn as nn
import yaml

from ..configs.base_config import InstantiateConfig
from ..darwinai.blockspecs import BlockSpec, BlockSpecConfig


class BlockSpecs(UserList):
    """A pure wrapper around a list of BlockSpec objects.

    This class satisfies the `Sequence[BlockSpec]` bound for `TBlockSpecs`,
    so it can be passed to `darwinai.torch.builder.build_model()`. All architectures
    that can be dynamically generated based on BlockSpec configurations should
    subclass this class.

    The constructor for this class is flexible. It can be passed a single sequence
    of BlockSpec objects (such as a list or tuple), like this:

        BlockSpecs([stage1, stage2, stage3])

    Alternatively, it can be passed one or more BlockSpec objects directly, like this:

        BlockSpecs(stage1, stage2, stage3)

    In the latter case, the BlockSpec arguments are collected into a list in the
    order they were passed.

    Args:
        blockspecs: One or more BlockSpec objects, or a single sequence of BlockSpec objects.
    """
    def __init__(self, *blockspecs: Union[BlockSpec, Sequence[BlockSpec]]):
        """Initialize a list of BlockSpec elements."""
        flat_blockspecs: List[BlockSpec] = []
        for item in blockspecs:
            if isinstance(item, Sequence):
                flat_blockspecs.extend(item)
            else:
                flat_blockspecs.append(item)
        super().__init__(flat_blockspecs)

    def to_config(self) -> "BlockSpecsConfig":
        """Return a `BlockSpecsConfig` object with the same fields as this object.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement function for serializing to config."
        )


# Type alias for `BlockSpecs` and its subclasses.
TBlockSpecs = TypeVar("TBlockSpecs", bound=BlockSpecs)


@dataclass
class BlockSpecsConfig(Generic[TBlockSpecs], InstantiateConfig):
    """A configuration holder for `BlockSpecs` objects.

    This class includes methods to convert the object to and from dictionary
    representations, which can be serialized and deserialized to and from YAML.

    Attributes:
        _target (Type[TBlockSpecs]): The class of the `BlockSpecs` object to be instantiated.
        _version (str): Version of the config for serialization compatibility.

    Methods:
        setup(**kwargs: Any) -> TBlockSpecs:
            Returns the instantiated `_target` object using `BlockSpecConfig` fields.
        to_dict() -> Dict[str, Dict[str, Union[int, bool]]]:
            Returns the public attributes of the BlockSpecs config as a dict.
        from_dict(data: Dict[str, Dict[str, Union[int, bool]]],
                  target: Type[TBlockSpecs]) -> "BlockSpecsConfig":
            Returns a BlockSpecsConfig object created from a dictionary.
        serialize() -> str:
            Returns the YAML string representation of the config.
        deserialize(yaml_path: Path, target: Type[TBlockSpecs]) -> "BlockSpecsConfig":
            Returns a BlockSpecsConfig object created from a YAML file.
    """

    _target: Type[TBlockSpecs]
    """Target class to instantiate (defined by subclasses)."""
    _version: str
    """Version of the config to enable backwards compatibility for serialization."""

    def setup(self, **kwargs: Any) -> TBlockSpecs:
        """Instantiate a `BlockSpecConfig` object for each field in the class.

        These are passed as keyword arguments to the constructor of `_target`.

        Returns:
            An instance of `_target`, initialized with the `BlockSpecConfig` fields.
        """
        blockspec_configs: Dict[str, BlockSpec] = {}
        for field in fields(self):
            value: Any = getattr(self, field.name)
            if isinstance(value, BlockSpecConfig):
                blockspec_configs[field.name] = value.setup()  # type: ignore
        return self._target(**blockspec_configs)

    def to_dict(self) -> Dict[str, Dict[str, Union[int, bool]]]:
        """Convert public attributes of the `BlockSpecsConfig` object to a dictionary.

        Returns:
            A dictionary representation of the `BlockSpecsConfig` object.
        """
        config = deepcopy(self)
        self_as_dict = {"_version": self._version}
        for field in fields(config):
            value: Any = getattr(config, field.name)
            if isinstance(value, BlockSpecConfig):
                self_as_dict[field.name] = value.to_dict()  # type: ignore
        self_as_dict["_class"] = self.__class__.__name__
        return self_as_dict  # type: ignore

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Dict[str, Union[int, bool]]],
        target: Type[TBlockSpecs],
    ) -> "BlockSpecsConfig":
        """Create a `BlockSpecsConfig` object from a dictionary.

        Args:
            data: The dictionary to convert into a `BlockSpecsConfig` object.
            target: The class to instantiate.

        Returns:
            An instance of `BlockSpecsConfig` initialized from the dictionary.
        """
        _version = data.pop("_version", "0.1.0")
        class_name = data.pop("_class")
        assert (
            class_name == cls.__name__
        ), f"Class {class_name} does not match expected {cls.__name__}"
        stage_configs = {
            name: BlockSpecConfig.from_dict(stage_config)
            for name, stage_config in data.items()
        }
        return cls(_target=target, _version=_version, **stage_configs)  # type: ignore

    def serialize(self) -> str:
        """Convert the `BlockSpecsConfig` object to a YAML string.

        Returns:
            A YAML string representation of the `BlockSpecsConfig` object.
        """
        return yaml.dump(self.to_dict())

    @classmethod
    def deserialize(
        cls, yaml_path: Path, target: Type[TBlockSpecs]
    ) -> "BlockSpecsConfig":
        """Create a `BlockSpecsConfig` object from a YAML file.

        Args:
            yaml_path: The path to the YAML file.
            target: The class to instantiate.

        Returns:
            Instance (or subclass) of `BlockSpecsConfig` initialized from the YAML file.
        """
        return cls.from_dict(
            data=yaml.load(yaml_path.read_text(), Loader=yaml.FullLoader), target=target
        )
