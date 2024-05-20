"""Module defining architecture search functions."""
import enum
from typing import Callable

from strenum import StrEnum

from gonas.search.blocks import BlockSpecsConfig


class ArchStyle(StrEnum):
    """Base class for architecture styles."""

    @staticmethod
    def auto() -> "ArchStyle":
        """Pure wrapper around `enum.auto`.

        Since `ArchStyle` derives from `StrEnum`, the value of the enum will
        default to the member name. This function is a workaround for mypy not
        being able to infer the return type of enum.auto() as it doesn't get
        generated until runtime.

        Example usage:
            >>> class DummyArchStyle(ArchStyle):
            ...     BIG_MLP = ArchStyle.auto()
            ...     SMALL_MLP = ArchStyle.auto()
            ...
            >>> DummyArchStyle.BIG_MLP
            <DummyArchStyle.BIG_MLP: 'BIG_MLP'>
            >>> DummyArchStyle.SMALL_MLP
            <DummyArchStyle.SMALL_MLP: 'SMALL_MLP'>
        """
        return enum.auto()  # type: ignore

    @property
    def baseline_style(self) -> "ArchStyle":
        """Return the baseline architecture style.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def pretrained_model(self) -> Callable:
        """Return a function for building the corresponding pretrained model."""
        raise NotImplementedError

    @property
    def blockspecs_config(self) -> BlockSpecsConfig:
        """Return the blockspecs config for the architecture style.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
