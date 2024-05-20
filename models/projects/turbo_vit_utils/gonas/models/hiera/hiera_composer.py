from dataclasses import dataclass, field
from typing import Type, Union

from composer.models.base import ComposerModel

from gonas.configs.base_config import InstantiateConfig
from gonas.models.hiera.hiera_compressible import HieraCompressible
from gonas.models.hiera.hiera_styles import HieraArchStyle


@dataclass
class HieraComposerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: HieraComposer)
    """Target class to instantiate."""
    model: Union[HieraCompressible, HieraArchStyle] = HieraArchStyle.BASE
    """Model to train/eval. If ArchStyle, will fetch a corresponding
    pretrained original Hiera model."""
    num_classes: int = 1000
    """Number of classes in the dataset."""


class HieraComposer(ComposerModel):
    def __init__(self, config: HieraComposerConfig) -> None:
        super().__init__()
        self.config = config

        # Instantiate the model.
        if isinstance(config.model, HieraArchStyle):
            self.model = config.model.pretrained_model(pretrained=True)
        else:
            self.model = config.model
