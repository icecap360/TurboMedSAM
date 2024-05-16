"""Module to enable search for generic architectures using DarwinAI SDK."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar, Union

import torch
import torch.nn as nn

import gonas.utils.misc_utils as mu
from gonas.configs.base_config import InstantiateConfig
from gonas.darwinai.blockspecs import BlockSpec, BlockSpecConfig, BuildMetrics
from gonas.search.arch_styles import ArchStyle
from gonas.search.blocks import BlockSpecs, BlockSpecsConfig, TBlockSpecs
from gonas.search.profiler import ModelProfiler


@dataclass
class BlocksSearcherConfig(InstantiateConfig):
    """Configuration class for the BlocksSearcher."""

    _target: Type
    """Target class to instantiate."""
    _model_type: Type[nn.Module]
    """Type of compressible model to search for."""
    _blockspecs_type: Type[BlockSpecs]
    """Type of BlockSpecs to use for search."""
    _blockspecs_config_type: Type[BlockSpecsConfig]
    """Type of BlockSpecsConfig to use for search."""
    arch_style: ArchStyle
    """Macro architecture style of block."""
    blocks_name: str
    """Human readable name for the blocks under search (used to name cache file)."""
    target_ratio: float = 1.0
    """Target flop ratio for search."""
    build_metric: BuildMetrics = BuildMetrics.FLOPS
    """Metric to optimize for during architecture search."""
    pretrained_model: Optional[Union[nn.Module, Path, bool]] = None
    """Pretrained model to use for search.

    If set, the generated architecture will be initialized based on the weights of the
    pretrained model. Can specify either a model instance or a path to a model
    checkpoint, or a boolean value. If boolean is set to True, will attempt to load
    weights from the default cached model that were generated from the previous search
    when a pretrained model was used.
    """
    use_cache: bool = False
    """Uses cached results from previous search.

    If no cached results are found, will run search to find optimal compressed model and
    cache results for future runs.
    """
    cache_dir: Path = Path(f"{mu.get_repo_working_dir()}/cache/configs")
    """Directory to store cached results."""


TCompressibleModel = TypeVar("TCompressibleModel", bound=nn.Module)


class BlocksSearcher(Generic[TCompressibleModel]):
    """Searches for architectures that can be generated based on BlockSpecsConfig."""

    _model_type: Type[TCompressibleModel]

    def __init__(
        self,
        config: BlocksSearcherConfig,
        input_shape: Sequence[int],
        **model_kwargs: Any,
    ) -> None:
        """Init with search configuration, input shape and model specific parameters.

        Args:
            config: Configuration for the BlocksSearcher.
            input_shape: Shape of the input data.
            model_kwargs: Additional keyword arguments for the model.
        """
        self.config = config
        self.input_shape = input_shape
        self.model_kwargs = model_kwargs
        self._loggable_dict: Dict[str, Any] = {
            "generated": True,
            "target_ratio": config.target_ratio,
            "build_metric": config.build_metric,
            "pretrained": config.pretrained_model,  # TODO(snair): Can this be serialized?
            "arch_style": config.arch_style,
        }

        self.cache_dir: Path = self.compose_cache_dir()

        self._generated_model: TCompressibleModel = self.search()
        self._loggable_dict[
            f"generated_{config.blocks_name}"
        ] = self.generated_model.loggable_dict
        # Compute comparisons for the target ratio against the baseline style.
        # This ratio is what build_model() actually optimizes for during search.
        baseline_arch_style = config.arch_style.baseline_style
        self.compute_comparisons(
            baseline_arch_style=baseline_arch_style,
            baseline_name=f"baseline_{config.blocks_name}_{baseline_arch_style.name.lower()}",
        )

    def searcher_fn(self, initial_blockspecs: BlockSpecs) -> TCompressibleModel:
        """Generate a compressible model based on the provided BlockSpecs.

        Wrapper function around DarwinAI's `build_model` so that the import scope can
        be limited to this function. This allows us to use this class without having to
        install DarwinAI.

        Args:
            initial_blockspecs: Initial blockspecs to use for architecture search.

        Returns:
            Compressed model for the specified target ratio.
        """
        from darwinai.torch.builder import build_model  # type: ignore

        # Ignore directive is needed because build_model() promises to return
        # an nn.Module when it actually returns a subclass (TCompressibleModel).
        pretrained_model = self.config.pretrained_model
        if isinstance(pretrained_model, Path):
            # Pretrained model architecture must match architecture corresponding to
            # initial blockspecs. If not, assertion will be raised by `build_model()`.
            pretrained_model_weights = torch.load(pretrained_model)
            pretrained_model = self.generate_model(initial_blockspecs)
            pretrained_model.load_state_dict(pretrained_model_weights)

        generated_model: TCompressibleModel = build_model(
            model_fn=self.generate_model,  # type: ignore
            initial_blockspecs=initial_blockspecs,  # type: ignore
            input_shape=self.input_shape,
            target_ratio=self.config.target_ratio,
            build_metric=self.config.build_metric,  # type: ignore
            pretrained_model=pretrained_model,  # type: ignore
        )  # type: ignore

        return generated_model

    def search(self) -> TCompressibleModel:
        """Perform a search for a compressed model.

        If cached results are available, will use cached results instead of running
        search.

        Returns:
            A compressed model for the specified target ratio.
        """
        # Return cached model if available. Otherwise, run search.
        if self.config.use_cache:
            cached_model = self.load_cached_model()
            if cached_model is not None:
                return cached_model

        # Get blockspecs for initial architecture based on architecture style.
        initial_blockspecs = self.config.arch_style.blockspecs_config.setup()

        # Run search to find optimal compressed model.
        generated_model: TCompressibleModel = self.searcher_fn(initial_blockspecs)

        # Cache generated model if enabled.
        if self.config.use_cache:
            self.cache_model(generated_model)

        return generated_model

    def generate_model(self, blockspecs: Sequence[BlockSpec]) -> TCompressibleModel:
        """Generate a compressible model from the provided BlockSpecs.

        Any additional keyword arguments required by the model constructor will be
        unpacked from the `model_kwargs` attribute.
        """
        # Typecast to the correct BlockSpecs type if necessary.
        if not isinstance(blockspecs, self.config._blockspecs_type):
            blockspecs = self.config._blockspecs_type(*blockspecs)

        return self.config._model_type(blockspecs=blockspecs, **self.model_kwargs)  # type: ignore

    def make_baseline_model(self, arch_style: ArchStyle) -> TCompressibleModel:
        """Generate baseline model for the specified architecture style.

        Args:
            arch_style: Architecture style to generate baseline model for.

        Returns:
            A model of type CompressibleModel with architecture corresponding to
            the specified style.
        """
        blockspecs = arch_style.blockspecs_config.setup()
        return self.generate_model(blockspecs)

    def compose_cache_dir(self) -> Path:
        """Return the directory to use for caching the generated model.

        Returns:
            Path for caching directory.
        """
        return (
            self.config.cache_dir
            / self.config.blocks_name
            / self.config.arch_style.name.lower()
        )

    def compose_cache_filename(self, ext: str = "yml") -> str:
        """Return the filename to use for caching the generated model.

        Args:
            ext: File extension for the cache file.

        Returns:
            Filename for cache file.
        """
        # Use "fr" for flops ratio and "pr" for params ratio.
        build_metric = "fr" if self.config.build_metric == BuildMetrics.FLOPS else "pr"
        arch_style = self.config.arch_style.name.lower()
        if self.config.pretrained_model:
            filename = f"{arch_style}-pretrained-{build_metric}_{self.config.target_ratio}.{ext}"
        else:
            filename = f"{arch_style}-{build_metric}_{self.config.target_ratio}.{ext}"

        return filename

    def load_cached_model(self) -> Optional[TCompressibleModel]:
        """Load a cached model from disk if available.

        Returns:
            A model of type CompressibleModel if cached model is available, else None.
        """
        cached_yaml_path: Path = self.cache_dir / self.compose_cache_filename(ext="yml")
        if not cached_yaml_path.exists():
            print(f"No cached model found at {cached_yaml_path}.")
            return None

        cached_blockspecs_config = self.config._blockspecs_config_type.deserialize(
            cached_yaml_path, target=self.config._blockspecs_type
        )
        cached_blockspecs = cached_blockspecs_config.setup()
        if not isinstance(cached_blockspecs, self.config._blockspecs_type):
            raise AssertionError(
                f"Expected blockspecs of type {self.config._blockspecs_type}, "
                f"got {type(cached_blockspecs)}"
            )

        # Generate model based on the cached blockspecs.
        model = self.generate_model(cached_blockspecs)

        # Try to load cached weights into model if set to True, and if model if exists.
        if self.config.pretrained_model == True:
            model = self.load_cached_weights(model)

        return model

    def load_cached_weights(self, model: TCompressibleModel) -> TCompressibleModel:
        """Load cached pretrained weights into model.

        Checks if weights are available.If not, the original model is
        returned and a warning is logged.

        Args:
            model: Model to load cached weights into.

        Returns:
            Model with cached weights loaded if available.
        """
        cached_weights_path: Path = self.cache_dir / self.compose_cache_filename(
            ext="pt"
        )
        if not cached_weights_path.exists():
            print(f"No cached weights found at {cached_weights_path}!")
            return model

        # Load the weights
        cached_weights = torch.load(cached_weights_path)
        model.load_state_dict(cached_weights)
        return model

    def cache_model(self, generated_model: TCompressibleModel) -> None:
        """Cache the generated model to disk.

        Args:
            generated_model: Model to cache.
        """
        # Create cache directory if it doesn't exist.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save model blockspecs.
        blockspecs_config = generated_model.blockspecs.to_config()  # type: ignore
        cached_blockspecs_path = self.cache_dir / self.compose_cache_filename(ext="yml")
        cached_blockspecs_path.write_text(blockspecs_config.serialize(), "utf8")

        # Save model weights if pretrained_model was used.
        if self.config.pretrained_model:
            cached_weights_path = self.cache_dir / self.compose_cache_filename(ext="pt")
            torch.save(generated_model.state_dict(), cached_weights_path)

    def compute_comparisons(
        self, baseline_arch_style: ArchStyle, baseline_name: str
    ) -> None:
        """Compute comparisons between the generated model and a baseline model.

        Args:
            baseline_arch_style: Baseline architecture style.
            baseline_name: Name of the baseline model.
        """
        baseline_model = self.make_baseline_model(baseline_arch_style)
        self._loggable_dict[baseline_name] = baseline_model.loggable_dict

        # Compute flop ratio comparison.
        self._loggable_dict[
            f"flop_ratio_vs_{baseline_name}"
        ] = ModelProfiler.compute_ratio(
            baseline=baseline_model.profiler,
            target=self._generated_model.profiler,
            metric="flops",
        )

        # Compute params ratio comparison.
        self._loggable_dict[
            f"params_ratio_vs_{baseline_name}"
        ] = ModelProfiler.compute_ratio(
            baseline=baseline_model.profiler,
            target=self._generated_model.profiler,
            metric="params",
        )

    @property
    def loggable_dict(self) -> Dict[str, Any]:
        """Return a dict with important information that might need to be logged.

        Returns:
            A dictionary with loggable information.
        """
        return self._loggable_dict

    @property
    def generated_model(self) -> TCompressibleModel:
        """Provide access to the generated model.

        Returns:
            The generated model.
        """
        return self._generated_model
