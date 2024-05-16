import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from thop import profile

from gonas.darwinai.blockspecs import BlockSpec
from gonas.search.blocks import BlockSpecs


class ModelProfiler:
    """Calculates the parameters and flops for a given model."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: List[int],
        record_params: bool = True,
        record_flops: bool = True,
        attributes_to_serialize: Optional[List[str]] = None,
    ):
        # Dictionary that holds profiled metrics and serialized attributes.
        self._loggable_dict: Dict[str, Any] = {}

        # Compute the model's parameters and flops.
        if record_params:
            self.model_params = self.get_params(model)
            self._loggable_dict["params"] = self.model_params

        if record_flops:
            self.model_flops = self.get_flops(model, input_shape)
            self._loggable_dict["flops"] = self.model_flops

        # Serialize the attributes that were specified.
        if attributes_to_serialize:
            self.model_attributes = self.get_attributes(model, attributes_to_serialize)
            self._loggable_dict["attributes"] = self.model_attributes

    @property
    def loggable_dict(self) -> Dict[str, Any]:
        """Return the serialized dictionary."""
        return self._loggable_dict

    @staticmethod
    def get_params(model: nn.Module):
        """Calculate the total parameters in the model."""
        total_parameters = 0
        for param in model.parameters():
            total_parameters += np.prod(param.size())
        return total_parameters

    @staticmethod
    def get_flops(model: nn.Module, input_shape: List[int]):
        """Calculate the flops in the model."""
        dummy_inputs = torch.zeros(*input_shape)
        flops = profile(model, inputs=(dummy_inputs,), verbose=False)[0]
        return math.floor(flops)

    @staticmethod
    def get_attributes(model: nn.Module, attributes: List[str]) -> Dict[str, Any]:
        """Return a subset of the model's attributes as a dict.

        Args:
            attributes: A list of attribute names from model to return as a dict.

        Returns:
            A dictionary of the specified attributes and their values.
        """
        attributes_dict: Dict[str, Any] = {}
        for attribute_name in attributes:
            if not hasattr(model, attribute_name):
                # Check if the attribute actually exists in model object.
                raise ValueError(
                    f"Attribute {attribute_name} not found in {model.__class__.__name__}"
                )

            attribute = getattr(model, attribute_name)
            if isinstance(attribute, BlockSpecs):
                attributes_dict[attribute_name] = attribute.to_config().to_dict()
            elif isinstance(attribute, BlockSpec):
                # If the attribute is a BlockSpec, convert it to a dict.
                attributes_dict[attribute_name] = {
                    "channels": attribute.channels,
                    "depth": attribute.depth,
                    "freeze_depth": attribute.freeze_depth,
                    "freeze_channels": attribute.freeze_channel,
                }
            elif isinstance(attribute, (int, float, str, bool)):
                # If the attribute is a primitive type, just add it to the dict.
                attributes_dict[attribute_name] = attribute
            elif isinstance(attribute, list):
                # If the attribute is a list, copy it to a new list.
                attributes_dict[attribute_name] = list(attribute)
            elif isinstance(attribute, dict):
                # If the attribute is a dict, copy it to a new dict.
                attributes_dict[attribute_name] = dict(attribute)
            else:
                raise ValueError(
                    f"Unsupported attribute type {type(attribute)} for converting to dictionary."
                )

        return attributes_dict

    @staticmethod
    def compute_ratio(
        baseline: "ModelProfiler", target: "ModelProfiler", metric: str
    ) -> float:
        """Compute the ratio of the metric between baseline and target model.

        Args:
            baseline: The baseline model profiler.
            target: The target model profiler.
            metric: The metric to compute the ratio for.

        Returns:
            The ratio of the metric between baseline and target model.
        """
        if metric not in baseline.loggable_dict:
            raise ValueError(f"Metric {metric} not found in baseline model.")

        if metric not in target.loggable_dict:
            raise ValueError(f"Metric {metric} not found in target model.")

        return target.loggable_dict[metric] / baseline.loggable_dict[metric]
