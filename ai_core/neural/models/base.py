"""Abstract base class and contract for SafeHER neural forensic image detectors.

Enforces strict input tensor shape, channel, dtype, and numerical validity checks,
and guarantees uniform metadata exposure across all backbones.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ai_core.neural.schemas import (
    MODEL_A_CLASS_MAPPING,
    MODEL_A_SUPPORTED_DOMAIN,
    InputConfiguration,
    ModelMetadata,
)


class BaseNeuralDetector(nn.Module, ABC):
    """Abstract base class for SafeHER neural image detectors."""

    def __init__(
        self,
        architecture: str,
        model_version: str = "1.0.0-m4.3",
        supported_domain: str = MODEL_A_SUPPORTED_DOMAIN,
        class_mapping: Optional[Dict[int, str]] = None,
        input_config: Optional[InputConfiguration] = None,
        is_trained: bool = False,
        weights_sha256: Optional[str] = None,
        training_dataset_id: Optional[str] = None,
    ):
        super().__init__()
        self._input_config = input_config or InputConfiguration()
        self._class_mapping = class_mapping or MODEL_A_CLASS_MAPPING
        self._architecture = architecture
        self._model_version = model_version
        self._supported_domain = supported_domain
        self._is_trained = is_trained
        self._weights_sha256 = weights_sha256
        self._training_dataset_id = training_dataset_id

    @property
    def input_config(self) -> InputConfiguration:
        """Expected input configuration."""
        return self._input_config

    @property
    def class_mapping(self) -> Dict[int, str]:
        """Class mapping dictionary."""
        return self._class_mapping

    @property
    def is_trained(self) -> bool:
        """Whether this detector's weights have been verified as trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value

    @property
    def weights_sha256(self) -> Optional[str]:
        """Cryptographic SHA-256 of the loaded weights."""
        return self._weights_sha256

    @weights_sha256.setter
    def weights_sha256(self, value: Optional[str]) -> None:
        self._weights_sha256 = value

    def validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate tensor type, dimensions, channels, spatial resolution, and numerical values.

        Raises:
            TypeError: If input is not a torch.Tensor or not a float dtype.
            ValueError: If dimensions, channels, resolution, or values are invalid.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}")

        if not x.is_floating_point():
            raise TypeError(
                f"Input tensor must have a floating-point dtype (e.g. torch.float32), got {x.dtype}"
            )

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor with layout (B, C, H, W), got {x.ndim}D tensor of shape {tuple(x.shape)}"
            )

        batch_size, channels, height, width = x.shape
        if batch_size < 1:
            raise ValueError(f"Batch dimension must be >= 1, got {batch_size}")

        expected_c = self._input_config.expected_channels
        if channels != expected_c:
            raise ValueError(
                f"Expected {expected_c} input channels (RGB), got {channels} in tensor of shape {tuple(x.shape)}"
            )

        expected_h, expected_w = self._input_config.spatial_resolution
        if (height, width) != (expected_h, expected_w):
            raise ValueError(
                f"Expected spatial resolution ({expected_h}, {expected_w}), "
                f"got ({height}, {width}) in tensor of shape {tuple(x.shape)}"
            )

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input tensor contains NaN or Inf values")

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate forensic feature representations before classification head."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass and return raw logits of shape (B, num_classes)."""
        pass

    def parameter_count(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        """Total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_metadata(self) -> ModelMetadata:
        """Construct and return the ModelMetadata descriptor."""
        return ModelMetadata(
            architecture=self._architecture,
            model_version=self._model_version,
            task_type="AI_GENERATED_IMAGE_DETECTION",
            supported_domain=self._supported_domain,
            class_mapping=self._class_mapping,
            input_config=self._input_config,
            is_trained=self._is_trained,
            weights_sha256=self._weights_sha256,
            training_dataset_id=self._training_dataset_id,
            parameter_count=self.parameter_count(),
        )
