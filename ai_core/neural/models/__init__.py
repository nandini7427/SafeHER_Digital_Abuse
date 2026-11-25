"""Model implementations and factory for SafeHER neural detectors."""

from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.models.efficientnet import EfficientNetB0Detector
from ai_core.neural.models.factory import (
    compute_file_sha256,
    create_image_detector,
    list_supported_architectures,
    load_detector_checkpoint,
)
from ai_core.neural.models.mobilenet_v3 import MobileNetV3LargeDetector

__all__ = [
    "BaseNeuralDetector",
    "EfficientNetB0Detector",
    "MobileNetV3LargeDetector",
    "compute_file_sha256",
    "create_image_detector",
    "list_supported_architectures",
    "load_detector_checkpoint",
]
