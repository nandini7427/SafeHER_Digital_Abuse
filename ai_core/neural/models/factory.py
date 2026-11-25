"""Model factory and checkpoint loading registry for SafeHER neural detectors.

Provides configurable instantiation for supported backbones, strict weights verification,
safe deserialization (SafeTensors / torch.load with weights_only=True), and rejection
of untrained, missing, or corrupted model artifacts.
"""

import hashlib
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch

from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.models.efficientnet import EfficientNetB0Detector
from ai_core.neural.models.mobilenet_v3 import MobileNetV3LargeDetector
from ai_core.neural.schemas import (
    MODEL_A_CLASS_MAPPING,
    MODEL_A_SUPPORTED_DOMAIN,
    InputConfiguration,
    NeuralArchitecture,
)

# Supported architecture registry
_ARCHITECTURE_REGISTRY: Dict[str, Type[BaseNeuralDetector]] = {
    NeuralArchitecture.MOBILENET_V3_LARGE.value: MobileNetV3LargeDetector,
    NeuralArchitecture.EFFICIENTNET_B0.value: EfficientNetB0Detector,
}


def list_supported_architectures() -> List[str]:
    """Return list of supported neural detector backbone identifiers."""
    return sorted(list(_ARCHITECTURE_REGISTRY.keys()))


def create_image_detector(
    architecture: Union[str, NeuralArchitecture] = NeuralArchitecture.MOBILENET_V3_LARGE,
    num_classes: int = 2,
    dropout_rate: float = 0.2,
    hidden_dim: int = 256,
    model_version: str = "1.0.0-m4.3",
    supported_domain: str = MODEL_A_SUPPORTED_DOMAIN,
    class_mapping: Optional[Dict[int, str]] = None,
    input_config: Optional[InputConfiguration] = None,
    is_trained: bool = False,
    weights_sha256: Optional[str] = None,
    training_dataset_id: Optional[str] = None,
) -> BaseNeuralDetector:
    """Instantiate a neural image detector by architecture identifier.

    Args:
        architecture: Backbone identifier ('mobilenet_v3_large' or 'efficientnet_b0').
        num_classes: Number of output classes (default 2: CAMERA vs AI_GENERATED).
        dropout_rate: Dropout probability in the forensic head.
        hidden_dim: Intermediate feature projection dimension in classification head.
        model_version: Semantic version of the model.
        supported_domain: Human-readable operational domain specification.
        class_mapping: Dict mapping logit indices to class names.
        input_config: Expected input tensor specification.
        is_trained: Flag indicating verified training status.
        weights_sha256: Expected SHA-256 hash of trained weights.
        training_dataset_id: Identifier of training dataset manifest.

    Returns:
        Instantiated BaseNeuralDetector subclass instance.

    Raises:
        ValueError: If requested architecture is not in the supported registry.
    """
    arch_key = architecture.value if isinstance(architecture, NeuralArchitecture) else str(architecture).lower()

    if arch_key not in _ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unsupported neural architecture '{arch_key}'. "
            f"Supported architectures: {list_supported_architectures()}"
        )

    model_class = _ARCHITECTURE_REGISTRY[arch_key]
    return model_class(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        model_version=model_version,
        supported_domain=supported_domain,
        class_mapping=class_mapping or MODEL_A_CLASS_MAPPING,
        input_config=input_config or InputConfiguration(),
        is_trained=is_trained,
        weights_sha256=weights_sha256,
        training_dataset_id=training_dataset_id,
    )


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 digest of a model weights file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _decode_safetensors_metadata(path: Path) -> Dict[str, Any]:
    """Read optional SafeTensors metadata and decode JSON-encoded values."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors package is required to inspect .safetensors metadata.") from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}

    decoded: Dict[str, Any] = {}
    for key, value in raw.items():
        try:
            decoded[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            decoded[key] = value
    return decoded


def _validate_training_provenance(metadata: Dict[str, Any], architecture: str) -> None:
    """Reject artifacts without explicit, completed training provenance."""
    required_hash = re.compile(r"^[a-fA-F0-9]{64}$")
    if metadata.get("is_trained") is not True:
        raise ValueError("Checkpoint is explicitly flagged as UNTRAINED or lacks is_trained=true.")
    if metadata.get("training_completed") is not True:
        raise ValueError("UNTRAINED: checkpoint must explicitly declare training_completed=true.")
    if metadata.get("architecture") != architecture:
        raise ValueError("Checkpoint architecture metadata does not match the requested model architecture.")
    if not isinstance(metadata.get("epoch"), int) or metadata["epoch"] < 1:
        raise ValueError("UNTRAINED: checkpoint must contain a positive completed training epoch.")
    if not isinstance(metadata.get("training_manifest_hash"), str) or not required_hash.fullmatch(metadata["training_manifest_hash"]):
        raise ValueError("UNTRAINED: checkpoint is missing a valid training manifest SHA-256.")
    if not isinstance(metadata.get("split_manifest_hash"), str) or not required_hash.fullmatch(metadata["split_manifest_hash"]):
        raise ValueError("UNTRAINED: checkpoint is missing a valid split manifest SHA-256.")
    versions = metadata.get("dataset_versions")
    if not isinstance(versions, list) or not versions or not all(isinstance(v, str) and v.strip() for v in versions):
        raise ValueError("UNTRAINED: checkpoint is missing dataset version provenance.")
    if metadata.get("approved_usage_state") != "COMMERCIAL_TRAINED":
        raise ValueError(
            "ARTIFACT_NOT_APPROVED: checkpoint must explicitly declare "
            "approved_usage_state=COMMERCIAL_TRAINED; research-only, pending-review, "
            "and missing approval states are rejected."
        )


def load_detector_checkpoint(
    model: BaseNeuralDetector,
    checkpoint_path: Union[str, Path],
    expected_sha256: Optional[str] = None,
    enforce_trained: bool = True,
) -> BaseNeuralDetector:
    """Load a detector using safe deserialization and explicit training provenance.

    Production callers should keep ``enforce_trained=True``. When it is enabled,
    raw state dictionaries and artifacts without complete training provenance are
    rejected instead of being implicitly marked as trained.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    actual_sha256 = compute_file_sha256(path)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch! Expected {expected_sha256}, got {actual_sha256}"
        )

    checkpoint_metadata: Dict[str, Any] = {}
    if path.suffix.lower() == ".safetensors":
        try:
            import safetensors.torch
            checkpoint_metadata = _decode_safetensors_metadata(path)
            state_dict = safetensors.torch.load_file(str(path), device="cpu")
        except ImportError as exc:
            raise RuntimeError("safetensors package is required to load .safetensors checkpoints.") from exc
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable SafeTensors checkpoint: {exc}") from exc
    elif path.suffix.lower() in {".pt", ".pth"}:
        try:
            loaded = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Unsafe or unreadable PyTorch checkpoint: {exc}") from exc

        if not isinstance(loaded, dict):
            raise ValueError(f"Unexpected checkpoint payload format: {type(loaded)}")
        if isinstance(loaded.get("model_state_dict"), dict):
            state_dict = loaded["model_state_dict"]
            checkpoint_metadata = loaded.get("metadata", {})
        elif isinstance(loaded.get("state_dict"), dict):
            state_dict = loaded["state_dict"]
            checkpoint_metadata = loaded.get("metadata", {})
        else:
            state_dict = loaded
            checkpoint_metadata = {}
    else:
        raise ValueError("Unsupported model artifact format. Use .safetensors, .pt, or .pth.")

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint does not contain a valid non-empty model state dictionary.")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state_dict.items()):
        raise ValueError("Checkpoint state dictionary contains unsupported entries.")

    if enforce_trained:
        if not isinstance(checkpoint_metadata, dict):
            raise ValueError("UNTRAINED: checkpoint training metadata is missing or invalid.")
        _validate_training_provenance(checkpoint_metadata, model.get_metadata().architecture)

    model.load_state_dict(state_dict, strict=True)
    model.is_trained = checkpoint_metadata.get("is_trained") is True
    model.weights_sha256 = actual_sha256
    model.eval()
    return model
