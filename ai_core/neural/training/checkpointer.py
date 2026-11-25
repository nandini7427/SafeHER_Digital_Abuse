"""Checkpoint persistence and serialization utilities for SafeHER neural detectors.

Supports safe serialization via SafeTensors and torch.save (weights_only=True compatible),
cryptographic SHA-256 hash generation, and metadata embedding.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.models.factory import compute_file_sha256


def save_detector_checkpoint(
    model: BaseNeuralDetector,
    output_path: Union[str, Path],
    extra_metadata: Optional[Dict[str, Any]] = None,
    use_safetensors: bool = False,
) -> Tuple[Path, str]:
    """Serialize detector weights with embedded metadata and compute SHA-256 digest.

    Args:
        model: BaseNeuralDetector instance whose weights will be serialized.
        output_path: Target file path (.pt, .pth, or .safetensors).
        extra_metadata: Optional supplemental metadata to embed in checkpoint.
        use_safetensors: Whether to serialize using safetensors format.

    Returns:
        Tuple of (resolved_path, sha256_hash).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta_dict = model.get_metadata().model_dump()
    if extra_metadata:
        meta_dict.update(extra_metadata)

    state_dict = model.state_dict()

    if use_safetensors or path.suffix.lower() == ".safetensors":
        try:
            import safetensors.torch
            # SafeTensors metadata must be string values
            str_meta = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta_dict.items()}
            safetensors.torch.save_file(state_dict, str(path), metadata=str_meta)
        except ImportError:
            raise RuntimeError("safetensors package required to export .safetensors files.")
    else:
        # Standard PyTorch checkpoint dictionary
        payload = {
            "state_dict": state_dict,
            "metadata": meta_dict,
            "architecture": model.get_metadata().architecture,
            "model_version": model.get_metadata().model_version,
            "is_trained": model.is_trained,
        }
        torch.save(payload, str(path))

    sha256_hash = compute_file_sha256(path)
    model.weights_sha256 = sha256_hash

    return path, sha256_hash
