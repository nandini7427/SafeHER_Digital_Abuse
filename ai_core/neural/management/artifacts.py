"""Secure model artifact validation, checksum verification, and safe deserialization.

Guarantees:
1. Safetensors loading as first-class safe format.
2. Safe PyTorch loading strictly using `weights_only=True`. Never unrestrictedly loads
   arbitrary pickle, joblib, or unvalidated Python code.
3. Streaming SHA-256 calculation and strict verification against metadata.
4. Exact file-size and format validation to detect truncated or corrupted files.
5. Configurable path whitelisting and protection against directory traversal attacks.
6. Architecture and tensor shape validation before weights injection into models.
"""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from ai_core.neural.management.metadata import (
    ModelArtifactFormat,
    ModelUsageState,
    SecureModelMetadata,
)
from ai_core.neural.models.base import BaseNeuralDetector


class ArtifactPathTraversalError(PermissionError):
    """Raised when an artifact path attempts directory traversal or escapes allowed directories."""
    pass


class IntegrityVerificationError(ValueError):
    """Raised when an artifact fails cryptographic SHA-256 or file-size verification."""
    pass


class CorruptArtifactError(ValueError):
    """Raised when an artifact bitstream is corrupted, malformed, or unreadable."""
    pass


class ArchitectureMismatchError(ValueError):
    """Raised when artifact architecture does not match target model backbone."""
    pass


class UnsafeFormatError(ValueError):
    """Raised when an untrusted or non-whitelisted model file format is supplied."""
    pass


def compute_streaming_sha256(
    file_path: Union[str, Path],
    chunk_size: int = 65536,
) -> Tuple[str, int]:
    """Compute streaming SHA-256 digest and exact byte size of a file.

    Args:
        file_path: Absolute or relative path to target file.
        chunk_size: Byte buffer chunk size for streaming.

    Returns:
        Tuple of (sha256_hex_digest, total_bytes_read).
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    hasher = hashlib.sha256()
    total_bytes = 0

    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
            total_bytes += len(chunk)

    return hasher.hexdigest().lower(), total_bytes


@dataclass
class ArtifactSecurityPolicy:
    """Security policy defining allowed artifact directories, formats, and validation rules."""

    allowed_directories: List[Path] = field(default_factory=list)
    allow_pytorch_weights: bool = True
    enforce_safetensors: bool = False
    enforce_exact_file_size: bool = True
    max_file_size_bytes: int = 500 * 1024 * 1024  # 500 MB max guard

    def __post_init__(self) -> None:
        """Resolve and canonicalize allowed directory paths."""
        resolved: List[Path] = []
        for d in self.allowed_directories:
            p = Path(d).resolve()
            resolved.append(p)
        self.allowed_directories = resolved

    def add_allowed_directory(self, directory: Union[str, Path]) -> None:
        """Add a trusted directory to the whitelist."""
        p = Path(directory).resolve()
        if p not in self.allowed_directories:
            self.allowed_directories.append(p)

    def validate_safe_path(self, path: Union[str, Path]) -> Path:
        """Validate that path exists within an allowed directory and has no traversal sequences.

        Raises:
            ArtifactPathTraversalError: If path escapes allowed directories or contains illegal traversal.
        """
        candidate = Path(path).resolve()

        # Check directory whitelist if configured
        if self.allowed_directories:
            is_permitted = any(
                candidate == allowed or allowed in candidate.parents
                for allowed in self.allowed_directories
            )
            if not is_permitted:
                raise ArtifactPathTraversalError(
                    f"Access denied: Artifact path '{candidate}' is outside permitted directories: "
                    f"{[str(d) for d in self.allowed_directories]}"
                )

        return candidate


class SecureArtifactManager:
    """Secure neural model artifact validation and deserialization manager."""

    def __init__(self, policy: Optional[ArtifactSecurityPolicy] = None) -> None:
        self.policy = policy or ArtifactSecurityPolicy()

    def verify_artifact(
        self,
        artifact_path: Union[str, Path],
        metadata: SecureModelMetadata,
    ) -> Path:
        """Verify artifact against cryptographic checksum, file size, and security policy.

        Returns:
            Validated Path object.

        Raises:
            FileNotFoundError: If artifact is missing.
            ArtifactPathTraversalError: If path violates security policy.
            IntegrityVerificationError: If SHA-256 or file size fails verification.
        """
        safe_path = self.policy.validate_safe_path(artifact_path)

        if not safe_path.is_file():
            raise FileNotFoundError(f"Model artifact file does not exist: {safe_path}")

        # Compute actual checksum and size
        actual_sha256, actual_size = compute_streaming_sha256(safe_path)

        # Check maximum allowed size
        if actual_size > self.policy.max_file_size_bytes:
            raise IntegrityVerificationError(
                f"Artifact size ({actual_size} bytes) exceeds maximum security limit "
                f"({self.policy.max_file_size_bytes} bytes)."
            )

        # Verify file size
        if self.policy.enforce_exact_file_size:
            if actual_size != metadata.expected_file_size_bytes:
                raise IntegrityVerificationError(
                    f"Artifact file size mismatch for '{safe_path.name}'! "
                    f"Expected {metadata.expected_file_size_bytes} bytes, got {actual_size} bytes."
                )

        # Verify SHA-256
        if actual_sha256 != metadata.sha256.lower():
            raise IntegrityVerificationError(
                f"Cryptographic SHA-256 verification FAILED for '{safe_path.name}'! "
                f"Expected: {metadata.sha256.lower()}, Actual: {actual_sha256}"
            )

        return safe_path

    def load_weights_safe(
        self,
        safe_path: Path,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Safely deserialize weights without executing arbitrary code.

        Supports:
        - .safetensors (via safetensors.torch.load_file)
        - .pt / .pth (strictly with torch.load(..., weights_only=True))

        Raises:
            UnsafeFormatError: If format is untrusted (.pkl, .joblib, etc.).
            CorruptArtifactError: If file bitstream is malformed or corrupted.
        """
        suffix = safe_path.suffix.lower()

        if suffix == ".safetensors":
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(str(safe_path), device=device)
                return state_dict
            except Exception as ex:
                raise CorruptArtifactError(f"Corrupted or invalid safetensors file '{safe_path}': {ex}")

        elif suffix in (".pt", ".pth", ".bin"):
            if self.policy.enforce_safetensors:
                raise UnsafeFormatError(
                    f"Policy enforces SafeTensors format. PyTorch format '{suffix}' is rejected."
                )
            if not self.policy.allow_pytorch_weights:
                raise UnsafeFormatError("PyTorch weights loading is disabled by security policy.")

            try:
                # Strictly enforce weights_only=True
                loaded = torch.load(str(safe_path), map_location=device, weights_only=True)
                if isinstance(loaded, dict) and "state_dict" in loaded:
                    return loaded["state_dict"]
                elif isinstance(loaded, dict):
                    return loaded
                else:
                    raise CorruptArtifactError(
                        f"PyTorch checkpoint payload must be a state_dict, got {type(loaded)}"
                    )
            except Exception as ex:
                if isinstance(ex, CorruptArtifactError):
                    raise
                raise CorruptArtifactError(f"Corrupted or unreadable PyTorch weights '{safe_path}': {ex}")

        else:
            raise UnsafeFormatError(
                f"Prohibited or unsafe model artifact extension '{suffix}'. "
                "Only .safetensors and safe PyTorch .pt/.pth (weights_only=True) are permitted. "
                "Untrusted pickle or arbitrary Python objects are strictly rejected."
            )

    def load_and_verify_model(
        self,
        artifact_path: Union[str, Path],
        metadata: SecureModelMetadata,
        model: BaseNeuralDetector,
        device: str = "cpu",
        strict: bool = True,
    ) -> BaseNeuralDetector:
        """Complete verification, safe loading, and parameter injection pipeline.

        Args:
            artifact_path: Path to serialized weights.
            metadata: Cryptographically verified metadata descriptor.
            model: Target BaseNeuralDetector instance.
            device: Target execution device ('cpu', 'cuda', etc.).
            strict: Enforce exact state_dict key matching.

        Returns:
            The loaded BaseNeuralDetector instance in eval mode.

        Raises:
            ArchitectureMismatchError: If architecture in metadata doesn't match model.
            IntegrityVerificationError: If SHA-256 or file size fails.
            CorruptArtifactError: If weights bitstream is corrupted.
        """
        # 1. Architecture check
        model_arch = model.get_metadata().architecture
        if metadata.architecture != model_arch:
            raise ArchitectureMismatchError(
                f"Model architecture mismatch! Metadata declares '{metadata.architecture}', "
                f"but target detector is '{model_arch}'."
            )

        # 2. Enforce the approved operational usage state. Research-only and
        # pending-review artifacts must never enter survivor-facing inference.
        if metadata.approved_usage_state == ModelUsageState.REJECTED:
            raise ValueError(
                f"Model artifact '{metadata.model_id}' is explicitly marked as REJECTED "
                "and cannot be loaded for inference."
            )
        if metadata.approved_usage_state != ModelUsageState.COMMERCIAL_TRAINED:
            raise ValueError(
                f"Model artifact '{metadata.model_id}' has usage state "
                f"'{metadata.approved_usage_state.value}'. Operational inference requires "
                "COMMERCIAL_TRAINED approval."
            )

        # 3. Path and checksum verification
        safe_path = self.verify_artifact(artifact_path, metadata)

        # 4. Ensure the declared serialization format matches the file.
        suffix = safe_path.suffix.lower()
        expected_format = (
            ModelArtifactFormat.SAFETENSORS
            if suffix == ".safetensors"
            else ModelArtifactFormat.PYTORCH_STATE_DICT
            if suffix in {".pt", ".pth", ".bin"}
            else None
        )
        if expected_format is None or metadata.format != expected_format:
            raise UnsafeFormatError(
                f"Artifact format mismatch: file extension '{suffix}' is not consistent "
                f"with declared format '{metadata.format.value}'."
            )

        # 5. Safe deserialization
        state_dict = self.load_weights_safe(safe_path, device=device)

        # 5. Injection into model
        try:
            model.load_state_dict(state_dict, strict=strict)
        except Exception as ex:
            raise CorruptArtifactError(f"Failed to load state_dict into {model_arch}: {ex}")

        model.is_trained = True
        model.weights_sha256 = metadata.sha256
        model.eval()
        return model
