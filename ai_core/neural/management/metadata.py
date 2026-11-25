"""Secure model metadata schemas, serialization, and approved usage contracts.

Stores:
- Model identifier.
- Version.
- Architecture.
- Input dimensions.
- Preprocessing version.
- Class mappings.
- Training manifest hash.
- Evaluation report reference.
- Calibration artifact reference.
- SHA-256 digest.
- Expected file size in bytes.
- Approved usage state (COMMERCIAL_TRAINED, RESEARCH_RESTRICTED, PENDING_REVIEW, REJECTED).
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelUsageState(str, Enum):
    """Categorical compliance status governing model deployment tracks."""

    COMMERCIAL_TRAINED = "COMMERCIAL_TRAINED"
    RESEARCH_RESTRICTED = "RESEARCH_RESTRICTED"
    PENDING_REVIEW = "PENDING_REVIEW"
    REJECTED = "REJECTED"


class ModelArtifactFormat(str, Enum):
    """Permitted secure model serialization formats."""

    SAFETENSORS = "safetensors"
    PYTORCH_STATE_DICT = "pytorch_state_dict"


class SecureModelMetadata(BaseModel):
    """Cryptographically verifiable metadata descriptor for SafeHER model weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(
        ...,
        min_length=3,
        description="Unique identifier for the neural detector model.",
    )
    version: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9\.\-_]+)?$",
        description="Semantic version string of the model artifact.",
    )
    architecture: str = Field(
        ...,
        min_length=2,
        description="Explicit neural backbone architecture identifier.",
    )
    input_dimensions: Tuple[int, ...] = Field(
        default=(3, 224, 224),
        description="Expected tensor input dimensions (Channels, Height, Width).",
    )
    preprocessing_version: str = Field(
        default="v1-imagenet-norm",
        description="Identifier of deterministic preprocessing pipeline configuration.",
    )
    class_mappings: Dict[int, str] = Field(
        default_factory=lambda: {0: "CAMERA_PHOTOGRAPH", 1: "AI_GENERATED_IMAGE"},
        description="Explicit mapping from logit index to categorical class label.",
    )
    training_manifest_hash: str = Field(
        ...,
        pattern=r"^[a-fA-F0-9]{64}$",
        description="SHA-256 digest of the approved training dataset manifest.",
    )
    evaluation_report_reference: str = Field(
        ...,
        min_length=3,
        description="Identifier or relative path to the independent evaluation report.",
    )
    calibration_artifact_reference: Optional[str] = Field(
        default=None,
        description="Identifier or relative path to the bound calibration artifact.",
    )
    sha256: str = Field(
        ...,
        pattern=r"^[a-fA-F0-9]{64}$",
        description="Cryptographic SHA-256 digest of the raw weights file.",
    )
    expected_file_size_bytes: int = Field(
        ...,
        gt=0,
        description="Exact expected file size in bytes.",
    )
    approved_usage_state: ModelUsageState = Field(
        default=ModelUsageState.PENDING_REVIEW,
        description="Compliance usage authorization state.",
    )
    format: ModelArtifactFormat = Field(
        default=ModelArtifactFormat.SAFETENSORS,
        description="Serialization format of the artifact file.",
    )
    creation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC creation timestamp.",
    )
    metadata_hash: Optional[str] = Field(
        default=None,
        description="Self-referential digest of metadata fields excluding metadata_hash.",
    )

    @field_validator("sha256", "training_manifest_hash")
    @classmethod
    def normalize_hex_hash(cls, v: str) -> str:
        return v.lower().strip()

    def compute_metadata_digest(self) -> str:
        """Compute SHA-256 digest over the canonical JSON representation of metadata fields."""
        data = self.model_dump(exclude={"metadata_hash"})
        canonical_json = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def with_computed_hash(self) -> "SecureModelMetadata":
        """Return a copy of metadata with computed metadata_hash."""
        digest = self.compute_metadata_digest()
        dumped = self.model_dump(exclude={"metadata_hash"})
        dumped["metadata_hash"] = digest
        return SecureModelMetadata(**dumped)

    def verify_integrity(self) -> bool:
        """Verify that metadata_hash matches current field values if present."""
        if self.metadata_hash is None:
            return True
        return self.metadata_hash.lower() == self.compute_metadata_digest().lower()

    def save(self, path: Union[str, Path]) -> None:
        """Save metadata descriptor to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SecureModelMetadata":
        """Load and validate metadata descriptor from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Metadata file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = cls(**data)
        if not inst.verify_integrity():
            raise ValueError(
                f"Metadata file at '{p}' has been tampered with! "
                f"Computed hash {inst.compute_metadata_digest()} != recorded hash {inst.metadata_hash}."
            )
        return inst
