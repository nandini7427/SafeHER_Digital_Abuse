"""Reproducible and resource-bounded dataset loader for SafeHER image pipelines.

Implements PyTorch-compatible Dataset and DataLoader layers:
1. Dataset-manifest-driven loading (reads DatasetManifest or record sequences).
2. Streaming / on-demand image loading without buffering large datasets into memory.
3. Rights eligibility validation: blocks unapproved records from commercial training manifests.
4. Split validation: guarantees strict split separation (train vs val vs test).
5. Label validation: maps labels to binary or multiclass tensor targets.
6. Corrupted image detection: intercepts corrupted or truncated files and records exclusions.
7. Decompression bomb protection and RGB normalization.
8. CPU-only compatibility and bounded worker threading.
9. Audit exclusion tracking: Never silently skips invalid samples; logs every exclusion with a reason.
"""

from collections import defaultdict
from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.utils.data import DataLoader, Dataset

from ai_core.datasets.integrity import (
    CorruptedFileError,
    FileIntegrityError,
    InvalidFormatError,
    PathTraversalError,
    compute_file_sha256,
    resolve_safe_relative_path,
    validate_image_file,
)
from ai_core.datasets.manifest import (
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    RightsApprovalStatus,
    TargetUsageScope,
)
from ai_core.datasets.transforms import (
    ForensicTransformPipeline,
    TransformOperationRecord,
)

logger = logging.getLogger(__name__)


# Standard binary label mapping: 0 = Authentic, 1 = Manipulated
BINARY_LABEL_MAP: Dict[DatasetImageLabel, int] = {
    DatasetImageLabel.CAMERA_PHOTOGRAPH: 0,
    DatasetImageLabel.AI_GENERATED_IMAGE: 1,
    DatasetImageLabel.FACE_SWAP: 1,
    DatasetImageLabel.FACIAL_REENACTMENT: 1,
    DatasetImageLabel.FACIAL_INPAINTING: 1,
    DatasetImageLabel.LOCAL_IMAGE_EDIT: 1,
    DatasetImageLabel.IMAGE_COMPOSITE: 1,
}

# Fine-grained multiclass mapping
MULTICLASS_LABEL_MAP: Dict[DatasetImageLabel, int] = {
    DatasetImageLabel.CAMERA_PHOTOGRAPH: 0,
    DatasetImageLabel.AI_GENERATED_IMAGE: 1,
    DatasetImageLabel.FACE_SWAP: 2,
    DatasetImageLabel.FACIAL_REENACTMENT: 3,
    DatasetImageLabel.FACIAL_INPAINTING: 4,
    DatasetImageLabel.LOCAL_IMAGE_EDIT: 5,
    DatasetImageLabel.IMAGE_COMPOSITE: 6,
}


class SampleExclusionRecord(BaseModel):
    """Audit record capturing an excluded sample during dataset ingestion or runtime loading."""

    record_id: str = Field(..., description="Unique record identifier")
    image_ref: str = Field(..., description="Referenced image path or URI")
    exclusion_reason: str = Field(..., description="Human-readable reason for sample exclusion")
    error_type: str = Field(..., description="Category (e.g. 'RIGHTS_INELIGIBLE', 'CORRUPTED_FILE')")
    reason: Optional[str] = Field(None, description="Categorical reason code")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic parameters")

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        if self.reason is None:
            object.__setattr__(self, "reason", self.error_type)


class SafeHERDataset(Dataset):
    """Manifest-driven PyTorch Dataset with streaming loading, rights filtering, and exclusion tracking."""

    def __init__(
        self,
        records: Union[DatasetManifest, List[DatasetImageRecord], str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        split: Optional[DatasetSplit] = None,
        target_workload: TargetUsageScope = TargetUsageScope.PRODUCTION_TRAINING,
        transform_pipeline: Optional[ForensicTransformPipeline] = None,
        label_mode: str = "binary",
        strict_mode: bool = False,
        return_none_on_corrupt: bool = True,
        verify_sha256_on_load: bool = False,
        in_memory_images: Optional[Dict[str, Union[bytes, Image.Image]]] = None,
        max_image_dimension: int = 4096,
        max_image_pixels: Optional[int] = None,
    ):
        """Initialize SafeHERDataset.

        Args:
            records: DatasetManifest instance, record list, or filesystem path to manifest.json.
            root_dir: Filesystem root directory for resolving safe relative image paths.
            split: Target split filter (TRAIN, VAL, INTERNAL_TEST, etc.). If None, all eligible records are loaded.
            target_workload: Target workload scope (PRODUCTION_TRAINING, EVALUATION, RESEARCH).
            transform_pipeline: Configured ForensicTransformPipeline instance.
            label_mode: 'binary' (0=authentic, 1=manipulated) or 'multiclass'.
            strict_mode: If True, raises exceptions on corrupt images; if False, logs exclusions.
            return_none_on_corrupt: If True, returns None on loading failures to be filtered by safe_collate_fn.
            verify_sha256_on_load: If True, validates SHA-256 cryptographic digest upon loading from disk.
            in_memory_images: Optional dictionary mapping record_id to bytes/PIL for testing or cache.
            max_image_dimension: Pixel dimension limit safety guard against decompression bombs.
            max_image_pixels: Optional total pixel count limit (defaults to max_image_dimension^2).
        """
        self.root_dir = Path(root_dir) if root_dir else None
        self.split = split
        self.target_workload = target_workload
        self.label_mode = label_mode.lower()
        self.strict_mode = strict_mode
        self.return_none_on_corrupt = return_none_on_corrupt
        self.verify_sha256_on_load = verify_sha256_on_load
        self.in_memory_images = in_memory_images or {}
        self.max_image_dimension = max_image_dimension
        self.max_image_pixels = (
            max_image_pixels if max_image_pixels is not None else max_image_dimension * max_image_dimension
        )

        # Decompression bomb guard
        Image.MAX_IMAGE_PIXELS = self.max_image_pixels

        # Set default transform pipeline if none provided
        mode_str = "train" if split == DatasetSplit.TRAIN else "eval"
        self.transform_pipeline = transform_pipeline or ForensicTransformPipeline(mode=mode_str)

        # Exclusion tracking log
        self.exclusion_log: List[SampleExclusionRecord] = []

        # Load raw records
        raw_records = self._resolve_raw_records(records)

        # Filter by split and rights eligibility
        self.records: List[DatasetImageRecord] = self._filter_records(raw_records)

    def _resolve_raw_records(
        self,
        records_input: Union[DatasetManifest, List[DatasetImageRecord], str, Path],
    ) -> List[DatasetImageRecord]:
        """Resolve records from manifest, list, or JSON path."""
        if isinstance(records_input, DatasetManifest):
            return records_input.records
        if isinstance(records_input, list):
            return records_input
        if isinstance(records_input, (str, Path)):
            manifest = DatasetManifest.load(records_input)
            return manifest.records
        raise ValueError(f"Unsupported records input type: {type(records_input)}")

    def _filter_records(
        self,
        raw_records: List[DatasetImageRecord],
    ) -> List[DatasetImageRecord]:
        """Apply split validation, rights eligibility validation, and label validation."""
        filtered: List[DatasetImageRecord] = []
        is_commercial = self.target_workload in (
            TargetUsageScope.COMMERCIAL_TRAINING,
            TargetUsageScope.PRODUCTION_TRAINING,
        )

        for rec in raw_records:
            # 1. Split validation
            if self.split is not None and rec.split != self.split:
                self.exclusion_log.append(
                    SampleExclusionRecord(
                        record_id=rec.record_id,
                        image_ref=rec.image_path_or_ref,
                        exclusion_reason=(
                            f"Record split '{rec.split.value}' does not match requested split '{self.split.value}'."
                        ),
                        error_type="SPLIT_MISMATCH",
                        details={"record_split": rec.split.value, "requested_split": self.split.value},
                    )
                )
                continue

            # 2. Rights eligibility validation
            if is_commercial:
                if rec.rights_status not in (
                    RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
                    RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
                ):
                    self.exclusion_log.append(
                        SampleExclusionRecord(
                            record_id=rec.record_id,
                            image_ref=rec.image_path_or_ref,
                            exclusion_reason=(
                                f"Record rights status '{rec.rights_status.value}' is ineligible for "
                                f"commercial workload '{self.target_workload.value}'. Requires active commercial clearance."
                            ),
                            error_type="EXCLUDED_RIGHTS_RESTRICTION",
                            reason="EXCLUDED_RIGHTS_RESTRICTION",
                            details={"rights_status": rec.rights_status.value},
                        )
                    )
                    continue

            # 3. Label validation
            if self.label_mode == "binary":
                if rec.label not in BINARY_LABEL_MAP:
                    self.exclusion_log.append(
                        SampleExclusionRecord(
                            record_id=rec.record_id,
                            image_ref=rec.image_path_or_ref,
                            exclusion_reason=f"Label '{rec.label.value}' cannot be mapped to binary target.",
                            error_type="INVALID_LABEL",
                            reason="INVALID_LABEL",
                            details={"label": rec.label.value},
                        )
                    )
                    continue
            else:
                if rec.label not in MULTICLASS_LABEL_MAP:
                    self.exclusion_log.append(
                        SampleExclusionRecord(
                            record_id=rec.record_id,
                            image_ref=rec.image_path_or_ref,
                            exclusion_reason=f"Label '{rec.label.value}' cannot be mapped to multiclass target.",
                            error_type="INVALID_LABEL",
                            reason="INVALID_LABEL",
                            details={"label": rec.label.value},
                        )
                    )
                    continue

            filtered.append(rec)

        return filtered

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, record: DatasetImageRecord) -> Image.Image:
        """Load image on demand from in-memory cache or safe relative filesystem path."""
        # 1. In-memory override
        if record.record_id in self.in_memory_images:
            payload = self.in_memory_images[record.record_id]
            if isinstance(payload, Image.Image):
                img = payload
            else:
                img = Image.open(io.BytesIO(payload))
            if (img.width * img.height) > self.max_image_pixels:
                raise Image.DecompressionBombError(
                    f"In-memory image pixel count {img.width * img.height} exceeds limit {self.max_image_pixels}."
                )
            img.load()
            return img.convert("RGB")

        # 2. Filesystem loading
        if self.root_dir is None:
            raise FileNotFoundError(
                f"Cannot load on-disk asset for record '{record.record_id}': root_dir is not specified."
            )

        resolved_path = resolve_safe_relative_path(self.root_dir, record.image_path_or_ref)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Image asset does not exist on disk: {resolved_path}")

        # Optional SHA-256 verification
        if self.verify_sha256_on_load:
            from ai_core.datasets.integrity import verify_file_sha256

            is_valid, computed_hash = verify_file_sha256(resolved_path, record.sha256)
            if not is_valid:
                raise FileIntegrityError(
                    f"SHA-256 digest mismatch for record '{record.record_id}': expected {record.sha256}, got {computed_hash}."
                )

        # Open and load image safely
        try:
            with Image.open(resolved_path) as raw_img:
                raw_img.verify()

            # Re-open for actual raster decoding
            with Image.open(resolved_path) as img:
                if (img.width * img.height) > self.max_image_pixels:
                    raise Image.DecompressionBombError(
                        f"Image pixel count {img.width * img.height} exceeds limit {self.max_image_pixels}."
                    )
                img.load()
                return img.convert("RGB")
        except (Image.DecompressionBombError, FileIntegrityError):
            raise
        except Exception as exc:
            raise CorruptedFileError(f"Corrupted image stream in {resolved_path}: {exc}") from exc

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Load single sample on demand, execute transforms, and return tensor with audit metadata."""
        record = self.records[idx]

        try:
            pil_image = self._load_image(record)
        except Exception as exc:
            if isinstance(exc, Image.DecompressionBombError):
                err_type = "DECOMPRESSION_BOMB"
            elif isinstance(exc, FileIntegrityError):
                err_type = "INTEGRITY_MISMATCH"
            elif isinstance(exc, CorruptedFileError):
                err_type = "FILE_CORRUPTED"
            else:
                err_type = "CORRUPTED_OR_MISSING_IMAGE"

            self.exclusion_log.append(
                SampleExclusionRecord(
                    record_id=record.record_id,
                    image_ref=record.image_path_or_ref,
                    exclusion_reason=f"Runtime loading failure: {str(exc)}",
                    error_type=err_type,
                    reason=err_type,
                    details={"error": str(exc)},
                )
            )
            if self.strict_mode:
                raise
            if self.return_none_on_corrupt:
                return None
            # Return placeholder in tolerant mode
            pil_image = Image.new("RGB", self.transform_pipeline.target_resolution, color="black")

        # Label resolution
        if self.label_mode == "binary":
            target_label = BINARY_LABEL_MAP[record.label]
        else:
            target_label = MULTICLASS_LABEL_MAP[record.label]

        # Apply transforms
        tensor, operations = self.transform_pipeline(
            pil_image,
            metadata=record.metadata,
        )

        return {
            "image": tensor,
            "label": torch.tensor(target_label, dtype=torch.long),
            "record_id": record.record_id,
            "sha256": record.sha256,
            "metadata": record.metadata,
            "applied_transforms": [op.model_dump() for op in operations],
        }


def safe_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Custom PyTorch collate function preserving metadata, filtering corrupted None samples, and handling CPU batch tensor stacking."""
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("Entire batch consisted of corrupted or unreadable samples.")

    images = torch.stack([item["image"] for item in valid_batch], dim=0)
    labels = torch.stack([item["label"] for item in valid_batch], dim=0)
    record_ids = [item["record_id"] for item in valid_batch]
    sha256s = [item["sha256"] for item in valid_batch]
    metadata = [item["metadata"] for item in valid_batch]
    transforms = [item["applied_transforms"] for item in valid_batch]

    return {
        "images": images,
        "labels": labels,
        "record_ids": record_ids,
        "sha256s": sha256s,
        "metadata": metadata,
        "applied_transforms": transforms,
    }
    transforms = [item["applied_transforms"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "record_ids": record_ids,
        "sha256s": sha256s,
        "metadata": metadata,
        "applied_transforms": transforms,
    }


def create_safe_dataloader(
    dataset: SafeHERDataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Create a CPU-compatible, memory-bounded PyTorch DataLoader for SafeHER datasets.

    Args:
        dataset: Configured SafeHERDataset instance.
        batch_size: Batch size (default: 16).
        shuffle: Whether to shuffle records per epoch.
        num_workers: Subprocess worker count (default: 0 for synchronous thread safety; capped at 4).
        drop_last: Whether to drop incomplete final batch.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    safe_workers = min(max(0, num_workers), 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=safe_workers,
        drop_last=drop_last,
        collate_fn=safe_collate_fn,
        pin_memory=False,  # False for CPU-only execution
    )
