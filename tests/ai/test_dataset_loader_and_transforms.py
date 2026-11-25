"""Automated test suite for SafeHER Milestone 4.2 Reproducible Dataset Loader and Transform Pipeline.

Verifies:
1. Signal-Preserving Forensic Transforms:
   - Resizing, JPEG recompression, scale-bounded crop, mild blur (sigma <= 1.0),
     color variation, controlled noise (sigma <= 3.5), face-region cropping.
   - Operation audit tracking with TransformOperationRecord.
   - Strict separation: deterministic preprocessing in eval/test modes (no stochastic augmentations).
   - Bit-for-bit reproducibility with fixed random seed.
2. Label Mapping and Validation:
   - Binary label mapping (0: authentic, 1: manipulated).
   - Multiclass label mapping (0..6).
   - Rejection/exclusion of invalid or unknown labels.
3. SafeHERDataset & Resource Limits:
   - Streaming, on-demand image loading (bounded memory usage, no full-dataset buffering).
   - Rights eligibility enforcement (commercial training rejects research-only/unapproved).
   - Split filtering (TRAIN only loads train split, TEST only loads test split).
   - Non-silent exclusion policy: exclusion_log tracks every excluded sample with reason.
   - Decompression bomb protection (MAX_IMAGE_PIXELS threshold).
   - Graceful handling of corrupted files (corrupt streams logged, valid items load).
   - SHA-256 verification on load option.
4. PyTorch DataLoader & Safe Collation:
   - Batch consistency: images (B, 3, H, W), labels (B,), record_ids and metadata preserved.
   - Worker bounding (capped at CPU cores, max 4).
   - Safe collation with corrupted None-sample filtering.
"""

import hashlib
import io
import os
from pathlib import Path
import tempfile
from typing import List, Tuple
import numpy as np
from PIL import Image
import pytest
import torch

from ai_core.datasets.loader import (
    BINARY_LABEL_MAP,
    MULTICLASS_LABEL_MAP,
    SafeHERDataset,
    SampleExclusionRecord,
    create_safe_dataloader,
    safe_collate_fn,
)
from ai_core.datasets.manifest import (
    AttributionMetadata,
    CommercialUseStatus,
    ConsentStatus,
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    DatasetTaskType,
    GeneratorInfo,
    ManipulationCategory,
    ResearchUseStatus,
    RightsApprovalStatus,
    TargetUsageScope,
    UsageScope,
)
from ai_core.datasets.transforms import (
    ForensicTransformPipeline,
    ReproducibleColorVariation,
    ReproducibleControlledNoise,
    ReproducibleCrop,
    ReproducibleFaceCrop,
    ReproducibleJPEGRecompression,
    ReproducibleMildBlur,
    ReproducibleResize,
    ToTensorAndNormalize,
    TransformOperationRecord,
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def _create_synthetic_image(
    size: Tuple[int, int] = (128, 128),
    color: Tuple[int, int, int] = (120, 150, 180),
) -> Image.Image:
    """Create a synthetic PIL RGB image with simple patterns."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    # Add a recognizable pattern to test blurring / noise / transforms
    arr[10:30, 10:30] = (255, 0, 0)
    arr[40:60, 40:60] = (0, 255, 0)
    return Image.fromarray(arr, mode="RGB")


def _save_image_and_get_hash(img: Image.Image, path: Path) -> str:
    """Save PIL image to disk and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=95)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def _make_dummy_record(
    record_id: str,
    rel_path: str,
    sha256: str,
    label: DatasetImageLabel = DatasetImageLabel.CAMERA_PHOTOGRAPH,
    manipulation_category: ManipulationCategory = ManipulationCategory.NONE,
    split: DatasetSplit = DatasetSplit.TRAIN,
    rights_status: RightsApprovalStatus = RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
    commercial_status: CommercialUseStatus = CommercialUseStatus.PERMITTED,
    research_status: ResearchUseStatus = ResearchUseStatus.PERMITTED,
    source_dataset: str = "SafeHER-Benchmark",
) -> DatasetImageRecord:
    """Helper to build a valid DatasetImageRecord for dataset loader testing."""
    return DatasetImageRecord(
        record_id=record_id,
        image_path_or_ref=rel_path,
        sha256=sha256,
        label=label,
        task_type=DatasetTaskType.DEEPFAKE_DETECTION,
        manipulation_category=manipulation_category,
        dataset_name=source_dataset,
        source_creator="SafeHER Test Lab",
        source_url="https://safeher.local/test",
        source_license="CC-BY-4.0",
        commercial_use_status=commercial_status,
        research_use_status=research_status,
        rights_status=rights_status,
        allowed_usage_scopes=[UsageScope.COMMERCIAL_TRAINING, UsageScope.RESEARCH],
        split=split,
    )


# ============================================================================
# 1. Transform Operations Unit Tests
# ============================================================================

class TestTransformOperations:
    """Unit tests for individual signal-preserving forensic transforms."""

    def test_reproducible_resize(self):
        img = _create_synthetic_image(size=(100, 80))
        transform = ReproducibleResize(target_resolution=(224, 224))
        out, op = transform(img)

        assert out.size == (224, 224)
        assert op.operation_name == "resize"
        assert op.applied is True
        assert op.parameters["target_resolution"] == (224, 224)

    def test_reproducible_jpeg_recompression(self):
        img = _create_synthetic_image(size=(128, 128))
        transform = ReproducibleJPEGRecompression(min_quality=70, max_quality=90, p=1.0, seed=42)
        out, op = transform(img)

        assert out.size == img.size
        assert out.mode == "RGB"
        assert op.operation_name == "jpeg_recompression"
        assert op.applied is True
        assert 70 <= op.parameters["quality"] <= 90

    def test_jpeg_recompression_bypassed_by_probability(self):
        img = _create_synthetic_image(size=(64, 64))
        transform = ReproducibleJPEGRecompression(p=0.0)
        out, op = transform(img)

        assert op.applied is False
        assert op.parameters["quality"] is None

    def test_reproducible_mild_blur(self):
        img = _create_synthetic_image(size=(128, 128))
        transform = ReproducibleMildBlur(max_sigma=1.0, p=1.0, seed=42)
        out, op = transform(img)

        assert out.size == img.size
        assert op.operation_name == "mild_blur"
        assert op.applied is True
        assert 0.1 <= op.parameters["sigma"] <= 1.0

    def test_reproducible_color_variation(self):
        img = _create_synthetic_image(size=(128, 128))
        transform = ReproducibleColorVariation(
            brightness_range=(0.95, 1.05),
            contrast_range=(0.95, 1.05),
            saturation_range=(0.95, 1.05),
            p=1.0,
            seed=42,
        )
        out, op = transform(img)

        assert out.size == img.size
        assert op.operation_name == "color_variation"
        assert op.applied is True
        assert "brightness_factor" in op.parameters

    def test_reproducible_controlled_noise(self):
        img = _create_synthetic_image(size=(128, 128))
        transform = ReproducibleControlledNoise(max_sigma=2.5, p=1.0, seed=42)
        out, op = transform(img)

        assert out.size == img.size
        assert out.mode == "RGB"
        assert op.operation_name == "controlled_noise"
        assert op.applied is True
        assert 0.1 <= op.parameters["noise_sigma"] <= 2.5

    def test_reproducible_crop(self):
        img = _create_synthetic_image(size=(200, 200))
        transform = ReproducibleCrop(scale_range=(0.85, 0.95), p=1.0, seed=42)
        out, op = transform(img)

        assert out.size == img.size  # Rescaled back to original
        assert op.operation_name == "scale_bounded_crop"
        assert op.applied is True
        assert "crop_box" in op.parameters

    def test_face_crop_with_bounding_box(self):
        img = _create_synthetic_image(size=(300, 300))
        transform = ReproducibleFaceCrop(margin_ratio=0.1, target_resolution=(224, 224))
        # Bounding box [x1, y1, x2, y2]
        metadata = {"bounding_box": [50, 50, 150, 150]}
        out, op = transform(img, metadata=metadata)

        assert out.size == (224, 224)
        assert op.operation_name == "face_region_crop"
        assert op.applied is True
        assert op.parameters["method"] == "bounding_box"

    def test_face_crop_fallback_center_crop(self):
        img = _create_synthetic_image(size=(300, 300))
        transform = ReproducibleFaceCrop(margin_ratio=0.1, target_resolution=(224, 224))
        out, op = transform(img, metadata={})

        assert out.size == (224, 224)
        assert op.applied is True
        assert op.parameters["method"] == "center_crop_fallback"

    def test_to_tensor_and_normalize(self):
        img = _create_synthetic_image(size=(64, 64))
        transform = ToTensorAndNormalize()
        tensor, op = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == torch.float32
        assert op.operation_name == "to_tensor_and_normalize"
        assert op.applied is True


# ============================================================================
# 2. Forensic Transform Pipeline Unit Tests
# ============================================================================

class TestForensicTransformPipeline:
    """Tests for pipeline orchestrator, train vs eval modes, and reproducibility."""

    def test_pipeline_reproducibility_with_same_seed(self):
        img = _create_synthetic_image(size=(128, 128))
        pipe1 = ForensicTransformPipeline(mode="train", seed=12345)
        pipe2 = ForensicTransformPipeline(mode="train", seed=12345)

        t1, ops1 = pipe1(img)
        t2, ops2 = pipe2(img)

        assert torch.allclose(t1, t2, atol=1e-5), "Same seed must yield identical output tensor."
        assert len(ops1) == len(ops2)
        for o1, o2 in zip(ops1, ops2):
            assert o1.operation_name == o2.operation_name
            assert o1.applied == o2.applied

    def test_pipeline_divergence_with_different_seeds(self):
        img = _create_synthetic_image(size=(128, 128))
        pipe1 = ForensicTransformPipeline(mode="train", seed=111)
        pipe2 = ForensicTransformPipeline(mode="train", seed=999)

        t1, _ = pipe1(img)
        t2, _ = pipe2(img)

        assert not torch.allclose(t1, t2, atol=1e-3), "Different seeds should yield divergent tensors."

    def test_eval_mode_strictly_deterministic(self):
        """In eval/test mode, only deterministic resize and normalization must occur."""
        img = _create_synthetic_image(size=(128, 128))
        eval_pipe = ForensicTransformPipeline(mode="eval", target_resolution=(224, 224))
        test_pipe = ForensicTransformPipeline(mode="test", target_resolution=(224, 224))

        t_eval, ops_eval = eval_pipe(img)
        t_test, ops_test = test_pipe(img)

        assert torch.allclose(t_eval, t_test, atol=1e-6)
        # Verify operations applied: resize and to_tensor_and_normalize ONLY
        applied_ops = [op.operation_name for op in ops_eval if op.applied]
        assert applied_ops == ["resize", "to_tensor_and_normalize"]
        # Ensure stochastic operations were never applied
        for op in ops_eval:
            if op.operation_name in ("jpeg_recompression", "mild_blur", "controlled_noise", "scale_bounded_crop"):
                assert op.applied is False

    def test_invalid_target_resolution_rejected(self):
        with pytest.raises(ValueError, match="target_resolution must contain positive integers"):
            ForensicTransformPipeline(target_resolution=(-1, 224))

        with pytest.raises(ValueError, match="target_resolution must contain positive integers"):
            ForensicTransformPipeline(target_resolution=(0, 224))

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            ForensicTransformPipeline(mode="invalid_mode")


# ============================================================================
# 3. Label Mapping Tests
# ============================================================================

class TestLabelMapping:
    """Tests for binary and multiclass label mappings."""

    def test_binary_label_mapping(self):
        # Authentic / pristine
        assert BINARY_LABEL_MAP[DatasetImageLabel.CAMERA_PHOTOGRAPH] == 0

        # Manipulated / synthetic
        assert BINARY_LABEL_MAP[DatasetImageLabel.AI_GENERATED_IMAGE] == 1
        assert BINARY_LABEL_MAP[DatasetImageLabel.FACE_SWAP] == 1
        assert BINARY_LABEL_MAP[DatasetImageLabel.FACIAL_REENACTMENT] == 1
        assert BINARY_LABEL_MAP[DatasetImageLabel.FACIAL_INPAINTING] == 1
        assert BINARY_LABEL_MAP[DatasetImageLabel.LOCAL_IMAGE_EDIT] == 1
        assert BINARY_LABEL_MAP[DatasetImageLabel.IMAGE_COMPOSITE] == 1

        # Unknown / Unverified must raise KeyError
        with pytest.raises(KeyError):
            _ = BINARY_LABEL_MAP[DatasetImageLabel.UNKNOWN]

        with pytest.raises(KeyError):
            _ = BINARY_LABEL_MAP[DatasetImageLabel.UNVERIFIED]

    def test_multiclass_label_mapping(self):
        # Valid multiclass labels map to distinct integers 0..6
        labels = [
            DatasetImageLabel.CAMERA_PHOTOGRAPH,
            DatasetImageLabel.AI_GENERATED_IMAGE,
            DatasetImageLabel.FACE_SWAP,
            DatasetImageLabel.FACIAL_REENACTMENT,
            DatasetImageLabel.FACIAL_INPAINTING,
            DatasetImageLabel.LOCAL_IMAGE_EDIT,
            DatasetImageLabel.IMAGE_COMPOSITE,
        ]
        indices = [MULTICLASS_LABEL_MAP[lbl] for lbl in labels]
        assert sorted(indices) == list(range(7))

        with pytest.raises(KeyError):
            _ = MULTICLASS_LABEL_MAP[DatasetImageLabel.UNKNOWN]


# ============================================================================
# 4. SafeHERDataset Loading & Validation Tests
# ============================================================================

class TestSafeHERDataset:
    """Tests for SafeHERDataset streaming, rights validation, splits, and error handling."""

    @pytest.fixture
    def test_env(self, tmp_path):
        """Build a temporary directory with synthetic images and manifest."""
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)

        records = []
        # Sample 1: Authentic image in TRAIN split, approved for commercial
        p1 = images_dir / "auth_01.jpg"
        h1 = _save_image_and_get_hash(_create_synthetic_image(color=(100, 150, 200)), p1)
        r1 = _make_dummy_record(
            "REC-001",
            "images/auth_01.jpg",
            h1,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            split=DatasetSplit.TRAIN,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        )
        records.append(r1)

        # Sample 2: Manipulated image in TRAIN split, approved for commercial
        p2 = images_dir / "fake_01.jpg"
        h2 = _save_image_and_get_hash(_create_synthetic_image(color=(200, 50, 50)), p2)
        r2 = _make_dummy_record(
            "REC-002",
            "images/fake_01.jpg",
            h2,
            label=DatasetImageLabel.FACE_SWAP,
            manipulation_category=ManipulationCategory.DEEPFAKE_FACE_SWAP,
            split=DatasetSplit.TRAIN,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        )
        records.append(r2)

        # Sample 3: Research-only image in TRAIN split (e.g. academic dataset)
        p3 = images_dir / "research_01.jpg"
        h3 = _save_image_and_get_hash(_create_synthetic_image(color=(50, 200, 50)), p3)
        r3 = _make_dummy_record(
            "REC-003",
            "images/research_01.jpg",
            h3,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            split=DatasetSplit.TRAIN,
            rights_status=RightsApprovalStatus.RESEARCH_ONLY,
            commercial_status=CommercialUseStatus.PROHIBITED,
        )
        records.append(r3)

        # Sample 4: Authentic image in TEST split
        p4 = images_dir / "test_01.jpg"
        h4 = _save_image_and_get_hash(_create_synthetic_image(color=(80, 80, 80)), p4)
        r4 = _make_dummy_record(
            "REC-004",
            "images/test_01.jpg",
            h4,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            split=DatasetSplit.TEST,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        )
        records.append(r4)

        manifest = DatasetManifest(
            manifest_id="test-manifest-001",
            dataset_name="TestSafeHER",
            version="1.0.0",
            records=records,
        )

        return {
            "root": tmp_path,
            "manifest": manifest,
            "paths": [p1, p2, p3, p4],
            "hashes": [h1, h2, h3, h4],
        }

    def test_dataset_loading_train_split(self, test_env):
        dataset = SafeHERDataset(
            records=test_env["manifest"],
            root_dir=test_env["root"],
            split=DatasetSplit.TRAIN,
            target_workload=TargetUsageScope.RESEARCH,
            transform_pipeline=ForensicTransformPipeline(mode="eval", target_resolution=(224, 224)),
        )

        # In RESEARCH mode on TRAIN split: REC-001, REC-002, REC-003 are included (3 items)
        assert len(dataset) == 3
        sample = dataset[0]
        assert sample is not None
        tensor = sample["image"]
        label = sample["label"].item()
        record_id = sample["record_id"]

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert label == 0  # Camera photograph
        assert record_id == "REC-001"

    def test_rights_eligibility_commercial_training(self, test_env):
        """Commercial training must strictly exclude RESEARCH_ONLY records and log them."""
        dataset = SafeHERDataset(
            records=test_env["manifest"],
            root_dir=test_env["root"],
            split=DatasetSplit.TRAIN,
            target_workload=TargetUsageScope.PRODUCTION_TRAINING,
            transform_pipeline=ForensicTransformPipeline(mode="eval"),
        )

        # REC-003 is RESEARCH_ONLY, so it must be excluded
        assert len(dataset) == 2
        loaded_ids = [dataset[i]["record_id"] for i in range(len(dataset))]
        assert "REC-001" in loaded_ids
        assert "REC-002" in loaded_ids
        assert "REC-003" not in loaded_ids

        # Verify exclusion was logged non-silently
        assert len(dataset.exclusion_log) >= 1
        reasons = [rec.reason for rec in dataset.exclusion_log]
        assert "EXCLUDED_RIGHTS_RESTRICTION" in reasons

    def test_split_filtering(self, test_env):
        dataset_test = SafeHERDataset(
            records=test_env["manifest"],
            root_dir=test_env["root"],
            split=DatasetSplit.TEST,
            target_workload=TargetUsageScope.EVALUATION,
        )

        assert len(dataset_test) == 1
        sample = dataset_test[0]
        assert sample["record_id"] == "REC-004"

    def test_multiclass_label_output(self, test_env):
        dataset = SafeHERDataset(
            records=test_env["manifest"],
            root_dir=test_env["root"],
            split=DatasetSplit.TRAIN,
            target_workload=TargetUsageScope.RESEARCH,
            label_mode="multiclass",
        )

        # REC-002 has label FACE_SWAP -> MULTICLASS index 2
        sample = dataset[1]
        assert sample["record_id"] == "REC-002"
        assert sample["label"].item() == MULTICLASS_LABEL_MAP[DatasetImageLabel.FACE_SWAP]

    def test_sha256_verification_on_load(self, test_env):
        # Tamper with an image file
        p2 = test_env["paths"][1]
        with open(p2, "ab") as f:
            f.write(b"tamper_bytes_12345")

        dataset = SafeHERDataset(
            records=test_env["manifest"],
            root_dir=test_env["root"],
            split=DatasetSplit.TRAIN,
            target_workload=TargetUsageScope.RESEARCH,
            verify_sha256_on_load=True,
            return_none_on_corrupt=True,
        )

        # Loading index 1 (tampered file) should fail integrity check, log exclusion, and return None
        sample = dataset[1]
        assert sample is None
        assert len(dataset.exclusion_log) >= 1
        assert any(rec.reason == "INTEGRITY_MISMATCH" for rec in dataset.exclusion_log)

    def test_corrupted_image_handling(self, tmp_path):
        """Corrupted image bytes must be caught gracefully and recorded in exclusion_log."""
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")

        record = _make_dummy_record(
            "REC-CORRUPT",
            "corrupt.jpg",
            "0" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        )
        manifest = DatasetManifest(
            manifest_id="corrupt-manifest",
            dataset_name="CorruptTest",
            version="1.0",
            records=[record],
        )

        dataset = SafeHERDataset(
            records=manifest,
            root_dir=tmp_path,
            target_workload=TargetUsageScope.RESEARCH,
            verify_sha256_on_load=False,
            return_none_on_corrupt=True,
        )

        sample = dataset[0]
        assert sample is None
        assert len(dataset.exclusion_log) >= 1
        assert dataset.exclusion_log[0].reason == "FILE_CORRUPTED"
        assert dataset.exclusion_log[0].record_id == "REC-CORRUPT"

    def test_decompression_bomb_safety(self, tmp_path):
        """Images exceeding max_image_pixels are rejected with DECOMPRESSION_BOMB reason."""
        # Create a small valid image
        p = tmp_path / "small.jpg"
        _save_image_and_get_hash(_create_synthetic_image(size=(32, 32)), p)

        record = _make_dummy_record(
            "REC-BOMB",
            "small.jpg",
            "0" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        )
        manifest = DatasetManifest(
            manifest_id="bomb-manifest",
            dataset_name="BombTest",
            version="1.0",
            records=[record],
        )

        # Set max_image_pixels to a tiny threshold (e.g. 500 pixels < 32*32 = 1024)
        dataset = SafeHERDataset(
            records=manifest,
            root_dir=tmp_path,
            target_workload=TargetUsageScope.RESEARCH,
            max_image_pixels=500,
            return_none_on_corrupt=True,
        )

        sample = dataset[0]
        assert sample is None
        assert len(dataset.exclusion_log) >= 1
        assert dataset.exclusion_log[0].reason == "DECOMPRESSION_BOMB"


# ============================================================================
# 5. PyTorch DataLoader & Safe Collation Tests
# ============================================================================

class TestDataLoaderAndCollate:
    """Tests for PyTorch DataLoader creation, safe batch collation, and batch consistency."""

    def test_collate_fn_batch_consistency(self):
        batch = [
            {
                "image": torch.randn(3, 224, 224),
                "label": torch.tensor(0, dtype=torch.long),
                "record_id": "REC-1",
                "sha256": "abc1",
                "metadata": {},
                "applied_transforms": [],
            },
            {
                "image": torch.randn(3, 224, 224),
                "label": torch.tensor(1, dtype=torch.long),
                "record_id": "REC-2",
                "sha256": "abc2",
                "metadata": {},
                "applied_transforms": [],
            },
        ]

        collated = safe_collate_fn(batch)

        assert isinstance(collated["images"], torch.Tensor)
        assert collated["images"].shape == (2, 3, 224, 224)
        assert isinstance(collated["labels"], torch.Tensor)
        assert collated["labels"].shape == (2,)
        assert collated["labels"].dtype == torch.long
        assert collated["record_ids"] == ["REC-1", "REC-2"]
        assert len(collated["metadata"]) == 2

    def test_collate_fn_with_none_filtering(self):
        """Safe collate should filter out None samples from corrupted files."""
        batch = [
            {
                "image": torch.randn(3, 224, 224),
                "label": torch.tensor(0, dtype=torch.long),
                "record_id": "REC-1",
                "sha256": "abc1",
                "metadata": {},
                "applied_transforms": [],
            },
            None,  # Corrupted sample returned None
            {
                "image": torch.randn(3, 224, 224),
                "label": torch.tensor(1, dtype=torch.long),
                "record_id": "REC-3",
                "sha256": "abc3",
                "metadata": {},
                "applied_transforms": [],
            },
        ]

        collated = safe_collate_fn(batch)
        assert collated["images"].shape == (2, 3, 224, 224)
        assert collated["record_ids"] == ["REC-1", "REC-3"]

    def test_collate_fn_all_none_raises(self):
        with pytest.raises(ValueError, match="Entire batch consisted of corrupted or unreadable samples"):
            safe_collate_fn([None, None, None])

    def test_create_safe_dataloader_iteration(self, tmp_path):
        """End-to-end iteration through PyTorch DataLoader with SafeHERDataset."""
        images_dir = tmp_path / "data"
        images_dir.mkdir(parents=True)

        records = []
        for i in range(5):
            p = images_dir / f"img_{i}.jpg"
            lbl = DatasetImageLabel.CAMERA_PHOTOGRAPH if i % 2 == 0 else DatasetImageLabel.FACE_SWAP
            cat = ManipulationCategory.NONE if i % 2 == 0 else ManipulationCategory.DEEPFAKE_FACE_SWAP
            h = _save_image_and_get_hash(_create_synthetic_image(size=(100, 100)), p)
            rec = _make_dummy_record(
                f"REC-{i:03d}",
                f"data/img_{i}.jpg",
                h,
                label=lbl,
                manipulation_category=cat,
            )
            records.append(rec)

        manifest = DatasetManifest(
            manifest_id="m-iter-001",
            dataset_name="IterDataset",
            version="1.0",
            records=records,
        )

        dataset = SafeHERDataset(
            records=manifest,
            root_dir=tmp_path,
            split=DatasetSplit.TRAIN,
            target_workload=TargetUsageScope.RESEARCH,
            transform_pipeline=ForensicTransformPipeline(mode="eval", target_resolution=(224, 224)),
        )

        dataloader = create_safe_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 0 for single process test execution
        )

        batches = list(dataloader)
        # 5 samples with batch_size=2 -> 3 batches (sizes 2, 2, 1)
        assert len(batches) == 3
        assert batches[0]["images"].shape == (2, 3, 224, 224)
        assert batches[1]["images"].shape == (2, 3, 224, 224)
        assert batches[2]["images"].shape == (1, 3, 224, 224)
        assert batches[0]["labels"].shape == (2,)

    def test_worker_count_bounded(self, tmp_path):
        """Verify worker count is capped at CPU cores (max 4)."""
        manifest = DatasetManifest(
            manifest_id="m-workers",
            dataset_name="Workers",
            version="1.0",
            records=[],
        )
        dataset = SafeHERDataset(records=manifest, root_dir=tmp_path)
        loader = create_safe_dataloader(dataset, batch_size=2, num_workers=64)

        # Loader worker count should be bounded <= 4
        assert loader.num_workers <= 4
