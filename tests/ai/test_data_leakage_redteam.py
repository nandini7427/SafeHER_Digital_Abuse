"""Adversarial Red-Team Penetration Tests for SafeHER Data Leakage.

Audits all 11 leakage vectors with targeted adversarial fixtures:
1. Duplicate and near-duplicate images across splits.
2. Same person across training and testing.
3. Same source video across splits.
4. Same original and manipulated image across splits.
5. Generator-specific artefacts leaking labels.
6. File-path or metadata leakage.
7. Watermarks and text labels that expose the class.
8. Dataset-source confounding.
9. Augmentation leakage.
10. Threshold tuning on test data.
11. Validation/test contamination.
"""

import hashlib
import io
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import pytest
import torch

from ai_core.datasets.duplicates import NearDuplicateCandidate, compute_dhash
from ai_core.datasets.leakage import (
    LeakageDetector,
    LeakageType,
    LeakageViolation,
)
from ai_core.datasets.leakage_redteam import (
    DataLeakageRedTeamAuditor,
    LeakagePathStatus,
    detect_synthetic_corner_watermark,
)
from ai_core.datasets.loader import SafeHERDataset
from ai_core.datasets.manifest import (
    AttributionMetadata,
    CommercialUseStatus,
    ConsentStatus,
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    DatasetTaskType,
    ManipulationCategory,
    RightsApprovalStatus,
    TargetUsageScope,
)
from ai_core.datasets.splitter import (
    DatasetLeakageError,
    GroupAwareDatasetSplitter,
    SplitPolicy,
)
from ai_core.datasets.transforms import ForensicTransformPipeline
from ai_core.neural.calibration.calibrators import (
    IsotonicRegressionCalibrator,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
)
from ai_core.neural.training.config import TrainingPipelineConfig
from ai_core.neural.training.pipeline import ModelATrainingPipeline
from ai_core.neural.training.safety import enforce_training_safety_and_rights


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().lower()


def _make_dummy_image(color: tuple = (128, 128, 128), size: tuple = (128, 128)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _create_record(
    record_id: str,
    split: DatasetSplit = DatasetSplit.UNASSIGNED,
    label: DatasetImageLabel = DatasetImageLabel.CAMERA_PHOTOGRAPH,
    sha256_hash: str = None,
    image_path: str = None,
    grouping_id: str = None,
    consent_ref: str = None,
    original_ref: str = None,
    parent_ref: str = None,
    generator_name: str = None,
    source_dataset: str = "SafeHER-Benchmark",
    training_eligibility: Optional[bool] = None,
) -> DatasetImageRecord:
    is_test_split = split in (
        DatasetSplit.TEST,
        DatasetSplit.INTERNAL_TEST,
        DatasetSplit.HELD_OUT_GENERATOR_TEST,
        DatasetSplit.CROSS_DATASET_TEST,
        DatasetSplit.COMPRESSION_ROBUSTNESS_TEST,
    )
    eligible = False if is_test_split else (True if training_eligibility is None else training_eligibility)

    return DatasetImageRecord(
        record_id=record_id,
        image_path_or_ref=image_path or f"images/{record_id}.jpg",
        sha256=sha256_hash or _sha256(record_id),
        label=label,
        split=split,
        grouping_id=grouping_id,
        consent_reference=consent_ref,
        original_image_ref=original_ref,
        parent_image_ref=parent_ref,
        generator_name=generator_name,
        source_dataset=source_dataset,
        dataset_name=source_dataset,
        source_creator="SafeHER Red Team",
        source_license="CC0-1.0",
        attribution=AttributionMetadata(required=False, creator="SafeHER Red Team", attribution_text="SafeHER Red Team 2026"),
        attribution_requirements="SafeHER Red Team 2026",
        commercial_use_status=CommercialUseStatus.PERMITTED,
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        training_eligibility=eligible,
        evaluation_eligibility=True,
    )


# ------------------------------------------------------------------------------
# 1. Duplicate and Near-Duplicate Images Across Splits
# ------------------------------------------------------------------------------
def test_adversarial_exact_duplicate_leakage() -> None:
    """Adversarial test: Identical image SHA-256 injected into train and test splits."""
    shared_hash = _sha256("identical_image_bytes_42")

    rec_train = _create_record(
        record_id="REC-TRAIN-DUP-1",
        split=DatasetSplit.TRAIN,
        sha256_hash=shared_hash,
    )
    rec_test = _create_record(
        record_id="REC-TEST-DUP-2",
        split=DatasetSplit.INTERNAL_TEST,
        sha256_hash=shared_hash,
    )

    detector = LeakageDetector()
    report = detector.audit_splits([rec_train, rec_test])

    assert report.is_leakage_free is False
    assert any(v.leakage_type == LeakageType.EXACT_DUPLICATE_LEAKAGE for v in report.violations)

    manifest = DatasetManifest(
        manifest_version="1.0.0",
        dataset_name="SafeHER-RedTeam",
        records=[rec_train, rec_test],
    )
    with pytest.raises(ValueError, match="Exact cryptographic duplicate image"):
        enforce_training_safety_and_rights(manifest)


def test_adversarial_near_duplicate_leakage() -> None:
    """Adversarial test: Visually near-duplicate image pair crossing split boundary."""
    rec_train = _create_record("REC-TRAIN-1", split=DatasetSplit.TRAIN)
    rec_val = _create_record("REC-VAL-1", split=DatasetSplit.VAL)

    # Candidate pair with small Hamming distance (3 bits out of 64)
    near_dup = NearDuplicateCandidate(
        record_id_a="REC-TRAIN-1",
        record_id_b="REC-VAL-1",
        hash_a="1122334455667788",
        hash_b="1122334455667789",
        hamming_distance=3,
        similarity_score=0.953,
        category="RECOMPRESSION_VARIANT",
        requires_human_review=True,
        notes="Recompressed duplicate",
    )

    detector = LeakageDetector()
    report = detector.audit_splits([rec_train, rec_val], near_duplicate_pairs=[near_dup])

    assert report.is_leakage_free is False
    assert any(v.leakage_type == LeakageType.NEAR_DUPLICATE_LEAKAGE for v in report.violations)


# ------------------------------------------------------------------------------
# 2. Same Person Across Training and Testing
# ------------------------------------------------------------------------------
def test_adversarial_identity_and_consent_leakage() -> None:
    """Adversarial test: Same human subject actor placed in both train and test splits."""
    rec_train = _create_record(
        "REC-TRAIN-ACTOR-1",
        split=DatasetSplit.TRAIN,
        grouping_id="ACTOR-ALICE",
        consent_ref="CR-SAFEHER-ALICE",
    )
    rec_test = _create_record(
        "REC-TEST-ACTOR-2",
        split=DatasetSplit.INTERNAL_TEST,
        grouping_id="ACTOR-ALICE",
        consent_ref="CR-SAFEHER-ALICE",
    )

    detector = LeakageDetector()
    report = detector.audit_splits([rec_train, rec_test])

    assert report.is_leakage_free is False
    assert any(v.leakage_type == LeakageType.IDENTITY_LEAKAGE for v in report.violations)

    manifest = DatasetManifest(
        manifest_version="1.0.0",
        dataset_name="SafeHER-RedTeam",
        records=[rec_train, rec_test],
    )
    with pytest.raises(ValueError, match="Test Leakage Violation!.*grouping_id"):
        enforce_training_safety_and_rights(manifest)


# ------------------------------------------------------------------------------
# 3. Same Source Video Across Splits
# ------------------------------------------------------------------------------
def test_adversarial_source_video_leakage() -> None:
    """Adversarial test: Frames from the same source video crossing train and test partitions."""
    rec_train_frame = _create_record(
        "REC-FRAME-001",
        split=DatasetSplit.TRAIN,
        image_path="videos/video_089_frame_001.jpg",
    )
    rec_test_frame = _create_record(
        "REC-FRAME-045",
        split=DatasetSplit.INTERNAL_TEST,
        image_path="videos/video_089_frame_045.jpg",
    )

    detector = LeakageDetector()
    report = detector.audit_splits([rec_train_frame, rec_test_frame])

    assert report.is_leakage_free is False
    assert any(v.leakage_type == LeakageType.SOURCE_VIDEO_LEAKAGE for v in report.violations)

    # Verify that GroupAwareDatasetSplitter automatically clusters them
    splitter = GroupAwareDatasetSplitter(
        policy=SplitPolicy(
            train_ratio=0.5,
            val_ratio=0.25,
            internal_test_ratio=0.25,
            group_by_lineage=True,
        )
    )
    clusters = splitter._build_clusters([rec_train_frame, rec_test_frame])
    # Both frames should be in the same cluster
    assert len(clusters) == 1


# ------------------------------------------------------------------------------
# 4. Same Original and Manipulated Image Across Splits
# ------------------------------------------------------------------------------
def test_adversarial_parent_child_lineage_leakage() -> None:
    """Adversarial test: Pristine original parent in train, deepfake manipulated child in test."""
    rec_parent = _create_record(
        "REC-PARENT-PRISTINE",
        split=DatasetSplit.TRAIN,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        original_ref=None,
        parent_ref=None,
    )
    rec_child_fake = _create_record(
        "REC-CHILD-DEEPFAKE",
        split=DatasetSplit.INTERNAL_TEST,
        label=DatasetImageLabel.FACE_SWAP,
        original_ref="REC-PARENT-PRISTINE",
        parent_ref="REC-PARENT-PRISTINE",
    )

    # 1. Verify LeakageDetector intercepts it
    detector = LeakageDetector()
    report = detector.audit_splits([rec_parent, rec_child_fake])
    assert report.is_leakage_free is False
    assert any(v.leakage_type == LeakageType.LINEAGE_DERIVATIVE_LEAKAGE for v in report.violations)

    # 2. Verify enforce_training_safety_and_rights catches it
    manifest = DatasetManifest(
        manifest_version="1.0.0",
        dataset_name="SafeHER-RedTeam",
        records=[rec_parent, rec_child_fake],
    )
    with pytest.raises(ValueError, match="Test Leakage Violation! Test record.*derived variant"):
        enforce_training_safety_and_rights(manifest)

    # 3. Verify GroupAwareDatasetSplitter unions them together
    splitter = GroupAwareDatasetSplitter(policy=SplitPolicy(group_by_lineage=True))
    clusters = splitter._build_clusters([rec_parent, rec_child_fake])
    assert len(clusters) == 1


# ------------------------------------------------------------------------------
# 5. Generator-Specific Artefacts Leaking Labels
# ------------------------------------------------------------------------------
def test_adversarial_held_out_generator_leakage() -> None:
    """Adversarial test: Held-out generator asset improperly included in training split."""
    rec_train_heldout = _create_record(
        "REC-TRAIN-GEN-1",
        split=DatasetSplit.TRAIN,
        generator_name="flux",
    )
    detector = LeakageDetector(held_out_generators={"flux", "sdxl"})
    report = detector.audit_splits([rec_train_heldout])

    assert report.is_leakage_free is False
    assert any(v.leakage_type == LeakageType.HELD_OUT_GENERATOR_LEAKAGE for v in report.violations)


# ------------------------------------------------------------------------------
# 6. File-Path or Metadata Leakage
# ------------------------------------------------------------------------------
def test_adversarial_metadata_and_path_leakage() -> None:
    """Adversarial test: Ensure model feature tensors never ingest file paths or metadata strings."""
    img = _make_dummy_image()
    rec_path_leak = _create_record(
        "REC-LEAKY-PATH-1",
        split=DatasetSplit.TRAIN,
        image_path="datasets/fake_images/generator_flux_synthetic.jpg",
    )

    dataset = SafeHERDataset(
        records=[rec_path_leak],
        split=DatasetSplit.TRAIN,
        in_memory_images={"REC-LEAKY-PATH-1": img},
        target_workload=TargetUsageScope.PRODUCTION_TRAINING,
    )

    sample = dataset[0]
    assert sample is not None

    # Verify tensor payload contains ONLY pixel data and integer label
    assert "image" in sample
    assert "label" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)
    # Filepath string must never be part of tensor inputs
    assert "image_path_or_ref" not in sample


# ------------------------------------------------------------------------------
# 7. Watermarks and Text Labels That Expose the Class
# ------------------------------------------------------------------------------
def test_adversarial_watermark_detection() -> None:
    """Adversarial test: Artificial corner ribbon (e.g. DALL-E ribbon) detected and flagged."""
    # Create image with artificial solid-color corner stripes in bottom right
    img_with_ribbon = _make_dummy_image(color=(50, 50, 50), size=(256, 256))
    arr = np.array(img_with_ribbon)
    # Paint high-saturation distinct colored stripes in bottom-right corner (rows 230-255, cols 230-255)
    arr[230:255, 230:255, :] = [255, 220, 0]  # bright yellow ribbon
    arr[240:255, 240:255, :] = [0, 200, 255]  # bright cyan ribbon
    img_marked = Image.fromarray(arr)

    assert detect_synthetic_corner_watermark(img_marked) is True

    # Pristine natural image
    img_clean = _make_dummy_image(color=(120, 120, 120), size=(256, 256))
    assert detect_synthetic_corner_watermark(img_clean) is False


# ------------------------------------------------------------------------------
# 8. Dataset-Source Confounding
# ------------------------------------------------------------------------------
def test_adversarial_dataset_source_confounding() -> None:
    """Adversarial test: 100% of authentic samples from Dataset A and 100% of fakes from Dataset B."""
    records = []
    # 15 authentic from Dataset-A
    for i in range(15):
        records.append(
            _create_record(
                f"REC-AUTH-{i}",
                split=DatasetSplit.TRAIN,
                label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
                source_dataset="Dataset-Only-Authentic",
            )
        )
    # 15 manipulated from Dataset-B
    for i in range(15):
        records.append(
            _create_record(
                f"REC-MANIP-{i}",
                split=DatasetSplit.TRAIN,
                label=DatasetImageLabel.AI_GENERATED_IMAGE,
                source_dataset="Dataset-Only-Manipulated",
            )
        )

    detector = LeakageDetector()
    report = detector.audit_splits(records)

    assert any(v.leakage_type == LeakageType.DATASET_SOURCE_CONFOUNDING for v in report.violations)


# ------------------------------------------------------------------------------
# 9. Augmentation Leakage
# ------------------------------------------------------------------------------
def test_adversarial_augmentation_isolation() -> None:
    """Adversarial test: Evaluation pipelines use deterministic ImageNet constants without test fitting."""
    pipeline_eval = ForensicTransformPipeline(mode="eval", target_resolution=(224, 224))
    img = _make_dummy_image(color=(100, 150, 200), size=(300, 300))

    tensor1, _ = pipeline_eval(img)
    tensor2, _ = pipeline_eval(img)

    # Deterministic preprocessing: identical inputs produce identical tensors
    assert torch.equal(tensor1, tensor2)

    # Standard ImageNet normalization constants viewable on to_tensor
    assert pipeline_eval.to_tensor.mean.squeeze().tolist() == pytest.approx([0.485, 0.456, 0.406], abs=1e-4)
    assert pipeline_eval.to_tensor.std.squeeze().tolist() == pytest.approx([0.229, 0.224, 0.225], abs=1e-4)


# ------------------------------------------------------------------------------
# 10. Threshold Tuning on Test Data
# ------------------------------------------------------------------------------
def test_adversarial_threshold_tuning_on_test_data() -> None:
    """Adversarial test: Attempting to fit calibration or tune thresholds on test split is rejected."""
    calibrator = TemperatureScalingCalibrator(
        model_name="test_model",
        model_version="1.0.0",
        architecture="mobilenet_v3_large",
    )

    logits = np.array([[-1.5, 1.2], [2.0, -1.8], [0.1, -0.2]])
    labels = np.array([1, 0, 0])

    forbidden_splits = ["test", "internal_test", "held_out_generator_test", "cross_dataset_test"]
    for split_name in forbidden_splits:
        with pytest.raises(ValueError, match="STRICT SEPARATION VIOLATION: Calibration cannot be fitted on split"):
            calibrator.verify_split_isolation(split_name)

        with pytest.raises(ValueError, match="STRICT SEPARATION VIOLATION: Calibration cannot be fitted on split"):
            calibrator.fit(logits, labels, calibration_split=split_name)


# ------------------------------------------------------------------------------
# 11. Validation/Test Contamination
# ------------------------------------------------------------------------------
def test_adversarial_validation_test_contamination() -> None:
    """Adversarial test: Test split records are never loaded during training pipeline loops."""
    records = [
        _create_record("REC-T1", split=DatasetSplit.TRAIN),
        _create_record("REC-T2", split=DatasetSplit.TRAIN),
        _create_record("REC-V1", split=DatasetSplit.VAL),
        _create_record("REC-TEST-SECRET", split=DatasetSplit.INTERNAL_TEST, training_eligibility=False),
    ]
    manifest = DatasetManifest(
        manifest_version="1.0.0",
        dataset_name="SafeHER-RedTeam",
        records=records,
    )

    in_memory = {
        r.record_id: _make_dummy_image(size=(128, 128))
        for r in records
    }

    config = TrainingPipelineConfig(
        model_architecture="mobilenet_v3_large",
        num_classes=2,
        batch_size=2,
        epochs=1,
        enforce_approved_manifest=False,  # testing data loader isolation specifically
    )

    pipeline = ModelATrainingPipeline(
        config=config,
        manifest=manifest,
        in_memory_images=in_memory,
    )

    train_ids = [r.record_id for r in pipeline.train_dataset.records]
    val_ids = [r.record_id for r in pipeline.val_dataset.records]

    assert "REC-TEST-SECRET" not in train_ids
    assert "REC-TEST-SECRET" not in val_ids
    assert len(train_ids) == 2
    assert len(val_ids) == 1


# ------------------------------------------------------------------------------
# 12. Complete Red-Team Audit Summary Engine
# ------------------------------------------------------------------------------
def test_complete_red_team_leakage_audit() -> None:
    """End-to-end verification of DataLeakageRedTeamAuditor across all 11 vectors."""
    clean_records = [
        _create_record("REC-C1", split=DatasetSplit.TRAIN, grouping_id="PERSON-1"),
        _create_record("REC-C2", split=DatasetSplit.TRAIN, grouping_id="PERSON-1"),
        _create_record("REC-C3", split=DatasetSplit.VAL, grouping_id="PERSON-2"),
        _create_record("REC-C4", split=DatasetSplit.INTERNAL_TEST, grouping_id="PERSON-3", training_eligibility=False),
    ]

    auditor = DataLeakageRedTeamAuditor()
    summary = auditor.audit_manifest(clean_records)

    assert summary.total_vectors_audited == 11
    assert summary.is_audit_passed is True
    assert summary.vulnerable_count == 0

    markdown_report = summary.to_markdown()
    assert "# SafeHER Milestone 4 — Comprehensive Data Leakage Red-Team Audit Report" in markdown_report
    assert "Duplicate and Near-Duplicate Images Across Splits" in markdown_report
    assert "Same Original and Manipulated Image Across Splits" in markdown_report
    assert "Threshold Tuning on Test Data" in markdown_report
