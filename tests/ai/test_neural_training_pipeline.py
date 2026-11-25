"""Unit tests for SafeHER Reproducible Neural Training Pipeline (M4.4).

Verifies:
- Tiny fixture training run execution and metric generation.
- Deterministic seed behavior across identical runs.
- Checkpoint saving, SafeTensors export, and state resumption.
- Rejection of invalid, unapproved, or research-restricted manifests in commercial training.
- Unconditional quarantine of private survivor case evidence.
- Detection and rejection of train/test partition leakage (shared group/lineage IDs).
- Training failure recovery from saved checkpoints.
- Checkpoint metadata validation (hashes, versions, environment).
- Early stopping triggering upon validation metric plateau.
- Mixed precision CPU safety guards.
"""

import io
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import pytest
import torch

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
    ResearchUseStatus,
    RightsApprovalStatus,
    TargetUsageScope,
)
from ai_core.neural.training import (
    EarlyStoppingMetric,
    LRSchedulerType,
    ModelATrainingPipeline,
    TrainingPipelineConfig,
    TrainingPipelineResult,
)


def create_synthetic_image_bytes(color: Tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Create a 224x224 RGB image as encoded JPEG bytes."""
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def create_test_manifest_and_images() -> Tuple[DatasetManifest, Dict[str, bytes]]:
    """Build a synthetic, legally approved test manifest and in-memory image dict."""
    images: Dict[str, bytes] = {
        "images/train_cam_1.jpg": create_synthetic_image_bytes((120, 100, 80)),
        "images/train_cam_2.jpg": create_synthetic_image_bytes((130, 110, 90)),
        "images/train_ai_1.jpg": create_synthetic_image_bytes((20, 40, 220)),
        "images/train_ai_2.jpg": create_synthetic_image_bytes((30, 50, 230)),
        "images/val_cam_1.jpg": create_synthetic_image_bytes((140, 120, 100)),
        "images/val_ai_1.jpg": create_synthetic_image_bytes((40, 60, 240)),
        "images/test_cam_1.jpg": create_synthetic_image_bytes((150, 130, 110)),
        "images/test_ai_1.jpg": create_synthetic_image_bytes((50, 70, 250)),
        "REC-TR-001": create_synthetic_image_bytes((120, 100, 80)),
        "REC-TR-002": create_synthetic_image_bytes((130, 110, 90)),
        "REC-TR-003": create_synthetic_image_bytes((20, 40, 220)),
        "REC-TR-004": create_synthetic_image_bytes((30, 50, 230)),
        "REC-VAL-001": create_synthetic_image_bytes((140, 120, 100)),
        "REC-VAL-002": create_synthetic_image_bytes((40, 60, 240)),
        "REC-TST-001": create_synthetic_image_bytes((150, 130, 110)),
        "REC-TST-002": create_synthetic_image_bytes((50, 70, 250)),
    }

    records: List[DatasetImageRecord] = [
        # Train (Group A & B)
        DatasetImageRecord(
            record_id="REC-TR-001",
            image_id="IMG-TR-001",
            image_path_or_ref="images/train_cam_1.jpg",
            sha256="1" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.NONE,
            original_image_ref="ORIG-TR-001",
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=True,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            split=DatasetSplit.TRAIN,
            grouping_id="GRP-TRAIN-A",
        ),
        DatasetImageRecord(
            record_id="REC-TR-002",
            image_id="IMG-TR-002",
            image_path_or_ref="images/train_cam_2.jpg",
            sha256="2" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.NONE,
            original_image_ref="ORIG-TR-002",
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=True,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            split=DatasetSplit.TRAIN,
            grouping_id="GRP-TRAIN-A",
        ),
        DatasetImageRecord(
            record_id="REC-TR-003",
            image_id="IMG-TR-003",
            image_path_or_ref="images/train_ai_1.jpg",
            sha256="3" * 64,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=True,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            generator_name="adobe-firefly-v2",
            generator_version="2.1",
            split=DatasetSplit.TRAIN,
            grouping_id="GRP-TRAIN-B",
        ),
        DatasetImageRecord(
            record_id="REC-TR-004",
            image_id="IMG-TR-004",
            image_path_or_ref="images/train_ai_2.jpg",
            sha256="4" * 64,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=True,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            generator_name="adobe-firefly-v2",
            generator_version="2.1",
            split=DatasetSplit.TRAIN,
            grouping_id="GRP-TRAIN-B",
        ),
        # Validation (Group C & D)
        DatasetImageRecord(
            record_id="REC-VAL-001",
            image_id="IMG-VAL-001",
            image_path_or_ref="images/val_cam_1.jpg",
            sha256="5" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.NONE,
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=True,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            split=DatasetSplit.VAL,
            grouping_id="GRP-VAL-C",
        ),
        DatasetImageRecord(
            record_id="REC-VAL-002",
            image_id="IMG-VAL-002",
            image_path_or_ref="images/val_ai_1.jpg",
            sha256="6" * 64,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=True,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            generator_name="adobe-firefly-v2",
            generator_version="v5",
            split=DatasetSplit.VAL,
            grouping_id="GRP-VAL-D",
        ),
        # Internal Test (Group E & F - strictly isolated from train/val)
        DatasetImageRecord(
            record_id="REC-TST-001",
            image_id="IMG-TST-001",
            image_path_or_ref="images/test_cam_1.jpg",
            sha256="7" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.NONE,
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=False,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            split=DatasetSplit.INTERNAL_TEST,
            grouping_id="GRP-TEST-E",
        ),
        DatasetImageRecord(
            record_id="REC-TST-002",
            image_id="IMG-TST-002",
            image_path_or_ref="images/test_ai_1.jpg",
            sha256="8" * 64,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            dataset_name="TestCorpus",
            source_creator="TestAuthor",
            source_url="https://safeher.org/datasets/test",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=False,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            generator_name="adobe-firefly-v2",
            generator_version="v5",
            split=DatasetSplit.INTERNAL_TEST,
            grouping_id="GRP-TEST-F",
        ),
    ]

    manifest = DatasetManifest(
        manifest_id="MNF-M44-TEST-001",
        description="Synthetic legally approved test manifest for training pipeline tests",
        dataset_name="SafeHER-M4-Synthetic-Test",
        dataset_version="1.0.0",
        records=records,
        attribution=AttributionMetadata(
            required=False,
            source_dataset_name="SafeHER-Synthetic",
            original_provider="SafeHER-Engineering",
            curator_contact="engineering@safeher.org",
        ),
    )

    return manifest, images


# ============================================================================
# 1. Tiny Fixture Training Run Test
# ============================================================================

def test_tiny_fixture_training_run():
    """Verify end-to-end execution of a tiny fixture training run."""
    manifest, images = create_test_manifest_and_images()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = str(Path(tmpdir) / "checkpoints")
        log_dir = str(Path(tmpdir) / "logs")

        config = TrainingPipelineConfig(
            architecture="mobilenet_v3_large",
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            seed=42,
        )

        pipeline = ModelATrainingPipeline(
            config=config,
            manifest=manifest,
            in_memory_images=images,
        )

        result = pipeline.run()

        assert isinstance(result, TrainingPipelineResult)
        assert result.total_epochs_trained == 2
        assert result.completion_status == "COMPLETED"
        assert result.best_checkpoint_path is not None
        assert Path(result.best_checkpoint_path).is_file()
        assert len(result.best_checkpoint_sha256) == 64
        assert len(result.history) == 2
        assert result.history[0]["train_loss"] > 0.0
        assert pipeline.model.is_trained is True


# ============================================================================
# 2. Deterministic Seed Behaviour Test
# ============================================================================

def test_deterministic_seed_behaviour():
    """Verify that two training runs with identical seeds produce matching weights and loss."""
    manifest, images = create_test_manifest_and_images()

    losses_run1 = []
    losses_run2 = []

    with tempfile.TemporaryDirectory() as tmp1:
        config1 = TrainingPipelineConfig(
            architecture="mobilenet_v3_large",
            epochs=1,
            batch_size=2,
            checkpoint_dir=str(Path(tmp1) / "ckpt"),
            log_dir=str(Path(tmp1) / "logs"),
            seed=100,
        )
        p1 = ModelATrainingPipeline(config=config1, manifest=manifest, in_memory_images=images)
        r1 = p1.run()
        losses_run1 = [h["train_loss"] for h in r1.history]
        weights_p1 = [p.clone() for p in p1.model.parameters()]

    with tempfile.TemporaryDirectory() as tmp2:
        config2 = TrainingPipelineConfig(
            architecture="mobilenet_v3_large",
            epochs=1,
            batch_size=2,
            checkpoint_dir=str(Path(tmp2) / "ckpt"),
            log_dir=str(Path(tmp2) / "logs"),
            seed=100,
        )
        p2 = ModelATrainingPipeline(config=config2, manifest=manifest, in_memory_images=images)
        r2 = p2.run()
        losses_run2 = [h["train_loss"] for h in r2.history]
        weights_p2 = [p.clone() for p in p2.model.parameters()]

    # Identical losses and parameters
    assert losses_run1 == losses_run2
    for w1, w2 in zip(weights_p1, weights_p2):
        torch.testing.assert_close(w1, w2, atol=1e-5, rtol=1e-5)


# ============================================================================
# 3. Checkpoint Save/Load & Resume Tests
# ============================================================================

def test_checkpoint_save_and_resume():
    """Verify checkpoint save, resume-from-checkpoint, and completion of remaining epochs."""
    manifest, images = create_test_manifest_and_images()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = str(Path(tmpdir) / "checkpoints")
        log_dir = str(Path(tmpdir) / "logs")

        # Phase 1: Train 2 epochs
        config1 = TrainingPipelineConfig(
            architecture="mobilenet_v3_large",
            epochs=2,
            batch_size=2,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            seed=42,
        )
        p1 = ModelATrainingPipeline(config=config1, manifest=manifest, in_memory_images=images)
        r1 = p1.run()
        assert r1.total_epochs_trained == 2

        latest_ckpt = Path(ckpt_dir) / "mobilenet_v3_large_latest.pt"
        assert latest_ckpt.is_file()

        # Phase 2: Resume for 2 more epochs (total 4 epochs)
        config2 = TrainingPipelineConfig(
            architecture="mobilenet_v3_large",
            epochs=4,
            batch_size=2,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            seed=42,
        )
        p2 = ModelATrainingPipeline(config=config2, manifest=manifest, in_memory_images=images)
        r2 = p2.run(resume_from_checkpoint=latest_ckpt)

        # Resumed run should complete epochs 3 and 4
        assert r2.total_epochs_trained == 2
        assert len(p2.logger.history) == 2
        assert p2.logger.history[0]["epoch"] == 3
        assert p2.logger.history[1]["epoch"] == 4


# ============================================================================
# 4. Manifest Safety & Rights Validation Tests
# ============================================================================

def test_reject_research_only_in_commercial_training():
    """Verify pipeline rejects research-only dataset records when target_workload is PRODUCTION_TRAINING."""
    manifest, images = create_test_manifest_and_images()

    # Modify one training record to be research-only
    bad_rec = manifest.records[0].model_copy(update={
        "commercial_use_status": CommercialUseStatus.PROHIBITED,
        "rights_status": RightsApprovalStatus.RESEARCH_ONLY,
    })
    mod_records = [bad_rec] + list(manifest.records[1:])
    mod_manifest = manifest.model_copy(update={"records": mod_records})

    config = TrainingPipelineConfig(
        target_workload=TargetUsageScope.PRODUCTION_TRAINING,
        enforce_approved_manifest=True,
    )

    with pytest.raises(ValueError, match="Commercial Training Safety Gate Failed"):
        ModelATrainingPipeline(config=config, manifest=mod_manifest, in_memory_images=images)


def test_quarantine_private_survivor_case_evidence():
    """Verify pipeline unconditionally rejects private survivor case evidence."""
    manifest, images = create_test_manifest_and_images()

    # Inject private survivor reference into image_path_or_ref
    bad_rec = manifest.records[0].model_copy(update={
        "image_path_or_ref": "cases/2026/survivor_upload_001.jpg",
    })
    mod_records = [bad_rec] + list(manifest.records[1:])
    mod_manifest = manifest.model_copy(update={"records": mod_records})

    config = TrainingPipelineConfig(enforce_approved_manifest=True)

    with pytest.raises(ValueError, match="Private survivor case evidence detected"):
        ModelATrainingPipeline(config=config, manifest=mod_manifest, in_memory_images=images)


def test_no_test_split_leakage():
    """Verify pipeline rejects training manifests where train records share grouping_id with test records."""
    manifest, images = create_test_manifest_and_images()

    # Cause forbidden identity/group leakage: Train record gets test grouping_id
    bad_rec = manifest.records[0].model_copy(update={
        "grouping_id": "GRP-TEST-E",
    })
    mod_records = [bad_rec] + list(manifest.records[1:])
    mod_manifest = manifest.model_copy(update={"records": mod_records})

    config = TrainingPipelineConfig(enforce_approved_manifest=True)

    with pytest.raises(ValueError, match="Test Leakage Violation"):
        ModelATrainingPipeline(config=config, manifest=mod_manifest, in_memory_images=images)


# ============================================================================
# 5. Checkpoint Metadata Validation Tests
# ============================================================================

def test_checkpoint_metadata_validation():
    """Verify checkpoint contains complete manifest hashes, environment, and class mapping."""
    manifest, images = create_test_manifest_and_images()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = str(Path(tmpdir) / "checkpoints")
        log_dir = str(Path(tmpdir) / "logs")

        config = TrainingPipelineConfig(
            architecture="mobilenet_v3_large",
            epochs=1,
            batch_size=2,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            seed=42,
        )
        p = ModelATrainingPipeline(config=config, manifest=manifest, in_memory_images=images)
        p.run()

        latest_ckpt = Path(ckpt_dir) / "mobilenet_v3_large_latest.pt"
        payload = torch.load(str(latest_ckpt), map_location="cpu", weights_only=False)

        metadata = payload.get("metadata", {})
        assert metadata["architecture"] == "mobilenet_v3_large"
        assert len(metadata["training_manifest_hash"]) == 64
        assert len(metadata["split_manifest_hash"]) == 64
        assert metadata["random_seed"] == 42
        assert "training_environment" in metadata
        assert metadata["training_environment"]["torch_version"] == torch.__version__
        assert metadata["class_mapping"][0] == "CAMERA_PHOTOGRAPH"
        assert metadata["class_mapping"][1] == "AI_GENERATED_IMAGE"


# ============================================================================
# 6. Early Stopping & Mixed Precision Guard Tests
# ============================================================================

def test_early_stopping_triggered():
    """Verify early stopping triggers when validation loss plateaus."""
    manifest, images = create_test_manifest_and_images()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingPipelineConfig(
            architecture="mobilenet_v3_large",
            epochs=10,
            batch_size=2,
            early_stopping_patience=1,
            early_stopping_min_delta=100.0,  # Unattainable delta guarantees early stopping
            checkpoint_dir=str(Path(tmpdir) / "ckpt"),
            log_dir=str(Path(tmpdir) / "logs"),
        )
        pipeline = ModelATrainingPipeline(config=config, manifest=manifest, in_memory_images=images)
        result = pipeline.run()

        assert result.early_stopped is True
        assert result.completion_status == "EARLY_STOPPED"
        assert result.total_epochs_trained == 2  # Epoch 1 (initial best), Epoch 2 (no improvement >= 100 -> stops)


def test_mixed_precision_cpu_guard():
    """Verify that requesting mixed precision on CPU safely disables AMP without crashing."""
    manifest, images = create_test_manifest_and_images()

    config = TrainingPipelineConfig(
        device="cpu",
        use_mixed_precision=True,
    )
    pipeline = ModelATrainingPipeline(config=config, manifest=manifest, in_memory_images=images)
    assert pipeline.use_amp is False
    assert pipeline.scaler is None
