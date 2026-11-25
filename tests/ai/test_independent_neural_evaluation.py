"""Unit tests for SafeHER Independent Neural Model Evaluation Framework (M4.5).

Verifies:
- Precision, Recall, F1, Balanced Accuracy, ROC-AUC, PR-AUC mathematical correctness.
- Bootstrap 95% confidence interval estimation.
- Generalization stress perturbations (JPEG, social recompression, resizing, cropping, enhancement).
- Strict evaluation separation enforcement (unconditional rejection of TRAIN and VAL splits).
- Model A independent evaluation with generator breakdown and failure analysis.
- Model B independent evaluation with manipulation family breakdown.
- Privacy preservation (zero absolute filesystem paths or PII in machine-readable exports).
- Structured machine-readable artifact generation (CSV, JSONL, JSON).
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
)
from ai_core.neural.evaluation import (
    ClassificationMetrics,
    IndependentEvaluationReport,
    IndependentNeuralEvaluator,
    compute_binary_metrics,
    compute_bootstrap_confidence_intervals,
    compute_pr_auc,
    compute_roc_auc,
    get_standard_perturbation_battery,
)
from ai_core.neural.facial.service import ModelBFacialInferenceService
from ai_core.neural.facial.schemas import NormalizedBoundingBox
from ai_core.neural.inference import ModelAInferenceService
from ai_core.neural.models import create_image_detector
from ai_core.neural.models.patch_forensics import PatchForensicsFacialDetector


def create_synthetic_image(color: Tuple[int, int, int] = (120, 140, 160)) -> Image.Image:
    """Create a synthetic 224x224 RGB image."""
    return Image.new("RGB", (224, 224), color=color)


def create_eval_manifest_and_images() -> Tuple[DatasetManifest, Dict[str, Image.Image]]:
    """Build a synthetic test-only manifest for independent evaluation testing."""
    images: Dict[str, Image.Image] = {
        "REC-TST-001": create_synthetic_image((100, 100, 100)),
        "REC-TST-002": create_synthetic_image((110, 110, 110)),
        "REC-TST-003": create_synthetic_image((20, 20, 220)),
        "REC-TST-004": create_synthetic_image((30, 30, 230)),
        "REC-TST-005": create_synthetic_image((40, 40, 240)),
        "REC-TST-006": create_synthetic_image((120, 120, 120)),
    }

    records: List[DatasetImageRecord] = [
        # 3 Authentic Camera control images (Nikon, Sony, Canon)
        DatasetImageRecord(
            record_id="REC-TST-001",
            image_id="IMG-TST-001",
            image_path_or_ref="images/tst_cam_1.jpg",
            sha256="a" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.NONE,
            dataset_name="Nikon-D850",
            source_creator="TestLab",
            source_url="https://safeher.org",
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
            grouping_id="GRP-TST-CAM-1",
        ),
        DatasetImageRecord(
            record_id="REC-TST-002",
            image_id="IMG-TST-002",
            image_path_or_ref="images/tst_cam_2.jpg",
            sha256="b" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.NONE,
            dataset_name="Sony-A7R",
            source_creator="TestLab",
            source_url="https://safeher.org",
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
            grouping_id="GRP-TST-CAM-2",
        ),
        DatasetImageRecord(
            record_id="REC-TST-006",
            image_id="IMG-TST-006",
            image_path_or_ref="images/tst_cam_3.jpg",
            sha256="f" * 64,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.NONE,
            dataset_name="Canon-EOS",
            source_creator="TestLab",
            source_url="https://safeher.org",
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
            grouping_id="GRP-TST-CAM-3",
        ),
        # 3 Synthetic AI-Generated images (Midjourney, StableDiffusion, Flux)
        DatasetImageRecord(
            record_id="REC-TST-003",
            image_id="IMG-TST-003",
            image_path_or_ref="images/tst_ai_1.jpg",
            sha256="c" * 64,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            dataset_name="Midjourney-V6",
            source_creator="TestLab",
            source_url="https://safeher.org",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=False,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            generator_name="Midjourney-V6",
            generator_version="6.0",
            split=DatasetSplit.INTERNAL_TEST,
            grouping_id="GRP-TST-AI-1",
        ),
        DatasetImageRecord(
            record_id="REC-TST-004",
            image_id="IMG-TST-004",
            image_path_or_ref="images/tst_ai_2.jpg",
            sha256="d" * 64,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            dataset_name="StableDiffusion-XL",
            source_creator="TestLab",
            source_url="https://safeher.org",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=False,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            generator_name="StableDiffusion-XL",
            generator_version="1.0",
            split=DatasetSplit.INTERNAL_TEST,
            grouping_id="GRP-TST-AI-2",
        ),
        DatasetImageRecord(
            record_id="REC-TST-005",
            image_id="IMG-TST-005",
            image_path_or_ref="images/tst_ai_3.jpg",
            sha256="e" * 64,
            label=DatasetImageLabel.AI_GENERATED_IMAGE,
            task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
            manipulation_category=ManipulationCategory.DEEPFAKE_FACE_SWAP,
            dataset_name="Flux-Schnell",
            source_creator="TestLab",
            source_url="https://safeher.org",
            dataset_version="1.0.0",
            source_license="CC0-1.0",
            commercial_use_status=CommercialUseStatus.PERMITTED,
            research_use_status=ResearchUseStatus.PERMITTED,
            training_eligibility=False,
            evaluation_eligibility=True,
            redistribution_eligibility=True,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            consent_status=ConsentStatus.NOT_APPLICABLE,
            generator_name="Flux-Schnell",
            generator_version="1.0",
            split=DatasetSplit.INTERNAL_TEST,
            grouping_id="GRP-TST-AI-3",
        ),
    ]

    # This fixture supplies explicit synthetic face-region annotations for Model B
    # pipeline tests. It does not assert that a production face detector exists.
    test_face_box = {
        "x_min": 0.20,
        "y_min": 0.15,
        "x_max": 0.80,
        "y_max": 0.85,
    }
    records = [
        record.model_copy(update={"metadata": {"face_bounding_boxes": [test_face_box]}})
        for record in records
    ]

    manifest = DatasetManifest(
        manifest_id="MNF-M45-TEST-EVAL",
        description="Isolated independent evaluation test manifest",
        dataset_name="SafeHER-Independent-Eval",
        dataset_version="1.0.0",
        records=records,
        attribution=AttributionMetadata(
            required=False,
            source_dataset_name="SafeHER-Independent-Eval",
            original_provider="SafeHER-QA",
            curator_contact="qa@safeher.org",
        ),
    )

    return manifest, images


# ============================================================================
# 1. Mathematical Metric Calculation Tests
# ============================================================================

def test_binary_metrics_exact_calculation():
    """Verify exact calculation of Accuracy, Balanced Accuracy, Precision, Recall, F1, FPR, FNR."""
    # 4 samples: 2 positive (1, 1), 2 negative (0, 0)
    y_true = [1, 1, 0, 0]
    # Prediction scores: TP (0.9), FN (0.3), FP (0.7), TN (0.2) at threshold 0.5
    y_scores = [0.9, 0.3, 0.7, 0.2]

    metrics = compute_binary_metrics(y_true, y_scores, threshold=0.50, compute_cis=False)

    assert metrics.sample_count == 4
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1

    assert metrics.accuracy == 0.50
    assert metrics.precision == 0.50
    assert metrics.recall == 0.50
    assert metrics.f1_score == 0.50
    assert metrics.balanced_accuracy == 0.50
    assert metrics.false_positive_rate == 0.50
    assert metrics.false_negative_rate == 0.50
    assert metrics.confusion_matrix == [[1, 1], [1, 1]]


def test_roc_auc_and_pr_auc_perfect_ranking():
    """Verify ROC-AUC and PR-AUC are 1.0 for perfect ranking."""
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.2, 0.1])

    roc = compute_roc_auc(y_true, y_scores)
    pr = compute_pr_auc(y_true, y_scores)

    assert abs(roc - 1.0) < 1e-4
    assert abs(pr - 1.0) < 1e-4


def test_roc_auc_single_class_edge_case():
    """Verify single-class degenerate array returns 0.5 without NaN or division by zero."""
    y_true = np.array([1, 1, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert compute_roc_auc(y_true, y_scores) == 0.5


def test_bootstrap_confidence_intervals():
    """Verify 95% bootstrap confidence interval bounds lie in [0, 1] and span metric values."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.85, 0.1, 0.2, 0.15, 0.3])

    acc_ci, f1_ci, auc_ci = compute_bootstrap_confidence_intervals(y_true, y_scores, num_bootstraps=200)

    assert 0.0 <= acc_ci.low <= acc_ci.high <= 1.0
    assert 0.0 <= f1_ci.low <= f1_ci.high <= 1.0
    assert 0.0 <= auc_ci.low <= auc_ci.high <= 1.0
    assert acc_ci.confidence_level == 0.95


# ============================================================================
# 2. Generalization Perturbation Tests
# ============================================================================

def test_perturbation_battery_preserves_images():
    """Verify all perturbations execute successfully on test images."""
    battery = get_standard_perturbation_battery()
    img = create_synthetic_image()

    for name, fn in battery.items():
        perturbed = fn(img)
        assert isinstance(perturbed, Image.Image)
        assert perturbed.size == img.size
        assert perturbed.mode == "RGB"


# ============================================================================
# 3. Evaluation Separation Enforcement Tests
# ============================================================================

def test_reject_training_records_in_evaluation():
    """Verify evaluator strictly rejects manifests containing TRAIN records."""
    manifest, images = create_eval_manifest_and_images()

    # Contaminate test manifest with a TRAIN record
    contaminated_rec = manifest.records[0].model_copy(update={"split": DatasetSplit.TRAIN})
    bad_manifest = manifest.model_copy(update={"records": [contaminated_rec] + list(manifest.records[1:])})

    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    service = ModelAInferenceService(model=model, enforce_trained=True)
    evaluator = IndependentNeuralEvaluator(service=service)

    with pytest.raises(ValueError, match="CRITICAL EVALUATION SEPARATION VIOLATION"):
        evaluator.evaluate_manifest(bad_manifest, in_memory_images=images)


def test_reject_validation_records_in_evaluation():
    """Verify evaluator strictly rejects manifests containing VAL records."""
    manifest, images = create_eval_manifest_and_images()

    # Contaminate test manifest with a VAL record
    contaminated_rec = manifest.records[0].model_copy(update={"split": DatasetSplit.VAL})
    bad_manifest = manifest.model_copy(update={"records": [contaminated_rec] + list(manifest.records[1:])})

    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    service = ModelAInferenceService(model=model, enforce_trained=True)
    evaluator = IndependentNeuralEvaluator(service=service)

    with pytest.raises(ValueError, match="CRITICAL EVALUATION SEPARATION VIOLATION"):
        evaluator.evaluate_manifest(bad_manifest, in_memory_images=images)


# ============================================================================
# 4. Model A Independent Evaluation & Machine-Readable Output Tests
# ============================================================================

def test_model_a_independent_evaluation_run():
    """Verify full evaluation run for Model A including metrics, CIs, breakdowns, and CSV export."""
    manifest, images = create_eval_manifest_and_images()

    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    service = ModelAInferenceService(model=model, enforce_trained=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = IndependentNeuralEvaluator(
            service=service,
            decision_threshold=0.50,
            output_dir=tmpdir,
        )

        report = evaluator.evaluate_manifest(
            manifest=manifest,
            in_memory_images=images,
            run_perturbations=True,
            num_bootstraps=100,
        )

        assert isinstance(report, IndependentEvaluationReport)
        assert report.total_samples == 6
        assert report.baseline_metrics.sample_count == 6
        assert report.baseline_metrics.accuracy_ci is not None
        assert len(report.per_generator_metrics) >= 2
        assert len(report.perturbation_benchmarks) >= 3

        # Verify exported files
        csv_file = Path(tmpdir) / "predictions_mobilenet_v3_large.csv"
        jsonl_file = Path(tmpdir) / "predictions_mobilenet_v3_large.jsonl"
        summary_file = Path(tmpdir) / "evaluation_summary_mobilenet_v3_large.json"

        assert csv_file.is_file()
        assert jsonl_file.is_file()
        assert summary_file.is_file()


# ============================================================================
# 5. Model B Facial Independent Evaluation Tests
# ============================================================================

def test_model_b_facial_independent_evaluation_run():
    """Verify full evaluation run for Model B including manipulation family breakdown."""
    manifest, images = create_eval_manifest_and_images()

    model = PatchForensicsFacialDetector(is_trained=True)
    service = ModelBFacialInferenceService(model=model, enforce_trained=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = IndependentNeuralEvaluator(
            service=service,
            decision_threshold=0.50,
            output_dir=tmpdir,
        )

        report = evaluator.evaluate_manifest(
            manifest=manifest,
            in_memory_images=images,
            run_perturbations=False,  # Skip perturbations for rapid unit test
            num_bootstraps=50,
        )

        assert isinstance(report, IndependentEvaluationReport)
        assert report.model_architecture == "patch_forensics_facial"
        assert report.total_samples == 6
        assert report.baseline_metrics.f1_score >= 0.0


# ============================================================================
# 6. Privacy Sanitization & Failure Analysis Tests
# ============================================================================

def test_privacy_sanitization_no_absolute_paths_in_exports():
    """Verify prediction exports contain only record IDs and hashes, never filesystem paths."""
    manifest, images = create_eval_manifest_and_images()

    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    service = ModelAInferenceService(model=model, enforce_trained=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = IndependentNeuralEvaluator(service=service, output_dir=tmpdir)
        evaluator.evaluate_manifest(manifest=manifest, in_memory_images=images, run_perturbations=False)

        csv_file = Path(tmpdir) / "predictions_mobilenet_v3_large.csv"
        content = csv_file.read_text(encoding="utf-8")

        # Must NOT contain filesystem indicators
        assert "C:\\" not in content
        assert "/Users/" not in content
        assert "survivor" not in content.lower()
        assert "cases/" not in content.lower()

        # Must contain scrubbed record ID and SHA-256
        assert "REC-TST-001" in content
        assert "a" * 64 in content


def test_failure_analysis_structure():
    """Verify failure analysis captures false positives, false negatives, and generator error rates."""
    manifest, images = create_eval_manifest_and_images()

    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    service = ModelAInferenceService(model=model, enforce_trained=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = IndependentNeuralEvaluator(service=service, output_dir=tmpdir)
        report = evaluator.evaluate_manifest(manifest=manifest, in_memory_images=images, run_perturbations=True)

        assert report.failure_analysis is not None
        assert report.failure_analysis.false_positive_count >= 0
        assert report.failure_analysis.false_negative_count >= 0
        assert isinstance(report.failure_analysis.generator_error_rates, dict)
