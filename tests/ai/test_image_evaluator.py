"""Unit and regression tests for SafeHER Image Detection model evaluation pipeline."""

import pytest

from ai_core.evaluation.image_dataset import (
    ImageBenchmarkSample,
    apply_perturbation,
    build_forensic_benchmark_dataset,
    generate_authentic_image,
    generate_manipulated_image,
)
from ai_core.evaluation.image_evaluator import (
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_roc_auc,
    evaluate_image_detector,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector


def test_roc_auc_computation_cases():
    """Verify ROC-AUC rank statistic for perfect, inverted, tied, and uniform scores."""
    # 1. Perfect separation
    y_true = [True, True, False, False]
    scores_perfect = [0.95, 0.85, 0.15, 0.05]
    assert compute_roc_auc(y_true, scores_perfect) == 1.0

    # 2. Inverted separation
    scores_inverted = [0.05, 0.15, 0.85, 0.95]
    assert compute_roc_auc(y_true, scores_inverted) == 0.0

    # 3. Tied scores
    scores_tied = [0.5, 0.5, 0.5, 0.5]
    assert compute_roc_auc(y_true, scores_tied) == 0.5

    # 4. Single class fallback
    assert compute_roc_auc([True, True], [0.8, 0.9]) == 0.5
    assert compute_roc_auc([False, False], [0.2, 0.3]) == 0.5

    # 5. Invalid inputs
    with pytest.raises(ValueError, match="same length"):
        compute_roc_auc([True], [0.5, 0.6])
    with pytest.raises(ValueError, match="empty dataset"):
        compute_roc_auc([], [])


def test_compute_classification_metrics():
    """Verify precision, recall, F1, accuracy, FPR, FNR, and confusion matrix."""
    y_true = [True, True, False, False]
    y_pred = [True, False, False, True]  # TP=1, FN=1, TN=1, FP=1
    y_scores = [0.9, 0.3, 0.2, 0.8]

    metrics = compute_classification_metrics(y_true, y_pred, y_scores, threshold=0.5)

    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5
    assert metrics.false_positive_rate == 0.5
    assert metrics.false_negative_rate == 0.5
    assert metrics.decision_threshold == 0.5


def test_compute_classification_metrics_empty_or_mismatched():
    """Verify ValueError is raised on mismatched or empty inputs."""
    with pytest.raises(ValueError, match="equal lengths"):
        compute_classification_metrics([True], [True, False], [0.5])
    with pytest.raises(ValueError, match="empty dataset"):
        compute_classification_metrics([], [], [])


def test_calibration_metrics_calculation():
    """Verify Brier score, ECE, and reliability binning."""
    y_true = [True, True, False, False]
    scores = [0.9, 0.8, 0.2, 0.1]

    cal = compute_calibration_metrics(y_true, scores, num_bins=10)
    assert not cal.is_calibrated
    assert cal.brier_score >= 0.0
    assert 0.0 <= cal.expected_calibration_error <= 1.0
    assert 0.0 <= cal.maximum_calibration_error <= 1.0
    assert len(cal.reliability_bins) > 0
    assert "UNCALIBRATED" in cal.calibration_assessment


def test_dataset_split_separation_and_no_leakage():
    """Verify that dataset splits have disjoint sample IDs and distinct seeds."""
    dataset = build_forensic_benchmark_dataset(num_train=10, num_val=10, num_test=10)
    assert len(dataset) == 30

    train_ids = {s.sample_id for s in dataset if s.split == "train"}
    val_ids = {s.sample_id for s in dataset if s.split == "val"}
    test_ids = {s.sample_id for s in dataset if s.split == "test"}

    # No ID overlap
    assert len(train_ids) == 10
    assert len(val_ids) == 10
    assert len(test_ids) == 10
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Images must be non-empty valid bytes
    for s in dataset:
        assert len(s.image_bytes) > 100
        assert s.width == 256
        assert s.height == 256


def test_robustness_perturbations_execute_cleanly():
    """Verify all 9 perturbation types produce valid output image bytes."""
    data, _ = generate_authentic_image(seed=42)

    perturbations = [
        "jpeg_q50",
        "jpeg_q75",
        "jpeg_q90",
        "resize_down_up",
        "center_crop",
        "gaussian_blur",
        "brightness_shift",
        "contrast_shift",
        "screenshot_composite",
    ]

    for p in perturbations:
        out = apply_perturbation(data, p)
        assert isinstance(out, bytes)
        assert len(out) > 100


def test_evaluate_image_detector_end_to_end():
    """Verify evaluate_image_detector executes end-to-end and returns complete report."""
    detector = SafeHERImageDetector()
    mini_dataset = build_forensic_benchmark_dataset(num_train=4, num_val=4, num_test=6)

    report = evaluate_image_detector(detector=detector, dataset=mini_dataset)

    assert report.total_test_samples == 6
    assert report.clean_test_metrics.accuracy >= 0.0
    assert report.latency_metrics.sample_count == 6
    assert report.latency_metrics.avg_latency_ms > 0.0
    assert len(report.robustness_results) == 9
    assert not report.calibration_metrics.is_calibrated
