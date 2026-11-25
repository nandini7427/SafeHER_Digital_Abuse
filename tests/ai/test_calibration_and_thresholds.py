"""Unit tests for SafeHER Score Calibration, Reliability, and Threshold Validation."""

import math
from typing import List

import numpy as np
import pytest

from ai_core.evaluation.calibration import (
    IsotonicRegressionCalibrator,
    ModelOutputNature,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
    audit_calibration,
    compute_brier_score,
    compute_nll,
    compute_reliability_diagram,
    determine_output_nature,
    optimize_decision_thresholds,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector


def test_determine_output_nature_uncalibrated_heuristic():
    """Verify heuristic scores in [0, 1] are classified as uncalibrated scores, not probabilities."""
    scores = [0.10, 0.35, 0.70, 0.10]
    nature = determine_output_nature(scores, is_explicitly_calibrated=False)

    assert nature["output_nature"] == ModelOutputNature.UNCALIBRATED_SCORES
    assert nature["is_calibrated"] is False
    assert "NOT posterior probabilities" in nature["justification"]


def test_determine_output_nature_unbounded_logits():
    """Verify scores outside [0, 1] are classified as logits."""
    scores = [-2.5, 0.4, 3.8]
    nature = determine_output_nature(scores)

    assert nature["output_nature"] == ModelOutputNature.LOGITS
    assert nature["is_calibrated"] is False


def test_brier_score_and_nll_calculations():
    """Verify Brier score and Negative Log-Likelihood exact values."""
    y_true = [True, False, True, False]
    # Perfect predictions
    perf_scores = [1.0, 0.0, 1.0, 0.0]
    assert compute_brier_score(y_true, perf_scores) == 0.0
    assert compute_nll(y_true, perf_scores) < 1e-4

    # Imperfect predictions
    scores = [0.8, 0.2, 0.6, 0.4]
    expected_brier = ((0.8 - 1) ** 2 + (0.2 - 0) ** 2 + (0.6 - 1) ** 2 + (0.4 - 0) ** 2) / 4.0
    assert abs(compute_brier_score(y_true, scores) - expected_brier) < 1e-6


def test_reliability_diagram_binning():
    """Verify 10-bin partitioning and calibration gap calculation."""
    y_true = [True, True, False, False]
    scores = [0.15, 0.15, 0.85, 0.85]  # Highly inverted/overconfident

    ece, mce, bins = compute_reliability_diagram(y_true, scores, num_bins=10)
    assert len(bins) == 2
    assert ece > 0.0
    assert mce > 0.0


def test_temperature_scaling_fitting():
    """Verify temperature scaling calibrator fits optimal T on validation data."""
    val_true = [True] * 10 + [False] * 10
    val_scores = [0.85] * 10 + [0.15] * 10  # Well-separated

    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(val_scores, val_true)

    assert calibrator.is_fitted is True
    assert calibrator.temperature > 0.0

    test_scores = [0.85, 0.15]
    calibrated = calibrator.predict_proba(test_scores)
    assert len(calibrated) == 2
    assert 0.0 <= calibrated[0] <= 1.0
    assert 0.0 <= calibrated[1] <= 1.0


def test_platt_scaling_fitting():
    """Verify Platt scaling calibrator fits logistic sigmoid parameters."""
    val_true = [True] * 15 + [False] * 15
    val_scores = [0.75] * 15 + [0.25] * 15

    calibrator = PlattScalingCalibrator()
    calibrator.fit(val_scores, val_true)

    assert calibrator.is_fitted is True
    calibrated = calibrator.predict_proba([0.8, 0.2])
    assert len(calibrated) == 2
    assert calibrated[0] > calibrated[1]  # Monotonic ordering preserved


def test_isotonic_regression_monotonicity():
    """Verify PAVA enforces non-decreasing monotonic calibration."""
    val_true = [False, True, False, True, True]
    val_scores = [0.1, 0.2, 0.3, 0.7, 0.9]

    calibrator = IsotonicRegressionCalibrator()
    calibrator.fit(val_scores, val_true)

    assert calibrator.is_fitted is True
    test_inputs = [0.05, 0.15, 0.25, 0.50, 0.85, 0.95]
    calibrated = calibrator.predict_proba(test_inputs)

    # Monotonicity check: p(s_i) <= p(s_{i+1})
    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i + 1] + 1e-6


def test_threshold_selection_low_fpr():
    """Verify decision threshold optimization selects FPR <= 5% for survivor protection."""
    # Synthetic validation data: 20 authentic (scores 0.1 to 0.4), 20 manipulated (scores 0.3 to 0.9)
    rng = np.random.RandomState(42)
    val_true = [False] * 20 + [True] * 20
    val_scores = list(rng.uniform(0.05, 0.30, 20)) + list(rng.uniform(0.40, 0.90, 20))

    report = optimize_decision_thresholds(val_true, val_scores)

    assert report.low_fpr_threshold.false_positive_rate <= 0.05
    assert report.low_fpr_threshold.threshold >= 0.30
    assert report.balanced_threshold is not None
    assert report.high_sensitivity_threshold is not None


def test_safeher_detector_preserves_uncalibrated_contract():
    """Verify SafeHERImageDetector outputs maintain is_calibrated=False and confidence_score=None."""
    detector = SafeHERImageDetector()
    from PIL import Image
    import io
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)

    resp = detector.detect(buf.getvalue())
    assert resp.is_calibrated is False
    assert resp.confidence_score is None
    assert resp.raw_model_score >= 0.0
