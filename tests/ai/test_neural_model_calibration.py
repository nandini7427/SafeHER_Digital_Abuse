"""Unit and integration tests for SafeHER Neural Model Calibration and Safe Abstention.

Tests:
1. Conceptual separation (Logits vs Scores vs Probabilities vs Thresholds vs Uncertainty vs OOD).
2. Fitting algorithms (Temperature Scaling, Platt Scaling, Isotonic Regression PAVA).
3. Strict evaluation separation (rejecting calibration fitting on train/test splits).
4. Serialization and deserialization (JSON save/load, fidelity preservation).
5. Correct model-version linkage (rejection of mismatched model versions or architectures).
6. Missing calibration artifact handling (graceful fallback to MODEL_UNAVAILABLE).
7. Invalid calibration state (negative temperature, non-finite parameters).
8. Non-finite values guard (NaN, Inf rejection).
9. Distribution shift impact on ECE and Brier score.
10. Abstention behavior across all safe output states:
    - INDICATORS_DETECTED
    - NO_CLEAR_INDICATORS (verifying it NEVER certifies authenticity)
    - INCONCLUSIVE (borderline probability, high entropy, inter-model disagreement)
    - OUTSIDE_SUPPORTED_DOMAIN (OOD, unsupported input conditions)
    - MODEL_UNAVAILABLE (missing or incompatible calibration)
11. Stratified calibration by generator/manipulation category.
12. Backward compatibility with existing ModelInferenceResult schemas.
"""

import json
import math
from pathlib import Path
import tempfile
from typing import List

import numpy as np
import pytest

from ai_core.neural.calibration import (
    AbstentionPolicyConfig,
    AbstentionReason,
    BaseNeuralCalibrator,
    CalibratedAbstentionEngine,
    CalibratedPrediction,
    CalibrationAuditor,
    CalibratorMethod,
    InvalidCalibrationStateError,
    IsotonicRegressionCalibrator,
    ModelVersionMismatchError,
    OODStatus,
    OutputNature,
    PlattScalingCalibrator,
    SafeDecisionState,
    TemperatureScalingCalibrator,
    UncertaintyDecomposition,
    UnfittedCalibratorError,
    audit_calibration_metrics,
    compute_brier_score,
    compute_nll,
    compute_reliability_diagram,
    evaluate_calibration_by_category,
    evaluate_calibration_under_distribution_shift,
    validate_conceptual_separation,
)
from ai_core.neural.schemas import (
    InferenceStatus,
    InputConfiguration,
    ModelInferenceResult,
)


# ---------------------------------------------------------------------------
# 1. Conceptual Separation and Taxonomy Tests
# ---------------------------------------------------------------------------

def test_conceptual_separation_validations():
    """Verify that validation correctly enforces bounds for each conceptual output type."""
    # Logits can be any real number
    validate_conceptual_separation(OutputNature.RAW_LOGIT, -14.2)
    validate_conceptual_separation(OutputNature.RAW_LOGIT, 0.0)
    validate_conceptual_separation(OutputNature.RAW_LOGIT, 25.8)

    # Probabilities and scores must reside in [0, 1]
    validate_conceptual_separation(OutputNature.UNCALIBRATED_SCORE, 0.42)
    validate_conceptual_separation(OutputNature.CALIBRATED_PROBABILITY, 0.95)
    validate_conceptual_separation(OutputNature.DECISION_THRESHOLD, 0.80)

    # Uncertainty must be non-negative
    validate_conceptual_separation(OutputNature.MODEL_UNCERTAINTY, 0.85)

    # Violations
    with pytest.raises(ValueError, match="must reside in \\[0.0, 1.0\\]"):
        validate_conceptual_separation(OutputNature.CALIBRATED_PROBABILITY, 1.05)

    with pytest.raises(ValueError, match="must reside in \\[0.0, 1.0\\]"):
        validate_conceptual_separation(OutputNature.UNCALIBRATED_SCORE, -0.01)

    with pytest.raises(ValueError, match="Model uncertainty must be non-negative"):
        validate_conceptual_separation(OutputNature.MODEL_UNCERTAINTY, -0.1)

    # Non-finite rejection
    with pytest.raises(ValueError, match="Non-finite value"):
        validate_conceptual_separation(OutputNature.RAW_LOGIT, float("nan"))

    with pytest.raises(ValueError, match="Non-finite value"):
        validate_conceptual_separation(OutputNature.RAW_LOGIT, float("inf"))


# ---------------------------------------------------------------------------
# 2. Strict Partition Separation Tests
# ---------------------------------------------------------------------------

def test_calibrator_rejects_training_and_test_splits():
    """Verify calibrator raises error when fitting is attempted on train or test splits."""
    cal = TemperatureScalingCalibrator()
    dummy_logits = [0.5, -0.5, 1.2, -1.0]
    dummy_labels = [True, False, True, False]

    for forbidden in ["train", "training", "internal_test", "cross_dataset_test", "test"]:
        with pytest.raises(ValueError, match="STRICT SEPARATION VIOLATION"):
            cal.fit(dummy_logits, dummy_labels, calibration_split=forbidden)

    # Permitted split passes
    cal.fit(dummy_logits, dummy_labels, calibration_split="validation")
    assert cal.calibration_split == "validation"


# ---------------------------------------------------------------------------
# 3. Fitting Algorithms and Monotonicity Tests
# ---------------------------------------------------------------------------

def test_temperature_scaling_optimization():
    """Verify temperature scaling optimizes T > 0 on validation logits while preserving ranking."""
    rng = np.random.RandomState(42)
    # Overconfident model: model outputs high confidence (logits ~ +/- 3.5 = 97% confidence)
    # but empirical accuracy is only 75% (30/40 correct in each group).
    val_labels = [True] * 30 + [False] * 10 + [False] * 30 + [True] * 10
    val_logits = list(rng.normal(3.5, 0.4, 40)) + list(rng.normal(-3.5, 0.4, 40))

    cal = TemperatureScalingCalibrator(
        model_name="mobilenetv3_large_100",
        model_version="1.0.0",
    )
    cal.fit(val_logits, val_labels, calibration_split="val_split_a")

    assert cal.is_fitted is True
    assert cal.temperature > 1.0  # Temperature must increase to soften overconfident predictions

    test_logits = [-2.0, -0.5, 0.0, 0.5, 2.0]
    calibrated = cal.calibrate(test_logits)

    # Verify bounds and strict monotonic ordering
    assert all(0.0 <= p <= 1.0 for p in calibrated)
    for i in range(len(calibrated) - 1):
        assert calibrated[i] < calibrated[i + 1]


def test_platt_scaling_fitting():
    """Verify Platt scaling fits affine logistic parameters A and B."""
    val_labels = [True] * 30 + [False] * 30
    val_scores = [0.85] * 30 + [0.15] * 30

    cal = PlattScalingCalibrator(
        model_name="facial_patch_detector_v1",
        model_version="1.0.0",
    )
    cal.fit(val_scores, val_labels, calibration_split="validation")

    assert cal.is_fitted is True
    calibrated = cal.calibrate([0.9, 0.1])
    assert calibrated[0] > calibrated[1]
    assert 0.0 <= calibrated[0] <= 1.0
    assert 0.0 <= calibrated[1] <= 1.0


def test_isotonic_regression_monotonicity_guarantee():
    """Verify PAVA enforces non-decreasing monotonic calibration across all test inputs."""
    val_labels = [False, False, True, False, True, True, True, False, True, True] * 3
    val_scores = sorted(list(np.linspace(0.05, 0.95, 30)))

    cal = IsotonicRegressionCalibrator(
        model_name="mobilenetv3_large_100",
        model_version="1.0.0",
    )
    cal.fit(val_scores, val_labels, calibration_split="validation")

    assert cal.is_fitted is True
    dense_test = np.linspace(0.0, 1.0, 50)
    calibrated = cal.calibrate(dense_test)

    # Non-decreasing check
    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i + 1] + 1e-7


# ---------------------------------------------------------------------------
# 4. Serialization and Version Linkage Tests
# ---------------------------------------------------------------------------

def test_calibration_serialization_and_version_linkage():
    """Verify JSON save/load preserves state and enforces model-version linkage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "calibrator.json"

        orig_cal = TemperatureScalingCalibrator(
            temperature=1.45,
            model_name="mobilenetv3_large_100",
            model_version="1.0.0",
            architecture="mobilenet_v3_large",
            training_manifest_hash="abc123hash",
        )
        orig_cal.is_fitted = True
        orig_cal.calibration_split = "validation"
        orig_cal.fit_timestamp = "2026-11-25T00:00:00Z"
        orig_cal.validation_brier = 0.045
        orig_cal.validation_ece = 0.032
        orig_cal.save(save_path)

        assert save_path.exists()

        # Load back
        loaded_cal = TemperatureScalingCalibrator.load(save_path)
        assert loaded_cal.temperature == 1.45
        assert loaded_cal.model_name == "mobilenetv3_large_100"
        assert loaded_cal.model_version == "1.0.0"
        assert loaded_cal.training_manifest_hash == "abc123hash"
        assert loaded_cal.is_fitted is True

        # Test model-version linkage check: matching passes
        loaded_cal.verify_model_linkage("mobilenetv3_large_100", "1.0.0", "mobilenet_v3_large")

        # Mismatched model version raises ModelVersionMismatchError
        with pytest.raises(ModelVersionMismatchError, match="does not match target model version"):
            loaded_cal.verify_model_linkage("mobilenetv3_large_100", "2.0.0")

        # Mismatched architecture raises ModelVersionMismatchError
        with pytest.raises(ModelVersionMismatchError, match="does not match target architecture"):
            loaded_cal.verify_model_linkage("mobilenetv3_large_100", "1.0.0", "resnet50")


def test_unfitted_calibrator_rejection():
    """Verify unfitted calibrator rejects calibration and serialization."""
    cal = TemperatureScalingCalibrator()
    with pytest.raises(UnfittedCalibratorError):
        cal.calibrate([0.5])

    with pytest.raises(UnfittedCalibratorError):
        cal.save("dummy.json")


def test_invalid_calibration_state_rejection():
    """Verify negative temperature or non-finite parameters trigger InvalidCalibrationStateError."""
    with pytest.raises(InvalidCalibrationStateError, match="strictly positive finite number"):
        TemperatureScalingCalibrator(temperature=-0.5)

    with pytest.raises(InvalidCalibrationStateError):
        TemperatureScalingCalibrator(temperature=0.0)

    with pytest.raises(InvalidCalibrationStateError):
        TemperatureScalingCalibrator(temperature=float("nan"))

    with pytest.raises(InvalidCalibrationStateError):
        PlattScalingCalibrator(a=float("inf"), b=0.0)


# ---------------------------------------------------------------------------
# 5. Non-Finite Values and Input Safety Tests
# ---------------------------------------------------------------------------

def test_non_finite_values_handling_in_abstention():
    """Verify NaN or Inf in model outputs safely trigger INCONCLUSIVE with NON_FINITE_NUMERICAL_VALUE."""
    engine = CalibratedAbstentionEngine()

    res = engine.evaluate(raw_logit=float("nan"), uncalibrated_score=0.5)
    assert res.decision_state == SafeDecisionState.INCONCLUSIVE
    assert res.abstention_reason == AbstentionReason.NON_FINITE_NUMERICAL_VALUE

    res_inf = engine.evaluate(raw_logit=float("inf"), uncalibrated_score=0.5)
    assert res_inf.decision_state == SafeDecisionState.INCONCLUSIVE
    assert res_inf.abstention_reason == AbstentionReason.NON_FINITE_NUMERICAL_VALUE


# ---------------------------------------------------------------------------
# 6. Safe Output States & Abstention Engine Tests
# ---------------------------------------------------------------------------

def test_abstention_states_indicators_detected():
    """Verify high calibrated probability with low uncertainty produces INDICATORS_DETECTED."""
    cal = TemperatureScalingCalibrator(temperature=1.0)
    cal.is_fitted = True
    engine = CalibratedAbstentionEngine(calibrator=cal)

    res = engine.evaluate(
        calibrated_probability=0.92,
        uncalibrated_score=0.90,
        model_version="1.0.0",
    )
    assert res.decision_state == SafeDecisionState.INDICATORS_DETECTED
    assert res.abstention_reason == AbstentionReason.NONE
    assert res.calibrated_probability == 0.92


def test_abstention_states_no_clear_indicators_never_certifies_authenticity():
    """Verify low probability yields NO_CLEAR_INDICATORS and explicitly refuses authenticity certification."""
    cal = TemperatureScalingCalibrator(temperature=1.0)
    cal.is_fitted = True
    engine = CalibratedAbstentionEngine(calibrator=cal)

    res = engine.evaluate(
        calibrated_probability=0.08,
        uncalibrated_score=0.10,
        model_version="1.0.0",
    )
    assert res.decision_state == SafeDecisionState.NO_CLEAR_INDICATORS
    assert res.abstention_reason == AbstentionReason.NONE
    # Verify forensic disclaimer is present
    assert "Absence of detected manipulation indicators does NOT certify" in res.disclaimer
    # Verify limits documentation contains negative detection constraint
    assert any("NEVER a certification of authentic origin" in limit for limit in res.interpretation_limits)


def test_abstention_states_borderline_probability_triggers_inconclusive():
    """Verify probability in the indeterminate zone [tau_low, tau_high] triggers INCONCLUSIVE."""
    cal = TemperatureScalingCalibrator(temperature=1.0)
    cal.is_fitted = True
    config = AbstentionPolicyConfig(tau_low=0.20, tau_high=0.80)
    engine = CalibratedAbstentionEngine(config=config, calibrator=cal)

    # 0.50 is in (0.20, 0.80)
    res = engine.evaluate(calibrated_probability=0.50)
    assert res.decision_state == SafeDecisionState.INCONCLUSIVE
    # When probability is near 0.5, entropy is high (1.0 bit), which elevates uncertainty
    assert res.abstention_reason in (
        AbstentionReason.BORDERLINE_PROBABILITY,
        AbstentionReason.ELEVATED_ENTROPY_UNCERTAINTY,
    )


def test_abstention_states_inter_model_disagreement():
    """Verify divergence between Model A and Model B triggers INCONCLUSIVE with INTER_MODEL_DISAGREEMENT."""
    cal = TemperatureScalingCalibrator(temperature=1.0)
    cal.is_fitted = True
    config = AbstentionPolicyConfig(max_disagreement_threshold=0.30)
    engine = CalibratedAbstentionEngine(config=config, calibrator=cal)

    # High probability but severe disagreement (e.g. Model A says 0.85, Model B says 0.20 -> diff 0.65)
    res = engine.evaluate(
        calibrated_probability=0.85,
        disagreement_with_peer=0.65,
    )
    assert res.decision_state == SafeDecisionState.INCONCLUSIVE
    assert res.abstention_reason == AbstentionReason.INTER_MODEL_DISAGREEMENT


def test_abstention_states_out_of_distribution():
    """Verify OOD status triggers OUTSIDE_SUPPORTED_DOMAIN."""
    engine = CalibratedAbstentionEngine()
    ood = OODStatus(
        is_ood=True,
        domain_category="digital_illustration",
        reasons=["Detected vector art palette."],
    )

    res = engine.evaluate(
        calibrated_probability=0.88,
        ood_status=ood,
    )
    assert res.decision_state == SafeDecisionState.OUTSIDE_SUPPORTED_DOMAIN
    assert res.abstention_reason == AbstentionReason.OUT_OF_DISTRIBUTION


def test_abstention_states_unsupported_input_condition():
    """Verify unsupported conditions (e.g. extreme pose or corrupted file) trigger OUTSIDE_SUPPORTED_DOMAIN."""
    engine = CalibratedAbstentionEngine()

    res = engine.evaluate(
        calibrated_probability=0.85,
        input_conditions=["extreme_pose", "severe_occlusion"],
    )
    assert res.decision_state == SafeDecisionState.OUTSIDE_SUPPORTED_DOMAIN
    assert res.abstention_reason == AbstentionReason.UNSUPPORTED_INPUT_CONDITION


def test_abstention_states_missing_calibration_triggers_model_unavailable():
    """Verify missing calibration triggers MODEL_UNAVAILABLE."""
    # Engine without calibrator
    engine = CalibratedAbstentionEngine(calibrator=None)

    res = engine.evaluate(raw_logit=2.5, uncalibrated_score=0.92)
    assert res.decision_state == SafeDecisionState.MODEL_UNAVAILABLE
    assert res.abstention_reason == AbstentionReason.MISSING_CALIBRATION_ARTIFACT


def test_abstention_states_model_version_mismatch():
    """Verify calibrator version mismatch triggers MODEL_UNAVAILABLE with MODEL_VERSION_MISMATCH."""
    cal = TemperatureScalingCalibrator(
        model_name="mobilenetv3_large_100",
        model_version="1.0.0",
    )
    cal.is_fitted = True
    engine = CalibratedAbstentionEngine(calibrator=cal)

    # Calling inference with model version "2.0.0" against a "1.0.0" calibrator
    res = engine.evaluate(
        raw_logit=2.0,
        model_version="2.0.0",
    )
    assert res.decision_state == SafeDecisionState.MODEL_UNAVAILABLE
    assert res.abstention_reason == AbstentionReason.MODEL_VERSION_MISMATCH


# ---------------------------------------------------------------------------
# 7. Metrics & Distribution Shift Tests
# ---------------------------------------------------------------------------

def test_calibration_metrics_brier_and_ece():
    """Verify Brier score, ECE, and reliability diagram calculations."""
    y_true = [True, True, False, False]
    y_perfect = [1.0, 1.0, 0.0, 0.0]
    assert compute_brier_score(y_true, y_perfect) == 0.0
    assert compute_nll(y_true, y_perfect) < 1e-4

    ece, mce, bins = compute_reliability_diagram(y_true, y_perfect, num_bins=10)
    assert ece == 0.0
    assert mce == 0.0

    # Miscalibrated scores: predicted 0.9 for negative class, 0.1 for positive
    y_inverted = [0.1, 0.1, 0.9, 0.9]
    bad_ece, bad_mce, _ = compute_reliability_diagram(y_true, y_inverted, num_bins=10)
    assert bad_ece > 0.5
    assert bad_mce > 0.5


def test_calibration_by_category():
    """Verify stratified calibration by category (e.g. generator family)."""
    categories = ["Midjourney"] * 20 + ["SDXL"] * 20
    y_true = [True] * 18 + [False] * 2 + [True] * 15 + [False] * 5
    y_probs = [0.9] * 20 + [0.75] * 20

    stratified = evaluate_calibration_by_category(categories, y_true, y_probs)
    assert "Midjourney" in stratified
    assert "SDXL" in stratified
    assert stratified["Midjourney"]["sample_count"] == 20
    assert stratified["SDXL"]["sample_count"] == 20
    assert 0.0 <= stratified["Midjourney"]["brier_score"] <= 1.0


def test_calibration_under_distribution_shift():
    """Verify audit of calibration degradation under JPEG compression distribution shifts."""
    y_true = [True] * 25 + [False] * 25
    clean_probs = [0.95] * 25 + [0.05] * 25

    # Simulated perturbations: probabilities become attenuated towards 0.5 under compression
    shifted_dict = {
        "jpeg_q75": [0.85] * 25 + [0.15] * 25,
        "jpeg_q30": [0.60] * 25 + [0.40] * 25,
    }

    shifts = evaluate_calibration_under_distribution_shift(y_true, clean_probs, shifted_dict)
    assert "baseline_clean" in shifts
    assert "jpeg_q75" in shifts
    assert "jpeg_q30" in shifts

    # ECE and Brier score should increase under aggressive compression Q=30
    assert shifts["jpeg_q30"]["brier_score"] > shifts["baseline_clean"]["brier_score"]
    assert shifts["jpeg_q30"]["delta_brier"] > 0.0


# ---------------------------------------------------------------------------
# 8. Calibration Auditor & Model Selection Tests
# ---------------------------------------------------------------------------

def test_calibration_auditor_selection():
    """Verify CalibrationAuditor fits all candidates and selects the best calibrator on validation data."""
    rng = np.random.RandomState(42)
    val_labels = [True] * 35 + [False] * 35
    val_logits = list(rng.normal(2.5, 1.0, 35)) + list(rng.normal(-2.5, 1.0, 35))
    val_categories = ["AdobeFirefly"] * 35 + ["CameraNikon"] * 35

    auditor = CalibrationAuditor(
        model_name="mobilenetv3_large_100",
        model_version="1.0.0",
        architecture="mobilenet_v3_large",
    )

    report = auditor.audit_and_select(
        val_logits=val_logits,
        val_labels=val_labels,
        val_categories=val_categories,
        calibration_split="validation",
    )

    assert "uncalibrated_baseline" in report
    assert "candidates" in report
    assert len(report["candidates"]) >= 3  # Temperature, Platt, Isotonic
    assert report["selected_method"] in [
        CalibratorMethod.TEMPERATURE_SCALING.value,
        CalibratorMethod.PLATT_SCALING.value,
        CalibratorMethod.ISOTONIC_REGRESSION.value,
    ]
    assert report["selected_calibrator"].is_fitted is True
    assert "AdobeFirefly" in report["category_calibration"]


# ---------------------------------------------------------------------------
# 9. Backward Compatibility Tests
# ---------------------------------------------------------------------------

def test_backward_compatibility_with_model_inference_result():
    """Verify that existing ModelInferenceResult outputs can be seamlessly calibrated by the engine."""
    cal = TemperatureScalingCalibrator(temperature=1.2)
    cal.is_fitted = True
    engine = CalibratedAbstentionEngine(calibrator=cal)

    # Create standard ModelInferenceResult from ai_core.neural.schemas
    legacy_inference = ModelInferenceResult(
        raw_logits=[2.4],
        uncalibrated_scores={"AI_GENERATED_IMAGE": 0.9168, "CAMERA_PHOTOGRAPH": 0.0832},
        predicted_class="AI_GENERATED_IMAGE",
        model_version="1.0.0",
        architecture="mobilenet_v3_large",
        input_config=InputConfiguration(),
        status=InferenceStatus.READY,
        latency_ms=12.5,
    )

    # Ingest into abstention engine
    pred = engine.evaluate(
        raw_logit=legacy_inference.raw_logits[0],
        uncalibrated_score=legacy_inference.uncalibrated_scores["AI_GENERATED_IMAGE"],
        model_version=legacy_inference.model_version,
    )

    assert isinstance(pred, CalibratedPrediction)
    assert pred.decision_state == SafeDecisionState.INDICATORS_DETECTED
    assert 0.0 <= pred.calibrated_probability <= 1.0
    # Serialized dictionary contains all required fields
    d = pred.to_dict()
    assert d["decision_state"] == "INDICATORS_DETECTED"
    assert "interpretation_limits" in d
