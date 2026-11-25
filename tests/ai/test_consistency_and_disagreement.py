"""Unit and integration tests for Model Consistency, Disagreement Analysis, and Disagreement Policy.

Tests:
1. FrequencySpectralRunner 2D-FFT spectral feature extraction and scoring.
2. ModelDisagreementPolicy operational states (Consistent, Conflicting, Insufficient, Unavailable).
3. Strict enforcement of zero score averaging.
4. Inter-rater agreement statistics and Cohen's Kappa.
5. Correlated error calculations and phi correlation.
6. SafeHERImageDetector multi-detector consistency execution and fallback.
"""

from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
import pytest

from ai_core.image_intelligence.config import ImageDetectorConfig
from ai_core.image_intelligence.consistency import (
    DetectorSignal,
    DisagreementState,
    ModelConsistencyAssessment,
    ModelDisagreementPolicy,
    compute_correlated_errors,
    compute_inter_model_agreement,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import (
    BaseImageModelRunner,
    FrequencySpectralRunner,
    ModelPrediction,
)
from ai_core.image_intelligence.preprocessing import preprocess_image
from ai_core.image_intelligence.quality import (
    QualityAssessmentResult,
    evaluate_image_quality,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    ImageQualityIndicators,
    UncertaintyState,
)


def _make_dummy_quality_result(eligible: bool = True) -> QualityAssessmentResult:
    """Helper creating a dummy QualityAssessmentResult."""
    return QualityAssessmentResult(
        indicators=ImageQualityIndicators(
            width=256,
            height=256,
            megapixels=0.065536,
            aspect_ratio=1.0,
            resolution_ok=eligible,
            blur_score=200.0,
            is_blurry=not eligible,
            compression_artifact_score=0.5,
            has_heavy_compression=False,
            mean_luminance=128.0,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_level="normal",
            is_monochromatic_or_flat=False,
            is_screen_capture=False,
            face_detected=None,
            face_count=0,
            face_resolution_ok=None,
            face_size_ratio=None,
            corrupted_data_detected=False,
            domain_category="photographic_portrait",
            is_in_domain=eligible,
            overall_quality_score=1.0 if eligible else 0.2,
        ),
        warnings=[] if eligible else ["Severe blur defect"],
        uncertainty_state=UncertaintyState.ANALYSIS_USABLE_EVIDENCE
        if eligible
        else UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
        analysis_eligibility=eligible,
        reason_for_insufficient_evidence=None
        if eligible
        else "Blur exceeds forensic tolerance",
        recommended_next_step="Preserve original" if eligible else "Request re-capture",
        assessment_label=AssessmentLabel.NO_OBVIOUS_ARTIFACTS
        if eligible
        else AssessmentLabel.INCONCLUSIVE,
    )


def _make_dummy_prediction(
    score: float,
    name: str = "Test-Runner",
    version: str = "1.0.0",
) -> ModelPrediction:
    """Helper creating a dummy ModelPrediction."""
    return ModelPrediction(
        raw_score=score,
        manipulation_regions=[],
        auxiliary_metrics={"test_metric": 1.0},
        runner_name=name,
        runner_version=version,
        is_neural=False,
    )


# ---------------------------------------------------------------------------
# 1. FrequencySpectralRunner Tests
# ---------------------------------------------------------------------------


def test_frequency_spectral_runner_availability():
    """Verify FrequencySpectralRunner is available and initialized."""
    runner = FrequencySpectralRunner()
    assert runner.is_available() is True
    assert "READY" in runner.get_status()
    assert "2D-FFT" in runner.get_status()


def test_frequency_spectral_runner_prediction_natural_image():
    """Verify FrequencySpectralRunner executes cleanly on natural-like gradient image."""
    runner = FrequencySpectralRunner()

    # Generate synthetic smooth gradient image
    img = Image.new("RGB", (128, 128), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    for y in range(128):
        draw.line([(0, y), (128, y)], fill=(y * 2, 100, 200 - y))

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    preprocessed = preprocess_image(buf.getvalue())

    prediction = runner.predict(preprocessed)
    assert 0.05 <= prediction.raw_score <= 0.95
    assert prediction.runner_name == "SafeHER-Frequency-Spectral-Detector"
    assert "spectral_slope" in prediction.auxiliary_metrics
    assert "high_freq_ratio" in prediction.auxiliary_metrics
    assert "periodic_peaks_count" in prediction.auxiliary_metrics


def test_frequency_spectral_runner_small_image_fallback():
    """Verify runner handles very small image gracefully without crash."""
    runner = FrequencySpectralRunner()
    tiny = Image.new("RGB", (8, 8), color=(100, 100, 100))

    import io
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    # Preprocessing enforces min dimension, so test internal feature method directly
    features = runner._compute_spectral_features(tiny)
    assert features["spectral_slope"] == -2.0
    assert features["periodic_peaks_count"] == 0


# ---------------------------------------------------------------------------
# 2. ModelDisagreementPolicy Operational States Tests
# ---------------------------------------------------------------------------


def test_policy_insufficient_evidence():
    """Verify policy halts when image quality or domain is ineligible."""
    policy = ModelDisagreementPolicy()
    quality = _make_dummy_quality_result(eligible=False)
    pred1 = _make_dummy_prediction(0.80)
    pred2 = _make_dummy_prediction(0.75)

    result = policy.evaluate(quality, pred1, pred2)
    assert result.disagreement_state == DisagreementState.INSUFFICIENT_EVIDENCE
    assert result.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert result.uncertainty_state == UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE
    assert "halted" in result.policy_explanation


def test_policy_one_detector_unavailable():
    """Verify policy executes single-detector fallback when one runner is None."""
    policy = ModelDisagreementPolicy(primary_threshold=0.45)
    quality = _make_dummy_quality_result(eligible=True)
    pred1 = _make_dummy_prediction(0.60, name="Operational-Primary")

    result = policy.evaluate(quality, primary_prediction=pred1, secondary_prediction=None)
    assert result.disagreement_state == DisagreementState.ONE_DETECTOR_UNAVAILABLE
    assert result.assessment_label == AssessmentLabel.SUSPICIOUS_MANIPULATION
    assert result.uncertainty_state == UncertaintyState.MODEL_UNCERTAIN
    assert "Operational Fallback" in result.policy_explanation
    assert len(result.warnings) > 0


def test_policy_both_detectors_unavailable():
    """Verify policy handles total detector outage."""
    policy = ModelDisagreementPolicy()
    quality = _make_dummy_quality_result(eligible=True)

    result = policy.evaluate(quality, primary_prediction=None, secondary_prediction=None)
    assert result.disagreement_state == DisagreementState.ONE_DETECTOR_UNAVAILABLE
    assert result.uncertainty_state == UncertaintyState.MODEL_UNAVAILABLE
    assert result.assessment_label == AssessmentLabel.INCONCLUSIVE


def test_policy_consistent_both_flagged():
    """Verify policy produces corroborated positive when both detectors alarm."""
    policy = ModelDisagreementPolicy(primary_threshold=0.45, secondary_threshold=0.45)
    quality = _make_dummy_quality_result(eligible=True)
    pred1 = _make_dummy_prediction(0.70, name="ELA-Runner")
    pred2 = _make_dummy_prediction(0.65, name="FFT-Runner")

    result = policy.evaluate(quality, pred1, pred2)
    assert result.disagreement_state == DisagreementState.CONSISTENT_DETECTION_SIGNALS
    assert result.assessment_label == AssessmentLabel.SUSPICIOUS_MANIPULATION
    assert result.uncertainty_state == UncertaintyState.ANALYSIS_USABLE_EVIDENCE
    assert "Corroborated Multi-Domain Forensic Findings" in result.policy_explanation
    assert result.disagreement_reason is None


def test_policy_consistent_neither_flagged():
    """Verify policy produces consistent baseline when neither detector alarms."""
    policy = ModelDisagreementPolicy(primary_threshold=0.45, secondary_threshold=0.45)
    quality = _make_dummy_quality_result(eligible=True)
    pred1 = _make_dummy_prediction(0.20, name="ELA-Runner")
    pred2 = _make_dummy_prediction(0.25, name="FFT-Runner")

    result = policy.evaluate(quality, pred1, pred2)
    assert result.disagreement_state == DisagreementState.CONSISTENT_DETECTION_SIGNALS
    assert result.assessment_label == AssessmentLabel.NO_OBVIOUS_ARTIFACTS
    assert result.uncertainty_state == UncertaintyState.ANALYSIS_USABLE_EVIDENCE
    assert "Consistent Baseline Evidence" in result.policy_explanation


def test_policy_conflicting_signals_triggers_inconclusive():
    """Verify conflicting signals trigger INCONCLUSIVE with MODEL_UNCERTAIN and ZERO score averaging."""
    policy = ModelDisagreementPolicy(primary_threshold=0.45, secondary_threshold=0.45)
    quality = _make_dummy_quality_result(eligible=True)
    pred1 = _make_dummy_prediction(0.75, name="ELA-Runner")  # Flags anomaly
    pred2 = _make_dummy_prediction(0.15, name="FFT-Runner")  # Normal baseline

    result = policy.evaluate(quality, pred1, pred2)
    assert result.disagreement_state == DisagreementState.CONFLICTING_DETECTION_SIGNALS
    # Critical rule verification: must NOT select the alarming SUSPICIOUS_MANIPULATION call
    assert result.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert result.uncertainty_state == UncertaintyState.MODEL_UNCERTAIN
    assert "Conflicting Signals" in result.policy_explanation
    assert "Score averaging was strictly prevented" in result.warnings[1]
    assert abs(result.score_divergence - 0.60) < 1e-4

    # Ensure individual signals are preserved without modification
    assert result.primary_signal.raw_score == 0.75
    assert result.secondary_signal.raw_score == 0.15


# ---------------------------------------------------------------------------
# 3. Inter-Rater Agreement & Correlated Errors Metrics Tests
# ---------------------------------------------------------------------------


def test_compute_inter_model_agreement_perfect():
    """Verify agreement calculation on perfect concordance."""
    d1 = [True, False, True, False]
    d2 = [True, False, True, False]
    stats = compute_inter_model_agreement(d1, d2)

    assert stats["raw_agreement_rate"] == 1.0
    assert stats["disagreement_rate"] == 0.0
    assert stats["cohen_kappa"] == 1.0


def test_compute_inter_model_agreement_discordance():
    """Verify agreement calculation on partial discordance."""
    d1 = [True, True, False, False]
    d2 = [True, False, True, False]
    stats = compute_inter_model_agreement(d1, d2)

    assert stats["raw_agreement_rate"] == 0.5
    assert stats["disagreement_rate"] == 0.5
    assert stats["cohen_kappa"] == 0.0


def test_compute_correlated_errors():
    """Verify joint error and phi coefficient computation."""
    gt = [1, 1, 0, 0]
    p1 = [1, 0, 0, 1]  # Failed on index 1 and 3
    p2 = [1, 0, 0, 0]  # Failed on index 1 only

    corr = compute_correlated_errors(gt, p1, p2)
    assert corr["both_failed_count"] == 1
    assert corr["primary_only_failed_count"] == 1
    assert corr["secondary_only_failed_count"] == 0
    assert corr["joint_failure_rate"] == 0.25
    assert corr["jaccard_error_overlap"] == 0.5


# ---------------------------------------------------------------------------
# 4. SafeHERImageDetector Multi-Detector Integration Tests
# ---------------------------------------------------------------------------


def test_detector_detect_with_consistency():
    """Verify detect_with_consistency runs dual detectors and attaches agreement details."""
    detector = SafeHERImageDetector()

    # Generate synthetic camera image
    img = Image.new("RGB", (256, 256), color=(140, 150, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 100, 100], fill=(200, 80, 80))

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()

    resp = detector.detect_with_consistency(data)
    assert resp is not None
    assert resp.disagreement_state in [
        DisagreementState.CONSISTENT_DETECTION_SIGNALS.value,
        DisagreementState.CONFLICTING_DETECTION_SIGNALS.value,
        DisagreementState.INSUFFICIENT_EVIDENCE.value,
    ]
    assert resp.model_agreement_details is not None
    assert "disagreement_state" in resp.model_agreement_details
    assert resp.is_calibrated is False
    assert resp.confidence_score is None


def test_detector_evaluate_consistency():
    """Verify evaluate_consistency standalone method returns ModelConsistencyAssessment."""
    detector = SafeHERImageDetector()

    img = Image.new("RGB", (256, 256), color=(140, 150, 160))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)

    consistency_assessment = detector.evaluate_consistency(buf.getvalue())
    assert isinstance(consistency_assessment, ModelConsistencyAssessment)
    assert consistency_assessment.disagreement_state in DisagreementState
    d = consistency_assessment.to_dict()
    assert "score_divergence" in d
    assert "policy_explanation" in d
