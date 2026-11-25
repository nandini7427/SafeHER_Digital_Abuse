"""Unit and integration tests for SafeHER Adversarial Robustness and Transformation Testing."""

import hashlib
import io
import math

import numpy as np
from PIL import Image
import pytest

from ai_core.evaluation.adversarial_robustness import (
    TRANSFORMATION_REGISTRY,
    RobustnessDerivative,
    TransformationPairEvaluation,
    TransformationRobustnessSummary,
    create_robustness_derivative,
    evaluate_transformation_pair,
    summarize_transformation_robustness,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.schemas import AssessmentLabel, UncertaintyState


def _create_controlled_test_image(seed: int = 100) -> bytes:
    """Generate deterministic synthetic test image."""
    rng = np.random.RandomState(seed)
    w, h = 256, 256
    xx, yy = np.meshgrid(np.linspace(0, 6 * math.pi, w), np.linspace(0, 6 * math.pi, h))
    pattern = (np.sin(xx) * np.cos(yy) * 50.0 + 128.0).astype(np.float32)
    noise = rng.normal(0, 5.0, (h, w)).astype(np.float32)
    base = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([base, base, base], axis=2), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_transformation_registry_covers_all_12_required_transformations():
    """Verify registry contains all 12 transformations requested in prompt."""
    expected_transformations = {
        "jpeg_compression",
        "png_conversion",
        "resizing",
        "cropping",
        "blur",
        "brightness_changes",
        "contrast_changes",
        "color_transformations",
        "screenshot_conversion",
        "metadata_removal",
        "re_encoding",
        "minor_image_noise",
    }
    assert set(TRANSFORMATION_REGISTRY.keys()) == expected_transformations


def test_evidence_preservation_and_derivative_separation():
    """Verify pristine original bytes and SHA-256 are unchanged while derivative has distinct hash."""
    raw_bytes = _create_controlled_test_image(seed=200)
    orig_hash = hashlib.sha256(raw_bytes).hexdigest()

    for trans_name in TRANSFORMATION_REGISTRY.keys():
        deriv = create_robustness_derivative(
            sample_id="SAMPLE_TEST_001",
            raw_bytes=raw_bytes,
            transformation_name=trans_name,
            is_manipulated=False,
        )

        # 1. Original bytes intact
        assert hashlib.sha256(deriv.original_bytes).hexdigest() == orig_hash
        assert hashlib.sha256(raw_bytes).hexdigest() == orig_hash
        # 2. Derivative has separate bytes and SHA-256
        assert len(deriv.derivative_bytes) > 0
        assert deriv.derivative_sha256 == hashlib.sha256(deriv.derivative_bytes).hexdigest()
        # 3. Parameters recorded
        assert isinstance(deriv.transformation_params, dict)


def test_evaluate_transformation_pair_execution():
    """Verify pair evaluation executes detector cleanly without model failure."""
    raw_bytes = _create_controlled_test_image(seed=300)
    detector = SafeHERImageDetector()

    deriv = create_robustness_derivative(
        sample_id="SAMPLE_TEST_002",
        raw_bytes=raw_bytes,
        transformation_name="jpeg_compression",
        is_manipulated=False,
    )

    pair_eval = evaluate_transformation_pair(deriv, detector)

    assert pair_eval.model_failed is False
    assert 0.0 <= pair_eval.clean_score <= 1.0
    assert 0.0 <= pair_eval.perturbed_score <= 1.0
    assert isinstance(pair_eval.clean_label, AssessmentLabel)
    assert isinstance(pair_eval.perturbed_label, AssessmentLabel)
    assert pair_eval.is_false_positive_transition is False


def test_summarize_transformation_robustness_aggregation():
    """Verify aggregate summary metrics calculation."""
    evals = [
        TransformationPairEvaluation(
            sample_id="S1",
            transformation_name="blur",
            is_manipulated=False,
            clean_score=0.10,
            perturbed_score=0.10,
            score_delta=0.0,
            clean_label=AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
            perturbed_label=AssessmentLabel.INCONCLUSIVE,
            label_changed=True,
            is_false_positive_transition=False,
            is_false_negative_transition=False,
            clean_uncertainty=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
            perturbed_uncertainty=UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
            uncertainty_escalated=True,
            quality_warnings_triggered=["Excessive blur detected"],
            model_failed=False,
        ),
        TransformationPairEvaluation(
            sample_id="S2",
            transformation_name="blur",
            is_manipulated=True,
            clean_score=0.70,
            perturbed_score=0.70,
            score_delta=0.0,
            clean_label=AssessmentLabel.HIGHLY_LIKELY_MANIPULATED,
            perturbed_label=AssessmentLabel.HIGHLY_LIKELY_MANIPULATED,
            label_changed=False,
            is_false_positive_transition=False,
            is_false_negative_transition=False,
            clean_uncertainty=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
            perturbed_uncertainty=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
            uncertainty_escalated=False,
            quality_warnings_triggered=[],
            model_failed=False,
        ),
    ]

    summary = summarize_transformation_robustness(evals)

    assert summary.sample_count == 2
    assert summary.transformation_name == "blur"
    assert summary.label_flip_rate == 0.5
    assert summary.uncertainty_escalation_rate == 0.5
    assert summary.quality_warning_trigger_rate == 0.5
    assert summary.model_failure_count == 0
    assert summary.false_positive_transitions == 0


def test_quality_engine_intercepts_severe_blur_perturbation():
    """Verify blur transformation triggers quality engine warning and poor quality uncertainty."""
    raw_bytes = _create_controlled_test_image(seed=400)
    detector = SafeHERImageDetector()

    deriv = create_robustness_derivative(
        sample_id="SAMPLE_BLUR_001",
        raw_bytes=raw_bytes,
        transformation_name="blur",
        is_manipulated=False,
    )

    pair_eval = evaluate_transformation_pair(deriv, detector)

    # Blur should trigger quality engine
    assert any("blur" in w.lower() for w in pair_eval.quality_warnings_triggered)
    assert pair_eval.perturbed_uncertainty in (
        UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
        UncertaintyState.INSUFFICIENT_QUALITY,
    )
    assert pair_eval.perturbed_label == AssessmentLabel.INCONCLUSIVE
    assert pair_eval.is_false_positive_transition is False
