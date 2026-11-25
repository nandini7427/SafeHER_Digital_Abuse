"""Unit and integration tests for SafeHER Image Quality & Uncertainty Engine.

Verifies:
- All 6 discrete uncertainty states:
  1. ANALYSIS_USABLE_EVIDENCE
  2. ANALYSIS_POOR_QUALITY_EVIDENCE
  3. MODEL_UNCERTAIN
  4. MODEL_UNAVAILABLE
  5. OUTSIDE_SUPPORTED_DOMAIN
  6. ANALYSIS_CANNOT_BE_COMPLETED
- Resolution thresholds and extreme aspect ratios (cropping).
- Excessive blur detection via Laplacian variance.
- Compression artifacts via 8x8 DCT grid blockiness.
- Extreme brightness (overexposure) and darkness (underexposure).
- Monochromatic or flat non-photographic graphics.
- Insufficient face resolution and area ratio.
- Corrupted or truncated image streams.
- Core safety guardrails: poor quality images are never classified as authentic or fake.
- Strict evidence preservation: raw bytes and SHA-256 hashes are never mutated.
"""

import hashlib
import io
import math
from typing import Tuple
from unittest.mock import MagicMock

import numpy as np
from PIL import Image, ImageFilter
import pytest

from ai_core.image_intelligence.config import ImageDetectorConfig
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import BaseImageModelRunner, ModelPrediction
from ai_core.image_intelligence.preprocessing import preprocess_image
from ai_core.image_intelligence.quality import (
    BLUR_THRESHOLD_LAPLACIAN_VAR,
    COMPRESSION_BLOCKINESS_THRESHOLD,
    MIN_FACE_DIMENSION,
    MIN_FORENSIC_DIMENSION,
    OVEREXPOSURE_RATIO_THRESHOLD,
    UNDEREXPOSURE_RATIO_THRESHOLD,
    check_monochromatic_or_flat,
    compute_compression_blockiness,
    compute_exposure_metrics,
    compute_laplacian_variance,
    evaluate_face_sufficiency,
    evaluate_image_quality,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    ImageQualityIndicators,
    UncertaintyState,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_test_image(
    dimensions: Tuple[int, int] = (256, 256),
    format_name: str = "JPEG",
    quality: int = 90,
    add_texture: bool = True,
) -> Tuple[bytes, Image.Image]:
    """Generate a reproducible test image and return both raw bytes and PIL Image."""
    w, h = dimensions
    if add_texture:
        # High-frequency natural-like texture (sinusoids + noise)
        xx, yy = np.meshgrid(np.linspace(0, 8 * math.pi, w), np.linspace(0, 8 * math.pi, h))
        pattern = (np.sin(xx) * np.cos(yy) * 60.0 + 128.0).astype(np.float32)
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 8.0, (h, w)).astype(np.float32)
        base = np.clip(pattern + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.stack([base, base, base], axis=2), mode="RGB")
    else:
        img = Image.new("RGB", (w, h), color=(128, 128, 128))

    buf = io.BytesIO()
    if format_name.upper() in ("JPEG", "JPG"):
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=format_name.upper())

    return buf.getvalue(), img


# ---------------------------------------------------------------------------
# 1. Tests for the 6 Discrete Uncertainty States
# ---------------------------------------------------------------------------

def test_uncertainty_state_1_usable_evidence():
    """State 1: Usable evidence with optimal resolution, sharpness, and clear score."""
    raw_bytes, img = _create_synthetic_test_image(dimensions=(256, 256), format_name="JPEG")
    detector = SafeHERImageDetector()

    response = detector.detect(raw_bytes)

    assert response.uncertainty_state in (
        UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
        UncertaintyState.LOW,
    )
    assert response.analysis_eligibility is True
    assert response.reason_for_insufficient_evidence is None
    assert "usable for automated triage" in response.recommended_next_step.lower()
    assert response.quality_indicators is not None
    assert response.quality_indicators.resolution_ok is True
    assert response.quality_indicators.is_blurry is False


def test_uncertainty_state_2_poor_quality_evidence():
    """State 2: Poor evidence quality (excessive blur or low resolution) maps to INCONCLUSIVE.

    MANDATE: Must NOT classify poor-quality images as authentic or fake.
    """
    raw_bytes, img = _create_synthetic_test_image(dimensions=(256, 256), format_name="JPEG")
    # Apply severe Gaussian blur
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=10.0))
    buf = io.BytesIO()
    blurred_img.save(buf, format="JPEG", quality=90)
    blurred_bytes = buf.getvalue()

    detector = SafeHERImageDetector()
    response = detector.detect(blurred_bytes)

    assert response.uncertainty_state in (
        UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
        UncertaintyState.INSUFFICIENT_QUALITY,
    )
    assert response.analysis_eligibility is False
    assert response.reason_for_insufficient_evidence is not None
    assert "blurry" in response.reason_for_insufficient_evidence.lower()
    # NON-NEGOTIABLE SAFETY ASSERTIONS:
    assert response.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert response.assessment_label != AssessmentLabel.NO_OBVIOUS_ARTIFACTS
    assert response.assessment_label != AssessmentLabel.HIGHLY_LIKELY_MANIPULATED


def test_uncertainty_state_3_model_uncertain():
    """State 3: Image quality is usable, but model score falls in ambiguous zone (0.40 - 0.60)."""
    raw_bytes, img = _create_synthetic_test_image(dimensions=(256, 256), format_name="JPEG")

    # Mock runner that returns ambiguous score 0.52
    ambiguous_runner = MagicMock(spec=BaseImageModelRunner)
    ambiguous_runner.predict.return_value = ModelPrediction(
        raw_score=0.52,
        runner_name="MockAmbiguousRunner",
        runner_version="1.0.0",
        manipulation_regions=[],
        auxiliary_metrics={},
    )

    detector = SafeHERImageDetector(runner=ambiguous_runner)
    response = detector.detect(raw_bytes)

    assert response.uncertainty_state in (
        UncertaintyState.MODEL_UNCERTAIN,
        UncertaintyState.MODERATE,
    )
    assert response.analysis_eligibility is True
    assert response.reason_for_insufficient_evidence is not None
    assert "intermediate" in response.reason_for_insufficient_evidence.lower() or "decision" in response.reason_for_insufficient_evidence.lower()
    assert "corroborating" in response.recommended_next_step.lower()


def test_uncertainty_state_4_model_unavailable():
    """State 4: Model fails or is offline; triage flags MODEL_UNAVAILABLE."""
    raw_bytes, img = _create_synthetic_test_image(dimensions=(256, 256), format_name="JPEG")

    # Mock runner raising exception
    crashing_runner = MagicMock(spec=BaseImageModelRunner)
    crashing_runner.predict.side_effect = RuntimeError("GPU server unavailable")

    detector = SafeHERImageDetector(runner=crashing_runner)
    response = detector.detect(raw_bytes)

    # When primary fails and fallback is engaged, if runner failed:
    assert response.uncertainty_state in (
        UncertaintyState.MODEL_UNAVAILABLE,
        UncertaintyState.ANALYSIS_USABLE_EVIDENCE,  # fallback baseline
    )
    # Direct quality evaluation with model_failed=True explicitly yields MODEL_UNAVAILABLE
    direct_res = evaluate_image_quality(img=img, model_failed=True)
    assert direct_res.uncertainty_state == UncertaintyState.MODEL_UNAVAILABLE
    assert direct_res.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert direct_res.analysis_eligibility is False


def test_uncertainty_state_5_outside_supported_domain():
    """State 5: Image is flat/monochromatic or extreme banner crop outside forensic domain."""
    # Monochromatic solid grey image
    solid_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    buf = io.BytesIO()
    solid_img.save(buf, format="JPEG", quality=90)
    solid_bytes = buf.getvalue()

    detector = SafeHERImageDetector()
    response = detector.detect(solid_bytes)

    assert response.uncertainty_state == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN
    assert response.analysis_eligibility is False
    assert response.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert "outside the supported photographic forensic domain" in response.reason_for_insufficient_evidence
    assert any("monochromatic" in w.lower() or "flat" in w.lower() for w in response.quality_warnings)


def test_uncertainty_state_6_analysis_cannot_be_completed():
    """State 6: Corrupted or truncated byte stream prevents decoding."""
    corrupted_bytes = b"\xFF\xD8\xFF\xE0" + b"\x00" * 30  # Broken JPEG header

    detector = SafeHERImageDetector()
    # detect_safe captures the corruption without crashing and returns structured response
    response = detector.detect_safe(corrupted_bytes)

    assert response.uncertainty_state == UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED
    assert response.analysis_eligibility is False
    assert response.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert "cannot be completed" in response.reason_for_insufficient_evidence.lower()
    assert "re-export or re-download" in response.recommended_next_step.lower()


# ---------------------------------------------------------------------------
# 2. Granular Quality Check Tests
# ---------------------------------------------------------------------------

def test_quality_check_resolution_thresholds():
    """Verify resolution ok flag for images above and below 128px threshold."""
    # Low resolution 64x64
    _, small_img = _create_synthetic_test_image(dimensions=(64, 64))
    res_small = evaluate_image_quality(small_img)
    assert res_small.indicators.resolution_ok is False
    assert res_small.indicators.width == 64
    assert res_small.indicators.height == 64
    assert any("below the minimum forensic requirement" in w for w in res_small.warnings)

    # Valid resolution 256x256
    _, valid_img = _create_synthetic_test_image(dimensions=(256, 256))
    res_valid = evaluate_image_quality(valid_img)
    assert res_valid.indicators.resolution_ok is True


def test_quality_check_laplacian_blur():
    """Verify Laplacian variance correctly discriminates sharp vs heavily blurred imagery."""
    _, sharp_img = _create_synthetic_test_image(dimensions=(256, 256), add_texture=True)
    sharp_arr = np.array(sharp_img.convert("L"), dtype=np.float32)
    sharp_blur_score = compute_laplacian_variance(sharp_arr)
    assert sharp_blur_score > BLUR_THRESHOLD_LAPLACIAN_VAR

    # Blurred version
    blurred_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=8.0))
    blurred_arr = np.array(blurred_img.convert("L"), dtype=np.float32)
    blurred_blur_score = compute_laplacian_variance(blurred_arr)
    assert blurred_blur_score < BLUR_THRESHOLD_LAPLACIAN_VAR


def test_quality_check_compression_blockiness():
    """Verify 8x8 DCT grid blockiness calculation on low quality vs high quality JPEGs."""
    # High quality (Q=95)
    high_q_bytes, high_q_img = _create_synthetic_test_image(quality=95)
    gray_high = np.array(high_q_img.convert("L"), dtype=np.float32)
    blockiness_high = compute_compression_blockiness(gray_high)
    assert blockiness_high < COMPRESSION_BLOCKINESS_THRESHOLD

    # Low quality (Q=15) introduces severe blocking
    buf = io.BytesIO()
    high_q_img.save(buf, format="JPEG", quality=15)
    low_q_img = Image.open(buf)
    gray_low = np.array(low_q_img.convert("L"), dtype=np.float32)
    blockiness_low = compute_compression_blockiness(gray_low)
    # Low quality increases boundary step discontinuity ratio
    assert blockiness_low > 1.2


def test_quality_check_extreme_brightness_overexposure():
    """Verify blown-out highlight clipping triggers overexposed status."""
    # Overexposed image (> 30% pixels at 255)
    over_arr = np.full((128, 128), 255.0, dtype=np.float32)
    over_arr[0:20, 0:20] = 100.0  # small non-saturated patch
    mean_lum, over_ratio, under_ratio, level = compute_exposure_metrics(over_arr)

    assert level == "overexposed"
    assert over_ratio >= OVEREXPOSURE_RATIO_THRESHOLD
    assert mean_lum > 240.0


def test_quality_check_extreme_darkness_underexposure():
    """Verify crushed shadow clipping triggers underexposed status."""
    # Underexposed image (> 35% pixels at 0)
    under_arr = np.full((128, 128), 5.0, dtype=np.float32)
    under_arr[0:20, 0:20] = 120.0
    mean_lum, over_ratio, under_ratio, level = compute_exposure_metrics(under_arr)

    assert level == "underexposed"
    assert under_ratio >= UNDEREXPOSURE_RATIO_THRESHOLD
    assert mean_lum < 20.0


def test_quality_check_extreme_aspect_ratio_cropping():
    """Verify extreme aspect ratio (ribbon / banner crop) flags domain mismatch."""
    # Panoramic banner (800x80 px -> aspect ratio 10.0)
    banner_img = Image.new("RGB", (800, 80), color=(100, 150, 200))
    res = evaluate_image_quality(banner_img)

    assert res.indicators.aspect_ratio == 10.0
    assert res.uncertainty_state == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN
    assert any("extreme aspect ratio" in w.lower() for w in res.warnings)


def test_quality_check_monochromatic_flat_image():
    """Verify flat solid color blocks are flagged as monochromatic."""
    flat_rgb = np.full((128, 128, 3), 180.0, dtype=np.float32)
    assert check_monochromatic_or_flat(flat_rgb) is True

    # Textured image is NOT monochromatic
    rng = np.random.RandomState(42)
    textured_rgb = rng.uniform(50, 200, (128, 128, 3)).astype(np.float32)
    assert check_monochromatic_or_flat(textured_rgb) is False


def test_quality_check_face_sufficiency():
    """Verify face resolution and area ratio evaluation."""
    # Small face: 30x30 px inside 512x512 image
    # Normalized: (0.1, 0.1, 0.158, 0.158) -> ~30px
    small_face_box = [(0.1, 0.1, 0.1 + (30.0 / 512.0), 0.1 + (30.0 / 512.0))]
    detected, count, res_ok, ratio = evaluate_face_sufficiency(512, 512, small_face_box)

    assert detected is True
    assert count == 1
    assert res_ok is False  # 30px < MIN_FACE_DIMENSION (60px)

    # Adequate face: 150x150 px inside 512x512 image
    adequate_face_box = [(0.2, 0.2, 0.2 + (150.0 / 512.0), 0.2 + (150.0 / 512.0))]
    detected, count, res_ok, ratio = evaluate_face_sufficiency(512, 512, adequate_face_box)

    assert detected is True
    assert count == 1
    assert res_ok is True
    assert ratio is not None
    assert ratio > 0.05  # > 5% frame area


# ---------------------------------------------------------------------------
# 3. Core Safety Guardrails & Evidence Integrity Verification
# ---------------------------------------------------------------------------

def test_safety_guardrail_poor_quality_never_labeled_authentic():
    """Ensure poor quality images are never classified as authentic or no_obvious_artifacts."""
    # Create tiny 48x48 image
    raw_bytes, _ = _create_synthetic_test_image(dimensions=(48, 48), format_name="JPEG")
    detector = SafeHERImageDetector(config=ImageDetectorConfig(min_image_dimension=32))

    response = detector.detect(raw_bytes)

    # MUST be INCONCLUSIVE, NEVER authentic
    assert response.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert "authentic" not in response.assessment_label.value.lower()
    assert response.analysis_eligibility is False


def test_safety_guardrail_poor_quality_never_auto_labeled_fake():
    """Ensure poor quality images are not automatically labeled as manipulated or deepfake."""
    # Create blurred image with low anomaly score
    raw_bytes, img = _create_synthetic_test_image(dimensions=(256, 256), format_name="JPEG")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=12.0))
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG", quality=90)

    detector = SafeHERImageDetector()
    response = detector.detect(buf.getvalue())

    # MUST be INCONCLUSIVE, NEVER automatically flagged as manipulated
    assert response.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert response.assessment_label != AssessmentLabel.HIGHLY_LIKELY_MANIPULATED
    assert response.assessment_label != AssessmentLabel.SUSPICIOUS_MANIPULATION


def test_evidence_preservation_integrity():
    """Verify that quality evaluation preserves the original uploaded bytes and SHA-256 hash."""
    raw_bytes, _ = _create_synthetic_test_image(dimensions=(256, 256), format_name="PNG")
    original_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    detector = SafeHERImageDetector()
    response = detector.detect(raw_bytes)

    # Cryptographic hash matches original bytes exactly
    assert response.metadata_summary is not None
    assert response.metadata_summary.sha256 == original_sha256
    assert response.metadata_summary.size_bytes == len(raw_bytes)
