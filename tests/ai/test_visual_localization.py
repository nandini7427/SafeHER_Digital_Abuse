"""Unit and integration tests for Explainable Visual Manipulation Localization.

Tests:
1. ForensicELALocalizer spatial variance mapping and non-fabrication on clean images.
2. OcclusionSensitivityLocalizer black-box perturbation attribution.
3. GenericFaceMaskGuard filtering of un-manipulated face boxes.
4. Frequency-domain non-localizability handling (image-level explanation fallback).
5. Reproducibility, stability, and faithfulness validation.
6. SafeHERImageDetector visual explanation integration and byte preservation.
"""

import hashlib
import io

import numpy as np
from PIL import Image, ImageDraw
import pytest

from ai_core.image_intelligence.config import ImageDetectorConfig
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import (
    BaselineHeuristicRunner,
    FrequencySpectralRunner,
)
from ai_core.image_intelligence.preprocessing import preprocess_image
from ai_core.image_intelligence.schemas import ManipulationRegion
from ai_core.image_intelligence.visual_localization import (
    AttributionCategory,
    ForensicELALocalizer,
    GenericFaceMaskGuard,
    LocalizationMethod,
    OcclusionSensitivityLocalizer,
    VisualLocalizationEngine,
    VisualLocalizationResult,
)


def _make_clean_image_bytes(w: int = 256, h: int = 256) -> bytes:
    """Create clean photographic gradient image."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        arr[y, :, :] = int(100 + (y / h) * 60)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_spliced_composite_bytes(w: int = 256, h: int = 256) -> bytes:
    """Create synthetic composite with high variance patch in top-left quadrant."""
    rng = np.random.RandomState(42)
    base_arr = np.clip(130 + rng.normal(0, 2.0, (h, w, 3)), 0, 255).astype(np.uint8)
    base_img = Image.fromarray(base_arr, mode="RGB")

    donor_arr = np.clip(160 + rng.normal(0, 35.0, (128, 128, 3)), 0, 255).astype(np.uint8)
    donor_img = Image.fromarray(donor_arr, mode="RGB")

    comp = base_img.copy()
    comp.paste(donor_img, (0, 0))

    buf = io.BytesIO()
    comp.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Forensic ELA Localizer & Non-Fabrication Tests
# ---------------------------------------------------------------------------


def test_forensic_ela_localizer_clean_image_no_fabricated_heatmap():
    """Verify clean image produces is_localizable=False and continuous_heatmap=None."""
    clean_bytes = _make_clean_image_bytes()
    preprocessed = preprocess_image(clean_bytes)
    localizer = ForensicELALocalizer()

    result = localizer.localize(preprocessed)
    # Strictly zero fabricated heatmaps
    assert result.is_localizable is False
    assert result.continuous_heatmap is None
    assert len(result.regions) == 0
    assert result.localization_method == LocalizationMethod.NONE_NO_SIGNIFICANT_ANOMALY
    assert result.attribution_category == AttributionCategory.NOT_APPLICABLE


def test_forensic_ela_localizer_spliced_image_detects_regions():
    """Verify spliced image with high variance patch produces localized regions."""
    spliced_bytes = _make_spliced_composite_bytes()
    preprocessed = preprocess_image(spliced_bytes)
    localizer = ForensicELALocalizer()

    result = localizer.localize(preprocessed)
    assert result.is_localizable is True
    assert len(result.regions) > 0
    assert result.continuous_heatmap is not None
    assert result.localization_method == LocalizationMethod.FORENSIC_ELA_SPATIAL_VARIANCE
    assert result.attribution_category == AttributionCategory.SUSPECTED_MANIPULATION_REGION
    assert result.faithfulness_score > 0.0

    # Verify localized regions are in the top-left quadrant
    for r in result.regions:
        ymin, xmin, ymax, xmax = r.box_normalized
        assert ymin <= 0.5 and xmin <= 0.5


# ---------------------------------------------------------------------------
# 2. Occlusion Sensitivity Localizer Tests
# ---------------------------------------------------------------------------


def test_occlusion_sensitivity_localizer_attribution():
    """Verify OcclusionSensitivityLocalizer computes sensitivity map and top attribution patch."""
    spliced_bytes = _make_spliced_composite_bytes()
    preprocessed = preprocess_image(spliced_bytes)
    runner = BaselineHeuristicRunner()
    occ_localizer = OcclusionSensitivityLocalizer(runner=runner, patch_divisor=2, stride_divisor=2)

    result = occ_localizer.attribute(preprocessed)
    assert result.is_localizable is True
    assert result.localization_method == LocalizationMethod.OCCLUSION_SENSITIVITY
    assert result.attribution_category == AttributionCategory.MODEL_ATTENTION_ATTRIBUTION
    assert len(result.regions) == 1
    assert result.continuous_heatmap is not None
    assert result.faithfulness_score > 0.0


# ---------------------------------------------------------------------------
# 3. Generic Face Mask Guard Tests
# ---------------------------------------------------------------------------


def test_generic_face_mask_guard_blocks_unmanipulated_face_box():
    """Verify face bounding box is filtered out when it is not a genuine manipulation."""
    guard = GenericFaceMaskGuard()
    face_box = (0.2, 0.2, 0.8, 0.8)

    fake_manip_region = ManipulationRegion(
        box_normalized=face_box,
        anomaly_score=0.85,
        description="Detected face region",
    )

    filtered = guard.audit_regions_against_face_mask([fake_manip_region], [face_box])
    # The generic face box MUST be removed
    assert len(filtered) == 0


def test_generic_face_mask_guard_retains_genuine_forensic_regions():
    """Verify genuine variance-backed anomaly regions are not wrongly filtered."""
    guard = GenericFaceMaskGuard()
    face_box = (0.2, 0.2, 0.8, 0.8)

    genuine_region = ManipulationRegion(
        box_normalized=(0.0, 0.0, 0.25, 0.25),  # Distinct corner
        anomaly_score=0.85,
        description="Spatial compression variance outlier in quadrant (1,1)",
    )

    filtered = guard.audit_regions_against_face_mask([genuine_region], [face_box])
    assert len(filtered) == 1
    assert filtered[0].box_normalized == (0.0, 0.0, 0.25, 0.25)


# ---------------------------------------------------------------------------
# 4. Frequency-Domain Non-Localizability Handling Tests
# ---------------------------------------------------------------------------


def test_frequency_domain_detection_returns_image_level_explanation():
    """Verify frequency-only detections do not fabricate spatial heatmaps."""
    engine = VisualLocalizationEngine()
    clean_bytes = _make_clean_image_bytes()

    result = engine.generate_visual_explanation(
        data=clean_bytes,
        runner_name="SafeHER-Frequency-Spectral-Detector",
        raw_score=0.75,
        is_frequency_only=True,
    )

    assert result.is_localizable is False
    assert result.continuous_heatmap is None
    assert len(result.regions) == 0
    assert result.localization_method == LocalizationMethod.NONE_GLOBAL_FREQUENCY_ARTIFACTS
    assert "global periodic properties" in result.explanation_text
    assert "Displaying image-level forensic explanation instead" in result.explanation_text


# ---------------------------------------------------------------------------
# 5. Reproducibility, Stability, and Faithfulness Tests
# ---------------------------------------------------------------------------


def test_localization_reproducibility():
    """Verify running localization twice on identical bytes yields identical outputs."""
    spliced_bytes = _make_spliced_composite_bytes()
    engine = VisualLocalizationEngine()

    res1 = engine.generate_visual_explanation(spliced_bytes, "Baseline", 0.60)
    res2 = engine.generate_visual_explanation(spliced_bytes, "Baseline", 0.60)

    assert res1.is_localizable == res2.is_localizable
    assert len(res1.regions) == len(res2.regions)
    for r1, r2 in zip(res1.regions, res2.regions):
        assert r1.box_normalized == r2.box_normalized
        assert r1.anomaly_score == r2.anomaly_score
    assert res1.continuous_heatmap == res2.continuous_heatmap


def test_localization_stability_under_minor_perturbation():
    """Verify localized region remains in top quadrant under minor noise perturbation."""
    spliced_bytes = _make_spliced_composite_bytes()
    preprocessed = preprocess_image(spliced_bytes)
    w_img = preprocessed.working_image

    # Add minor noise
    noise_arr = np.array(w_img, dtype=np.float32) + np.random.normal(0, 0.5, (w_img.height, w_img.width, 3))
    noise_img = Image.fromarray(np.clip(noise_arr, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    noise_img.save(buf, format="JPEG", quality=85)
    noise_bytes = buf.getvalue()

    engine = VisualLocalizationEngine()
    res_orig = engine.generate_visual_explanation(spliced_bytes, "Baseline", 0.60)
    res_noise = engine.generate_visual_explanation(noise_bytes, "Baseline", 0.60)

    assert res_orig.is_localizable is True
    assert res_noise.is_localizable is True
    # The detected regions must remain in the top-left area
    assert res_noise.regions[0].box_normalized[0] <= 0.5
    assert res_noise.regions[0].box_normalized[1] <= 0.5


# ---------------------------------------------------------------------------
# 6. SafeHERImageDetector Integration & Provenance Invariance Tests
# ---------------------------------------------------------------------------


def test_detector_integration_populates_visual_explanation():
    """Verify SafeHERImageDetector.detect() includes visual_explanation."""
    spliced_bytes = _make_spliced_composite_bytes()
    detector = SafeHERImageDetector()

    response = detector.detect(spliced_bytes)
    assert response.visual_explanation is not None
    assert "is_localizable" in response.visual_explanation
    assert "localization_method" in response.visual_explanation
    assert "limitations" in response.visual_explanation
    assert len(response.visual_explanation["limitations"]) > 0


def test_detector_localize_manipulation_standalone():
    """Verify detector.localize_manipulation() method executes cleanly."""
    spliced_bytes = _make_spliced_composite_bytes()
    detector = SafeHERImageDetector()

    loc_res = detector.localize_manipulation(spliced_bytes)
    assert isinstance(loc_res, VisualLocalizationResult)
    assert loc_res.is_localizable is True
    assert loc_res.preservation_verified is True


def test_sha256_provenance_invariance_during_localization():
    """Verify original bytes and SHA-256 hash are 100% unchanged during localization."""
    spliced_bytes = _make_spliced_composite_bytes()
    initial_sha = hashlib.sha256(spliced_bytes).hexdigest()

    detector = SafeHERImageDetector()
    _ = detector.detect(spliced_bytes)
    _ = detector.localize_manipulation(spliced_bytes)

    post_sha = hashlib.sha256(spliced_bytes).hexdigest()
    assert initial_sha == post_sha
