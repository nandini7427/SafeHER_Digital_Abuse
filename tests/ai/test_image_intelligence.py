"""Comprehensive unit tests for SafeHER Image Intelligence and Deepfake Detection Engine.

Tests all preprocessing boundaries, magic byte verification, corrupt image rejection,
EXIF sanitization, model runner execution, uncalibrated score contracts, uncertainty states,
explainability outputs, and resilient fallback mechanisms without network access.
"""

import io
from typing import Tuple
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pytest

from ai_core.image_intelligence.config import (
    ImageDetectorConfig,
    MANDATORY_IMAGE_AI_DISCLAIMER,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.explainability import (
    determine_assessment_label_and_uncertainty,
)
from ai_core.image_intelligence.model_runner import (
    BaseImageModelRunner,
    BaselineHeuristicRunner,
    ModelPrediction,
    NeuralVisionModelRunner,
)
from ai_core.image_intelligence.preprocessing import (
    PreprocessedImage,
    identify_magic_format,
    preprocess_image,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    ImageAssessmentResponse,
    ImageMetadataSummary,
    ManipulationRegion,
    ManipulationRegionType,
    UncertaintyState,
)
from ai_core.image_intelligence.service import (
    ImageIntelligenceService,
    get_image_detector,
    set_image_detector,
)


def _create_synthetic_image_bytes(
    dimensions: Tuple[int, int] = (256, 256),
    format_name: str = "JPEG",
    mode: str = "RGB",
    color: object = None,
) -> bytes:
    """Helper to generate in-memory synthetic image bytes."""
    if color is None:
        color = 128 if mode == "L" else (120, 150, 180)
    img = Image.new(mode, dimensions, color=color)
    # Add some non-trivial drawing to give texture
    draw = ImageDraw.Draw(img)
    x1 = max(1, dimensions[0] // 2)
    y1 = max(1, dimensions[1] // 2)
    draw.rectangle([0, 0, x1, y1], fill=200 if mode == "L" else (200, 50, 50))
    draw.line([0, 0, dimensions[0], dimensions[1]], fill=150 if mode == "L" else (50, 200, 50), width=3)

    buf = io.BytesIO()
    if format_name.upper() == "JPEG" and mode not in ("RGB", "L"):
        # JPEG requires RGB or L
        img = img.convert("RGB")
    img.save(buf, format=format_name)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Configuration Validation Tests
# ---------------------------------------------------------------------------


def test_image_config_validation():
    """Verify ImageDetectorConfig validates boundaries and types."""
    cfg = ImageDetectorConfig(
        max_image_dimension=2048,
        min_image_dimension=64,
        max_file_size_bytes=10 * 1024 * 1024,
        device="cpu",
        num_threads=2,
    )
    assert cfg.max_image_dimension == 2048
    assert cfg.min_image_dimension == 64
    assert cfg.device == "cpu"

    with pytest.raises(ValueError, match="min_image_dimension cannot exceed"):
        ImageDetectorConfig(min_image_dimension=1000, max_image_dimension=500)

    with pytest.raises(ValueError, match="device must be one of"):
        ImageDetectorConfig(device="tpu")

    with pytest.raises(ValueError, match="ela_quality cannot exceed 100"):
        ImageDetectorConfig(ela_quality=150)


# ---------------------------------------------------------------------------
# 2. Magic Byte & Format Validation Tests
# ---------------------------------------------------------------------------


def test_identify_magic_formats():
    """Verify magic bytes detection for JPEG, PNG, WEBP, BMP."""
    jpeg_bytes = _create_synthetic_image_bytes(format_name="JPEG")
    assert identify_magic_format(jpeg_bytes) == "JPEG"

    png_bytes = _create_synthetic_image_bytes(format_name="PNG")
    assert identify_magic_format(png_bytes) == "PNG"

    bmp_bytes = _create_synthetic_image_bytes(format_name="BMP")
    assert identify_magic_format(bmp_bytes) == "BMP"

    webp_bytes = _create_synthetic_image_bytes(format_name="WEBP")
    assert identify_magic_format(webp_bytes) == "WEBP"

    # Non-image streams
    assert identify_magic_format(b"Hello world this is plain text") is None
    assert identify_magic_format(b"%PDF-1.4 file header") is None
    assert identify_magic_format(b"") is None


def test_invalid_image_format_rejection():
    """Verify non-image bytes raise ValueError."""
    fake_payload = b"<!DOCTYPE html><html><body><h1>Not an image</h1></body></html>"
    with pytest.raises(ValueError, match="Unsupported or invalid image file format"):
        preprocess_image(fake_payload)


def test_empty_image_data_rejection():
    """Verify empty image payload raises ValueError."""
    with pytest.raises(ValueError, match="Image data cannot be empty"):
        preprocess_image(b"")


# ---------------------------------------------------------------------------
# 3. Corrupt Image Stream Tests
# ---------------------------------------------------------------------------


def test_corrupt_image_rejection():
    """Verify truncated or corrupted image data is caught."""
    valid_jpeg = _create_synthetic_image_bytes(format_name="JPEG")
    # Truncate image to simulate severed upload stream
    corrupt_jpeg = valid_jpeg[: len(valid_jpeg) // 4]
    with pytest.raises(ValueError, match="Corrupt or unreadable image stream"):
        preprocess_image(corrupt_jpeg)


# ---------------------------------------------------------------------------
# 4. Dimension & Size Limit Tests
# ---------------------------------------------------------------------------


def test_oversized_payload_rejection():
    """Verify files exceeding max_file_size_bytes are rejected."""
    cfg = ImageDetectorConfig(max_file_size_bytes=1024)  # 1 KB limit
    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")
    assert len(img_bytes) > 1024

    with pytest.raises(ValueError, match="exceeds the maximum limit"):
        preprocess_image(img_bytes, config=cfg)


def test_small_image_dimensions_rejection():
    """Verify images below min_image_dimension are rejected."""
    small_bytes = _create_synthetic_image_bytes(dimensions=(16, 16), format_name="PNG")
    cfg = ImageDetectorConfig(min_image_dimension=32)
    with pytest.raises(ValueError, match="smaller than the required minimum"):
        preprocess_image(small_bytes, config=cfg)


def test_oversized_image_dimensions_rejection():
    """Verify images exceeding max_image_dimension are rejected."""
    cfg = ImageDetectorConfig(max_image_dimension=300)
    large_bytes = _create_synthetic_image_bytes(dimensions=(350, 350), format_name="PNG")
    with pytest.raises(ValueError, match="exceed the maximum allowable limit"):
        preprocess_image(large_bytes, config=cfg)


# ---------------------------------------------------------------------------
# 5. Color Conversion & Working Copy Tests
# ---------------------------------------------------------------------------


def test_rgba_to_rgb_conversion():
    """Verify RGBA images are converted to RGB working copy without modifying raw bytes."""
    rgba_bytes = _create_synthetic_image_bytes(dimensions=(128, 128), format_name="PNG", mode="RGBA")
    preprocessed = preprocess_image(rgba_bytes)

    assert preprocessed.working_image.mode == "RGB"
    assert preprocessed.metadata.color_mode == "RGBA"
    assert any("converted from RGBA to standard RGB" in w for w in preprocessed.warnings)
    # Original raw bytes preserved exactly
    assert preprocessed.raw_bytes == rgba_bytes


def test_grayscale_to_rgb_conversion():
    """Verify grayscale (L) images are converted to RGB working copy."""
    gray_bytes = _create_synthetic_image_bytes(dimensions=(128, 128), format_name="PNG", mode="L")
    preprocessed = preprocess_image(gray_bytes)

    assert preprocessed.working_image.mode == "RGB"
    assert preprocessed.metadata.color_mode == "L"
    assert any("converted from L to standard RGB" in w for w in preprocessed.warnings)


# ---------------------------------------------------------------------------
# 6. EXIF Handling & Privacy Sanitization Tests
# ---------------------------------------------------------------------------


def test_exif_sanitization_strips_gps():
    """Verify GPSInfo and PII tags are purged from sanitized metadata."""
    from PIL import ExifTags
    img = Image.new("RGB", (128, 128), color=(100, 100, 100))
    exif = img.getexif()
    # Add Make (safe tag 0x010F = 271)
    exif[271] = "SafeHER Camera"
    # Add Model (safe tag 0x0110 = 272)
    exif[272] = "TestModel-100"
    # Add GPS IFD (tag 34853 = GPSInfo)
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = "N"

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    jpeg_with_exif = buf.getvalue()

    preprocessed = preprocess_image(jpeg_with_exif)
    assert preprocessed.metadata.has_exif is True
    # Safe tags retained
    assert preprocessed.metadata.sanitized_exif.get("Make") == "SafeHER Camera"
    assert preprocessed.metadata.sanitized_exif.get("Model") == "TestModel-100"
    # Sensitive GPS tag strictly stripped
    assert "GPSInfo" not in preprocessed.metadata.sanitized_exif
    assert 34853 not in preprocessed.metadata.sanitized_exif


# ---------------------------------------------------------------------------
# 7. Model Runner & Heuristic Tests
# ---------------------------------------------------------------------------


def test_baseline_heuristic_runner_execution():
    """Verify BaselineHeuristicRunner computes ELA metrics and outputs valid scores."""
    runner = BaselineHeuristicRunner()
    assert runner.is_available() is True
    assert "READY" in runner.get_status()

    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")
    preprocessed = preprocess_image(img_bytes)

    prediction = runner.predict(preprocessed)
    assert 0.0 <= prediction.raw_score <= 1.0
    assert "SafeHER-Baseline-ELA-Heuristics" in prediction.runner_name
    assert "ela_mean_diff" in prediction.auxiliary_metrics
    assert "ela_variance" in prediction.auxiliary_metrics


def test_neural_runner_unavailable_by_default():
    """Verify NeuralVisionModelRunner is gracefully disabled when not configured."""
    runner = NeuralVisionModelRunner(config=ImageDetectorConfig(enable_neural_detector=False))
    assert runner.is_available() is False
    assert "DISABLED" in runner.get_status()

    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")
    preprocessed = preprocess_image(img_bytes)
    with pytest.raises(RuntimeError, match="not available"):
        runner.predict(preprocessed)


def test_neural_runner_missing_file_graceful_status():
    """Verify NeuralVisionModelRunner reports missing file without crashing."""
    cfg = ImageDetectorConfig(
        enable_neural_detector=True,
        model_name_or_path="nonexistent_models/weights.pth",
    )
    runner = NeuralVisionModelRunner(config=cfg)
    assert runner.is_available() is False
    assert "not found at path" in runner.get_status()


# ---------------------------------------------------------------------------
# 8. Detector Fallback & Resilient Execution Tests
# ---------------------------------------------------------------------------


def test_detector_fallback_when_primary_runner_raises():
    """Verify SafeHERImageDetector falls back to baseline when a custom runner raises an exception."""
    failing_runner = MagicMock(spec=BaseImageModelRunner)
    failing_runner.predict.side_effect = RuntimeError("CUDA OOM or simulated model crash")

    detector = SafeHERImageDetector(runner=failing_runner)
    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")

    # Should NOT crash; falls back gracefully to baseline ELA runner
    response = detector.detect(img_bytes)
    assert response is not None
    assert response.raw_model_score >= 0.0
    assert response.is_calibrated is False
    assert response.confidence_score is None


# ---------------------------------------------------------------------------
# 9. Calibration & Uncertainty Contracts
# ---------------------------------------------------------------------------


def test_uncalibrated_score_contract():
    """Verify is_calibrated is strictly False and confidence_score is None."""
    detector = SafeHERImageDetector()
    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")
    response = detector.detect(img_bytes)

    assert response.is_calibrated is False
    assert response.confidence_score is None
    assert 0.0 <= response.raw_model_score <= 1.0


def test_conflicting_signals_do_not_claim_authentic():
    """A flat synthetic graphic with conflicting ELA/FFT signals must remain inconclusive."""
    detector = SafeHERImageDetector()
    # This simple synthetic graphic is not representative of a natural camera photograph.
    # It intentionally exercises the policy that detector disagreement must elevate uncertainty.
    clean_bytes = _create_synthetic_image_bytes(
        dimensions=(256, 256), format_name="PNG", color=(200, 200, 200)
    )
    response = detector.detect(clean_bytes)

    assert response.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert response.uncertainty_state == UncertaintyState.MODEL_UNCERTAIN
    assert response.disagreement_state == "conflicting_detection_signals"
    assert "does not prove an image is authentic" in response.disclaimer.lower()
    assert "scores are not averaged" in response.explanation.lower()


def test_uncertainty_state_low_resolution():
    """Verify small images under 128px trigger INSUFFICIENT_QUALITY uncertainty state."""
    img_bytes = _create_synthetic_image_bytes(dimensions=(64, 64), format_name="PNG")
    preprocessed = preprocess_image(img_bytes, config=ImageDetectorConfig(min_image_dimension=32))

    label, uncertainty = determine_assessment_label_and_uncertainty(
        raw_score=0.20,
        preprocessed=preprocessed,
        manipulation_regions=[],
    )
    assert uncertainty in (UncertaintyState.INSUFFICIENT_QUALITY, UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE)
    assert label == AssessmentLabel.INCONCLUSIVE


def test_uncertainty_state_moderate_boundary():
    """Verify scores near the 0.50 decision boundary produce MODERATE uncertainty."""
    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")
    preprocessed = preprocess_image(img_bytes)

    label, uncertainty = determine_assessment_label_and_uncertainty(
        raw_score=0.52,
        preprocessed=preprocessed,
        manipulation_regions=[],
    )
    assert uncertainty in (UncertaintyState.MODERATE, UncertaintyState.MODEL_UNCERTAIN)


# ---------------------------------------------------------------------------
# 10. Explainability & Disclaimer Verification Tests
# ---------------------------------------------------------------------------


def test_forensic_explanation_structure():
    """Verify explanation includes examination scope, evidence, and human review mandate."""
    detector = SafeHERImageDetector()
    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")
    response = detector.detect(img_bytes)

    exp = response.explanation
    # Must explain scope
    assert "evaluated this JPEG image" in exp
    assert "Error Level Analysis" in exp
    # Must explain safety and human verification
    assert "cannot determine identity, consent" in exp
    assert "Independent verification by a qualified digital forensic analyst is required" in exp
    # Model limitations must be populated
    assert len(response.model_limitations) >= 4


def test_mandatory_disclaimer_immutability():
    """Verify mandatory disclaimer is preserved and cannot be tampered with."""
    detector = SafeHERImageDetector()
    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")
    response = detector.detect(img_bytes)

    assert response.disclaimer == MANDATORY_IMAGE_AI_DISCLAIMER

    # Schema must reject tampered disclaimer
    with pytest.raises(ValueError, match="disclaimer must exactly match"):
        ImageAssessmentResponse(
            detector_name="Test",
            detector_version="1.0",
            detection_task=DetectionTask.HYBRID_MANIPULATION,
            raw_model_score=0.5,
            assessment_label=AssessmentLabel.INCONCLUSIVE,
            uncertainty_state=UncertaintyState.LOW,
            explanation="Test explanation",
            disclaimer="Tampered disclaimer text",
        )


# ---------------------------------------------------------------------------
# 11. Service Layer & Telemetry Tests
# ---------------------------------------------------------------------------


def test_image_service_singleton_and_assessment():
    """Verify ImageIntelligenceService lifecycle and privacy-preserving assessment."""
    service = ImageIntelligenceService()
    test_image_id = uuid4()
    img_bytes = _create_synthetic_image_bytes(dimensions=(256, 256), format_name="JPEG")

    response = service.assess_image(data=img_bytes, image_id=test_image_id)
    assert response.image_id == test_image_id
    assert response.metadata_summary is not None
    assert response.metadata_summary.format == "JPEG"
    assert response.metadata_summary.dimensions == (256, 256)


def test_image_service_validation_rejection():
    """Verify ImageIntelligenceService bubbles ValueError on invalid data."""
    service = ImageIntelligenceService()
    with pytest.raises(ValueError, match="Unsupported or invalid image file format"):
        service.assess_image(b"not a valid image")


def test_set_image_detector_override():
    """Verify set_image_detector allows mocking/overriding singleton detector."""
    mock_detector = MagicMock(spec=SafeHERImageDetector)
    set_image_detector(mock_detector)
    assert get_image_detector() == mock_detector
    # Reset back to None so other tests are unaffected
    set_image_detector(None)
