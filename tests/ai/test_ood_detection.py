"""Unit tests for SafeHER Out-of-Distribution (OOD) Detection and Domain Triage.

Verifies:
- Accurate classification of supported in-domain and unsupported OOD categories.
- Avoidance of forced binary classification on unsupported domains (returns INCONCLUSIVE).
- Strict non-throwing triage with ANALYSIS_POOR_QUALITY_EVIDENCE or OUTSIDE_SUPPORTED_DOMAIN.
- Cryptographic SHA-256 byte invariance of evidence inputs.
- Privacy-safe logging without victim image content exposure.
"""

import hashlib
import io
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image
import pytest

from ai_core.evaluation.ood_benchmark import (
    generate_digital_illustration,
    generate_document_or_scanned_text,
    generate_drawing_or_sketch,
    generate_extremely_compressed,
    generate_landscape_without_subject,
    generate_medical_imagery,
    generate_meme_or_text_graphic,
    generate_photographic_portrait,
    generate_photographic_portrait_medium,
    generate_screenshot_or_ui,
    generate_unsupported_format,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.ood_detector import (
    DomainCategory,
    OutOfDistributionDetector,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    UncertaintyState,
)


def test_ood_detector_accepts_in_domain_portraits():
    """Verify in-domain photographic portraits are correctly accepted."""
    detector = OutOfDistributionDetector()

    # Portrait close-up
    portrait_bytes = generate_photographic_portrait(seed=42)
    img = Image.open(io.BytesIO(portrait_bytes))
    res = detector.classify_domain(img=img, format_name="JPEG", has_exif=True)

    assert res.is_in_domain is True
    assert res.category == DomainCategory.PHOTOGRAPHIC_PORTRAIT
    assert res.uncertainty_state == UncertaintyState.ANALYSIS_USABLE_EVIDENCE
    assert res.analysis_eligibility is True
    assert res.reason is None

    # Portrait medium-shot
    portrait_med_bytes = generate_photographic_portrait_medium(seed=42)
    img_med = Image.open(io.BytesIO(portrait_med_bytes))
    res_med = detector.classify_domain(img=img_med, format_name="JPEG", has_exif=True)

    assert res_med.is_in_domain is True
    assert res_med.category == DomainCategory.PHOTOGRAPHIC_PORTRAIT
    assert res_med.analysis_eligibility is True


@pytest.mark.parametrize(
    "gen_fn,expected_cat,expected_uncertainty",
    [
        (generate_drawing_or_sketch, DomainCategory.DRAWING_OR_SKETCH, UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
        (generate_digital_illustration, DomainCategory.DIGITAL_ILLUSTRATION, UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
        (generate_meme_or_text_graphic, DomainCategory.MEME_OR_TEXT_GRAPHIC, UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
        (generate_screenshot_or_ui, DomainCategory.SCREENSHOT_OR_UI, UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
        (generate_medical_imagery, DomainCategory.MEDICAL_IMAGERY, UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
        (generate_document_or_scanned_text, DomainCategory.DOCUMENT_OR_SCANNED_TEXT, UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
        (generate_landscape_without_subject, DomainCategory.LANDSCAPE_WITHOUT_SUBJECT, UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
        (generate_extremely_compressed, DomainCategory.EXTREMELY_COMPRESSED, UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE),
    ],
)
def test_ood_detector_identifies_unsupported_categories(gen_fn, expected_cat, expected_uncertainty):
    """Verify each unsupported category is rejected with the correct category and uncertainty state."""
    detector = OutOfDistributionDetector()
    data = gen_fn(seed=101)
    img = Image.open(io.BytesIO(data))

    fmt = "PNG" if expected_cat in (DomainCategory.DIGITAL_ILLUSTRATION, DomainCategory.SCREENSHOT_OR_UI) else "JPEG"
    res = detector.classify_domain(img=img, format_name=fmt, has_exif=False)

    assert res.is_in_domain is False
    assert res.category == expected_cat
    assert res.uncertainty_state == expected_uncertainty
    assert res.analysis_eligibility is False
    assert res.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert len(res.warnings) > 0


def test_ood_detector_handles_unsupported_or_corrupted_format():
    """Verify corrupted stream triggers UNSUPPORTED_FORMAT."""
    detector = OutOfDistributionDetector()
    corrupt_bytes = generate_unsupported_format(seed=42)

    # Calling with corrupted flag
    dummy_img = Image.new("RGB", (128, 128))
    res = detector.classify_domain(img=dummy_img, is_corrupted=True)

    assert res.is_in_domain is False
    assert res.category == DomainCategory.UNSUPPORTED_FORMAT
    assert res.uncertainty_state == UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED
    assert res.analysis_eligibility is False
    assert res.assessment_label == AssessmentLabel.INCONCLUSIVE


def test_detector_end_to_end_avoids_forced_binary_classification_on_ood():
    """CRITICAL MANDATE: Verify end-to-end detector strictly returns INCONCLUSIVE for OOD images."""
    det = SafeHERImageDetector()

    # 1. Document / Scanned Paper
    doc_bytes = generate_document_or_scanned_text(seed=42)
    res_doc = det.detect(doc_bytes)
    assert res_doc.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert res_doc.uncertainty_state == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN
    assert res_doc.analysis_eligibility is False
    assert res_doc.is_calibrated is False
    assert res_doc.confidence_score is None
    assert "document" in res_doc.quality_indicators.domain_category

    # 2. Medical Scan
    med_bytes = generate_medical_imagery(seed=42)
    res_med = det.detect(med_bytes)
    assert res_med.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert res_med.uncertainty_state == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN
    assert res_med.analysis_eligibility is False
    assert "medical" in res_med.quality_indicators.domain_category

    # 3. Drawing / Sketch
    draw_bytes = generate_drawing_or_sketch(seed=42)
    res_draw = det.detect(draw_bytes)
    assert res_draw.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert res_draw.analysis_eligibility is False
    assert "drawing" in res_draw.quality_indicators.domain_category

    # 4. Landscape / Scenery
    land_bytes = generate_landscape_without_subject(seed=42)
    res_land = det.detect(land_bytes)
    assert res_land.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert res_land.analysis_eligibility is False
    assert "landscape" in res_land.quality_indicators.domain_category


def test_evidence_sha256_invariance_across_ood_evaluation():
    """Verify input evidence bytes are strictly immutable and SHA-256 invariant."""
    det = SafeHERImageDetector()
    sample_data = generate_digital_illustration(seed=99)

    initial_hash = hashlib.sha256(sample_data).hexdigest()
    res = det.detect_safe(sample_data)
    post_hash = hashlib.sha256(sample_data).hexdigest()

    assert initial_hash == post_hash
    assert res.metadata_summary is not None
    assert res.metadata_summary.sha256 == initial_hash


def test_privacy_safe_logging_does_not_leak_image_bytes(caplog):
    """Verify that domain classification logs do not expose raw pixel bytes or base64 strings."""
    caplog.set_level(logging.INFO)
    det = SafeHERImageDetector()
    meme_bytes = generate_meme_or_text_graphic(seed=42)

    res = det.detect(meme_bytes)
    assert res.assessment_label == AssessmentLabel.INCONCLUSIVE

    # Ensure log output contains no base64 fragments or raster arrays
    for record in caplog.records:
        msg = record.getMessage()
        assert "data:image" not in msg
        assert len(msg) < 500  # No massive dumps
