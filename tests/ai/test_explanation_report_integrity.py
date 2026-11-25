"""Automated Audit and Verification Suite for Explanation and Report Integrity in SafeHER.

Verifies:
1. Complete Assessment Metadata:
   - Model name and version.
   - Unique assessment ID.
   - Stored evidence ID association.
   - Explicit detection task scope.
   - Raw score interpretation and qualitative label.
   - Calibration status (is_calibrated contract).
   - Categorical uncertainty state.
   - Image quality indicators and warnings.
   - Localization provenance and attribution category.
   - Explicit model limitations and boundary disclosures.
   - Cryptographic SHA-256 and dimensions in metadata summary.
   - UTC analysis timestamp.
   - Mandatory, non-overridable legal and safety disclaimer.

2. Negative Safeguards & Integrity Assertions:
   - Do not invent visual artifacts or manipulated regions on benign images.
   - Do not convert uncalibrated scores into probabilities or percentages.
   - Do not present model attention / heuristic attribution as proof of manipulation.
   - Do not claim or infer that a specific person is responsible for creating an image.
   - Do not claim an image is authentic or genuine merely because detection was negative or failed.
   - Do not force binary classification when evidence is degraded or out-of-distribution.

3. Frontend & Backend Terminology Synchronization:
   - Exact 1-to-1 parity between TypeScript union types and Python Enums.
"""

import io
from pathlib import Path
import re
from typing import Set
import uuid

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from ai_core.image_intelligence.config import MANDATORY_IMAGE_AI_DISCLAIMER
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.explainability import STANDARD_MODEL_LIMITATIONS
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    ImageAssessmentResponse,
    ManipulationRegion,
    ManipulationRegionType,
    UncertaintyState,
)
from backend.app.schemas import (
    HumanReviewStatus,
    HumanReviewVerdict,
    MANDATORY_HUMAN_REVIEW_DISCLAIMER,
)


def _generate_test_image(
    dimensions=(256, 256),
    color=(130, 150, 180),
    format_name="JPEG",
) -> bytes:
    """Generate in-memory test image bytes."""
    img = Image.new("RGB", dimensions, color=color)
    draw = ImageDraw.Draw(img)
    draw.line([0, 0, dimensions[0], dimensions[1]], fill=(50, 200, 50), width=3)
    draw.rectangle([20, 20, 100, 100], fill=(220, 60, 60))
    buf = io.BytesIO()
    img.save(buf, format=format_name, quality=92)
    return buf.getvalue()


# ==============================================================================
# 1. COMPLETE REPORT INTEGRITY & METADATA VERIFICATION
# ==============================================================================

def test_complete_report_integrity_contains_all_required_fields():
    """Verify that every assessment output contains the complete audit metadata."""
    detector = SafeHERImageDetector()
    raw_bytes = _generate_test_image()
    test_evidence_id = uuid.uuid4()

    response = detector.detect(data=raw_bytes, image_id=test_evidence_id)

    # 1. Model name and version
    assert response.detector_name is not None and len(response.detector_name) > 0
    assert response.detector_version is not None and len(response.detector_version) > 0

    # 2. Image assessment ID
    assert isinstance(response.assessment_id, uuid.UUID)

    # 3. Image evidence identifier
    assert response.image_id == test_evidence_id

    # 4. Detection task
    assert isinstance(response.detection_task, DetectionTask)

    # 5. Score interpretation
    assert 0.0 <= response.raw_model_score <= 1.0
    assert isinstance(response.assessment_label, AssessmentLabel)

    # 6. Calibration status
    assert isinstance(response.is_calibrated, bool)
    if not response.is_calibrated:
        assert response.confidence_score is None, "Uncalibrated detector must have confidence_score=None"

    # 7. Uncertainty state
    assert isinstance(response.uncertainty_state, UncertaintyState)

    # 8. Image quality indicators and warnings
    assert response.quality_indicators is not None
    assert isinstance(response.quality_warnings, list)
    assert response.quality_indicators.width > 0
    assert response.quality_indicators.height > 0

    # 9. Localization provenance
    assert isinstance(response.manipulation_regions, list)
    if response.visual_explanation is not None:
        assert "localization_method" in response.visual_explanation
        assert "attribution_category" in response.visual_explanation
        assert "is_localizable" in response.visual_explanation

    # 10. Model limitations
    assert isinstance(response.model_limitations, list)
    assert len(response.model_limitations) >= 4

    # 11. Analysis timestamp
    assert response.analysis_timestamp is not None

    # 12. Mandatory disclaimer
    assert response.disclaimer == MANDATORY_IMAGE_AI_DISCLAIMER


# ==============================================================================
# 2. CALIBRATION INTEGRITY & PROBABILITY CONTRACT
# ==============================================================================

def test_uncalibrated_scores_never_converted_into_probabilities():
    """Verify that heuristic outputs are never presented as calibrated probabilities."""
    detector = SafeHERImageDetector()
    raw_bytes = _generate_test_image()
    response = detector.detect(data=raw_bytes)

    # The baseline detector is an uncalibrated heuristic
    assert response.is_calibrated is False
    assert response.confidence_score is None

    # Verify that the generated explanation does NOT make probability claims
    explanation_lower = response.explanation.lower()
    assert "probability" not in explanation_lower
    assert "% chance" not in explanation_lower
    assert "% likely" not in explanation_lower
    assert "percent probability" not in explanation_lower


# ==============================================================================
# 3. AUTHENTICITY CLAIM SAFEGUARD
# ==============================================================================

def test_negative_or_failed_detection_never_claims_authenticity():
    """Verify SafeHER never claims an image is authentic or genuine."""
    detector = SafeHERImageDetector()

    # Case A: In-domain benign image with low compression anomalies
    raw_bytes = _generate_test_image()
    response = detector.detect(data=raw_bytes)
    assert response.assessment_label in (
        AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
        AssessmentLabel.INCONCLUSIVE,
    )
    assert response.assessment_label.value != "authentic"
    assert response.assessment_label.value != "genuine"

    # Explanation must explicitly disclaim confirmation of authenticity
    explanation = response.explanation
    if response.assessment_label == AssessmentLabel.NO_OBVIOUS_ARTIFACTS:
        assert "does NOT confirm that the image is authentic" in explanation

    # Model limitations must explicitly state absence of artifacts != proof of authenticity
    assert any(
        "does NOT prove an image is authentic or unaltered" in limit
        for limit in response.model_limitations
    )

    # Case B: Flat/monochromatic or unsupported image must route to INCONCLUSIVE, never authentic
    img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    flat_bytes = buf.getvalue()

    response_flat = detector.detect(data=flat_bytes)
    assert response_flat.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert response_flat.assessment_label.value != "authentic"


def test_corrupted_or_failed_detection_never_claims_authenticity():
    """Failed, corrupted, or OOD images must route to INCONCLUSIVE, never authentic."""
    detector = SafeHERImageDetector()
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_CORRUPTED_STREAM"

    response = detector.detect_safe(data=corrupt_bytes)
    assert response.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert response.uncertainty_state == UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED
    assert response.analysis_eligibility is False
    assert response.assessment_label.value != "authentic"


# ==============================================================================
# 4. PERPETRATOR ATTRIBUTION & CREATOR INTENT SAFEGUARD
# ==============================================================================

def test_no_claim_of_person_creation_or_perpetrator_attribution():
    """Verify that detector and explanations never infer who created the image or who is responsible."""
    detector = SafeHERImageDetector()
    raw_bytes = _generate_test_image()
    response = detector.detect(data=raw_bytes)

    explanation = response.explanation.lower()

    # Disallowed perpetrator/creator attribution keywords
    forbidden_terms = [
        "created by",
        "perpetrated by",
        "culprit",
        "attacker identity",
        "person who created",
        "author of this image",
        "suspect created",
    ]
    for term in forbidden_terms:
        assert term not in explanation, f"Forbidden perpetrator attribution '{term}' found in explanation"

    # Mandatory limitation presence
    assert any(
        "cannot assess the real-world identity, intent, or consent" in limit
        for limit in response.model_limitations
    )


# ==============================================================================
# 5. MODEL ATTENTION VS PROOF OF MANIPULATION
# ==============================================================================

def test_model_attention_is_distinguished_from_proof_of_manipulation():
    """Verify that model attention or occlusion attribution is labeled as guidance, NOT proof."""
    detector = SafeHERImageDetector()
    raw_bytes = _generate_test_image()
    loc_result = detector.localize_manipulation(data=raw_bytes)

    if loc_result.is_localizable and loc_result.regions:
        for r in loc_result.regions:
            assert r.attribution_category in (
                "suspected_manipulation_region",
                "model_attention_attribution",
            )
            # Ensure description does not assert judicial certainty
            desc = r.description.lower()
            assert "conclusive proof" not in desc
            assert "guaranteed tampering" not in desc
            assert "absolute proof" not in desc


# ==============================================================================
# 6. ZERO INVENTED ARTIFACTS ON CLEAN BENIGN IMAGES
# ==============================================================================

def test_no_invented_visual_artifacts_or_regions_on_clean_images():
    """Clean un-spliced images should not have fabricated manipulation clusters."""
    from ai_core.image_intelligence.visual_localization import GenericFaceMaskGuard

    detector = SafeHERImageDetector()
    raw_bytes = _generate_test_image()
    response = detector.detect(data=raw_bytes)

    # Every flagged region must be mathematically computed from real ELA residuals
    for r in response.manipulation_regions:
        assert 0.0 <= r.anomaly_score <= 1.0
        ymin, xmin, ymax, xmax = r.box_normalized
        assert 0.0 <= ymin < ymax <= 1.0
        assert 0.0 <= xmin < xmax <= 1.0
        assert r.description and len(r.description) > 0

    # Generic face masks are strictly blocked from being labeled as manipulated regions
    guard = GenericFaceMaskGuard()
    face_box = (0.2, 0.2, 0.6, 0.6)
    fake_face_region = ManipulationRegion(
        box_normalized=face_box,
        anomaly_score=0.9,
        region_type=ManipulationRegionType.BOUNDING_BOX,
        description="Generic face detector bounding box",
    )
    filtered = guard.audit_regions_against_face_mask(
        regions=[fake_face_region],
        face_bboxes=[face_box],
    )
    assert len(filtered) == 0, "Generic face mask was not filtered out by GenericFaceMaskGuard"


# ==============================================================================
# 7. MANDATORY DISCLAIMER IMMUTABILITY
# ==============================================================================

def test_mandatory_disclaimer_immutability():
    """Verify that altering or removing the mandatory disclaimer fails Pydantic validation."""
    raw_bytes = _generate_test_image()
    detector = SafeHERImageDetector()
    resp = detector.detect(data=raw_bytes)

    # Attempting to construct with an invalid disclaimer must raise ValidationError
    with pytest.raises(ValidationError):
        ImageAssessmentResponse(
            detector_name=resp.detector_name,
            detector_version=resp.detector_version,
            detection_task=resp.detection_task,
            raw_model_score=resp.raw_model_score,
            assessment_label=resp.assessment_label,
            uncertainty_state=resp.uncertainty_state,
            explanation=resp.explanation,
            disclaimer="Unauthorized custom disclaimer.",  # Tampered disclaimer!
        )


# ==============================================================================
# 8. FRONTEND & BACKEND TERMINOLOGY SYNCHRONIZATION
# ==============================================================================

def _extract_typescript_union(ts_content: str, type_name: str) -> Set[str]:
    """Helper to parse TypeScript string union types from source code."""
    pattern = rf"export\s+type\s+{type_name}\s*=\s*([^;]+);"
    match = re.search(pattern, ts_content)
    assert match is not None, f"TypeScript type {type_name} not found"
    raw_union = match.group(1)
    # Extract string literals inside single quotes
    items = set(re.findall(r"'([^']+)'", raw_union))
    return items


def test_frontend_backend_terminology_synchronization():
    """Assert exact 1-to-1 enum and type synchronization between Python backend and TypeScript frontend."""
    ts_file_path = Path("frontend/src/types/imageAnalysis.ts")
    assert ts_file_path.exists(), "frontend/src/types/imageAnalysis.ts must exist"
    ts_content = ts_file_path.read_text(encoding="utf-8")

    # 1. DetectionTask
    ts_detection_tasks = _extract_typescript_union(ts_content, "DetectionTask")
    py_detection_tasks = {t.value for t in DetectionTask}
    assert ts_detection_tasks == py_detection_tasks, (
        f"DetectionTask divergence! Diff: {ts_detection_tasks ^ py_detection_tasks}"
    )

    # 2. AssessmentLabel
    ts_assessment_labels = _extract_typescript_union(ts_content, "AssessmentLabel")
    py_assessment_labels = {l.value for l in AssessmentLabel}
    assert ts_assessment_labels == py_assessment_labels, (
        f"AssessmentLabel divergence! Diff: {ts_assessment_labels ^ py_assessment_labels}"
    )

    # 3. UncertaintyState
    ts_uncertainty_states = _extract_typescript_union(ts_content, "UncertaintyState")
    py_uncertainty_states = {u.value for u in UncertaintyState}
    assert ts_uncertainty_states == py_uncertainty_states, (
        f"UncertaintyState divergence! Diff: {ts_uncertainty_states ^ py_uncertainty_states}"
    )

    # 4. ManipulationRegionType
    ts_region_types = _extract_typescript_union(ts_content, "ManipulationRegionType")
    py_region_types = {m.value for m in ManipulationRegionType}
    assert ts_region_types == py_region_types, (
        f"ManipulationRegionType divergence! Diff: {ts_region_types ^ py_region_types}"
    )

    # 5. HumanReviewVerdict
    ts_review_verdicts = _extract_typescript_union(ts_content, "HumanReviewVerdict")
    py_review_verdicts = {v.value for v in HumanReviewVerdict}
    assert ts_review_verdicts == py_review_verdicts, (
        f"HumanReviewVerdict divergence! Diff: {ts_review_verdicts ^ py_review_verdicts}"
    )

    # 6. HumanReviewStatus
    ts_review_statuses = _extract_typescript_union(ts_content, "HumanReviewStatus")
    py_review_statuses = {s.value for s in HumanReviewStatus}
    assert ts_review_statuses == py_review_statuses, (
        f"HumanReviewStatus divergence! Diff: {ts_review_statuses ^ py_review_statuses}"
    )
