"""Unit tests for SafeHER Model B Facial Manipulation Detector.

Verifies:
- Bounding box coordinate calculations and validation.
- Face candidate detection, prominence sorting, and primary face selection.
- Quality and visibility condition assessment (low-resolution, border truncation, normal).
- Face crop extraction with boundary context margins and source coordinate preservation.
- Dense patch-level spatial anomaly mapping (14x14 grid) and fine-grained category classification.
- Zero-face, multi-face, and low-res handling with appropriate uncertainty triage.
- Rejection of missing, unweighted, or untrained model artifacts.
- Spatial patch anomaly localization and source image re-projection.
- Evaluation metrics (precision, recall, F1, IoU, compression robustness) and legal feasibility blocker documentation.
"""

import io
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import pytest
import torch

from ai_core.neural.facial import (
    FaceCandidate,
    FaceCondition,
    FaceDetectionStatus,
    FaceRegionPreprocessor,
    FacialEvaluationReport,
    FacialManipulationInferenceResult,
    FacialManipulationType,
    FacialUncertaintyState,
    LocalizedManipulationRegion,
    ModelBFacialEvaluator,
    ModelBFacialInferenceService,
    NormalizedBoundingBox,
    PatchForensicsFacialDetector,
    compute_bounding_box_iou,
)


def create_test_face_image(width: int = 400, height: int = 400) -> Image.Image:
    """Create a synthetic RGB test image simulating face and background."""
    arr = np.full((height, width, 3), 50, dtype=np.uint8)  # Dark background
    # Central face-like region with skin tone
    arr[int(height * 0.2):int(height * 0.8), int(width * 0.2):int(width * 0.8)] = [210, 160, 130]
    return Image.fromarray(arr, mode="RGB")


# ============================================================================
# 1. Normalized Bounding Box Tests
# ============================================================================

def test_normalized_bounding_box_valid():
    """Verify valid normalized bounding box properties and pixel conversion."""
    box = NormalizedBoundingBox(x_min=0.1, y_min=0.2, x_max=0.7, y_max=0.8)
    assert abs(box.width - 0.6) < 1e-5
    assert abs(box.height - 0.6) < 1e-5
    assert abs(box.area - 0.36) < 1e-5

    x1, y1, x2, y2 = box.to_pixel_coordinates(400, 400)
    assert (x1, y1, x2, y2) == (40, 80, 280, 320)


def test_normalized_bounding_box_invalid_bounds():
    """Verify bounding box rejects inverted bounds where max < min."""
    with pytest.raises(ValueError, match="x_max .* cannot be less than x_min"):
        NormalizedBoundingBox(x_min=0.8, y_min=0.2, x_max=0.3, y_max=0.8)

    with pytest.raises(ValueError, match="y_max .* cannot be less than y_min"):
        NormalizedBoundingBox(x_min=0.1, y_min=0.8, x_max=0.5, y_max=0.2)


def test_bounding_box_iou():
    """Verify IoU calculation between matching, overlapping, and disjoint boxes."""
    b1 = NormalizedBoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    b2 = NormalizedBoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    assert abs(compute_bounding_box_iou(b1, b2) - 1.0) < 1e-5

    # Disjoint
    b3 = NormalizedBoundingBox(x_min=0.6, y_min=0.6, x_max=1.0, y_max=1.0)
    assert compute_bounding_box_iou(b1, b3) == 0.0

    # Half overlap
    b4 = NormalizedBoundingBox(x_min=0.25, y_min=0.0, x_max=0.75, y_max=0.5)
    iou = compute_bounding_box_iou(b1, b4)
    assert 0.3 < iou < 0.4


# ============================================================================
# 2. Face Preprocessor Candidate Detection & Condition Tests
# ============================================================================

def test_detect_face_candidates_multi_face():
    """Verify preprocessor handles multiple detected faces and sorts by prominence."""
    preprocessor = FaceRegionPreprocessor()
    img = create_test_face_image(400, 400)

    # Supply two faces: Face A (small background face) and Face B (large foreground face)
    small_box = NormalizedBoundingBox(x_min=0.05, y_min=0.05, x_max=0.20, y_max=0.20)
    large_box = NormalizedBoundingBox(x_min=0.30, y_min=0.20, x_max=0.80, y_max=0.80)

    status, candidates = preprocessor.detect_face_candidates(img, provided_boxes=[small_box, large_box])

    assert status == FaceDetectionStatus.MULTIPLE_FACES_DETECTED
    assert len(candidates) == 2
    # The larger face must be primary (first in list)
    assert candidates[0].is_primary is True
    assert candidates[0].bounding_box == large_box
    assert candidates[1].is_primary is False


def test_detect_face_candidates_zero_face():
    """Verify preprocessor reports NO_FACE_DETECTED when provided box list is empty."""
    preprocessor = FaceRegionPreprocessor()
    img = create_test_face_image(400, 400)

    status, candidates = preprocessor.detect_face_candidates(img, provided_boxes=[])
    assert status == FaceDetectionStatus.NO_FACE_DETECTED
    assert len(candidates) == 0


def test_face_condition_low_resolution():
    """Verify that a face smaller than min_face_resolution is flagged LOW_RESOLUTION."""
    preprocessor = FaceRegionPreprocessor(min_face_resolution=64)
    img = create_test_face_image(400, 400)

    # 400 * 0.1 = 40 pixels (< 64)
    low_res_box = NormalizedBoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    status, candidates = preprocessor.detect_face_candidates(img, provided_boxes=[low_res_box])

    assert len(candidates) == 1
    assert FaceCondition.LOW_RESOLUTION in candidates[0].face_conditions


def test_face_condition_partially_visible():
    """Verify that a face touching the image border is flagged PARTIALLY_VISIBLE."""
    preprocessor = FaceRegionPreprocessor()
    img = create_test_face_image(400, 400)

    # Touches left boundary (x_min = 0.0)
    border_box = NormalizedBoundingBox(x_min=0.0, y_min=0.2, x_max=0.5, y_max=0.7)
    status, candidates = preprocessor.detect_face_candidates(img, provided_boxes=[border_box])

    assert len(candidates) == 1
    assert FaceCondition.PARTIALLY_VISIBLE in candidates[0].face_conditions


# ============================================================================
# 3. Crop Extraction & Coordinate Preservation Tests
# ============================================================================

def test_extract_face_crop_with_margin():
    """Verify crop extraction with 20% margin and tensor output shape."""
    preprocessor = FaceRegionPreprocessor(margin_ratio=0.20, target_resolution=(224, 224))
    img = create_test_face_image(400, 400)

    candidate = FaceCandidate(
        face_id="FACE-001",
        bounding_box=NormalizedBoundingBox(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7),
        confidence=1.0,
        face_conditions=[FaceCondition.NORMAL],
    )

    tensor, expanded_box = preprocessor.extract_face_crop(img, candidate, apply_margin=True)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32

    # Width: 0.4. Margin: 0.4 * 0.2 = 0.08. Expanded: 0.3 - 0.08 = 0.22 to 0.7 + 0.08 = 0.78
    assert abs(expanded_box.x_min - 0.22) < 1e-3
    assert abs(expanded_box.x_max - 0.78) < 1e-3


def test_map_crop_coords_to_source():
    """Verify mathematical re-projection from crop coordinates to source image coordinates."""
    preprocessor = FaceRegionPreprocessor()
    expanded_box = NormalizedBoundingBox(x_min=0.20, y_min=0.20, x_max=0.60, y_max=0.60)
    # Patch anomaly localized in center of face crop: [0.25, 0.25, 0.75, 0.75]
    crop_box = NormalizedBoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

    src_box = preprocessor.map_crop_coords_to_source(crop_box, expanded_box)

    # src_x_min = 0.20 + 0.25 * 0.40 = 0.30
    # src_x_max = 0.20 + 0.75 * 0.40 = 0.50
    assert abs(src_box.x_min - 0.30) < 1e-3
    assert abs(src_box.x_max - 0.50) < 1e-3
    assert abs(src_box.y_min - 0.30) < 1e-3
    assert abs(src_box.y_max - 0.50) < 1e-3


# ============================================================================
# 4. Neural Architecture (PatchForensicsFacialDetector) Tests
# ============================================================================

def test_patch_forensics_forward_pass_and_validation():
    """Verify dual-head forward pass, spatial anomaly map dimensions, and input validation."""
    model = PatchForensicsFacialDetector(num_categories=5, is_trained=True)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    patch_map, cat_logits = model(x)

    assert patch_map.shape == (2, 1, 14, 14)
    assert cat_logits.shape == (2, 5)

    # Sigmoidal output for patch anomaly map
    assert torch.all(patch_map >= 0.0) and torch.all(patch_map <= 1.0)

    # Input validation rejects 3D tensor
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model.validate_input_tensor(torch.randn(3, 224, 224))


# ============================================================================
# 5. Model B Inference Service Tests
# ============================================================================

def test_service_zero_face_detected():
    """Verify service returns NO_FACE_DETECTED uncertainty and disclosures when no faces exist."""
    model = PatchForensicsFacialDetector(is_trained=True)
    service = ModelBFacialInferenceService(model=model, enforce_trained=True)

    img = create_test_face_image(400, 400)
    result = service.analyze_image(image=img, provided_boxes=[])

    assert result.face_detection_status == FaceDetectionStatus.NO_FACE_DETECTED
    assert result.faces_detected_count == 0
    assert result.uncertainty == FacialUncertaintyState.NO_FACE_IN_FRAME
    assert result.inference_status == "NO_FACE_DETECTED"
    assert len(result.manipulation_category_indicators) == 0
    assert result.spatial_anomaly_map is None
    assert result.predicted_category == FacialManipulationType.NOT_ASSESSED
    assert result.model_available is True


def test_model_b_rejects_checkpoint_without_commercial_usage_approval():
    """Model B must reject trained-looking artifacts without explicit permitted-use metadata."""
    from ai_core.neural.training.checkpointer import save_detector_checkpoint

    model = PatchForensicsFacialDetector(is_trained=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model_b_research_only.pt"
        save_detector_checkpoint(
            model,
            checkpoint_path,
            extra_metadata={
                "is_trained": True,
                "training_completed": True,
                "epoch": 2,
                "training_manifest_hash": "a" * 64,
                "split_manifest_hash": "b" * 64,
                "dataset_versions": ["research-fixture-v1"],
                "approved_usage_state": "RESEARCH_RESTRICTED",
            },
        )
        service = ModelBFacialInferenceService(checkpoint_path=checkpoint_path, enforce_trained=True)
        assert service.model is None
        assert service.operational_status == "UNTRAINED_MODEL_REJECTED"

        result = service.analyze_image(
            create_test_face_image(),
            provided_boxes=[NormalizedBoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)],
        )
        assert result.predicted_category == FacialManipulationType.NOT_ASSESSED
        assert result.inference_status == "UNTRAINED_MODEL_REJECTED"
        assert result.manipulation_category_indicators == {}


def test_service_does_not_mistake_center_crop_for_face_detection():
    """Without detector output or annotated boxes, Model B must not invent a face region."""
    model = PatchForensicsFacialDetector(is_trained=True)
    service = ModelBFacialInferenceService(model=model, enforce_trained=True)

    result = service.analyze_image(image=create_test_face_image(400, 400))

    assert result.face_detection_status == FaceDetectionStatus.LOCALIZATION_UNAVAILABLE
    assert result.uncertainty == FacialUncertaintyState.FACE_LOCALIZATION_UNAVAILABLE
    assert result.inference_status == "FACE_LOCALIZATION_UNAVAILABLE"
    # The trained model object is available, but no analysis ran without face localization.
    assert result.model_available is True
    assert result.predicted_category == FacialManipulationType.NOT_ASSESSED
    assert result.manipulation_category_indicators == {}


def test_service_multi_face_analyzes_primary():
    """Verify multi-face scenes analyze the primary face without silent unrelated cropping."""
    model = PatchForensicsFacialDetector(is_trained=True)
    service = ModelBFacialInferenceService(model=model, enforce_trained=True)

    img = create_test_face_image(400, 400)
    box_small = NormalizedBoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    box_large = NormalizedBoundingBox(x_min=0.3, y_min=0.3, x_max=0.8, y_max=0.8)

    result = service.analyze_image(image=img, provided_boxes=[box_small, box_large])

    assert result.face_detection_status == FaceDetectionStatus.MULTIPLE_FACES_DETECTED
    assert result.faces_detected_count == 2
    assert result.face_bounding_box == box_large  # Primary face chosen
    assert result.inference_status == "READY"
    assert result.spatial_anomaly_map is not None
    assert len(result.spatial_anomaly_map) == 14


def test_service_low_resolution_uncertainty():
    """Verify low resolution face triggers POOR_RESOLUTION uncertainty triage."""
    model = PatchForensicsFacialDetector(is_trained=True)
    service = ModelBFacialInferenceService(model=model, min_face_resolution=64, enforce_trained=True)

    img = create_test_face_image(400, 400)
    tiny_box = NormalizedBoundingBox(x_min=0.1, y_min=0.1, x_max=0.18, y_max=0.18)  # 32x32px

    result = service.analyze_image(image=img, provided_boxes=[tiny_box])
    assert result.uncertainty == FacialUncertaintyState.POOR_RESOLUTION
    assert any("below the recommended 64x64px" in lim for lim in result.analysis_limitations)


def test_service_rejects_untrained_model():
    """Verify service rejects explicitly untrained Model B artifacts."""
    untrained_model = PatchForensicsFacialDetector(is_trained=False)
    service = ModelBFacialInferenceService(model=untrained_model, enforce_trained=True)

    img = create_test_face_image(400, 400)
    result = service.analyze_image(
        image=img,
        provided_boxes=[NormalizedBoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)],
    )

    assert result.inference_status == "UNTRAINED_MODEL_REJECTED"
    assert result.uncertainty == FacialUncertaintyState.UNTRAINED_MODEL_REJECTED
    assert result.manipulation_category_indicators == {}
    assert result.model_available is False
    assert result.predicted_category == FacialManipulationType.NOT_ASSESSED


def test_service_rejects_missing_checkpoint():
    """Verify service reports MODEL_UNAVAILABLE when weights file is not found."""
    service = ModelBFacialInferenceService(checkpoint_path="missing_facial.pt", enforce_trained=True)

    img = create_test_face_image(400, 400)
    result = service.analyze_image(
        image=img,
        provided_boxes=[NormalizedBoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)],
    )

    assert result.inference_status == "MODEL_UNAVAILABLE"
    assert result.uncertainty == FacialUncertaintyState.MODEL_UNAVAILABLE
    assert result.model_available is False
    assert result.predicted_category == FacialManipulationType.NOT_ASSESSED


# ============================================================================
# 6. Evaluator & Legal Data Feasibility Blocker Tests
# ============================================================================

def test_evaluator_and_legal_feasibility_blocker():
    """Verify Model B evaluator computes metrics and records legal commercial blocker."""
    model = PatchForensicsFacialDetector(is_trained=True)
    service = ModelBFacialInferenceService(model=model, enforce_trained=True)
    evaluator = ModelBFacialEvaluator(service)

    img = create_test_face_image(400, 400)
    samples = [
        {
            "image": img,
            "is_manipulated": False,
            "category": FacialManipulationType.NONE_AUTHENTIC,
            "face_bbox": NormalizedBoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8),
        },
        {
            "image": img,
            "is_manipulated": True,
            "category": FacialManipulationType.FACE_SWAP,
            "face_bbox": NormalizedBoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8),
            "ground_truth_bbox": NormalizedBoundingBox(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7),
        },
    ]

    report = evaluator.evaluate_dataset(samples)

    assert isinstance(report, FacialEvaluationReport)
    assert report.total_samples == 2
    assert report.commercial_data_feasibility_status == "COMMERCIAL_TRAINING_BLOCKED"
    assert "SafeHER Commercial Deployment Feasibility Blocker" in report.commercial_data_feasibility_blocker
    assert "FaceForensics++" in report.commercial_data_feasibility_blocker


def test_evaluator_counts_unassessed_samples_as_abstentions():
    """Unavailable model output must not be counted as authentic or manipulated."""
    service = ModelBFacialInferenceService(checkpoint_path="missing_facial_model.pt", enforce_trained=True)
    evaluator = ModelBFacialEvaluator(service)
    image = create_test_face_image(400, 400)
    samples = [
        {
            "image": image,
            "is_manipulated": False,
            "category": FacialManipulationType.NONE_AUTHENTIC,
            "face_bbox": NormalizedBoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8),
        },
        {
            "image": image,
            "is_manipulated": True,
            "category": FacialManipulationType.FACE_SWAP,
            "face_bbox": NormalizedBoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8),
        },
    ]

    report = evaluator.evaluate_dataset(samples)

    assert report.total_samples == 2
    assert report.assessed_samples == 0
    assert report.abstained_samples == 2
    assert report.coverage == 0.0
    assert report.true_positives == report.false_positives == 0
    assert report.true_negatives == report.false_negatives == 0
    assert report.precision is None
    assert report.recall is None
    assert report.f1_score is None
    assert report.compression_abstained_samples == 2
    assert report.unseen_generalization_accuracy is None
