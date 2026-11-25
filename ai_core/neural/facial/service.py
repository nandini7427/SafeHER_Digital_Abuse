"""Inference service for SafeHER Model B Facial Manipulation Detector.

Provides end-to-end facial manipulation triage:
- Face detection status, multi-face handling, and quality/condition assessment (low-res, truncation).
- Dense patch-level anomaly localization with re-projection to original source image coordinates.
- Fine-grained category indicators (face swap, reenactment, inpainting, localized replacement).
- Uncertainty quantification and mandatory forensic limitation disclosures.
- Strict rejection gates for missing or untrained model weights with zero score fabrication.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from ai_core.neural.facial.preprocessor import FaceRegionPreprocessor
from ai_core.neural.facial.schemas import (
    MANDATORY_FACIAL_DISCLAIMERS,
    FaceCandidate,
    FaceCondition,
    FaceDetectionStatus,
    FacialManipulationInferenceResult,
    FacialManipulationType,
    FacialUncertaintyState,
    LocalizedManipulationRegion,
    NormalizedBoundingBox,
)
from ai_core.neural.models.factory import load_detector_checkpoint
from ai_core.neural.models.patch_forensics import (
    MODEL_B_CLASS_MAPPING,
    PatchForensicsFacialDetector,
)


class ModelBFacialInferenceService:
    """SafeHER Model B Facial Manipulation Inference and Localization Service."""

    def __init__(
        self,
        model: Optional[PatchForensicsFacialDetector] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        expected_sha256: Optional[str] = None,
        enforce_trained: bool = True,
        patch_anomaly_threshold: float = 0.50,
        min_face_resolution: int = 64,
        margin_ratio: float = 0.20,
    ):
        """Initialize ModelBFacialInferenceService.

        Args:
            model: Optional pre-instantiated PatchForensicsFacialDetector.
            checkpoint_path: Optional path to serialized weights file.
            expected_sha256: Optional expected SHA-256 digest of weights file.
            enforce_trained: Legacy compatibility parameter; operational inference always rejects unverified weights.
            patch_anomaly_threshold: Threshold on spatial patch anomaly scores to delineate regions.
            min_face_resolution: Pixel edge length below which face is flagged LOW_RESOLUTION.
            margin_ratio: Expansion margin around face bounding box to capture blending seams.
        """
        self.enforce_trained = enforce_trained
        self.patch_anomaly_threshold = patch_anomaly_threshold
        self.preprocessor = FaceRegionPreprocessor(
            min_face_resolution=min_face_resolution,
            margin_ratio=margin_ratio,
        )

        self.model: Optional[PatchForensicsFacialDetector] = None
        self.operational_status: str = "READY"
        self.model_version: str = "1.0.0-m4.4-model-b"
        self.architecture: str = "patch_forensics_facial"

        if model is not None:
            self.model = model
            self.architecture = model.get_metadata().architecture
            self.model_version = model.get_metadata().model_version
            if not model.is_trained:
                self.model = None
                self.operational_status = "UNTRAINED_MODEL_REJECTED"
            else:
                self.model.eval()
                self.operational_status = "READY"

        elif checkpoint_path is not None:
            path = Path(checkpoint_path)
            if not path.is_file():
                self.operational_status = "MODEL_UNAVAILABLE"
            else:
                try:
                    det = PatchForensicsFacialDetector()
                    # Always enforce provenance at the inference boundary. The
                    # legacy enforce_trained=False switch must never activate
                    # random or unverifiable weights for survivor-facing output.
                    self.model = load_detector_checkpoint(
                        model=det,
                        checkpoint_path=path,
                        expected_sha256=expected_sha256,
                        enforce_trained=True,
                    )
                    self.architecture = self.model.get_metadata().architecture
                    self.model_version = self.model.get_metadata().model_version
                    self.operational_status = "READY"
                except ValueError as exc:
                    message = str(exc)
                    if "UNTRAINED" in message.upper() or "ARTIFACT_NOT_APPROVED" in message.upper():
                        self.operational_status = "UNTRAINED_MODEL_REJECTED"
                    elif "SHA-256 mismatch" in message:
                        self.operational_status = "INTEGRITY_VERIFICATION_FAILED"
                    else:
                        self.operational_status = "ERROR"
                    self.model = None
                except Exception:
                    # Do not expose artifact parsing details or leave a partially
                    # initialized model available after any loading failure.
                    self.operational_status = "ERROR"
                    self.model = None

        else:
            self.operational_status = (
                "MODEL_UNAVAILABLE" if enforce_trained else "UNTRAINED_MODEL_REJECTED"
            )
            self.model = None

    def analyze_image(
        self,
        image: Union[Image.Image, np.ndarray],
        provided_boxes: Optional[List[NormalizedBoundingBox]] = None,
        face_selection: str = "primary",
    ) -> FacialManipulationInferenceResult:
        """Execute facial manipulation triage and spatial patch localization on image.

        Args:
            image: Input image (PIL Image or 3D RGB NumPy array).
            provided_boxes: Optional explicit face bounding boxes.
            face_selection: Strategy for multi-face scenes ('primary' for largest face).

        Returns:
            FacialManipulationInferenceResult structured payload.
        """
        start_time = time.perf_counter()

        # 1. Detect and assess face candidate regions
        detect_status, candidates = self.preprocessor.detect_face_candidates(
            image=image,
            provided_boxes=provided_boxes,
        )

        # Handle missing face regions without pretending that a central crop is a face.
        if not candidates:
            if detect_status == FaceDetectionStatus.INVALID_IMAGE:
                no_face_uncertainty = FacialUncertaintyState.INVALID_IMAGE
                no_face_status = "INVALID_IMAGE"
                localization_limitations = ["The image is too small or invalid for facial-region analysis."]
            elif detect_status == FaceDetectionStatus.LOCALIZATION_UNAVAILABLE:
                no_face_uncertainty = FacialUncertaintyState.FACE_LOCALIZATION_UNAVAILABLE
                no_face_status = "FACE_LOCALIZATION_UNAVAILABLE"
                localization_limitations = [
                    "No verified face detector output or annotated face bounding box was supplied.",
                    "A central image crop is not treated as a detected face; facial manipulation analysis was not run.",
                ]
            else:
                no_face_uncertainty = FacialUncertaintyState.NO_FACE_IN_FRAME
                no_face_status = "NO_FACE_DETECTED"
                localization_limitations = [
                    "No human face region was supplied or detected in the input frame.",
                    "Facial manipulation analysis requires at least one verified face candidate region.",
                ]
            return FacialManipulationInferenceResult(
                face_detection_status=detect_status,
                faces_detected_count=0,
                analyzed_face_id=None,
                face_bounding_box=None,
                manipulation_category_indicators={},
                predicted_category=FacialManipulationType.NOT_ASSESSED,
                spatial_anomaly_map=None,
                localized_regions=[],
                uncertainty=no_face_uncertainty,
                model_available=bool(self.operational_status == "READY" and self.model is not None),
                face_conditions=[],
                model_version=self.model_version,
                architecture=self.architecture,
                analysis_limitations=localization_limitations,
                inference_status=no_face_status,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        # Select target face candidate
        if face_selection == "primary":
            target_candidate = candidates[0]  # First candidate is sorted largest/primary
        else:
            target_candidate = candidates[0]

        # Assess face conditions
        conditions = list(target_candidate.face_conditions)
        limitations = list(MANDATORY_FACIAL_DISCLAIMERS)

        uncertainty = FacialUncertaintyState.USABLE_EVIDENCE
        if FaceCondition.LOW_RESOLUTION in conditions:
            uncertainty = FacialUncertaintyState.POOR_RESOLUTION
            limitations.append(
                f"Face crop resolution ({target_candidate.crop_resolution[0]}x{target_candidate.crop_resolution[1]}) "
                f"is below the recommended 64x64px limit; high-frequency blending seams may be blurred."
            )
        elif FaceCondition.PARTIALLY_VISIBLE in conditions:
            uncertainty = FacialUncertaintyState.PARTIALLY_VISIBLE_FACE
            limitations.append(
                "Face bounding box touches the image border; boundary blending artifacts may be truncated."
            )

        # Handle model unavailability or rejection
        if self.operational_status != "READY" or self.model is None:
            unc_state = (
                FacialUncertaintyState.UNTRAINED_MODEL_REJECTED
                if self.operational_status == "UNTRAINED_MODEL_REJECTED"
                else FacialUncertaintyState.MODEL_UNAVAILABLE
            )
            return FacialManipulationInferenceResult(
                face_detection_status=detect_status,
                faces_detected_count=len(candidates),
                analyzed_face_id=target_candidate.face_id,
                face_bounding_box=target_candidate.bounding_box,
                manipulation_category_indicators={},
                predicted_category=FacialManipulationType.NOT_ASSESSED,
                spatial_anomaly_map=None,
                localized_regions=[],
                uncertainty=unc_state,
                model_available=False,
                face_conditions=conditions,
                model_version=self.model_version,
                architecture=self.architecture,
                analysis_limitations=limitations,
                inference_status=self.operational_status,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        # 2. Extract face crop with blending margin
        crop_tensor, expanded_box = self.preprocessor.extract_face_crop(
            image=image,
            candidate=target_candidate,
            apply_margin=True,
        )

        # 3. Model forward pass
        with torch.no_grad():
            patch_anomaly_map, category_logits = self.model(crop_tensor)

        # Category softmax probabilities
        probs = torch.softmax(category_logits[0], dim=-1).cpu().numpy()
        category_indicators: Dict[str, float] = {}
        for idx, score in enumerate(probs):
            cat_name = MODEL_B_CLASS_MAPPING.get(idx, f"category_{idx}")
            category_indicators[cat_name] = round(float(score), 4)

        pred_idx = int(np.argmax(probs))
        predicted_category_name = MODEL_B_CLASS_MAPPING.get(pred_idx, FacialManipulationType.NONE_AUTHENTIC.value)
        try:
            predicted_category = FacialManipulationType(predicted_category_name)
        except ValueError:
            predicted_category = FacialManipulationType.UNKNOWN_MANIPULATION

        # 4. Dense Patch Anomaly Map extraction (14x14 grid)
        patch_grid = patch_anomaly_map[0, 0].cpu().numpy()  # (14, 14)
        spatial_map_list: List[List[float]] = [
            [round(float(val), 4) for val in row] for row in patch_grid
        ]

        # 5. Extract Localized Manipulation Bounding Boxes
        localized_regions: List[LocalizedManipulationRegion] = []
        anomalous_indices = np.argwhere(patch_grid >= self.patch_anomaly_threshold)

        if len(anomalous_indices) > 0 and predicted_category != FacialManipulationType.NONE_AUTHENTIC:
            min_r = int(np.min(anomalous_indices[:, 0]))
            max_r = int(np.max(anomalous_indices[:, 0]))
            min_c = int(np.min(anomalous_indices[:, 1]))
            max_c = int(np.max(anomalous_indices[:, 1]))

            grid_h, grid_w = patch_grid.shape

            # Bounding box within the face crop [0, 1]
            crop_rel_box = NormalizedBoundingBox(
                x_min=round(min_c / grid_w, 4),
                y_min=round(min_r / grid_h, 4),
                x_max=round(min(1.0, (max_c + 1) / grid_w), 4),
                y_max=round(min(1.0, (max_r + 1) / grid_h), 4),
            )

            # Map back to source image coordinates
            src_box = self.preprocessor.map_crop_coords_to_source(crop_rel_box, expanded_box)

            anomaly_density = float(np.mean(patch_grid[anomalous_indices[:, 0], anomalous_indices[:, 1]]))
            patch_coverage = float(len(anomalous_indices) / (grid_h * grid_w))

            region = LocalizedManipulationRegion(
                bounding_box=src_box,
                manipulation_type=predicted_category,
                anomaly_density=round(anomaly_density, 4),
                patch_coverage=round(patch_coverage, 4),
            )
            localized_regions.append(region)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FacialManipulationInferenceResult(
            face_detection_status=detect_status,
            faces_detected_count=len(candidates),
            analyzed_face_id=target_candidate.face_id,
            face_bounding_box=target_candidate.bounding_box,
            manipulation_category_indicators=category_indicators,
            predicted_category=predicted_category,
            spatial_anomaly_map=spatial_map_list,
            localized_regions=localized_regions,
            uncertainty=uncertainty,
            model_available=True,
            face_conditions=conditions,
            model_version=self.model_version,
            architecture=self.architecture,
            analysis_limitations=limitations,
            inference_status="READY",
            latency_ms=round(elapsed_ms, 2),
        )

    # Standard predictor interface alias
    predict = analyze_image
