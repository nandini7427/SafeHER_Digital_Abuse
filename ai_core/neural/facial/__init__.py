"""SafeHER Model B Facial Manipulation Detection Subsystem."""

from ai_core.neural.facial.evaluator import (
    FacialEvaluationReport,
    ModelBFacialEvaluator,
    compute_bounding_box_iou,
)
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
from ai_core.neural.facial.service import ModelBFacialInferenceService
from ai_core.neural.models.patch_forensics import (
    MODEL_B_CLASS_MAPPING,
    MODEL_B_LABEL_TO_INDEX,
    MODEL_B_SUPPORTED_DOMAIN,
    PatchForensicsFacialDetector,
)

__all__ = [
    "FaceCandidate",
    "FaceCondition",
    "FaceDetectionStatus",
    "FaceRegionPreprocessor",
    "FacialEvaluationReport",
    "FacialManipulationInferenceResult",
    "FacialManipulationType",
    "FacialUncertaintyState",
    "LocalizedManipulationRegion",
    "MANDATORY_FACIAL_DISCLAIMERS",
    "MODEL_B_CLASS_MAPPING",
    "MODEL_B_LABEL_TO_INDEX",
    "MODEL_B_SUPPORTED_DOMAIN",
    "ModelBFacialEvaluator",
    "ModelBFacialInferenceService",
    "NormalizedBoundingBox",
    "PatchForensicsFacialDetector",
    "compute_bounding_box_iou",
]
