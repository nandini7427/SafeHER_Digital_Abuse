"""Pydantic schemas and metadata contracts for SafeHER Model B Facial Manipulation Detection.

Defines typed contracts for face candidate regions, bounding boxes in source image
coordinates, dense patch-level anomaly maps, manipulation categories, uncertainty
states, and forensic limitation disclosures.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FacialManipulationType(str, Enum):
    """Categorical taxonomy of facial manipulation and synthesis techniques."""

    # Legacy model class name retained for checkpoint/API compatibility. It means
    # only that this classifier selected its non-manipulation class; it is not
    # an authenticity determination.
    NONE_AUTHENTIC = "none_authentic"
    NOT_ASSESSED = "not_assessed"
    FACE_SWAP = "face_swap"
    FACIAL_REENACTMENT = "facial_reenactment"
    FACIAL_INPAINTING = "facial_inpainting"
    LOCALIZED_FACE_REPLACEMENT = "localized_face_replacement"
    UNKNOWN_MANIPULATION = "unknown_manipulation"


class FaceCondition(str, Enum):
    """Quality and visibility conditions of a detected face region."""

    NORMAL = "normal"
    LOW_RESOLUTION = "low_resolution"
    PARTIALLY_VISIBLE = "partially_visible"
    EXTREME_POSE = "extreme_pose"
    SEVERE_OCCLUSION = "severe_occlusion"


class FaceDetectionStatus(str, Enum):
    """Operational status of face extraction on input imagery."""

    FACE_DETECTED = "face_detected"
    MULTIPLE_FACES_DETECTED = "multiple_faces_detected"
    NO_FACE_DETECTED = "no_face_detected"
    LOCALIZATION_UNAVAILABLE = "localization_unavailable"
    INVALID_IMAGE = "invalid_image"


class FacialUncertaintyState(str, Enum):
    """Categorical triage of forensic uncertainty for facial manipulation analysis."""

    USABLE_EVIDENCE = "usable_evidence"
    POOR_RESOLUTION = "poor_resolution"
    PARTIALLY_VISIBLE_FACE = "partially_visible_face"
    NO_FACE_IN_FRAME = "no_face_in_frame"
    FACE_LOCALIZATION_UNAVAILABLE = "face_localization_unavailable"
    INVALID_IMAGE = "invalid_image"
    MODEL_UNCERTAIN = "model_uncertain"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNTRAINED_MODEL_REJECTED = "untrained_model_rejected"


MANDATORY_FACIAL_DISCLAIMERS: List[str] = [
    "SafeHER Facial Forensic Disclaimer: Detected manipulation scores are uncalibrated raw triage indicators and do NOT constitute calibrated statistical probabilities.",
    "A localized anomaly bounding box reflects elevated patch-level forensic inconsistency; it does NOT constitute legal or absolute proof of localized editing.",
    "Absence of detected face swap or facial reenactment artifacts does NOT certify that a face or person is authentic.",
    "Severe compression, camera blur, or extreme facial angles can degrade edge blending seam detection and produce false boundary signals.",
]


class NormalizedBoundingBox(BaseModel):
    """Normalized bounding box coordinates in [0.0, 1.0] relative to image dimensions."""

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(..., ge=0.0, le=1.0, description="Left edge normalized coordinate.")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Top edge normalized coordinate.")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Right edge normalized coordinate.")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Bottom edge normalized coordinate.")

    @field_validator("x_max")
    @classmethod
    def validate_x_bounds(cls, v: float, info: Any) -> float:
        x_min = info.data.get("x_min", 0.0)
        if v < x_min:
            raise ValueError(f"x_max ({v}) cannot be less than x_min ({x_min})")
        return v

    @field_validator("y_max")
    @classmethod
    def validate_y_bounds(cls, v: float, info: Any) -> float:
        y_min = info.data.get("y_min", 0.0)
        if v < y_min:
            raise ValueError(f"y_max ({v}) cannot be less than y_min ({y_min})")
        return v

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixel_coordinates(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute integer pixel bounds (x1, y1, x2, y2)."""
        x1 = max(0, min(image_width - 1, int(round(self.x_min * image_width))))
        y1 = max(0, min(image_height - 1, int(round(self.y_min * image_height))))
        x2 = max(x1 + 1, min(image_width, int(round(self.x_max * image_width))))
        y2 = max(y1 + 1, min(image_height, int(round(self.y_max * image_height))))
        return x1, y1, x2, y2


class FaceCandidate(BaseModel):
    """Detected face candidate region with quality indicators and source coordinates."""

    model_config = ConfigDict(frozen=True)

    face_id: str = Field(..., description="Unique identifier for candidate face in frame.")
    bounding_box: NormalizedBoundingBox = Field(..., description="Source image bounding box.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score.")
    face_conditions: List[FaceCondition] = Field(
        default_factory=list,
        description="Assessed environmental/quality conditions (e.g. low-res, truncated).",
    )
    crop_resolution: Tuple[int, int] = Field(
        default=(0, 0),
        description="Native spatial pixel resolution of extracted crop (W, H).",
    )
    is_primary: bool = Field(
        default=False,
        description="Whether this is the primary (largest/most prominent) face in frame.",
    )


class LocalizedManipulationRegion(BaseModel):
    """Spatial region identified with statistically elevated manipulation anomalies."""

    model_config = ConfigDict(frozen=True)

    bounding_box: NormalizedBoundingBox = Field(
        ...,
        description="Region coordinates projected onto original source image frame.",
    )
    manipulation_type: FacialManipulationType = Field(
        ...,
        description="Categorical manipulation type suspected in this region.",
    )
    anomaly_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized density of anomalous patch activations in region.",
    )
    patch_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of face patch grid covered by this anomaly cluster.",
    )
    disclaimer: str = Field(
        default="Bounding box indicates statistically elevated spatial anomaly density; not legal proof of editing.",
        description="Mandatory disclaimer.",
    )


class FacialManipulationInferenceResult(BaseModel):
    """Complete inference payload returned by SafeHER Model B Facial Manipulation Detector."""

    model_config = ConfigDict(frozen=True)

    face_detection_status: FaceDetectionStatus = Field(
        ...,
        description="Outcome of face localization.",
    )
    faces_detected_count: int = Field(
        ...,
        description="Number of face candidate regions identified in frame.",
    )
    analyzed_face_id: Optional[str] = Field(
        default=None,
        description="Identifier of face subjected to forensic evaluation.",
    )
    face_bounding_box: Optional[NormalizedBoundingBox] = Field(
        default=None,
        description="Source bounding box of evaluated face.",
    )
    manipulation_category_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Uncalibrated per-category manipulation scores in [0.0, 1.0].",
    )
    predicted_category: FacialManipulationType = Field(
        default=FacialManipulationType.NOT_ASSESSED,
        description="Highest-scoring category, or NOT_ASSESSED when no valid model analysis ran.",
    )
    spatial_anomaly_map: Optional[List[List[float]]] = Field(
        default=None,
        description="2D normalized grid of patch-level anomaly activations.",
    )
    localized_regions: List[LocalizedManipulationRegion] = Field(
        default_factory=list,
        description="List of detected localized manipulation bounding boxes.",
    )
    uncertainty: FacialUncertaintyState = Field(
        ...,
        description="Triage uncertainty state.",
    )
    model_available: bool = Field(
        default=False,
        description="Whether a verified trained Model B artifact was available for this request.",
    )
    face_conditions: List[FaceCondition] = Field(
        default_factory=list,
        description="Conditions present on the evaluated face (e.g. low resolution).",
    )
    model_version: str = Field(
        default="1.0.0-m4.4-model-b",
        description="Semantic version string of Model B artifact.",
    )
    architecture: str = Field(
        default="patch_forensics_facial",
        description="Model B neural backbone identifier.",
    )
    analysis_limitations: List[str] = Field(
        default_factory=lambda: list(MANDATORY_FACIAL_DISCLAIMERS),
        description="Documented forensic limitations of this analysis.",
    )
    inference_status: str = Field(
        ...,
        description="Operational status ('READY', 'NO_FACE_DETECTED', 'MODEL_UNAVAILABLE', etc.).",
    )
    latency_ms: float = Field(
        ...,
        description="Execution duration on CPU in milliseconds.",
    )
    disclaimer: str = Field(
        default=MANDATORY_FACIAL_DISCLAIMERS[0],
        description="Mandatory forensic disclaimer.",
    )
