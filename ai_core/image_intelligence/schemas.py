"""Pydantic v2 schemas for SafeHER Image Intelligence and Visual Manipulation Detection."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_core.image_intelligence.config import MANDATORY_IMAGE_AI_DISCLAIMER


class DetectionTask(str, Enum):
    """Forensic computer vision task scope."""

    AI_GENERATED_IMAGE = "ai_generated_image"
    FACE_MANIPULATION = "face_manipulation"
    HYBRID_MANIPULATION = "hybrid_manipulation"
    FORENSIC_TRIAGE = "forensic_triage"


class AssessmentLabel(str, Enum):
    """Qualitative assessment label.

    NOTE: SafeHER never claims an image is 'AUTHENTIC' because negative
    detector output does not constitute proof of authenticity.
    """

    NO_OBVIOUS_ARTIFACTS = "no_obvious_artifacts"
    SUSPICIOUS_MANIPULATION = "suspicious_manipulation"
    HIGHLY_LIKELY_MANIPULATED = "highly_likely_manipulated"
    INCONCLUSIVE = "inconclusive"


class UncertaintyState(str, Enum):
    """Categorical triage of forensic uncertainty, evidence usability, and model status."""

    # 6 SafeHER Milestone 3 Core Quality & Uncertainty States
    ANALYSIS_USABLE_EVIDENCE = "analysis_usable_evidence"
    ANALYSIS_POOR_QUALITY_EVIDENCE = "analysis_poor_quality_evidence"
    MODEL_UNCERTAIN = "model_uncertain"
    MODEL_UNAVAILABLE = "model_unavailable"
    OUTSIDE_SUPPORTED_DOMAIN = "outside_supported_domain"
    ANALYSIS_CANNOT_BE_COMPLETED = "analysis_cannot_be_completed"

    # Compatibility aliases
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    INSUFFICIENT_QUALITY = "insufficient_quality"


class ImageQualityIndicators(BaseModel):
    """Granular technical metrics assessing image quality, integrity, and domain suitability."""

    width: int = Field(..., ge=0, description="Image width in pixels")
    height: int = Field(..., ge=0, description="Image height in pixels")
    megapixels: float = Field(..., ge=0.0, description="Total megapixels (width * height / 1e6)")
    aspect_ratio: float = Field(..., gt=0.0, description="Aspect ratio (width / height)")
    resolution_ok: bool = Field(..., description="Whether image resolution meets minimum forensic requirements (>= 128x128)")

    blur_score: float = Field(..., ge=0.0, description="Laplacian variance edge energy; lower values indicate severe blur")
    is_blurry: bool = Field(..., description="True if blur score falls below the threshold (< 15.0)")

    compression_artifact_score: float = Field(..., ge=0.0, description="Discontinuity ratio along 8x8 DCT grid boundaries")
    has_heavy_compression: bool = Field(..., description="True if severe blockiness artifacts are detected (> 2.5)")

    mean_luminance: float = Field(..., ge=0.0, le=255.0, description="Mean pixel luminance across the frame (0.0=black, 255.0=white)")
    overexposed_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of pixels near highlight saturation (>= 250)")
    underexposed_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of pixels near shadow clipping (<= 10)")
    exposure_level: str = Field(..., description="'normal', 'overexposed', 'underexposed', or 'extreme_contrast'")

    is_monochromatic_or_flat: bool = Field(..., description="True if image has near-zero variance across channels")
    is_screen_capture: bool = Field(..., description="True if aspect ratio, dimensions, or UI characteristics match screen captures")

    face_detected: Optional[bool] = Field(None, description="True/False if face detection was performed, None otherwise")
    face_count: int = Field(0, ge=0, description="Number of evaluated faces in the frame")
    face_resolution_ok: Optional[bool] = Field(None, description="Whether detected faces meet minimum micro-forensic size (>= 60x60)")
    face_size_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Area ratio of largest face relative to total frame")

    corrupted_data_detected: bool = Field(False, description="True if truncated streams or byte corruption was detected")
    domain_category: Optional[str] = Field(None, description="Assessed domain category (photographic, drawing, document, medical, screenshot, etc.)")
    is_in_domain: Optional[bool] = Field(None, description="Whether image belongs to the validated photographic evidence domain")
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Normalized composite quality index (0.0=unusable, 1.0=pristine)")

    model_config = ConfigDict(frozen=True)


class ManipulationRegionType(str, Enum):
    """Type of localized visual manipulation region."""

    BOUNDING_BOX = "bounding_box"
    GRID_CELL = "grid_cell"
    ATTRIBUTION_PATCH = "attribution_patch"


class ManipulationRegion(BaseModel):
    """Specific spatial region flagged for manipulation or compression anomalies."""

    box_normalized: Tuple[float, float, float, float] = Field(
        ...,
        description="Normalized coordinates (ymin, xmin, ymax, xmax) in the range 0.0 to 1.0",
    )
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Localized anomaly score in this spatial region",
    )
    region_type: ManipulationRegionType = Field(
        default=ManipulationRegionType.BOUNDING_BOX,
        description="Spatial representation type",
    )
    attribution_category: str = Field(
        default="suspected_manipulation_region",
        description="Distinguishes 'model_attention_attribution' from 'suspected_manipulation_region'",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Technical description of the localized anomaly",
    )

    @field_validator("box_normalized")
    @classmethod
    def validate_box_coordinates(cls, box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        ymin, xmin, ymax, xmax = box
        for coord in (ymin, xmin, ymax, xmax):
            if not (0.0 <= coord <= 1.0):
                raise ValueError("Normalized box coordinates must be between 0.0 and 1.0")
        if ymin >= ymax:
            raise ValueError("ymin must be strictly less than ymax")
        if xmin >= xmax:
            raise ValueError("xmin must be strictly less than xmax")
        return box

    model_config = ConfigDict(frozen=True)


class ImageMetadataSummary(BaseModel):
    """Extracted non-sensitive image metadata."""

    format: str = Field(..., description="Detected image format (JPEG, PNG, etc.)")
    dimensions: Tuple[int, int] = Field(..., description="Image dimensions as (width, height)")
    size_bytes: int = Field(..., ge=0, description="Raw file size in bytes")
    sha256: str = Field(..., min_length=64, max_length=64, description="Cryptographic SHA-256 hash")
    has_exif: bool = Field(False, description="Whether EXIF metadata was present")
    sanitized_exif: Dict[str, Any] = Field(default_factory=dict, description="Safe, sanitized EXIF tags")
    color_mode: str = Field(..., description="Color mode (e.g. RGB, RGBA, L)")

    model_config = ConfigDict(frozen=True)


class ImageAssessmentResponse(BaseModel):
    """Structured, explainable response schema for deepfake and image manipulation assessments."""

    assessment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique assessment identifier",
    )
    image_id: Optional[UUID] = Field(
        default=None,
        description="Optional ID referencing the stored evidence record",
    )
    detector_name: str = Field(
        ...,
        min_length=1,
        description="Name of the vision detector engine executed",
    )
    detector_version: str = Field(
        ...,
        min_length=1,
        description="Version string of the detector engine",
    )
    detection_task: DetectionTask = Field(
        ...,
        description="Forensic detection task scope",
    )
    raw_model_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw uncalibrated score from the detector (0.0=lowest suspicion, 1.0=highest)",
    )
    is_calibrated: bool = Field(
        False,
        description="False for uncalibrated heuristics; do not present scores as probabilities when false.",
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Statistical calibrated confidence; populated strictly when calibration is proven.",
    )
    assessment_label: AssessmentLabel = Field(
        ...,
        description="Qualitative categorization of visual manipulation risk",
    )
    uncertainty_state: UncertaintyState = Field(
        ...,
        description="Categorical triage of forensic confidence and signal quality",
    )
    manipulation_regions: List[ManipulationRegion] = Field(
        default_factory=list,
        description="List of localized spatial anomaly regions if supported and detected",
    )
    preprocessing_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings encountered during image normalization or validation",
    )
    quality_indicators: Optional[ImageQualityIndicators] = Field(
        default=None,
        description="Granular technical quality indicators and empirical forensic metrics",
    )
    quality_warnings: List[str] = Field(
        default_factory=list,
        description="Explicit human-readable quality, degradation, or domain mismatch warnings",
    )
    analysis_eligibility: bool = Field(
        default=True,
        description="Whether the image meets quality and domain criteria for reliable forensic analysis",
    )
    reason_for_insufficient_evidence: Optional[str] = Field(
        default=None,
        description="Clear explanation when evidence quality is poor or analysis cannot be reliably completed",
    )
    recommended_next_step: str = Field(
        default="Preserve this original file with its SHA-256 hash. If quality is poor, request an uncompressed original.",
        description="Actionable next step recommendation for survivors, advocates, or investigators",
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Plain-language forensic explanation of what was examined and found",
    )
    disagreement_state: Optional[str] = Field(
        default=None,
        description="Model consistency state: consistent_detection_signals, conflicting_detection_signals, insufficient_evidence, or one_detector_unavailable",
    )
    model_agreement_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed multi-model consistency and disagreement metrics if multiple detectors were evaluated",
    )
    visual_explanation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured visual localization metadata including localization method, attribution category, faithfulness metric, and limitations",
    )
    model_limitations: List[str] = Field(
        default_factory=list,
        description="Explicit boundaries and limitations of the detector model",
    )
    metadata_summary: Optional[ImageMetadataSummary] = Field(
        default=None,
        description="Non-sensitive technical metadata of the evaluated image",
    )
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the analysis turn",
    )
    disclaimer: str = Field(
        default=MANDATORY_IMAGE_AI_DISCLAIMER,
        min_length=1,
        frozen=True,
        validate_default=True,
        description="Mandatory legal, forensic, and safety disclaimer",
    )

    @field_validator("disclaimer", mode="before")
    @classmethod
    def validate_mandatory_disclaimer(cls, value: object) -> str:
        """Require the exact configured disclaimer."""
        if value != MANDATORY_IMAGE_AI_DISCLAIMER:
            raise ValueError("disclaimer must exactly match the configured mandatory image AI disclaimer")
        return MANDATORY_IMAGE_AI_DISCLAIMER

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)
