"""Pydantic v2 schemas for Integrated Neural and Forensic Image Triage.

Maintains independent result containers for:
1. AI-Generated Image Detection (Model A)
2. Facial Manipulation Detection (Model B)
3. Error Level Analysis (ELA)
4. Fast Fourier Transform Spectral Analysis (FFT)
5. Out-of-Distribution Assessment (OOD)
6. Technical Image Quality Indicators

Extends ImageAssessmentResponse with additive fields, preserving 100% backward
compatibility with existing Milestone 3 API consumers.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    ImageAssessmentResponse,
    ImageMetadataSummary,
    ImageQualityIndicators,
    ManipulationRegion,
    UncertaintyState,
)


class ForensicModality(str, Enum):
    """Categorical modalities evaluated in the integrated pipeline."""

    NEURAL_AI_DETECTOR = "neural_ai_detector"
    NEURAL_FACIAL_DETECTOR = "neural_facial_detector"
    ERROR_LEVEL_ANALYSIS = "error_level_analysis"
    FREQUENCY_SPECTRAL_ANALYSIS = "frequency_spectral_analysis"
    OUT_OF_DISTRIBUTION_ASSESSMENT = "out_of_distribution_assessment"
    IMAGE_QUALITY_EVALUATION = "image_quality_evaluation"


class AIGeneratedResult(BaseModel):
    """Independent result container for Model A (AI-Generated vs Camera Photograph)."""

    model_config = ConfigDict(frozen=True)

    is_available: bool = Field(..., description="Whether Model A was evaluated.")
    model_version: str = Field(default="1.0.0", description="Model version string.")
    architecture: str = Field(default="mobilenet_v3_large", description="Neural backbone.")
    raw_logits: List[float] = Field(default_factory=list, description="Raw linear head logits.")
    uncalibrated_score: float = Field(default=0.0, description="Softmax activation for synthetic class.")
    calibrated_probability: Optional[float] = Field(None, description="Calibrated empirical probability.")
    decision_state: str = Field(default="NO_CLEAR_INDICATORS", description="Categorical triage decision.")
    uncertainty: Dict[str, Any] = Field(default_factory=dict, description="Entropy and variance metrics.")
    disclaimer: str = Field(default="Preliminary AI-generated image triage indicator.", description="Disclaimer.")


class FacialManipulationResult(BaseModel):
    """Independent result container for Model B (Facial Manipulation & Blending Seams)."""

    model_config = ConfigDict(frozen=True)

    is_available: bool = Field(..., description="Whether a verified trained Model B artifact was available; decision_state indicates whether image analysis ran.")
    model_version: str = Field(default="1.0.0", description="Model version string.")
    face_detected: bool = Field(default=False, description="True if a verified human face region was localized.")
    face_count: int = Field(default=0, description="Number of verified candidate face regions in frame.")
    face_detection_status: str = Field(
        default="localization_unavailable",
        description="Face localization outcome; distinguishes unavailable localization from a verified no-face result.",
    )
    analysis_limitations: List[str] = Field(
        default_factory=list,
        description="Specific reasons or limitations that prevented or constrained facial analysis.",
    )
    face_bounding_box: Optional[Tuple[float, float, float, float]] = Field(
        None, description="Normalized source image bounds (ymin, xmin, ymax, xmax)."
    )
    raw_score: float = Field(default=0.0, description="Uncalibrated facial manipulation score.")
    calibrated_probability: Optional[float] = Field(None, description="Calibrated probability if fitted.")
    decision_state: str = Field(default="NO_CLEAR_INDICATORS", description="Categorical triage decision.")
    manipulation_type: str = Field(default="not_assessed", description="Predicted category, or not_assessed when unavailable.")
    localized_regions: List[Dict[str, Any]] = Field(default_factory=list, description="Anomalous boundary patches.")
    face_conditions: List[str] = Field(default_factory=list, description="Assessed face quality conditions.")
    disclaimer: str = Field(default="Localized facial manipulation anomaly indicator.", description="Disclaimer.")


class ELAAnalysisResult(BaseModel):
    """Independent result container for Error Level Analysis (compression discrepancy)."""

    model_config = ConfigDict(frozen=True)

    raw_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Normalized ELA anomaly score.")
    mean_error: float = Field(..., description="Global mean pixel recompression error.")
    variance: float = Field(..., description="Spatial variance of recompression error.")
    max_local_discrepancy: float = Field(..., description="Peak error among spatial grid blocks.")
    suspicious_blocks_count: int = Field(default=0, description="Number of blocks exceeding anomaly threshold.")
    has_compression_anomalies: bool = Field(default=False, description="Whether anomalous recompression is detected.")
    quality_used: int = Field(default=90, description="JPEG quality tier used for re-encoding.")
    scale_used: float = Field(default=15.0, description="Visualization error multiplier.")


class FFTAnalysisResult(BaseModel):
    """Independent result container for Fast Fourier Transform spectral analysis."""

    model_config = ConfigDict(frozen=True)

    raw_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Normalized FFT anomaly score.")
    high_freq_energy_ratio: float = Field(..., description="Fraction of energy residing in high-frequency band.")
    azimuthal_spectral_discrepancy: float = Field(..., description="Directional variance across polar radial bins.")
    grid_peak_prominence: float = Field(..., description="Prominence of periodic lattice peaks (GAN/diffusion grid).")
    has_frequency_artifacts: bool = Field(default=False, description="Whether spectral anomaly peaks exceed baseline.")


class OODEvaluationResult(BaseModel):
    """Independent result container for Out-of-Distribution (OOD) domain suitability."""

    model_config = ConfigDict(frozen=True)

    is_ood: bool = Field(..., description="True if image falls outside supported photographic domain.")
    domain_category: str = Field(..., description="Assessed domain (e.g. photographic, drawing, meme, UI).")
    suitability_status: str = Field(..., description="'IN_DOMAIN_USABLE' or 'OUT_OF_DOMAIN_UNSUPPORTED'.")
    reasons: List[str] = Field(default_factory=list, description="Detailed explanatory reasons if OOD.")


class ImageQualityResult(BaseModel):
    """Independent result container for image technical quality assessment."""

    model_config = ConfigDict(frozen=True)

    resolution_ok: bool = Field(..., description="True if image meets minimum dimensions (>= 128x128).")
    is_blurry: bool = Field(..., description="True if edge variance indicates severe blur.")
    has_heavy_compression: bool = Field(..., description="True if severe JPEG blocking is present.")
    is_corrupted: bool = Field(default=False, description="True if byte stream corruption detected.")
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Normalized composite quality index.")
    analysis_eligibility: bool = Field(default=True, description="Whether quality is sufficient for forensic triage.")
    warnings: List[str] = Field(default_factory=list, description="Quality warning strings.")


class NeuralForensicDisagreement(BaseModel):
    """Detailed synthesis of inter-modal agreements and disagreements."""

    model_config = ConfigDict(frozen=True)

    disagreement_type: str = Field(
        ...,
        description="e.g. 'FULL_CONSENSUS', 'NEURAL_FORENSIC_DISAGREEMENT', 'MODALITY_UNAVAILABLE'.",
    )
    neural_verdict: str = Field(..., description="Summary verdict from neural models.")
    forensic_verdict: str = Field(..., description="Summary verdict from ELA/FFT heuristics.")
    confidence_delta: float = Field(default=0.0, description="Absolute difference between primary indicators.")
    synthesis_rationale: str = Field(..., description="Forensic explanation of how conflict was handled.")
    human_review_required: bool = Field(default=False, description="True if human casework review is mandatory.")
    recommended_action: str = Field(..., description="Actionable recommendation for investigator or caseworker.")


class NeuralInferenceMetadata(BaseModel):
    """Operational, versioning, uncertainty, and calibration metadata for backend neural inference."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(
        ...,
        description="Operational status: 'active', 'disabled', 'model_unavailable', 'checksum_mismatch', 'timeout', or 'error'.",
    )
    model_a_version: Optional[str] = Field(
        default=None,
        description="Semantic version of Model A (AI-generated detector) if evaluated.",
    )
    model_b_version: Optional[str] = Field(
        default=None,
        description="Semantic version of Model B (Facial manipulation detector) if evaluated.",
    )
    supported_tasks: List[str] = Field(
        default_factory=lambda: [
            "AI_GENERATED_IMAGE_DETECTION",
            "FACIAL_MANIPULATION_DETECTION",
        ],
        description="Supported neural assessment tasks.",
    )
    uncertainty_state: str = Field(
        default="DETERMINISTIC",
        description="Uncertainty qualification: e.g. 'LOW_UNCERTAINTY', 'ELEVATED_UNCERTAINTY', 'HIGH_UNCERTAINTY', 'INCONCLUSIVE'.",
    )
    calibration_status: str = Field(
        default="UNAVAILABLE",
        description="Empirical calibration state: 'CALIBRATED_TEMPERATURE_SCALING', 'CALIBRATED_ISOTONIC', 'UNAVAILABLE', or 'NOT_APPLICABLE'.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Operational warnings, degradation notices, or forensic caveat strings.",
    )


class IntegratedForensicAssessment(ImageAssessmentResponse):
    """Complete integrated forensic assessment combining neural and heuristic modalities.

    Inherits all fields from ImageAssessmentResponse to guarantee 100% backward
    compatibility, adding rich independent result sub-objects for granular inspection.
    """

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    # Additive Milestone 4.7 Result Containers
    ai_generated_result: Optional[AIGeneratedResult] = Field(
        default=None,
        description="Independent Model A AI-generated image detection result.",
    )
    facial_manipulation_result: Optional[FacialManipulationResult] = Field(
        default=None,
        description="Independent Model B facial manipulation and seam detection result.",
    )
    ela_analysis_result: Optional[ELAAnalysisResult] = Field(
        default=None,
        description="Independent Error Level Analysis (ELA) result.",
    )
    fft_analysis_result: Optional[FFTAnalysisResult] = Field(
        default=None,
        description="Independent Fast Fourier Transform (FFT) spectral analysis result.",
    )
    ood_evaluation_result: Optional[OODEvaluationResult] = Field(
        default=None,
        description="Independent Out-of-Distribution domain suitability evaluation.",
    )
    neural_forensic_disagreement: Optional[NeuralForensicDisagreement] = Field(
        default=None,
        description="Multi-modal consensus and disagreement resolution analysis.",
    )
    human_review_guidance: Optional[str] = Field(
        default=None,
        description="Structured forensic review guidance for caseworkers when signals conflict.",
    )
    is_neural_enabled: bool = Field(
        default=False,
        description="Whether neural detectors were active during this evaluation turn.",
    )
    neural_metadata: Optional[NeuralInferenceMetadata] = Field(
        default=None,
        description="Summary neural inference operational status, versioning, uncertainty, and calibration metadata.",
    )
