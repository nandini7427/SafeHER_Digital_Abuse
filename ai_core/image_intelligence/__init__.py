"""SafeHER Image Intelligence and Visual Manipulation Detection Module."""

from ai_core.image_intelligence.config import (
    ImageDetectorConfig,
    MANDATORY_IMAGE_AI_DISCLAIMER,
    default_image_config,
)
from ai_core.image_intelligence.detector import (
    BaseImageDetector,
    SafeHERImageDetector,
)
from ai_core.image_intelligence.explainability import (
    STANDARD_MODEL_LIMITATIONS,
    build_forensic_explanation,
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
from ai_core.image_intelligence.quality import (
    QualityAssessmentResult,
    evaluate_image_quality,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    ImageAssessmentResponse,
    ImageMetadataSummary,
    ImageQualityIndicators,
    ManipulationRegion,
    ManipulationRegionType,
    UncertaintyState,
)
from ai_core.image_intelligence.service import (
    ImageIntelligenceService,
    get_image_detector,
    image_service,
    set_image_detector,
)

__all__ = [
    "AssessmentLabel",
    "BaseImageDetector",
    "BaseImageModelRunner",
    "BaselineHeuristicRunner",
    "DetectionTask",
    "ImageAssessmentResponse",
    "ImageDetectorConfig",
    "ImageIntelligenceService",
    "ImageMetadataSummary",
    "ImageQualityIndicators",
    "MANDATORY_IMAGE_AI_DISCLAIMER",
    "ManipulationRegion",
    "ManipulationRegionType",
    "ModelPrediction",
    "NeuralVisionModelRunner",
    "PreprocessedImage",
    "QualityAssessmentResult",
    "STANDARD_MODEL_LIMITATIONS",
    "SafeHERImageDetector",
    "UncertaintyState",
    "build_forensic_explanation",
    "default_image_config",
    "determine_assessment_label_and_uncertainty",
    "evaluate_image_quality",
    "get_image_detector",
    "identify_magic_format",
    "image_service",
    "preprocess_image",
    "set_image_detector",
]
