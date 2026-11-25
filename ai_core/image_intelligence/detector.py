"""Detector classes orchestrating SafeHER Image Intelligence and Visual Manipulation Triage."""

from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Tuple
from uuid import UUID

from ai_core.image_intelligence.config import ImageDetectorConfig, default_image_config
from ai_core.image_intelligence.consistency import (
    DisagreementState,
    ModelConsistencyAssessment,
    ModelDisagreementPolicy,
)
from ai_core.image_intelligence.explainability import (
    STANDARD_MODEL_LIMITATIONS,
    build_forensic_explanation,
    determine_assessment_label_and_uncertainty,
)
from ai_core.image_intelligence.model_runner import (
    BaseImageModelRunner,
    BaselineHeuristicRunner,
    FrequencySpectralRunner,
    NeuralVisionModelRunner,
)
from ai_core.image_intelligence.visual_localization import (
    VisualLocalizationEngine,
    VisualLocalizationResult,
)
from ai_core.image_intelligence.preprocessing import PreprocessedImage, preprocess_image
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
    UncertaintyState,
)

logger = logging.getLogger("safeher.image_intelligence.detector")


class BaseImageDetector(ABC):
    """Abstract base class for SafeHER image manipulation detectors."""

    def __init__(self, config: Optional[ImageDetectorConfig] = None) -> None:
        self.config = config or default_image_config

    @abstractmethod
    def detect(self, data: bytes, image_id: Optional[UUID] = None) -> ImageAssessmentResponse:
        """Analyze raw image bytes and return an explainable, structured assessment."""
        pass


class SafeHERImageDetector(BaseImageDetector):
    """Primary SafeHER visual manipulation anomaly triage engine.

    Orchestrates secure preprocessing, heuristic model runner execution with resilient fallback,
    quality evaluation, explainability generation, and strict compliance with the uncalibrated scoring contract.
    Operates as an image-forensics anomaly triage tool rather than a trained neural deepfake classifier.
    """

    def __init__(
        self,
        config: Optional[ImageDetectorConfig] = None,
        runner: Optional[BaseImageModelRunner] = None,
        secondary_runner: Optional[BaseImageModelRunner] = None,
        enable_multi_detector: Optional[bool] = None,
    ) -> None:
        super().__init__(config=config)
        self._custom_runner = runner
        self._custom_secondary_runner = secondary_runner
        self._fallback_runner = BaselineHeuristicRunner(config=self.config)
        self._default_secondary_runner = FrequencySpectralRunner(config=self.config)
        self.enable_multi_detector = (
            enable_multi_detector if enable_multi_detector is not None else self.config.enable_multi_detector
        )
        self.consistency_policy = ModelDisagreementPolicy()
        self.localization_engine = VisualLocalizationEngine(config=self.config)

    @property
    def runner(self) -> BaseImageModelRunner:
        """Return the active model runner, prioritizing custom runner or neural runner if configured."""
        if self._custom_runner is not None:
            return self._custom_runner

        if self.config.enable_neural_detector:
            neural = NeuralVisionModelRunner(config=self.config)
            if neural.is_available():
                return neural
            logger.info("Neural runner not available (%s); using baseline fallback.", neural.get_status())

        return self._fallback_runner

    @property
    def secondary_runner(self) -> Optional[BaseImageModelRunner]:
        """Return the secondary frequency-domain model runner if enabled."""
        if not self.enable_multi_detector:
            return None
        if self._custom_secondary_runner is not None:
            return self._custom_secondary_runner
        return self._default_secondary_runner

    def detect(self, data: bytes, image_id: Optional[UUID] = None) -> ImageAssessmentResponse:
        """Execute full forensic triage on raw input bytes.

        Args:
            data: Raw uploaded image bytes.
            image_id: Optional UUID of the associated stored evidence.

        Returns:
            Structured, explainable ImageAssessmentResponse.

        Raises:
            ValueError: If image validation or preprocessing fails.
        """
        # 1. Deterministic evidence preprocessing (working copy isolation)
        preprocessed = preprocess_image(data=data, config=self.config)

        # 2. Model runner prediction with graceful fallback
        active_runner = self.runner
        runner_failed = False
        try:
            prediction = active_runner.predict(preprocessed)
        except Exception as exc:
            logger.warning(
                "Primary image runner (%s) raised exception: %s. Falling back to deterministic baseline.",
                getattr(active_runner, "__class__", type(active_runner)).__name__,
                str(exc),
            )
            runner_failed = True
            try:
                prediction = self._fallback_runner.predict(preprocessed)
            except Exception as fb_exc:
                logger.error(
                    "Fallback baseline runner also failed: %s. Emitting safe unassessed error response.",
                    str(fb_exc),
                )
                quality_res = evaluate_image_quality(
                    img=preprocessed.working_image,
                    raw_score=None,
                    face_bboxes=None,
                    model_failed=True,
                    is_corrupted=False,
                    format_name=preprocessed.metadata.format,
                    has_exif=preprocessed.metadata.has_exif,
                )
                return ImageAssessmentResponse(
                    image_id=image_id,
                    detector_name="SafeHER-Resilience-Safeguard",
                    detector_version="0.0.0-unhealthy",
                    detection_task=DetectionTask.HYBRID_MANIPULATION,
                    raw_model_score=0.0,
                    is_calibrated=False,
                    confidence_score=None,
                    assessment_label=AssessmentLabel.INCONCLUSIVE,
                    uncertainty_state=UncertaintyState.MODEL_UNAVAILABLE,
                    quality_indicators=quality_res.indicators,
                    quality_warnings=preprocessed.warnings + ["AI subsystem unavailable; both primary and fallback runners failed."],
                    analysis_eligibility=False,
                    reason_for_insufficient_evidence="AI analysis subsystem failure: Both primary and fallback forensic models were unavailable or encountered runtime errors.",
                    recommended_next_step="System administrators should inspect server logs and health endpoints. Users should retry once service is restored.",
                    manipulation_regions=[],
                    preprocessing_warnings=preprocessed.warnings,
                    explanation=(
                        "CRITICAL FAILURE: The automated forensic analysis could not be completed because "
                        "the AI detection subsystem is offline or unhealthy. This unassessed image cannot "
                        "and must not be treated as authentic or verified."
                    ),
                    disagreement_state="ONE_DETECTOR_UNAVAILABLE",
                    model_agreement_details={},
                    visual_explanation={"is_localizable": False, "regions": [], "method": "none"},
                    model_limitations=STANDARD_MODEL_LIMITATIONS,
                    metadata_summary=preprocessed.metadata,
                )

        # Secondary runner execution for cross-domain consistency
        secondary_prediction: Optional[ModelPrediction] = None
        sec_runner = self.secondary_runner
        if sec_runner is not None and not runner_failed:
            try:
                secondary_prediction = sec_runner.predict(preprocessed)
            except Exception as exc:
                logger.warning(
                    "Secondary frequency runner raised exception: %s. Continuing in single-detector fallback.",
                    str(exc),
                )
                secondary_prediction = None

        # 3. Quality evaluation with model outputs
        quality_res = evaluate_image_quality(
            img=preprocessed.working_image,
            raw_score=prediction.raw_score,
            face_bboxes=None,
            model_failed=runner_failed,
            is_corrupted=False,
            format_name=preprocessed.metadata.format,
            has_exif=preprocessed.metadata.has_exif,
        )

        # 4. Multi-detector consistency & disagreement policy evaluation
        consistency_res = self.consistency_policy.evaluate(
            quality_result=quality_res,
            primary_prediction=prediction,
            secondary_prediction=secondary_prediction,
        )

        # Resolve assessment label and uncertainty under SafeHER policy
        if not quality_res.analysis_eligibility:
            uncertainty = quality_res.uncertainty_state
            label = quality_res.assessment_label
        elif consistency_res.disagreement_state == DisagreementState.CONFLICTING_DETECTION_SIGNALS:
            uncertainty = consistency_res.uncertainty_state
            label = consistency_res.assessment_label
        elif quality_res.analysis_eligibility and prediction.manipulation_regions and prediction.raw_score < 0.35:
            uncertainty = UncertaintyState.MODEL_UNCERTAIN
            label = AssessmentLabel.SUSPICIOUS_MANIPULATION
        else:
            uncertainty = quality_res.uncertainty_state
            label = quality_res.assessment_label

        # 5. Visual Explanation & Manipulation Localization
        visual_loc_res = self.localization_engine.generate_visual_explanation(
            data=data,
            runner_name=prediction.runner_name,
            raw_score=prediction.raw_score,
            is_frequency_only=("Frequency" in prediction.runner_name),
            face_bboxes=None,
        )
        final_regions = (
            visual_loc_res.regions
            if visual_loc_res.is_localizable
            else prediction.manipulation_regions
        )

        # 6. Forensic explainability assembly
        preprocessing_warnings = list(dict.fromkeys(preprocessed.warnings))
        quality_warnings = list(dict.fromkeys(quality_res.warnings + consistency_res.warnings))
        combined_warnings = list(dict.fromkeys(preprocessing_warnings + quality_warnings))
        explanation = build_forensic_explanation(
            detector_name=prediction.runner_name,
            raw_score=prediction.raw_score,
            label=label,
            uncertainty=uncertainty,
            manipulation_regions=final_regions,
            preprocessed=preprocessed,
            auxiliary_metrics=prediction.auxiliary_metrics,
            quality_warnings=quality_res.warnings,
            reason_for_insufficient_evidence=quality_res.reason_for_insufficient_evidence,
        )

        if consistency_res.disagreement_state == DisagreementState.CONFLICTING_DETECTION_SIGNALS:
            explanation += f"\n\nModel Disagreement Policy Notice: {consistency_res.policy_explanation}"

        # 7. Schema assembly with strict uncalibrated contract
        return ImageAssessmentResponse(
            image_id=image_id,
            detector_name=prediction.runner_name,
            detector_version=prediction.runner_version,
            detection_task=DetectionTask.HYBRID_MANIPULATION,
            raw_model_score=prediction.raw_score,
            is_calibrated=False,  # Strictly False by architectural policy
            confidence_score=None,  # Not calibrated; do not fabricate probabilities
            assessment_label=label,
            uncertainty_state=uncertainty,
            quality_indicators=quality_res.indicators,
            quality_warnings=quality_warnings,
            analysis_eligibility=quality_res.analysis_eligibility,
            reason_for_insufficient_evidence=quality_res.reason_for_insufficient_evidence,
            recommended_next_step=quality_res.recommended_next_step,
            manipulation_regions=final_regions,
            preprocessing_warnings=preprocessing_warnings,
            explanation=explanation,
            disagreement_state=consistency_res.disagreement_state.value,
            model_agreement_details=consistency_res.to_dict(),
            visual_explanation=visual_loc_res.to_dict(),
            model_limitations=STANDARD_MODEL_LIMITATIONS,
            metadata_summary=preprocessed.metadata,
        )

    def evaluate_consistency(
        self,
        data: bytes,
    ) -> ModelConsistencyAssessment:
        """Evaluate multi-detector consistency and disagreement metrics on image bytes."""
        preprocessed = preprocess_image(data=data, config=self.config)
        primary_pred = self.runner.predict(preprocessed)
        sec_runner = self.secondary_runner
        secondary_pred = sec_runner.predict(preprocessed) if sec_runner else None
        quality_res = evaluate_image_quality(
            img=preprocessed.working_image,
            raw_score=primary_pred.raw_score,
            face_bboxes=None,
            model_failed=False,
            is_corrupted=False,
            format_name=preprocessed.metadata.format,
            has_exif=preprocessed.metadata.has_exif,
        )
        return self.consistency_policy.evaluate(
            quality_result=quality_res,
            primary_prediction=primary_pred,
            secondary_prediction=secondary_pred,
        )

    def detect_with_consistency(
        self,
        data: bytes,
        image_id: Optional[UUID] = None,
    ) -> ImageAssessmentResponse:
        """Execute detection with multi-detector consistency and disagreement policy explicitly enforced."""
        orig_multi = self.enable_multi_detector
        self.enable_multi_detector = True
        try:
            return self.detect(data=data, image_id=image_id)
        finally:
            self.enable_multi_detector = orig_multi

    def localize_manipulation(
        self,
        data: bytes,
        face_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> VisualLocalizationResult:
        """Execute scientifically grounded visual localization on raw image bytes."""
        preprocessed = preprocess_image(data=data, config=self.config)
        prediction = self.runner.predict(preprocessed)
        return self.localization_engine.generate_visual_explanation(
            data=data,
            runner_name=prediction.runner_name,
            raw_score=prediction.raw_score,
            is_frequency_only=("Frequency" in prediction.runner_name),
            face_bboxes=face_bboxes,
        )

    def evaluate_quality(
        self,
        data: bytes,
        face_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> QualityAssessmentResult:
        """Evaluate image bytes for quality, integrity, and domain suitability without running full detection."""
        try:
            preprocessed = preprocess_image(data=data, config=self.config)
            return evaluate_image_quality(
                img=preprocessed.working_image,
                face_bboxes=face_bboxes,
                format_name=preprocessed.metadata.format,
                has_exif=preprocessed.metadata.has_exif,
            )
        except Exception as exc:
            return QualityAssessmentResult(
                indicators=ImageQualityIndicators(
                    width=0,
                    height=0,
                    megapixels=0.0,
                    aspect_ratio=1.0,
                    resolution_ok=False,
                    blur_score=0.0,
                    is_blurry=True,
                    compression_artifact_score=1.0,
                    has_heavy_compression=False,
                    mean_luminance=0.0,
                    overexposed_ratio=0.0,
                    underexposed_ratio=0.0,
                    exposure_level="normal",
                    is_monochromatic_or_flat=False,
                    is_screen_capture=False,
                    face_detected=None,
                    face_count=0,
                    face_resolution_ok=None,
                    face_size_ratio=None,
                    corrupted_data_detected=True,
                    domain_category="unsupported_format",
                    is_in_domain=False,
                    overall_quality_score=0.0,
                ),
                warnings=[f"Payload processing error: {str(exc)}"],
                uncertainty_state=UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED,
                analysis_eligibility=False,
                reason_for_insufficient_evidence=f"Analysis cannot be completed: {str(exc)}",
                recommended_next_step="Re-export or re-download the original file to ensure it is not truncated or corrupted.",
                assessment_label=AssessmentLabel.INCONCLUSIVE,
            )

    def detect_safe(self, data: bytes, image_id: Optional[UUID] = None) -> ImageAssessmentResponse:
        """Execute detection with guaranteed non-throwing return, capturing ANALYSIS_CANNOT_BE_COMPLETED."""
        try:
            return self.detect(data=data, image_id=image_id)
        except Exception as exc:
            quality_res = self.evaluate_quality(data=data)
            return ImageAssessmentResponse(
                image_id=image_id,
                detector_name="SafeHER Image Quality Engine",
                detector_version="1.0.0",
                detection_task=DetectionTask.FORENSIC_TRIAGE,
                raw_model_score=0.0,
                is_calibrated=False,
                confidence_score=None,
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                uncertainty_state=UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED,
                quality_indicators=quality_res.indicators,
                quality_warnings=quality_res.warnings,
                analysis_eligibility=False,
                reason_for_insufficient_evidence=f"Analysis cannot be completed: {str(exc)}",
                recommended_next_step=quality_res.recommended_next_step,
                manipulation_regions=[],
                preprocessing_warnings=quality_res.warnings,
                explanation=f"Analysis could not be completed because the image data is corrupted, empty, or unreadable: {str(exc)}",
                model_limitations=STANDARD_MODEL_LIMITATIONS,
                metadata_summary=None,
            )
