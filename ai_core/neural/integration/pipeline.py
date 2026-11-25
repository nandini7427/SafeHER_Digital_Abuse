"""Integrated Neural and Forensic Image Triage Pipeline.

Integrates:
1. Error Level Analysis (ELA) - Compression discrepancy heuristics.
2. Fast Fourier Transform (FFT) - Radial and azimuthal spectral anomaly detection.
3. Out-of-Distribution (OOD) Assessment - Domain suitability validation.
4. Image Quality Assessment - Blur, resolution, and exposure verification.
5. Visual Localization - Micro-forensic spatial anomaly bounding boxes.
6. Model A - AI-generated vs camera photograph detection (SecureInferenceEngine).
7. Model B - Facial manipulation and blending seam detection (ModelBFacialInferenceService).

Guarantees:
- Neural execution is strictly gated behind `enable_neural: bool = False` (default False).
- Existing Milestone 3 API response fields are 100% preserved.
- Independent result sub-objects are maintained without averaging raw scores.
- Non-independent nature of ELA and FFT is explicitly accounted for.
- Full forensic disagreement policy with safe abstention and caseworker guidance.
"""

from datetime import datetime, timezone
import io
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import numpy as np
from PIL import Image, ImageChops
import torch

from ai_core.image_intelligence.config import (
    ImageDetectorConfig,
    MANDATORY_IMAGE_AI_DISCLAIMER,
    default_image_config,
)
from ai_core.image_intelligence.consistency import DisagreementState
from ai_core.image_intelligence.explainability import (
    STANDARD_MODEL_LIMITATIONS,
    build_forensic_explanation,
)
from ai_core.image_intelligence.model_runner import (
    BaselineHeuristicRunner,
    FrequencySpectralRunner,
)
from ai_core.image_intelligence.ood_detector import (
    DomainCategory,
    OutOfDistributionDetector,
)
from ai_core.image_intelligence.preprocessing import (
    PreprocessedImage,
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
from ai_core.image_intelligence.visual_localization import VisualLocalizationEngine
from ai_core.neural.calibration.abstention import (
    CalibratedAbstentionEngine,
    OODStatus,
    SafeDecisionState,
)
from ai_core.neural.facial.service import ModelBFacialInferenceService
from ai_core.neural.integration.disagreement import (
    ConsensusEvaluation,
    ForensicDisagreementPolicy,
)
from ai_core.neural.integration.schemas import (
    AIGeneratedResult,
    ELAAnalysisResult,
    FacialManipulationResult,
    FFTAnalysisResult,
    ImageQualityResult,
    IntegratedForensicAssessment,
    NeuralForensicDisagreement,
    OODEvaluationResult,
)
from ai_core.neural.management.engine import (
    InferenceTimeoutError,
    SecureEngineConfig,
    SecureInferenceEngine,
)
from ai_core.neural.schemas import InputConfiguration

logger = logging.getLogger("safeher.neural.integration")


class IntegratedForensicPipeline:
    """Production pipeline integrating classical forensics with deep neural detectors."""

    def __init__(
        self,
        config: Optional[ImageDetectorConfig] = None,
        enable_neural: bool = False,  # Gated by default
        model_a_engine: Optional[SecureInferenceEngine] = None,
        model_b_service: Optional[ModelBFacialInferenceService] = None,
        model_a_calibrator: Optional[Any] = None,
        disagreement_policy: Optional[ForensicDisagreementPolicy] = None,
    ) -> None:
        self.config = config or default_image_config
        self.enable_neural = bool(enable_neural)

        # Baseline heuristic and frequency runners (Milestone 3)
        self.ela_runner = BaselineHeuristicRunner(config=self.config)
        self.fft_runner = FrequencySpectralRunner(config=self.config)
        self.ood_detector = OutOfDistributionDetector()
        self.localization_engine = VisualLocalizationEngine(config=self.config)
        self.disagreement_policy = disagreement_policy or ForensicDisagreementPolicy()

        # Neural engines (Milestone 4)
        self.model_a_engine = model_a_engine
        self.model_b_service = model_b_service
        self.calibrated_abstention_engine = CalibratedAbstentionEngine(calibrator=model_a_calibrator)

    def analyze(
        self,
        data: bytes,
        image_id: Optional[UUID] = None,
    ) -> IntegratedForensicAssessment:
        """Execute complete multi-modal forensic assessment on raw image bytes.

        Args:
            data: Raw uploaded image bytes.
            image_id: Optional UUID associated with the evidence record.

        Returns:
            IntegratedForensicAssessment payload with additive independent results.
        """
        start_wall_time = time.perf_counter()

        # -------------------------------------------------------------------
        # 1. Deterministic Evidence Preprocessing & Working Copy Isolation
        # -------------------------------------------------------------------
        try:
            preprocessed = preprocess_image(data=data, config=self.config)
        except Exception as prep_exc:
            # Handle corrupt/unreadable bytes
            return self._build_corrupt_response(image_id=image_id, error_msg=str(prep_exc))

        # -------------------------------------------------------------------
        # 2. Out-of-Distribution (OOD) Domain Evaluation
        # -------------------------------------------------------------------
        ood_assessment = self.ood_detector.classify_domain(preprocessed.working_image)
        reasons_list = [ood_assessment.reason] if ood_assessment.reason else list(ood_assessment.warnings)
        ood_result = OODEvaluationResult(
            is_ood=bool(not ood_assessment.is_in_domain),
            domain_category=ood_assessment.category.value,
            suitability_status="IN_DOMAIN_USABLE" if ood_assessment.is_in_domain else "OUT_OF_DOMAIN_UNSUPPORTED",
            reasons=reasons_list,
        )

        # -------------------------------------------------------------------
        # 3. Technical Quality & Degradation Evaluation
        # -------------------------------------------------------------------
        raw_quality = evaluate_image_quality(
            img=preprocessed.working_image,
            raw_score=0.0,
            face_bboxes=None,
            model_failed=False,
            is_corrupted=False,
            format_name=preprocessed.metadata.format,
            has_exif=preprocessed.metadata.has_exif,
        )
        quality_result = ImageQualityResult(
            resolution_ok=raw_quality.indicators.resolution_ok,
            is_blurry=raw_quality.indicators.is_blurry,
            has_heavy_compression=raw_quality.indicators.has_heavy_compression,
            is_corrupted=raw_quality.indicators.corrupted_data_detected,
            overall_quality_score=raw_quality.indicators.overall_quality_score,
            analysis_eligibility=raw_quality.analysis_eligibility,
            warnings=list(raw_quality.warnings),
        )

        # -------------------------------------------------------------------
        # 4. Independent ELA Analysis Execution
        # -------------------------------------------------------------------
        ela_pred = self.ela_runner.predict(preprocessed)
        ela_metrics = ela_pred.auxiliary_metrics
        ela_result = ELAAnalysisResult(
            raw_score=float(np.clip(ela_pred.raw_score, 0.0, 1.0)),
            mean_error=float(ela_metrics.get("ela_mean", 0.0)),
            variance=float(ela_metrics.get("ela_variance", 0.0)),
            max_local_discrepancy=float(ela_metrics.get("max_grid_mean", 0.0)),
            suspicious_blocks_count=len(ela_pred.manipulation_regions),
            has_compression_anomalies=bool(ela_pred.raw_score > 0.40),
            quality_used=self.config.ela_quality,
            scale_used=self.config.ela_scale,
        )

        # -------------------------------------------------------------------
        # 5. Independent FFT Spectral Analysis Execution
        # -------------------------------------------------------------------
        fft_pred = self.fft_runner.predict(preprocessed)
        fft_metrics = fft_pred.auxiliary_metrics
        fft_result = FFTAnalysisResult(
            raw_score=float(np.clip(fft_pred.raw_score, 0.0, 1.0)),
            high_freq_energy_ratio=float(fft_metrics.get("high_frequency_ratio", 0.0)),
            azimuthal_spectral_discrepancy=float(fft_metrics.get("azimuthal_discrepancy", 0.0)),
            grid_peak_prominence=float(fft_metrics.get("spectral_peak_prominence", 0.0)),
            has_frequency_artifacts=bool(fft_pred.raw_score > 0.40),
        )

        # -------------------------------------------------------------------
        # 6. Neural Detectors Execution (Behind Feature Flag)
        # -------------------------------------------------------------------
        ai_gen_result: Optional[AIGeneratedResult] = None
        facial_result: Optional[FacialManipulationResult] = None
        timeout_occurred = False
        artifact_invalid = False

        if self.enable_neural:
            # 6.1 Model A: AI-Generated Image Detector
            if self.model_a_engine is not None:
                try:
                    # Convert preprocessed working image to normalized tensor (1, 3, 224, 224)
                    resized_rgb = preprocessed.working_image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    np_arr = np.array(resized_rgb, dtype=np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    norm_arr = (np_arr - mean) / std
                    tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0).float()

                    model_a_pred = self.model_a_engine.predict(tensor)
                    raw_score = model_a_pred.uncalibrated_scores.get("AI_GENERATED_IMAGE", 0.0)

                    # Pass through calibrated abstention engine
                    cal_pred = self.calibrated_abstention_engine.evaluate(
                        raw_logit=model_a_pred.raw_logits[1] if len(model_a_pred.raw_logits) > 1 else None,
                        uncalibrated_score=raw_score,
                        model_version=model_a_pred.model_version,
                        ood_status=OODStatus(
                            is_ood=ood_result.is_ood,
                            domain_category=ood_result.domain_category,
                        ),
                    )

                    ai_gen_result = AIGeneratedResult(
                        is_available=True,
                        model_version=model_a_pred.model_version,
                        architecture=model_a_pred.architecture,
                        raw_logits=model_a_pred.raw_logits,
                        uncalibrated_score=round(raw_score, 4),
                        calibrated_probability=round(cal_pred.calibrated_probability, 4) if cal_pred.calibrated_probability is not None else None,
                        decision_state=cal_pred.decision_state.value,
                        uncertainty=cal_pred.uncertainty.__dict__,
                        disclaimer=cal_pred.disclaimer,
                    )
                except InferenceTimeoutError:
                    timeout_occurred = True
                    ai_gen_result = AIGeneratedResult(is_available=False, decision_state="TIMEOUT")
                except Exception as model_a_err:
                    logger.warning("Model A execution failed: %s", model_a_err)
                    ai_gen_result = AIGeneratedResult(is_available=False, decision_state="ERROR")

            # 6.2 Model B: Facial Manipulation Detector
            if self.model_b_service is not None and not timeout_occurred:
                try:
                    infer_fn = getattr(self.model_b_service, "predict", None) or getattr(self.model_b_service, "analyze_image")
                    facial_pred = infer_fn(preprocessed.working_image)
                    face_box_tuple = None
                    if facial_pred.face_bounding_box:
                        b = facial_pred.face_bounding_box
                        face_box_tuple = (b.y_min, b.x_min, b.y_max, b.x_max)

                    top_cat_score = max(facial_pred.manipulation_category_indicators.values()) if facial_pred.manipulation_category_indicators else 0.0

                    facial_model_available = bool(getattr(facial_pred, "model_available", False))
                    facial_result = FacialManipulationResult(
                        is_available=facial_model_available,
                        model_version=facial_pred.model_version,
                        face_detected=(facial_pred.faces_detected_count > 0),
                        face_count=facial_pred.faces_detected_count,
                        face_detection_status=facial_pred.face_detection_status.value,
                        analysis_limitations=list(facial_pred.analysis_limitations),
                        face_bounding_box=face_box_tuple,
                        raw_score=round(top_cat_score, 4) if facial_model_available else 0.0,
                        calibrated_probability=None,
                        decision_state=facial_pred.inference_status,
                        manipulation_type=(
                            facial_pred.predicted_category.value
                            if facial_model_available and facial_pred.faces_detected_count > 0
                            else "not_assessed"
                        ),
                        localized_regions=[r.model_dump() for r in facial_pred.localized_regions],
                        face_conditions=[c.value for c in facial_pred.face_conditions],
                        disclaimer=facial_pred.disclaimer,
                    )
                except Exception as model_b_err:
                    logger.warning("Model B facial service failed: %s", model_b_err)
                    facial_result = FacialManipulationResult(is_available=False, decision_state="ERROR")

        # -------------------------------------------------------------------
        # 7. Multi-Modal Disagreement & Consensus Resolution
        # -------------------------------------------------------------------
        consensus = self.disagreement_policy.evaluate_consensus(
            ood_result=ood_result,
            quality_result=quality_result,
            ela_result=ela_result,
            fft_result=fft_result,
            ai_gen_result=ai_gen_result,
            facial_result=facial_result,
            timeout_occurred=timeout_occurred,
            artifact_invalid=artifact_invalid,
        )

        # -------------------------------------------------------------------
        # 8. Visual Localization & Region Attribution
        # -------------------------------------------------------------------
        visual_loc_res = self.localization_engine.generate_visual_explanation(
            data=data,
            runner_name="Integrated-Neural-Forensic-Pipeline",
            raw_score=consensus.synthesized_score,
            is_frequency_only=False,
            face_bboxes=None,
        )

        # Merge localized regions
        final_regions: List[ManipulationRegion] = list(visual_loc_res.regions if visual_loc_res.is_localizable else ela_pred.manipulation_regions)
        if facial_result and facial_result.localized_regions:
            for reg_dict in facial_result.localized_regions:
                box_d = reg_dict.get("bounding_box", {})
                if box_d:
                    final_regions.append(
                        ManipulationRegion(
                            box_normalized=(
                                float(box_d.get("y_min", 0.0)),
                                float(box_d.get("x_min", 0.0)),
                                float(box_d.get("y_max", 1.0)),
                                float(box_d.get("x_max", 1.0)),
                            ),
                            anomaly_score=float(reg_dict.get("anomaly_density", 0.5)),
                            region_type=ManipulationRegionType.BOUNDING_BOX,
                            attribution_category="facial_blending_seam",
                            description=f"Suspected facial editing boundary ({reg_dict.get('manipulation_type', 'face_edit')})",
                        )
                    )

        # -------------------------------------------------------------------
        # 9. Forensic Explainability Assembly
        # -------------------------------------------------------------------
        explanation = build_forensic_explanation(
            detector_name="SafeHER Integrated Neural & Forensic Triage Engine",
            raw_score=consensus.synthesized_score,
            label=consensus.assessment_label,
            uncertainty=consensus.uncertainty_state,
            manipulation_regions=final_regions,
            preprocessed=preprocessed,
            auxiliary_metrics={
                "ela_variance": ela_result.variance,
                "fft_high_freq": fft_result.high_freq_energy_ratio,
                "neural_active": self.enable_neural,
            },
            quality_warnings=consensus.warnings,
            reason_for_insufficient_evidence=consensus.human_review_guidance if consensus.assessment_label == AssessmentLabel.INCONCLUSIVE else None,
        )

        total_elapsed_ms = (time.perf_counter() - start_wall_time) * 1000.0

        # -------------------------------------------------------------------
        # 10. Construct Fully Backward-Compatible Integrated Response
        # -------------------------------------------------------------------
        return IntegratedForensicAssessment(
            assessment_id=uuid4(),
            image_id=image_id,
            detector_name="SafeHER-Integrated-Neural-Forensic-Engine",
            detector_version="1.0.0-m4.7",
            detection_task=DetectionTask.HYBRID_MANIPULATION,
            raw_model_score=round(float(consensus.synthesized_score), 4),
            is_calibrated=consensus.is_calibrated,
            confidence_score=round(float(consensus.confidence_score), 4) if consensus.confidence_score is not None else None,
            assessment_label=consensus.assessment_label,
            uncertainty_state=consensus.uncertainty_state,
            manipulation_regions=final_regions,
            preprocessing_warnings=preprocessed.warnings,
            quality_indicators=raw_quality.indicators,
            quality_warnings=consensus.warnings,
            analysis_eligibility=quality_result.analysis_eligibility and not ood_result.is_ood,
            reason_for_insufficient_evidence=consensus.human_review_guidance if consensus.assessment_label == AssessmentLabel.INCONCLUSIVE else None,
            recommended_next_step=(
                "Caseworker review recommended: Check visual boundary anomalies and camera EXIF."
                if consensus.disagreement.human_review_required
                else "Preserve original file with cryptographic SHA-256 hash."
            ),
            explanation=explanation,
            disagreement_state=consensus.disagreement.disagreement_type,
            model_agreement_details=consensus.disagreement.model_dump(),
            visual_explanation=visual_loc_res.to_dict(),
            model_limitations=list(STANDARD_MODEL_LIMITATIONS),
            metadata_summary=preprocessed.metadata,
            analysis_timestamp=datetime.now(timezone.utc),
            disclaimer=MANDATORY_IMAGE_AI_DISCLAIMER,
            # Additive M4.7 Fields
            ai_generated_result=ai_gen_result,
            facial_manipulation_result=facial_result,
            ela_analysis_result=ela_result,
            fft_analysis_result=fft_result,
            ood_evaluation_result=ood_result,
            neural_forensic_disagreement=consensus.disagreement,
            human_review_guidance=consensus.human_review_guidance,
            is_neural_enabled=self.enable_neural,
        )

    def _build_corrupt_response(self, image_id: Optional[UUID], error_msg: str) -> IntegratedForensicAssessment:
        """Safe non-crashing response for corrupt or completely unreadable bitstreams."""
        return IntegratedForensicAssessment(
            assessment_id=uuid4(),
            image_id=image_id,
            detector_name="SafeHER-Resilience-Safeguard",
            detector_version="1.0.0-m4.7",
            detection_task=DetectionTask.HYBRID_MANIPULATION,
            raw_model_score=0.0,
            is_calibrated=False,
            confidence_score=None,
            assessment_label=AssessmentLabel.INCONCLUSIVE,
            uncertainty_state=UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED,
            manipulation_regions=[],
            preprocessing_warnings=[f"Preprocess error: {error_msg}"],
            quality_indicators=None,
            quality_warnings=["File is corrupted, truncated, or in an unreadable format."],
            analysis_eligibility=False,
            reason_for_insufficient_evidence=f"Corrupt image file: {error_msg}",
            recommended_next_step="Request the original camera photograph directly from the source.",
            explanation="CRITICAL: The uploaded image file could not be decoded or processed.",
            disagreement_state="CORRUPT_FILE",
            model_agreement_details={},
            visual_explanation={"is_localizable": False, "regions": [], "method": "none"},
            model_limitations=list(STANDARD_MODEL_LIMITATIONS),
            metadata_summary=None,
            analysis_timestamp=datetime.now(timezone.utc),
            disclaimer=MANDATORY_IMAGE_AI_DISCLAIMER,
            is_neural_enabled=self.enable_neural,
        )
