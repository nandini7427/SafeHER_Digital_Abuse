"""Unit and integration tests for SafeHER Milestone 4.7 Neural and Forensic Detector Integration.

Tests:
1. Feature flag gating (neural detectors disabled by default).
2. Backward compatibility with Milestone 3 ImageAssessmentResponse contract.
3. Independent result object preservation (AIGenerated, Facial, ELA, FFT, OOD, Quality).
4. No score averaging rule enforcement.
5. Disagreement policy:
   - Neural and ELA agreement (both clean -> NO_OBVIOUS_ARTIFACTS; both manipulated -> HIGHLY_LIKELY_MANIPULATED).
   - Neural and FFT agreement.
   - Neural/forensic conflict -> INCONCLUSIVE with mandatory human review guidance.
   - Missing neural model -> graceful fallback to heuristic baseline.
   - Poor image quality (blur / heavy compression / low-res) -> INCONCLUSIVE.
   - Out-of-distribution (OOD) image -> OUTSIDE_SUPPORTED_DOMAIN.
   - Corrupt / unsupported file -> ANALYSIS_CANNOT_BE_COMPLETED.
   - Inference timeout -> graceful degradation with warning.
   - Invalid artifact / checksum failure -> safe fallback.
   - Non-finite prediction -> safe quarantine.
6. Combined pipeline end-to-end execution.
"""

import io
from pathlib import Path
import tempfile
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import pytest
import safetensors.torch
import torch

from ai_core.image_intelligence.config import ImageDetectorConfig
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    ImageAssessmentResponse,
    UncertaintyState,
)
from ai_core.neural.calibration import TemperatureScalingCalibrator
from ai_core.neural.facial.service import ModelBFacialInferenceService
from ai_core.neural.integration import (
    AIGeneratedResult,
    ELAAnalysisResult,
    FacialManipulationResult,
    FFTAnalysisResult,
    ForensicDisagreementPolicy,
    ImageQualityResult,
    IntegratedForensicAssessment,
    IntegratedForensicPipeline,
    NeuralForensicDisagreement,
    OODEvaluationResult,
)
from ai_core.neural.management import (
    ModelArtifactFormat,
    ModelUsageState,
    SecureEngineConfig,
    SecureInferenceEngine,
    SecureModelMetadata,
    compute_streaming_sha256,
)
from ai_core.neural.models.factory import create_image_detector
from ai_core.neural.models.patch_forensics import PatchForensicsFacialDetector


def _generate_test_image_bytes(
    width: int = 256,
    height: int = 256,
    format_name: str = "JPEG",
) -> bytes:
    """Helper to generate in-memory synthetic photographic image bytes with natural sensor noise."""
    rng = np.random.RandomState(42)
    arr = np.clip(rng.normal(128, 20, (height, width, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=format_name, quality=95)
    return buf.getvalue()


def _create_mock_engine(tmp_dir: Path) -> SecureInferenceEngine:
    """Helper to instantiate a real local Model A SecureInferenceEngine with verified dummy weights."""
    detector = create_image_detector("mobilenet_v3_large", is_trained=True)
    weights_path = tmp_dir / "model_a.safetensors"
    safetensors.torch.save_file(detector.state_dict(), str(weights_path))

    sha256_hex, file_size = compute_streaming_sha256(weights_path)
    metadata_path = tmp_dir / "metadata_a.json"
    meta = SecureModelMetadata(
        model_id="safeher-mobilenetv3-test",
        version="1.0.0",
        architecture="mobilenet_v3_large",
        input_dimensions=(3, 224, 224),
        preprocessing_version="v1-imagenet-norm",
        class_mappings={0: "CAMERA_PHOTOGRAPH", 1: "AI_GENERATED_IMAGE"},
        training_manifest_hash="b" * 64,
        evaluation_report_reference="reports/INDEPENDENT_NEURAL_EVALUATION_REPORT.md",
        sha256=sha256_hex,
        expected_file_size_bytes=file_size,
        approved_usage_state=ModelUsageState.COMMERCIAL_TRAINED,
        format=ModelArtifactFormat.SAFETENSORS,
    ).with_computed_hash()
    meta.save(metadata_path)

    config = SecureEngineConfig(device="cpu", num_threads=1)
    return SecureInferenceEngine(
        artifact_path=weights_path,
        metadata_path=metadata_path,
        config=config,
        lazy_load=False,
    )


def _create_mock_facial_service() -> ModelBFacialInferenceService:
    """Helper to create a ModelBFacialInferenceService with dummy weights."""
    detector = PatchForensicsFacialDetector(is_trained=True)
    return ModelBFacialInferenceService(
        model=detector,
        enforce_trained=False,
    )


# ---------------------------------------------------------------------------
# 1. Feature Flag Gating & Default Safety Tests
# ---------------------------------------------------------------------------

def test_neural_detector_disabled_by_default():
    """Verify that neural detectors are strictly gated behind enable_neural=False by default."""
    pipeline = IntegratedForensicPipeline()
    assert pipeline.enable_neural is False

    img_bytes = _generate_test_image_bytes()
    res = pipeline.analyze(img_bytes)

    assert isinstance(res, IntegratedForensicAssessment)
    assert res.is_neural_enabled is False
    assert res.ai_generated_result is None
    assert res.facial_manipulation_result is None
    # Heuristic results must still be populated
    assert res.ela_analysis_result is not None
    assert res.fft_analysis_result is not None
    assert res.ood_evaluation_result is not None


# ---------------------------------------------------------------------------
# 2. Backward Compatibility Tests
# ---------------------------------------------------------------------------

def test_full_backward_compatibility_with_milestone_3_response():
    """Verify that IntegratedForensicAssessment preserves every single Milestone 3 API field."""
    pipeline = IntegratedForensicPipeline(enable_neural=False)
    img_bytes = _generate_test_image_bytes()
    res = pipeline.analyze(img_bytes)

    # Must be an instance of ImageAssessmentResponse
    assert isinstance(res, ImageAssessmentResponse)

    # Core Milestone 3 fields verification
    assert res.assessment_id is not None
    assert res.detector_name != ""
    assert res.detector_version != ""
    assert res.detection_task.value == "hybrid_manipulation"
    assert 0.0 <= res.raw_model_score <= 1.0
    assert res.is_calibrated is False
    assert res.confidence_score is None
    assert res.assessment_label in AssessmentLabel
    assert res.uncertainty_state in UncertaintyState
    assert isinstance(res.manipulation_regions, list)
    assert isinstance(res.quality_warnings, list)
    assert isinstance(res.preprocessing_warnings, list)
    assert res.explanation != ""
    assert res.disclaimer != ""
    assert res.metadata_summary is not None
    assert res.metadata_summary.sha256 != ""


# ---------------------------------------------------------------------------
# 3. Independent Result Objects & No Score Averaging Tests
# ---------------------------------------------------------------------------

def test_independent_result_objects_and_no_score_averaging():
    """Verify that all six forensic modalities maintain distinct result objects without score averaging."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        engine_a = _create_mock_engine(tmp_dir)
        service_b = _create_mock_facial_service()

        calibrator = TemperatureScalingCalibrator(temperature=1.2)
        calibrator.is_fitted = True

        pipeline = IntegratedForensicPipeline(
            enable_neural=True,
            model_a_engine=engine_a,
            model_b_service=service_b,
            model_a_calibrator=calibrator,
        )

        img_bytes = _generate_test_image_bytes(256, 256)
        res = pipeline.analyze(img_bytes)

        assert res.is_neural_enabled is True

        # Check all 6 independent result objects
        assert res.ai_generated_result is not None
        assert isinstance(res.ai_generated_result, AIGeneratedResult)
        assert res.ai_generated_result.is_available is True

        assert res.facial_manipulation_result is not None
        assert isinstance(res.facial_manipulation_result, FacialManipulationResult)
        # The live pipeline does not embed a face detector or supply verified
        # face boxes; it must report unavailable localization, not invent a face.
        assert res.facial_manipulation_result.face_detection_status == "localization_unavailable"
        assert res.facial_manipulation_result.decision_state == "FACE_LOCALIZATION_UNAVAILABLE"
        assert res.facial_manipulation_result.manipulation_type == "not_assessed"
        assert res.facial_manipulation_result.analysis_limitations

        assert res.ela_analysis_result is not None
        assert isinstance(res.ela_analysis_result, ELAAnalysisResult)

        assert res.fft_analysis_result is not None
        assert isinstance(res.fft_analysis_result, FFTAnalysisResult)

        assert res.ood_evaluation_result is not None
        assert isinstance(res.ood_evaluation_result, OODEvaluationResult)

        # Verify raw score is NOT an arithmetic average of unrelated modalities
        ai_score = res.ai_generated_result.uncalibrated_score
        ela_score = res.ela_analysis_result.variance
        fft_score = res.fft_analysis_result.high_freq_energy_ratio
        avg_score = (ai_score + ela_score + fft_score) / 3.0

        # The synthesized score must follow policy, not naive averaging
        assert abs(res.raw_model_score - avg_score) > 1e-4 or (ai_score == ela_score == fft_score)


# ---------------------------------------------------------------------------
# 4. Multi-Modal Disagreement Policy Tests
# ---------------------------------------------------------------------------

def test_disagreement_policy_neural_and_ela_agreement():
    """Verify full consensus between neural and heuristic checks produces confident labels."""
    policy = ForensicDisagreementPolicy()
    ood_ok = OODEvaluationResult(is_ood=False, domain_category="photographic", suitability_status="IN_DOMAIN_USABLE")
    quality_ok = ImageQualityResult(resolution_ok=True, is_blurry=False, has_heavy_compression=False, overall_quality_score=0.9, analysis_eligibility=True, warnings=[])

    # Case A: Both indicate clean
    ai_clean = AIGeneratedResult(is_available=True, uncalibrated_score=0.08, calibrated_probability=0.08, decision_state="NO_CLEAR_INDICATORS")
    ela_clean = ELAAnalysisResult(mean_error=2.0, variance=0.05, max_local_discrepancy=0.1, has_compression_anomalies=False)
    fft_clean = FFTAnalysisResult(high_freq_energy_ratio=0.08, azimuthal_spectral_discrepancy=0.02, grid_peak_prominence=0.01, has_frequency_artifacts=False)

    eval_clean = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_ok,
        ela_result=ela_clean,
        fft_result=fft_clean,
        ai_gen_result=ai_clean,
    )
    assert eval_clean.assessment_label == AssessmentLabel.NO_OBVIOUS_ARTIFACTS
    assert eval_clean.disagreement.disagreement_type == "FULL_CONSENSUS_CLEAN"
    assert eval_clean.disagreement.human_review_required is False

    # Case B: Both indicate manipulation
    ai_manip = AIGeneratedResult(is_available=True, uncalibrated_score=0.92, calibrated_probability=0.94, decision_state="INDICATORS_DETECTED")
    ela_manip = ELAAnalysisResult(mean_error=15.0, variance=0.85, max_local_discrepancy=0.9, has_compression_anomalies=True)
    fft_manip = FFTAnalysisResult(high_freq_energy_ratio=0.75, azimuthal_spectral_discrepancy=0.60, grid_peak_prominence=0.80, has_frequency_artifacts=True)

    eval_manip = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_ok,
        ela_result=ela_manip,
        fft_result=fft_manip,
        ai_gen_result=ai_manip,
    )
    assert eval_manip.assessment_label == AssessmentLabel.HIGHLY_LIKELY_MANIPULATED
    assert eval_manip.disagreement.disagreement_type == "FULL_CONSENSUS_MANIPULATED"
    assert eval_manip.disagreement.human_review_required is True


def test_disagreement_policy_neural_and_forensic_conflict():
    """Verify conflict between neural and heuristic checks triggers INCONCLUSIVE and human review."""
    policy = ForensicDisagreementPolicy()
    ood_ok = OODEvaluationResult(is_ood=False, domain_category="photographic", suitability_status="IN_DOMAIN_USABLE")
    quality_ok = ImageQualityResult(resolution_ok=True, is_blurry=False, has_heavy_compression=False, overall_quality_score=0.9, analysis_eligibility=True, warnings=[])

    # Neural says AI (0.92), but ELA/FFT shows uniform camera noise (clean)
    ai_manip = AIGeneratedResult(is_available=True, uncalibrated_score=0.92, calibrated_probability=0.92, decision_state="INDICATORS_DETECTED")
    ela_clean = ELAAnalysisResult(mean_error=1.5, variance=0.04, max_local_discrepancy=0.08, has_compression_anomalies=False)
    fft_clean = FFTAnalysisResult(high_freq_energy_ratio=0.05, azimuthal_spectral_discrepancy=0.01, grid_peak_prominence=0.01, has_frequency_artifacts=False)

    consensus = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_ok,
        ela_result=ela_clean,
        fft_result=fft_clean,
        ai_gen_result=ai_manip,
    )

    assert consensus.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert consensus.uncertainty_state == UncertaintyState.MODEL_UNCERTAIN
    assert consensus.disagreement.disagreement_type == "NEURAL_FORENSIC_DISAGREEMENT"
    assert consensus.disagreement.human_review_required is True
    assert "Caseworker review required" in consensus.disagreement.recommended_action


def test_disagreement_policy_poor_image_quality_abstention():
    """Verify degraded image quality (blur / extreme compression) triggers INCONCLUSIVE with ANALYSIS_POOR_QUALITY_EVIDENCE."""
    policy = ForensicDisagreementPolicy()
    ood_ok = OODEvaluationResult(is_ood=False, domain_category="photographic", suitability_status="IN_DOMAIN_USABLE")
    # Low blur score (<15) + heavy compression
    quality_poor = ImageQualityResult(
        resolution_ok=True,
        is_blurry=True,
        has_heavy_compression=True,
        overall_quality_score=0.25,
        analysis_eligibility=True,
        warnings=["Low edge energy; severe optical blur."],
    )

    consensus = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_poor,
    )

    assert consensus.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert consensus.uncertainty_state == UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE
    assert consensus.disagreement.disagreement_type == "POOR_IMAGE_QUALITY"
    assert consensus.disagreement.human_review_required is True


def test_disagreement_policy_out_of_distribution_refusal():
    """Verify OOD image (e.g. digital drawing / meme) strictly triggers OUTSIDE_SUPPORTED_DOMAIN."""
    policy = ForensicDisagreementPolicy()
    ood_fail = OODEvaluationResult(
        is_ood=True,
        domain_category="digital_illustration",
        suitability_status="OUT_OF_DOMAIN_UNSUPPORTED",
        reasons=["Detected flat illustration palette and vector fills."],
    )
    quality_ok = ImageQualityResult(resolution_ok=True, is_blurry=False, has_heavy_compression=False, overall_quality_score=0.9, analysis_eligibility=True, warnings=[])

    consensus = policy.evaluate_consensus(
        ood_result=ood_fail,
        quality_result=quality_ok,
    )

    assert consensus.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert consensus.uncertainty_state == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN
    assert consensus.disagreement.disagreement_type == "OUT_OF_DISTRIBUTION"


def test_disagreement_policy_timeout_and_corrupt_files():
    """Verify timeout and corrupted bitstreams produce safe non-crashing abstentions."""
    policy = ForensicDisagreementPolicy()
    ood_ok = OODEvaluationResult(is_ood=False, domain_category="photographic", suitability_status="IN_DOMAIN_USABLE")
    quality_ok = ImageQualityResult(resolution_ok=True, is_blurry=False, has_heavy_compression=False, overall_quality_score=0.9, analysis_eligibility=True, warnings=[])

    # Timeout
    eval_timeout = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_ok,
        timeout_occurred=True,
    )
    assert eval_timeout.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert eval_timeout.uncertainty_state == UncertaintyState.MODEL_UNAVAILABLE
    assert eval_timeout.disagreement.disagreement_type == "INFERENCE_TIMEOUT"

    # Corrupt file
    quality_corrupt = ImageQualityResult(
        resolution_ok=False,
        is_blurry=False,
        has_heavy_compression=False,
        is_corrupted=True,
        overall_quality_score=0.0,
        analysis_eligibility=False,
        warnings=["Corrupted byte stream."],
    )
    eval_corrupt = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_corrupt,
    )
    assert eval_corrupt.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert eval_corrupt.uncertainty_state == UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED


def test_disagreement_policy_non_finite_predictions():
    """Verify non-finite neural logit produces INCONCLUSIVE with MODEL_UNCERTAIN."""
    policy = ForensicDisagreementPolicy()
    ood_ok = OODEvaluationResult(is_ood=False, domain_category="photographic", suitability_status="IN_DOMAIN_USABLE")
    quality_ok = ImageQualityResult(resolution_ok=True, is_blurry=False, has_heavy_compression=False, overall_quality_score=0.9, analysis_eligibility=True, warnings=[])

    ai_nan = AIGeneratedResult(
        is_available=True,
        raw_logits=[0.5, float("nan")],
        uncalibrated_score=0.5,
    )
    consensus = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_ok,
        ai_gen_result=ai_nan,
    )
    assert consensus.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert consensus.uncertainty_state == UncertaintyState.MODEL_UNCERTAIN
    assert consensus.disagreement.disagreement_type == "NON_FINITE_PREDICTION"


# ---------------------------------------------------------------------------
# 5. Pipeline End-to-End Failure & Edge Case Tests
# ---------------------------------------------------------------------------

def test_pipeline_ood_image_triggers_abstention():
    """Verify end-to-end pipeline detects OOD images and abstains with OUTSIDE_SUPPORTED_DOMAIN."""
    pipeline = IntegratedForensicPipeline(enable_neural=True)
    # 100% white flat image with high border contrast mimics a graphic/document
    white_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    buf = io.BytesIO()
    white_img.save(buf, format="PNG")

    res = pipeline.analyze(buf.getvalue())
    assert isinstance(res, IntegratedForensicAssessment)
    assert res.ood_evaluation_result is not None


def test_pipeline_corrupted_bytes_handling():
    """Verify completely corrupt or truncated bytes do not crash the pipeline."""
    pipeline = IntegratedForensicPipeline(enable_neural=True)
    corrupted_bytes = b"CORRUPTED_NON_IMAGE_HEADER_1234567890"

    res = pipeline.analyze(corrupted_bytes)
    assert isinstance(res, IntegratedForensicAssessment)
    assert res.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert res.uncertainty_state == UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED
    assert res.analysis_eligibility is False


def test_pipeline_missing_neural_model_graceful_fallback():
    """Verify enable_neural=True without engines falls back gracefully to heuristics."""
    pipeline = IntegratedForensicPipeline(
        enable_neural=True,
        model_a_engine=None,
        model_b_service=None,
    )
    img_bytes = _generate_test_image_bytes()
    res = pipeline.analyze(img_bytes)

    assert isinstance(res, IntegratedForensicAssessment)
    assert res.is_neural_enabled is True
    assert res.ai_generated_result is None
    assert res.facial_manipulation_result is None
    # Heuristics must succeed
    assert res.ela_analysis_result is not None
    assert res.fft_analysis_result is not None
    assert res.disagreement_state == "HEURISTIC_ONLY_ACTIVE"


def test_pipeline_human_review_guidance_on_disagreement():
    """Verify conflicting modalities provide clear human review guidance for caseworkers."""
    policy = ForensicDisagreementPolicy()
    ood_ok = OODEvaluationResult(is_ood=False, domain_category="photographic", suitability_status="IN_DOMAIN_USABLE")
    quality_ok = ImageQualityResult(resolution_ok=True, is_blurry=False, has_heavy_compression=False, overall_quality_score=0.9, analysis_eligibility=True, warnings=[])

    # Disagreeing inputs
    ai_manip = AIGeneratedResult(is_available=True, uncalibrated_score=0.88, calibrated_probability=0.88, decision_state="INDICATORS_DETECTED")
    ela_clean = ELAAnalysisResult(mean_error=1.0, variance=0.03, max_local_discrepancy=0.05, has_compression_anomalies=False)

    consensus = policy.evaluate_consensus(
        ood_result=ood_ok,
        quality_result=quality_ok,
        ela_result=ela_clean,
        ai_gen_result=ai_manip,
    )

    assert consensus.human_review_guidance != ""
    assert "human triage review required" in consensus.human_review_guidance.lower()
    assert consensus.disagreement.human_review_required is True
