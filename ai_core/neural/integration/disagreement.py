"""Forensic consensus and disagreement resolution policy.

Defines explicit handling for:
1. Neural and ELA agreement.
2. Neural and FFT agreement.
3. Neural/forensic disagreement (NEVER averages raw scores from unrelated systems).
4. Missing neural model (graceful fallback).
5. Poor image quality (attenuation of confidence, safe abstention).
6. Out-of-distribution (OOD) images (strict refusal to force classification).
7. Unsupported image types / corrupted files.
8. Inference timeouts.
9. Invalid artifacts / checksum failures.
10. Non-finite predictions (NaN / Inf).

Crucial Rules:
- Never average raw scores from unrelated systems.
- Do not treat ELA and FFT as statistically independent proofs.
- Negative detection must NEVER be converted into an authenticity certification.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    UncertaintyState,
)
from ai_core.neural.integration.schemas import (
    AIGeneratedResult,
    ELAAnalysisResult,
    FacialManipulationResult,
    FFTAnalysisResult,
    ImageQualityResult,
    NeuralForensicDisagreement,
    OODEvaluationResult,
)


@dataclass(frozen=True)
class ConsensusEvaluation:
    """Consensus outcome from multi-modal forensic evaluation."""

    assessment_label: AssessmentLabel
    uncertainty_state: UncertaintyState
    disagreement: NeuralForensicDisagreement
    synthesized_score: float
    is_calibrated: bool
    confidence_score: Optional[float]
    human_review_guidance: str
    warnings: List[str]


class ForensicDisagreementPolicy:
    """Evaluates multi-modal indicators and resolves agreements and conflicts according to SafeHER policy."""

    def __init__(
        self,
        neural_high_threshold: float = 0.80,
        neural_low_threshold: float = 0.20,
        ela_anomaly_threshold: float = 0.45,
        fft_anomaly_threshold: float = 0.40,
    ) -> None:
        self.neural_high_threshold = neural_high_threshold
        self.neural_low_threshold = neural_low_threshold
        self.ela_anomaly_threshold = ela_anomaly_threshold
        self.fft_anomaly_threshold = fft_anomaly_threshold

    def evaluate_consensus(
        self,
        ood_result: OODEvaluationResult,
        quality_result: ImageQualityResult,
        ela_result: Optional[ELAAnalysisResult] = None,
        fft_result: Optional[FFTAnalysisResult] = None,
        ai_gen_result: Optional[AIGeneratedResult] = None,
        facial_result: Optional[FacialManipulationResult] = None,
        timeout_occurred: bool = False,
        artifact_invalid: bool = False,
    ) -> ConsensusEvaluation:
        """Synthesize multi-modal findings into a safe, explainable forensic assessment.

        Strictly avoids averaging scores and enforces safe abstention for all conflict cases.
        """
        warnings: List[str] = list(quality_result.warnings)

        # -------------------------------------------------------------------
        # Rule 1: Out-of-Distribution (OOD) Guard
        # -------------------------------------------------------------------
        if ood_result.is_ood:
            disagreement = NeuralForensicDisagreement(
                disagreement_type="OUT_OF_DISTRIBUTION",
                neural_verdict="ABSTAINED_OOD",
                forensic_verdict="ABSTAINED_OOD",
                confidence_delta=0.0,
                synthesis_rationale=(
                    f"Input image classified as '{ood_result.domain_category}', which falls outside "
                    "the validated photographic evidence domain. Automated deepfake and forensic "
                    "classification is strictly prohibited on non-photographic domains."
                ),
                human_review_required=True,
                recommended_action=(
                    "Do not rely on automated forensic models for drawings, memes, or screenshots. "
                    "If authentic camera evidence exists, request the original unmodified file."
                ),
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                disagreement=disagreement,
                synthesized_score=0.0,
                is_calibrated=False,
                confidence_score=None,
                human_review_guidance=(
                    "Out-of-distribution input: Models abstain from classification to prevent false allegations."
                ),
                warnings=warnings + ood_result.reasons,
            )

        # -------------------------------------------------------------------
        # Rule 2: Corrupt or Unsupported Image Guard
        # -------------------------------------------------------------------
        if quality_result.is_corrupted:
            disagreement = NeuralForensicDisagreement(
                disagreement_type="UNSUPPORTED_OR_CORRUPT_INPUT",
                neural_verdict="CANNOT_COMPLETE",
                forensic_verdict="CANNOT_COMPLETE",
                confidence_delta=0.0,
                synthesis_rationale=(
                    "Input image failed technical eligibility checks (corrupt bitstream or unreadable format). "
                    "Forensic analysis cannot be completed."
                ),
                human_review_required=True,
                recommended_action="Request an uncorrupted original image file with verified SHA-256 hash.",
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                uncertainty_state=UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED,
                disagreement=disagreement,
                synthesized_score=0.0,
                is_calibrated=False,
                confidence_score=None,
                human_review_guidance="Corrupt image: Analysis aborted to prevent misleading indicators.",
                warnings=warnings,
            )

        # -------------------------------------------------------------------
        # Rule 3: Severe Quality Degradation (Low Resolution, Blur, Quantization)
        # -------------------------------------------------------------------
        if not quality_result.resolution_ok or quality_result.is_blurry or quality_result.has_heavy_compression:
            deg_reasons = []
            if not quality_result.resolution_ok:
                deg_reasons.append("resolution below forensic minimum (<128x128)")
            if quality_result.is_blurry:
                deg_reasons.append("severe motion or optical blur")
            if quality_result.has_heavy_compression:
                deg_reasons.append("heavy JPEG compression artifacts (Q<=30)")

            disagreement = NeuralForensicDisagreement(
                disagreement_type="POOR_IMAGE_QUALITY",
                neural_verdict="ATTENUATED_CONFIDENCE",
                forensic_verdict="ATTENUATED_CONFIDENCE",
                confidence_delta=0.0,
                synthesis_rationale=(
                    f"Evidence quality degraded by {', '.join(deg_reasons)}. "
                    "High-frequency blending seams and PRNU sensor signatures are heavily attenuated."
                ),
                human_review_required=True,
                recommended_action=(
                    "Preserve original file hash. Automated models cannot establish reliable evidence on "
                    "heavily degraded files; manual caseworker inspection is required."
                ),
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                uncertainty_state=UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
                disagreement=disagreement,
                synthesized_score=0.0,
                is_calibrated=False,
                confidence_score=None,
                human_review_guidance="Evidence quality degraded; triage score attenuated to INCONCLUSIVE.",
                warnings=warnings + [f"Quality warning: {r}" for r in deg_reasons],
            )

        # -------------------------------------------------------------------
        # Rule 4: System Failure Modes (Timeout, Invalid Artifact, Non-Finite)
        # -------------------------------------------------------------------
        if timeout_occurred:
            disagreement = NeuralForensicDisagreement(
                disagreement_type="INFERENCE_TIMEOUT",
                neural_verdict="TIMEOUT",
                forensic_verdict="EVALUATED_BASELINE",
                confidence_delta=0.0,
                synthesis_rationale="Neural inference watchdog timed out. Falling back to deterministic heuristics.",
                human_review_required=True,
                recommended_action="System load high or complex image structure; retry with smaller batch.",
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                uncertainty_state=UncertaintyState.MODEL_UNAVAILABLE,
                disagreement=disagreement,
                synthesized_score=float(ela_result.raw_score) if ela_result else 0.0,
                is_calibrated=False,
                confidence_score=None,
                human_review_guidance="Inference timeout: Neural model could not complete in allotted window.",
                warnings=warnings + ["Neural execution watchdog timeout triggered."],
            )

        if artifact_invalid:
            disagreement = NeuralForensicDisagreement(
                disagreement_type="INVALID_ARTIFACT",
                neural_verdict="CHECKSUM_FAILED",
                forensic_verdict="EVALUATED_BASELINE",
                confidence_delta=0.0,
                synthesis_rationale="Neural weights failed cryptographic integrity or version verification.",
                human_review_required=True,
                recommended_action="Inspect model weights and metadata checksums on server.",
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                uncertainty_state=UncertaintyState.MODEL_UNAVAILABLE,
                disagreement=disagreement,
                synthesized_score=float(ela_result.raw_score) if ela_result else 0.0,
                is_calibrated=False,
                confidence_score=None,
                human_review_guidance="Model integrity failure: Neural weights rejected by security policy.",
                warnings=warnings + ["Model weights failed cryptographic verification."],
            )

        # Check for non-finite values in neural outputs
        if ai_gen_result and ai_gen_result.is_available:
            for logit in ai_gen_result.raw_logits:
                if not math.isfinite(logit):
                    disagreement = NeuralForensicDisagreement(
                        disagreement_type="NON_FINITE_PREDICTION",
                        neural_verdict="NON_FINITE_LOGIT",
                        forensic_verdict="EVALUATED",
                        confidence_delta=0.0,
                        synthesis_rationale="Neural linear head emitted non-finite logit (NaN or Inf).",
                        human_review_required=True,
                        recommended_action="Check input tensor normalization and numerical bounds.",
                    )
                    return ConsensusEvaluation(
                        assessment_label=AssessmentLabel.INCONCLUSIVE,
                        uncertainty_state=UncertaintyState.MODEL_UNCERTAIN,
                        disagreement=disagreement,
                        synthesized_score=0.0,
                        is_calibrated=False,
                        confidence_score=None,
                        human_review_guidance="Non-finite prediction encountered; output quarantined.",
                        warnings=warnings + ["Numerical instability: Non-finite prediction intercepted."],
                    )

        # -------------------------------------------------------------------
        # Rule 5: Modality Availability Extraction
        # -------------------------------------------------------------------
        has_neural_ai = bool(ai_gen_result and ai_gen_result.is_available)
        has_neural_facial = bool(facial_result and facial_result.is_available and facial_result.face_detected)
        has_ela = bool(ela_result is not None)
        has_fft = bool(fft_result is not None)

        neural_prob: Optional[float] = None
        if has_neural_ai:
            neural_prob = ai_gen_result.calibrated_probability if ai_gen_result.calibrated_probability is not None else ai_gen_result.uncalibrated_score

        ela_score = float(ela_result.raw_score) if ela_result else 0.0
        fft_score = float(fft_result.raw_score) if fft_result else 0.0

        # If neural models are completely absent, fall back gracefully to heuristics
        if not has_neural_ai and not has_neural_facial:
            heuristic_anom = (ela_score > self.ela_anomaly_threshold) or (fft_score > self.fft_anomaly_threshold)
            disagreement = NeuralForensicDisagreement(
                disagreement_type="HEURISTIC_ONLY_ACTIVE",
                neural_verdict="NOT_ENABLED",
                forensic_verdict="SUSPICIOUS" if heuristic_anom else "NO_OBVIOUS_ARTIFACTS",
                confidence_delta=0.0,
                synthesis_rationale="Neural detectors inactive or in development placeholder mode; baseline heuristics evaluated.",
                human_review_required=heuristic_anom,
                recommended_action="Review ELA and FFT spatial anomaly maps manually.",
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.SUSPICIOUS_MANIPULATION if heuristic_anom else AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
                uncertainty_state=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
                disagreement=disagreement,
                synthesized_score=float(max(ela_score, fft_score)),
                is_calibrated=False,
                confidence_score=None,
                human_review_guidance="Heuristic evaluation only: ELA and FFT indicate compression/spectral features.",
                warnings=warnings,
            )

        # -------------------------------------------------------------------
        # Rule 6: Neural and ELA/FFT Consensus & Disagreement Evaluation
        # -------------------------------------------------------------------
        neural_indicates_manip = (
            (neural_prob is not None and neural_prob >= self.neural_high_threshold)
            or (has_neural_facial and facial_result.raw_score >= self.neural_high_threshold)
        )
        neural_indicates_clean = (
            (neural_prob is not None and neural_prob <= self.neural_low_threshold)
            and (not has_neural_facial or facial_result.raw_score <= self.neural_low_threshold)
        )

        forensic_indicates_manip = (
            (has_ela and ela_result.has_compression_anomalies)
            or (has_fft and fft_result.has_frequency_artifacts)
        )
        forensic_indicates_clean = (
            (not has_ela or not ela_result.has_compression_anomalies)
            and (not has_fft or not fft_result.has_frequency_artifacts)
        )

        # Case 6.1: Full Consensus - Manipulation Detected
        if neural_indicates_manip and forensic_indicates_manip:
            disagreement = NeuralForensicDisagreement(
                disagreement_type="FULL_CONSENSUS_MANIPULATED",
                neural_verdict="INDICATORS_DETECTED",
                forensic_verdict="ANOMALOUS_COMPRESSION_AND_SPECTRUM",
                confidence_delta=round(abs((neural_prob or 0.85) - max(ela_score, fft_score)), 4),
                synthesis_rationale=(
                    "Both neural representations (diffusion textures/face seams) and heuristic metrics "
                    "(ELA compression variance / FFT spectral lattice) exhibit statistically significant anomalies. "
                    "Note: ELA and FFT both measure compression/frequency artifacts and are not statistically independent proofs."
                ),
                human_review_required=True,
                recommended_action="High-priority forensic casework review: Flagged visual anomalies present across multiple modalities.",
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.HIGHLY_LIKELY_MANIPULATED,
                uncertainty_state=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
                disagreement=disagreement,
                synthesized_score=neural_prob or (facial_result.raw_score if facial_result else 0.85),
                is_calibrated=bool(ai_gen_result and ai_gen_result.calibrated_probability is not None),
                confidence_score=ai_gen_result.calibrated_probability if ai_gen_result else None,
                human_review_guidance="Consensus across neural and forensic modalities indicates probable manipulation.",
                warnings=warnings,
            )

        # Case 6.2: Full Consensus - No Obvious Indicators
        if neural_indicates_clean and forensic_indicates_clean:
            disagreement = NeuralForensicDisagreement(
                disagreement_type="FULL_CONSENSUS_CLEAN",
                neural_verdict="NO_CLEAR_INDICATORS",
                forensic_verdict="UNIFORM_COMPRESSION_AND_SPECTRUM",
                confidence_delta=0.0,
                synthesis_rationale=(
                    "Both neural detectors and heuristic forensic checks show no obvious indicators of "
                    "synthetic generation or localized face editing. CRITICAL: This does NOT certify that the image "
                    "is authentic or untampered."
                ),
                human_review_required=False,
                recommended_action="Preserve original file with cryptographic hash.",
            )
            return ConsensusEvaluation(
                assessment_label=AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
                uncertainty_state=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
                disagreement=disagreement,
                synthesized_score=neural_prob or 0.10,
                is_calibrated=bool(ai_gen_result and ai_gen_result.calibrated_probability is not None),
                confidence_score=ai_gen_result.calibrated_probability if ai_gen_result else None,
                human_review_guidance="No clear indicators detected across modalities. Never treated as an authenticity certification.",
                warnings=warnings,
            )

        # Case 6.3: Neural / Forensic Disagreement (Conflict)
        # CRITICAL: Never average raw scores! Produce INCONCLUSIVE and human review guidance.
        delta_val = round(abs((neural_prob or 0.5) - max(ela_score, fft_score)), 4)
        disagreement = NeuralForensicDisagreement(
            disagreement_type="NEURAL_FORENSIC_DISAGREEMENT",
            neural_verdict="INDICATORS_DETECTED" if neural_indicates_manip else "NO_CLEAR_INDICATORS",
            forensic_verdict="ANOMALOUS" if forensic_indicates_manip else "CLEAN",
            confidence_delta=delta_val,
            synthesis_rationale=(
                "Inter-modal disagreement detected! Neural detector and heuristic forensic checks disagree. "
                "In accordance with SafeHER policy, scores from unrelated systems are NEVER averaged together. "
                "The assessment is marked INCONCLUSIVE, and manual forensic caseworker review is required."
            ),
            human_review_required=True,
            recommended_action=(
                "Caseworker review required: Inspect image PRNU sensor noise and camera EXIF to resolve "
                "conflict between neural representations and compression heuristics."
            ),
        )
        return ConsensusEvaluation(
            assessment_label=AssessmentLabel.INCONCLUSIVE,
            uncertainty_state=UncertaintyState.MODEL_UNCERTAIN,
            disagreement=disagreement,
            synthesized_score=neural_prob or 0.50,
            is_calibrated=False,  # Uncalibrated consensus
            confidence_score=None,
            human_review_guidance=(
                f"Conflicting signals: Neural model ({disagreement.neural_verdict}) disagrees with "
                f"forensics ({disagreement.forensic_verdict}). Mandatory human triage review required."
            ),
            warnings=warnings + ["Modality conflict: Neural and heuristic detectors produced opposing verdicts."],
        )
