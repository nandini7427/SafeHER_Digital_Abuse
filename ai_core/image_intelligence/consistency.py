"""Model Consistency, Disagreement Analysis, and Disagreement Policy Engine for SafeHER.

Evaluates inter-model consistency across complementary forensic modalities (spatial ELA vs. frequency FFT),
computes empirical agreement statistics, and enforces survivor-protective disagreement policies
without arbitrary score averaging, forced binary determinations, or false claims of statistical independence.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai_core.image_intelligence.model_runner import ModelPrediction
from ai_core.image_intelligence.quality import QualityAssessmentResult
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    UncertaintyState,
)

logger = logging.getLogger("safeher.image_intelligence.consistency")


class DisagreementState(str, Enum):
    """Categorical states of multi-detector agreement and evidence concordance."""

    CONSISTENT_DETECTION_SIGNALS = "consistent_detection_signals"
    CONFLICTING_DETECTION_SIGNALS = "conflicting_detection_signals"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ONE_DETECTOR_UNAVAILABLE = "one_detector_unavailable"


@dataclass(frozen=True)
class DetectorSignal:
    """Standardized representation of a single forensic detector's signal."""

    detector_name: str
    detector_version: str
    forensic_domain: str  # e.g., "spatial_compression_ela", "frequency_spectral_fft", "neural_representation"
    raw_score: float
    decision_threshold: float
    is_anomaly_flagged: bool
    auxiliary_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConsistencyAssessment:
    """Forensic outcome of multi-detector consistency evaluation under SafeHER policy."""

    disagreement_state: DisagreementState
    assessment_label: AssessmentLabel
    uncertainty_state: UncertaintyState
    primary_signal: Optional[DetectorSignal]
    secondary_signal: Optional[DetectorSignal]
    score_divergence: float
    policy_explanation: str
    disagreement_reason: Optional[str]
    recommended_action: str
    warnings: List[str]
    correlated_risk_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert consistency assessment to a JSON-serializable dictionary."""
        return {
            "disagreement_state": self.disagreement_state.value,
            "assessment_label": self.assessment_label.value,
            "uncertainty_state": self.uncertainty_state.value,
            "primary_signal": {
                "name": self.primary_signal.detector_name,
                "domain": self.primary_signal.forensic_domain,
                "score": self.primary_signal.raw_score,
                "threshold": self.primary_signal.decision_threshold,
                "flagged": self.primary_signal.is_anomaly_flagged,
            }
            if self.primary_signal
            else None,
            "secondary_signal": {
                "name": self.secondary_signal.detector_name,
                "domain": self.secondary_signal.forensic_domain,
                "score": self.secondary_signal.raw_score,
                "threshold": self.secondary_signal.decision_threshold,
                "flagged": self.secondary_signal.is_anomaly_flagged,
            }
            if self.secondary_signal
            else None,
            "score_divergence": round(self.score_divergence, 4),
            "policy_explanation": self.policy_explanation,
            "disagreement_reason": self.disagreement_reason,
            "recommended_action": self.recommended_action,
            "warnings": self.warnings,
            "correlated_risk_factors": self.correlated_risk_factors,
        }


class ModelDisagreementPolicy:
    """Enforces SafeHER's multi-detector disagreement and evidence concordance policy.

    Guiding Principles:
    1. Zero Score Averaging: Arbitrary mathematical blending of uncalibrated heuristic scores
       from distinct physical domains (e.g. spatial ELA vs. Fourier power spectrum) is strictly prohibited.
    2. Zero Silent Alarming: When models produce conflicting signals, the system MUST NOT silently
       select the more alarming determination. Doing so artificially inflates false-positive rates
       and inflicts psychological and legal harm on survivors.
    3. Transparent Conflict Disclosure: Conflicting signals trigger explicit UNCERTAINTY disclosure
       with plain-language documentation of why the detectors diverged.
    4. Conservative Corroboration: Positive manipulation calls require corroborating forensic signals
       or qualified human forensic verification.
    """

    DEFAULT_PRIMARY_THRESHOLD: float = 0.45
    DEFAULT_SECONDARY_THRESHOLD: float = 0.45

    def __init__(
        self,
        primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
        secondary_threshold: float = DEFAULT_SECONDARY_THRESHOLD,
    ) -> None:
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold

    def evaluate(
        self,
        quality_result: QualityAssessmentResult,
        primary_prediction: Optional[ModelPrediction] = None,
        secondary_prediction: Optional[ModelPrediction] = None,
    ) -> ModelConsistencyAssessment:
        """Evaluate multi-detector signals according to SafeHER disagreement policy."""
        # --- STATE 1: Insufficient Evidence (Domain Mismatch or Severe Quality Defect) ---
        if not quality_result.analysis_eligibility:
            return ModelConsistencyAssessment(
                disagreement_state=DisagreementState.INSUFFICIENT_EVIDENCE,
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                uncertainty_state=quality_result.uncertainty_state,
                primary_signal=self._make_signal(
                    primary_prediction,
                    self.primary_threshold,
                    "spatial_compression_ela",
                )
                if primary_prediction
                else None,
                secondary_signal=self._make_signal(
                    secondary_prediction,
                    self.secondary_threshold,
                    "frequency_spectral_fft",
                )
                if secondary_prediction
                else None,
                score_divergence=0.0,
                policy_explanation=(
                    f"Evidence triage halted: {quality_result.reason_for_insufficient_evidence}. "
                    "In accordance with SafeHER policy, multi-detector consistency is not evaluated "
                    "on out-of-distribution media or poor-quality evidence."
                ),
                disagreement_reason="Analysis halted due to insufficient evidentiary quality or out-of-distribution media.",
                recommended_action=quality_result.recommended_next_step,
                warnings=quality_result.warnings,
                correlated_risk_factors=["out_of_distribution_or_poor_quality"],
            )

        # --- STATE 2: One Detector Unavailable ---
        if primary_prediction is None or secondary_prediction is None:
            available_pred = primary_prediction or secondary_prediction
            if available_pred is None:
                return ModelConsistencyAssessment(
                    disagreement_state=DisagreementState.ONE_DETECTOR_UNAVAILABLE,
                    assessment_label=AssessmentLabel.INCONCLUSIVE,
                    uncertainty_state=UncertaintyState.MODEL_UNAVAILABLE,
                    primary_signal=None,
                    secondary_signal=None,
                    score_divergence=0.0,
                    policy_explanation="No operational detector was available to evaluate the evidence.",
                    disagreement_reason="Both primary and secondary model runners were unavailable.",
                    recommended_action="Check system model dependencies or retain evidence for manual review.",
                    warnings=["All automated forensic detectors were unavailable."],
                    correlated_risk_factors=["total_detector_outage"],
                )

            # Single detector fallback with explicit elevated uncertainty
            is_primary = primary_prediction is not None
            domain = "spatial_compression_ela" if is_primary else "frequency_spectral_fft"
            threshold = self.primary_threshold if is_primary else self.secondary_threshold
            sig = self._make_signal(available_pred, threshold, domain)

            label = (
                AssessmentLabel.SUSPICIOUS_MANIPULATION
                if sig.is_anomaly_flagged
                else AssessmentLabel.NO_OBVIOUS_ARTIFACTS
            )
            # When operating with only one detector, certainty cannot be high
            uncertainty = (
                UncertaintyState.MODEL_UNCERTAIN
                if sig.is_anomaly_flagged
                else UncertaintyState.ANALYSIS_USABLE_EVIDENCE
            )

            warnings = [
                f"Multi-detector cross-verification unavailable: only {sig.detector_name} executed.",
                "Evidence assessed under single-detector operational fallback with elevated uncertainty.",
            ]

            return ModelConsistencyAssessment(
                disagreement_state=DisagreementState.ONE_DETECTOR_UNAVAILABLE,
                assessment_label=label,
                uncertainty_state=uncertainty,
                primary_signal=sig if is_primary else None,
                secondary_signal=sig if not is_primary else None,
                score_divergence=0.0,
                policy_explanation=(
                    f"Operational Fallback: Evaluated solely with {sig.detector_name}. "
                    "Independent multi-domain cross-verification could not be conducted."
                ),
                disagreement_reason="Secondary verification detector was unavailable or disabled.",
                recommended_action="Independent forensic review is advised because multi-detector cross-corroboration was incomplete.",
                warnings=warnings,
                correlated_risk_factors=["single_detector_exposure"],
            )

        # --- Extract Signals from Both Operational Detectors ---
        sig1 = self._make_signal(primary_prediction, self.primary_threshold, "spatial_compression_ela")
        sig2 = self._make_signal(secondary_prediction, self.secondary_threshold, "frequency_spectral_fft")
        score_diff = abs(sig1.raw_score - sig2.raw_score)

        # Check for correlated failure risk factors
        risk_factors: List[str] = []
        if quality_result.indicators.has_heavy_compression:
            risk_factors.append("heavy_jpeg_recompression_degradation")
        if quality_result.indicators.is_blurry:
            risk_factors.append("high_frequency_attenuation_from_blur")
        if quality_result.indicators.exposure_level in ("overexposed", "underexposed"):
            risk_factors.append("luminance_clipping_spectral_distortion")

        # --- STATE 3: Consistent Detection Signals ---
        if sig1.is_anomaly_flagged == sig2.is_anomaly_flagged:
            if sig1.is_anomaly_flagged:
                # Both models detect anomalies in their respective physical domains
                label = AssessmentLabel.SUSPICIOUS_MANIPULATION
                uncertainty = UncertaintyState.ANALYSIS_USABLE_EVIDENCE
                explanation = (
                    f"Corroborated Multi-Domain Forensic Findings: Both the spatial compression detector "
                    f"({sig1.detector_name}, score {sig1.raw_score:.2f}) and the frequency spectral detector "
                    f"({sig2.detector_name}, score {sig2.raw_score:.2f}) corroborate the presence of "
                    "structural anomalies."
                )
                recommended_action = "Preserve original file and document dual-model corroboration for legal or casework review."
            else:
                # Neither model detects anomalies above the decision threshold
                label = AssessmentLabel.NO_OBVIOUS_ARTIFACTS
                uncertainty = UncertaintyState.ANALYSIS_USABLE_EVIDENCE
                explanation = (
                    f"Consistent Baseline Evidence: Neither the spatial compression detector "
                    f"({sig1.detector_name}, score {sig1.raw_score:.2f}) nor the frequency spectral detector "
                    f"({sig2.detector_name}, score {sig2.raw_score:.2f}) detected anomalous manipulation "
                    "patterns above operational thresholds."
                )
                recommended_action = "Preserve original file. Note that lack of obvious artifacts does not guarantee authenticity."

            return ModelConsistencyAssessment(
                disagreement_state=DisagreementState.CONSISTENT_DETECTION_SIGNALS,
                assessment_label=label,
                uncertainty_state=uncertainty,
                primary_signal=sig1,
                secondary_signal=sig2,
                score_divergence=score_diff,
                policy_explanation=explanation,
                disagreement_reason=None,
                recommended_action=recommended_action,
                warnings=[],
                correlated_risk_factors=risk_factors,
            )

        # --- STATE 4: Conflicting Detection Signals ---
        # One detector flagged anomaly; the other observed baseline
        # CRITICAL RULE: Never average scores; never silently select the alarming call.
        conflict_detail = (
            f"{sig1.detector_name} ({sig1.forensic_domain}) scored {sig1.raw_score:.2f} "
            f"(flagged={sig1.is_anomaly_flagged}), whereas {sig2.detector_name} ({sig2.forensic_domain}) "
            f"scored {sig2.raw_score:.2f} (flagged={sig2.is_anomaly_flagged})."
        )

        policy_explanation = (
            "Forensic Detectors Produced Conflicting Signals across physical domains: "
            f"{conflict_detail} In accordance with SafeHER's survivor-protective policy, "
            "scores are not averaged and conflicting evidence is triaged as uncertain rather "
            "than assuming the most alarming determination."
        )

        disagreement_reason = (
            f"Domain divergence between spatial compression analysis and Fourier spectral analysis. "
            f"Divergence magnitude: {score_diff:.3f}."
        )

        warnings = [
            "Forensic detectors produced conflicting signals across spatial and frequency domains.",
            "Score averaging was strictly prevented to avoid fabricating ungrounded composite scores.",
            "Triage designated as Inconclusive / Model Uncertain to prevent survivor harm.",
        ]

        return ModelConsistencyAssessment(
            disagreement_state=DisagreementState.CONFLICTING_DETECTION_SIGNALS,
            assessment_label=AssessmentLabel.INCONCLUSIVE,
            uncertainty_state=UncertaintyState.MODEL_UNCERTAIN,
            primary_signal=sig1,
            secondary_signal=sig2,
            score_divergence=score_diff,
            policy_explanation=policy_explanation,
            disagreement_reason=disagreement_reason,
            recommended_action="Submit for expert human forensic review. Request the original camera capture if available.",
            warnings=warnings,
            correlated_risk_factors=risk_factors,
        )

    @staticmethod
    def _make_signal(
        pred: ModelPrediction,
        threshold: float,
        domain: str,
    ) -> DetectorSignal:
        return DetectorSignal(
            detector_name=pred.runner_name,
            detector_version=pred.runner_version,
            forensic_domain=domain,
            raw_score=pred.raw_score,
            decision_threshold=threshold,
            is_anomaly_flagged=(pred.raw_score >= threshold),
            auxiliary_metrics=pred.auxiliary_metrics,
        )


# --- Statistical Evaluation Helpers ---

def compute_inter_model_agreement(
    primary_decisions: List[bool],
    secondary_decisions: List[bool],
) -> Dict[str, float]:
    """Compute empirical inter-rater agreement metrics: raw agreement, disagreement rate, and Cohen's Kappa."""
    if len(primary_decisions) != len(secondary_decisions):
        raise ValueError("Decisions lists must have identical lengths.")

    n = len(primary_decisions)
    if n == 0:
        return {
            "sample_count": 0,
            "raw_agreement_rate": 1.0,
            "disagreement_rate": 0.0,
            "cohen_kappa": 1.0,
        }

    # Confusion matrix between raters
    # a: both True, b: primary True & secondary False
    # c: primary False & secondary True, d: both False
    a = sum(1 for p, s in zip(primary_decisions, secondary_decisions) if p and s)
    b = sum(1 for p, s in zip(primary_decisions, secondary_decisions) if p and not s)
    c = sum(1 for p, s in zip(primary_decisions, secondary_decisions) if not p and s)
    d = sum(1 for p, s in zip(primary_decisions, secondary_decisions) if not p and not s)

    po = (a + d) / n  # Observed agreement
    disagreement_rate = (b + c) / n

    # Expected agreement by chance
    p_true_prim = (a + b) / n
    p_false_prim = (c + d) / n
    p_true_sec = (a + c) / n
    p_false_sec = (b + d) / n
    pe = (p_true_prim * p_true_sec) + (p_false_prim * p_false_sec)

    if pe >= 1.0 or abs(1.0 - pe) < 1e-9:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "sample_count": n,
        "both_flagged": a,
        "primary_only_flagged": b,
        "secondary_only_flagged": c,
        "neither_flagged": d,
        "raw_agreement_rate": round(po, 4),
        "disagreement_rate": round(disagreement_rate, 4),
        "cohen_kappa": round(kappa, 4),
    }


def compute_correlated_errors(
    ground_truth: List[int],
    primary_preds: List[bool],
    secondary_preds: List[bool],
) -> Dict[str, Any]:
    """Analyze correlated errors between two detectors."""
    n = len(ground_truth)
    if n == 0:
        return {"sample_count": 0, "correlated_error_rate": 0.0}

    primary_errors = [bool(p != (gt == 1)) for p, gt in zip(primary_preds, ground_truth)]
    secondary_errors = [bool(s != (gt == 1)) for s, gt in zip(secondary_preds, ground_truth)]

    # Joint errors: both models failed on the same sample
    both_failed = sum(1 for e1, e2 in zip(primary_errors, secondary_errors) if e1 and e2)
    prim_only_failed = sum(1 for e1, e2 in zip(primary_errors, secondary_errors) if e1 and not e2)
    sec_only_failed = sum(1 for e1, e2 in zip(primary_errors, secondary_errors) if not e1 and e2)
    neither_failed = sum(1 for e1, e2 in zip(primary_errors, secondary_errors) if not e1 and not e2)

    total_unique_failures = both_failed + prim_only_failed + sec_only_failed
    jaccard_error_overlap = both_failed / max(total_unique_failures, 1)

    # Correlated error correlation coefficient (phi coefficient between error arrays)
    e1_arr = np.array(primary_errors, dtype=np.float32)
    e2_arr = np.array(secondary_errors, dtype=np.float32)
    std1, std2 = float(np.std(e1_arr)), float(np.std(e2_arr))
    if std1 > 0 and std2 > 0:
        error_correlation = float(np.corrcoef(e1_arr, e2_arr)[0, 1])
    else:
        error_correlation = 0.0

    return {
        "sample_count": n,
        "both_failed_count": both_failed,
        "primary_only_failed_count": prim_only_failed,
        "secondary_only_failed_count": sec_only_failed,
        "neither_failed_count": neither_failed,
        "joint_failure_rate": round(both_failed / n, 4),
        "jaccard_error_overlap": round(jaccard_error_overlap, 4),
        "error_correlation_phi": round(error_correlation, 4),
    }
