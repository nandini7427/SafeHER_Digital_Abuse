"""Taxonomy, definitions, and output state contracts for SafeHER Model Calibration.

Strictly distinguishes between:
1. Raw Logits: Unbounded real numbers z in (-inf, +inf) directly output by linear layers.
2. Uncalibrated Model Scores: Softmax or sigmoid activations in [0, 1] without empirical calibration.
3. Calibrated Probabilities: Posterior probabilities P(Y=1 | X) empirically adjusted via held-out validation mapping.
4. Decision Thresholds: Operational cutoffs tau chosen for specific risk-benefit utility tradeoffs.
5. Model Uncertainty: Quantitative dispersion across entropy, variance, and multi-model disagreement.
6. Out-of-Distribution (OOD) Status: Determination of whether an input violates operational domain assumptions.

CRITICAL RULE:
Never use these terms interchangeably. Low model scores must NEVER be converted into an authenticity certification.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple


class OutputNature(str, Enum):
    """Formal taxonomy of mathematical output entities in machine learning inference."""

    RAW_LOGIT = "RAW_LOGIT"
    UNCALIBRATED_SCORE = "UNCALIBRATED_SCORE"
    CALIBRATED_PROBABILITY = "CALIBRATED_PROBABILITY"
    DECISION_THRESHOLD = "DECISION_THRESHOLD"
    MODEL_UNCERTAINTY = "MODEL_UNCERTAINTY"
    OUT_OF_DISTRIBUTION_STATUS = "OUT_OF_DISTRIBUTION_STATUS"


class SafeDecisionState(str, Enum):
    """Categorical triage states enforcing safe abstention and ethical reporting."""

    INDICATORS_DETECTED = "INDICATORS_DETECTED"
    NO_CLEAR_INDICATORS = "NO_CLEAR_INDICATORS"
    INCONCLUSIVE = "INCONCLUSIVE"
    OUTSIDE_SUPPORTED_DOMAIN = "OUTSIDE_SUPPORTED_DOMAIN"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class AbstentionReason(str, Enum):
    """Explicit rationale for model abstention or inconclusive triage."""

    NONE = "NONE"
    BORDERLINE_PROBABILITY = "BORDERLINE_PROBABILITY"
    ELEVATED_ENTROPY_UNCERTAINTY = "ELEVATED_ENTROPY_UNCERTAINTY"
    ELEVATED_PREDICTION_VARIANCE = "ELEVATED_PREDICTION_VARIANCE"
    INTER_MODEL_DISAGREEMENT = "INTER_MODEL_DISAGREEMENT"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    UNSUPPORTED_INPUT_CONDITION = "UNSUPPORTED_INPUT_CONDITION"
    NON_FINITE_NUMERICAL_VALUE = "NON_FINITE_NUMERICAL_VALUE"
    MISSING_CALIBRATION_ARTIFACT = "MISSING_CALIBRATION_ARTIFACT"
    MODEL_VERSION_MISMATCH = "MODEL_VERSION_MISMATCH"


class CalibratorMethod(str, Enum):
    """Supported scientific calibration mapping algorithms."""

    TEMPERATURE_SCALING = "TEMPERATURE_SCALING"
    PLATT_SCALING = "PLATT_SCALING"
    ISOTONIC_REGRESSION = "ISOTONIC_REGRESSION"
    UNCALIBRATED_RAW = "UNCALIBRATED_RAW"


PROBABILITY_INTERPRETATION_LIMITS: List[str] = [
    "A calibrated probability represents empirical event frequency under conditions matching the held-out validation distribution.",
    "A calibrated probability does NOT represent metaphysical certainty or absolute ontological truth.",
    "NO_CLEAR_INDICATORS indicates the absence of detectable artifacts within the evaluated model domain; it is NEVER a certification of authentic origin.",
    "Distribution shifts (such as aggressive social media recompression or unseen generator architectures) degrade calibration and increase epistemic uncertainty.",
    "Calibration maps model scores to empirical risk bands; high-stakes victim triage decisions require multidisciplinary human-in-the-loop verification.",
]

DEFAULT_FORENSIC_DISCLAIMER: str = (
    "SafeHER Forensic Triage Notice: This evaluation reflects calibrated empirical probability "
    "within the supported operational domain. Absence of detected manipulation indicators does NOT "
    "certify that an image or video is authentic."
)


@dataclass(frozen=True)
class UncertaintyDecomposition:
    """Decomposition of model uncertainty into entropy, patch variance, and multi-model disagreement."""

    entropy: float  # Shannon entropy in bits: -p log2 p - (1-p) log2(1-p)
    variance: float = 0.0  # Variance across crops, patches, or test-time augmentations
    disagreement: float = 0.0  # Divergence between Model A and Model B: |p_A - p_B|
    is_elevated: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_probability(
        cls,
        probability: float,
        variance: float = 0.0,
        disagreement: float = 0.0,
        max_entropy_threshold: float = 0.85,
        max_variance_threshold: float = 0.08,
        max_disagreement_threshold: float = 0.35,
    ) -> "UncertaintyDecomposition":
        """Compute Shannon entropy and assess whether composite uncertainty is elevated."""
        p = max(1e-7, min(1.0 - 1e-7, float(probability)))
        entropy = - (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

        is_elevated = (
            entropy >= max_entropy_threshold
            or variance >= max_variance_threshold
            or disagreement >= max_disagreement_threshold
        )

        return cls(
            entropy=round(float(entropy), 4),
            variance=round(float(variance), 4),
            disagreement=round(float(disagreement), 4),
            is_elevated=bool(is_elevated),
            details={
                "entropy_threshold": max_entropy_threshold,
                "variance_threshold": max_variance_threshold,
                "disagreement_threshold": max_disagreement_threshold,
            },
        )


@dataclass(frozen=True)
class OODStatus:
    """Out-of-distribution (OOD) and domain suitability assessment."""

    is_ood: bool
    domain_category: str
    anomaly_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibratedPrediction:
    """Rigorous calibrated prediction payload guaranteeing clear conceptual separation."""

    raw_logit: float
    uncalibrated_score: float
    calibrated_probability: float
    decision_state: SafeDecisionState
    abstention_reason: AbstentionReason
    uncertainty: UncertaintyDecomposition
    ood_status: OODStatus
    decision_threshold: float
    calibrator_method: CalibratorMethod
    model_version: str
    disclaimer: str = DEFAULT_FORENSIC_DISCLAIMER
    interpretation_limits: List[str] = field(
        default_factory=lambda: list(PROBABILITY_INTERPRETATION_LIMITS)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to serializable dictionary."""
        return {
            "raw_logit": self.raw_logit,
            "uncalibrated_score": self.uncalibrated_score,
            "calibrated_probability": self.calibrated_probability,
            "decision_state": self.decision_state.value,
            "abstention_reason": self.abstention_reason.value,
            "uncertainty": {
                "entropy": self.uncertainty.entropy,
                "variance": self.uncertainty.variance,
                "disagreement": self.uncertainty.disagreement,
                "is_elevated": self.uncertainty.is_elevated,
            },
            "ood_status": {
                "is_ood": self.ood_status.is_ood,
                "domain_category": self.ood_status.domain_category,
                "anomaly_score": self.ood_status.anomaly_score,
                "reasons": self.ood_status.reasons,
            },
            "decision_threshold": self.decision_threshold,
            "calibrator_method": self.calibrator_method.value,
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
            "interpretation_limits": self.interpretation_limits,
        }


def validate_conceptual_separation(output_nature: OutputNature, value: float) -> None:
    """Verify that values conform strictly to their declared conceptual nature.

    Raises:
        ValueError: If a value violates the mathematical or semantic bounds of its nature.
    """
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite value ({value}) encountered for output nature {output_nature.value}. "
            "Model outputs must be strictly finite numbers."
        )

    if output_nature == OutputNature.RAW_LOGIT:
        # Logits can be any real value in (-inf, +inf)
        return

    if output_nature in (
        OutputNature.UNCALIBRATED_SCORE,
        OutputNature.CALIBRATED_PROBABILITY,
        OutputNature.DECISION_THRESHOLD,
    ):
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{output_nature.value} must reside in [0.0, 1.0], but received {value}."
            )

    if output_nature == OutputNature.MODEL_UNCERTAINTY:
        if value < 0.0:
            raise ValueError(
                f"Model uncertainty must be non-negative, but received {value}."
            )
