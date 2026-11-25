"""Calibrated abstention engine and safe forensic decision policy.

Enforces five categorical decision states:
1. INDICATORS_DETECTED: High calibrated probability (p >= tau_high), low uncertainty, in-domain.
2. NO_CLEAR_INDICATORS: Low calibrated probability (p <= tau_low), low uncertainty, in-domain.
   (CRITICAL: Never converts low scores into an authenticity certification).
3. INCONCLUSIVE: Indeterminate probability band (tau_low < p < tau_high), elevated entropy,
   patch variance, or inter-model disagreement.
4. OUTSIDE_SUPPORTED_DOMAIN: Out-of-distribution (OOD) status, unsupported format, or corrupt data.
5. MODEL_UNAVAILABLE: Missing model weights, missing calibration artifact, or numerical failure.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ai_core.neural.calibration.calibrators import (
    BaseNeuralCalibrator,
    ModelVersionMismatchError,
    UnfittedCalibratorError,
)
from ai_core.neural.calibration.taxonomy import (
    AbstentionReason,
    CalibratedPrediction,
    CalibratorMethod,
    DEFAULT_FORENSIC_DISCLAIMER,
    OODStatus,
    PROBABILITY_INTERPRETATION_LIMITS,
    SafeDecisionState,
    UncertaintyDecomposition,
)


@dataclass(frozen=True)
class AbstentionPolicyConfig:
    """Operational threshold configuration for calibrated abstention."""

    tau_low: float = 0.20  # Below this -> NO_CLEAR_INDICATORS (if uncertainty is low)
    tau_high: float = 0.80  # Above this -> INDICATORS_DETECTED (if uncertainty is low)
    max_entropy_threshold: float = 0.85  # Maximum allowable Shannon entropy in bits
    max_variance_threshold: float = 0.08  # Maximum allowable patch/crop variance
    max_disagreement_threshold: float = 0.35  # Maximum allowable inter-model divergence |p_A - p_B|
    unsupported_conditions: Tuple[str, ...] = (
        "unsupported_format",
        "corrupt_file",
        "face_not_detected",
        "extreme_pose",
        "severe_occlusion",
    )


class CalibratedAbstentionEngine:
    """Evaluates neural model outputs under strict calibration and safety abstention policies."""

    def __init__(
        self,
        config: Optional[AbstentionPolicyConfig] = None,
        calibrator: Optional[BaseNeuralCalibrator] = None,
    ) -> None:
        self.config = config or AbstentionPolicyConfig()
        self.calibrator = calibrator

    def evaluate(
        self,
        raw_logit: Optional[float] = None,
        uncalibrated_score: Optional[float] = None,
        calibrated_probability: Optional[float] = None,
        model_version: str = "1.0.0",
        ood_status: Optional[OODStatus] = None,
        disagreement_with_peer: Optional[float] = None,
        patch_variance: float = 0.0,
        input_conditions: Optional[List[str]] = None,
        calibrator_override: Optional[BaseNeuralCalibrator] = None,
    ) -> CalibratedPrediction:
        """Evaluate raw outputs and return safe calibrated prediction.

        Ensures that:
        1. Non-finite values are caught and rejected.
        2. Missing calibration produces MODEL_UNAVAILABLE.
        3. OOD samples produce OUTSIDE_SUPPORTED_DOMAIN.
        4. Model disagreements or elevated uncertainty produce INCONCLUSIVE.
        5. Low scores produce NO_CLEAR_INDICATORS without certifying authenticity.
        """
        active_calibrator = calibrator_override or self.calibrator
        active_ood = ood_status or OODStatus(is_ood=False, domain_category="photographic")
        active_conditions = [c.lower() for c in (input_conditions or [])]

        # 1. Check for non-finite values in inputs
        for val, name in [
            (raw_logit, "raw_logit"),
            (uncalibrated_score, "uncalibrated_score"),
            (calibrated_probability, "calibrated_probability"),
            (disagreement_with_peer, "disagreement_with_peer"),
            (patch_variance, "patch_variance"),
        ]:
            if val is not None and not math.isfinite(val):
                return CalibratedPrediction(
                    raw_logit=float(raw_logit or 0.0),
                    uncalibrated_score=float(uncalibrated_score or 0.5),
                    calibrated_probability=0.5,
                    decision_state=SafeDecisionState.INCONCLUSIVE,
                    abstention_reason=AbstentionReason.NON_FINITE_NUMERICAL_VALUE,
                    uncertainty=UncertaintyDecomposition(entropy=1.0, is_elevated=True),
                    ood_status=active_ood,
                    decision_threshold=self.config.tau_high,
                    calibrator_method=CalibratorMethod.UNCALIBRATED_RAW,
                    model_version=model_version,
                )

        # 2. Check for missing calibration artifact
        if calibrated_probability is None:
            if active_calibrator is None or not active_calibrator.is_fitted:
                return CalibratedPrediction(
                    raw_logit=float(raw_logit or 0.0),
                    uncalibrated_score=float(uncalibrated_score or 0.5),
                    calibrated_probability=float(uncalibrated_score or 0.5),
                    decision_state=SafeDecisionState.MODEL_UNAVAILABLE,
                    abstention_reason=AbstentionReason.MISSING_CALIBRATION_ARTIFACT,
                    uncertainty=UncertaintyDecomposition(entropy=1.0, is_elevated=True),
                    ood_status=active_ood,
                    decision_threshold=self.config.tau_high,
                    calibrator_method=CalibratorMethod.UNCALIBRATED_RAW,
                    model_version=model_version,
                )

            # Check model version linkage
            try:
                active_calibrator.verify_model_linkage(
                    target_model_name=active_calibrator.model_name,
                    target_model_version=model_version,
                )
            except ModelVersionMismatchError:
                return CalibratedPrediction(
                    raw_logit=float(raw_logit or 0.0),
                    uncalibrated_score=float(uncalibrated_score or 0.5),
                    calibrated_probability=float(uncalibrated_score or 0.5),
                    decision_state=SafeDecisionState.MODEL_UNAVAILABLE,
                    abstention_reason=AbstentionReason.MODEL_VERSION_MISMATCH,
                    uncertainty=UncertaintyDecomposition(entropy=1.0, is_elevated=True),
                    ood_status=active_ood,
                    decision_threshold=self.config.tau_high,
                    calibrator_method=active_calibrator.method(),
                    model_version=model_version,
                )

            # Produce calibrated probability from raw logit or score
            input_val = raw_logit if raw_logit is not None else (uncalibrated_score or 0.5)
            cal_probs = active_calibrator.calibrate([input_val])
            p_val = float(cal_probs[0])
            method_used = active_calibrator.method()
        else:
            p_val = float(calibrated_probability)
            method_used = (
                active_calibrator.method()
                if active_calibrator
                else CalibratorMethod.TEMPERATURE_SCALING
            )

        # 3. Check Out-of-Distribution status
        if active_ood.is_ood:
            return CalibratedPrediction(
                raw_logit=float(raw_logit or 0.0),
                uncalibrated_score=float(uncalibrated_score or p_val),
                calibrated_probability=p_val,
                decision_state=SafeDecisionState.OUTSIDE_SUPPORTED_DOMAIN,
                abstention_reason=AbstentionReason.OUT_OF_DISTRIBUTION,
                uncertainty=UncertaintyDecomposition.from_probability(p_val),
                ood_status=active_ood,
                decision_threshold=self.config.tau_high,
                calibrator_method=method_used,
                model_version=model_version,
            )

        # 4. Check Unsupported input conditions
        for cond in active_conditions:
            if cond in self.config.unsupported_conditions:
                return CalibratedPrediction(
                    raw_logit=float(raw_logit or 0.0),
                    uncalibrated_score=float(uncalibrated_score or p_val),
                    calibrated_probability=p_val,
                    decision_state=SafeDecisionState.OUTSIDE_SUPPORTED_DOMAIN,
                    abstention_reason=AbstentionReason.UNSUPPORTED_INPUT_CONDITION,
                    uncertainty=UncertaintyDecomposition.from_probability(p_val),
                    ood_status=active_ood,
                    decision_threshold=self.config.tau_high,
                    calibrator_method=method_used,
                    model_version=model_version,
                )

        # 5. Compute uncertainty decomposition
        disagree_val = float(disagreement_with_peer) if disagreement_with_peer is not None else 0.0
        uncertainty = UncertaintyDecomposition.from_probability(
            probability=p_val,
            variance=patch_variance,
            disagreement=disagree_val,
            max_entropy_threshold=self.config.max_entropy_threshold,
            max_variance_threshold=self.config.max_variance_threshold,
            max_disagreement_threshold=self.config.max_disagreement_threshold,
        )

        # 6. Check inter-model disagreement
        if disagree_val >= self.config.max_disagreement_threshold:
            return CalibratedPrediction(
                raw_logit=float(raw_logit or 0.0),
                uncalibrated_score=float(uncalibrated_score or p_val),
                calibrated_probability=p_val,
                decision_state=SafeDecisionState.INCONCLUSIVE,
                abstention_reason=AbstentionReason.INTER_MODEL_DISAGREEMENT,
                uncertainty=uncertainty,
                ood_status=active_ood,
                decision_threshold=self.config.tau_high,
                calibrator_method=method_used,
                model_version=model_version,
            )

        # 7. Check elevated entropy uncertainty
        if uncertainty.entropy >= self.config.max_entropy_threshold:
            return CalibratedPrediction(
                raw_logit=float(raw_logit or 0.0),
                uncalibrated_score=float(uncalibrated_score or p_val),
                calibrated_probability=p_val,
                decision_state=SafeDecisionState.INCONCLUSIVE,
                abstention_reason=AbstentionReason.ELEVATED_ENTROPY_UNCERTAINTY,
                uncertainty=uncertainty,
                ood_status=active_ood,
                decision_threshold=self.config.tau_high,
                calibrator_method=method_used,
                model_version=model_version,
            )

        # 8. Check elevated patch/crop variance
        if uncertainty.variance >= self.config.max_variance_threshold:
            return CalibratedPrediction(
                raw_logit=float(raw_logit or 0.0),
                uncalibrated_score=float(uncalibrated_score or p_val),
                calibrated_probability=p_val,
                decision_state=SafeDecisionState.INCONCLUSIVE,
                abstention_reason=AbstentionReason.ELEVATED_PREDICTION_VARIANCE,
                uncertainty=uncertainty,
                ood_status=active_ood,
                decision_threshold=self.config.tau_high,
                calibrator_method=method_used,
                model_version=model_version,
            )

        # 9. Check borderline probability interval [tau_low, tau_high]
        if self.config.tau_low < p_val < self.config.tau_high:
            return CalibratedPrediction(
                raw_logit=float(raw_logit or 0.0),
                uncalibrated_score=float(uncalibrated_score or p_val),
                calibrated_probability=p_val,
                decision_state=SafeDecisionState.INCONCLUSIVE,
                abstention_reason=AbstentionReason.BORDERLINE_PROBABILITY,
                uncertainty=uncertainty,
                ood_status=active_ood,
                decision_threshold=self.config.tau_high,
                calibrator_method=method_used,
                model_version=model_version,
            )

        # 10. Clear indicator detection (p >= tau_high)
        if p_val >= self.config.tau_high:
            return CalibratedPrediction(
                raw_logit=float(raw_logit or 0.0),
                uncalibrated_score=float(uncalibrated_score or p_val),
                calibrated_probability=p_val,
                decision_state=SafeDecisionState.INDICATORS_DETECTED,
                abstention_reason=AbstentionReason.NONE,
                uncertainty=uncertainty,
                ood_status=active_ood,
                decision_threshold=self.config.tau_high,
                calibrator_method=method_used,
                model_version=model_version,
            )

        # 11. No clear indicators (p <= tau_low) - NEVER an authenticity certificate
        return CalibratedPrediction(
            raw_logit=float(raw_logit or 0.0),
            uncalibrated_score=float(uncalibrated_score or p_val),
            calibrated_probability=p_val,
            decision_state=SafeDecisionState.NO_CLEAR_INDICATORS,
            abstention_reason=AbstentionReason.NONE,
            uncertainty=uncertainty,
            ood_status=active_ood,
            decision_threshold=self.config.tau_low,
            calibrator_method=method_used,
            model_version=model_version,
        )
