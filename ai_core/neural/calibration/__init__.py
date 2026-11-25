"""SafeHER Neural Model Calibration and Safe Abstention Framework."""

from ai_core.neural.calibration.abstention import (
    AbstentionPolicyConfig,
    CalibratedAbstentionEngine,
)
from ai_core.neural.calibration.auditor import (
    CalibrationAuditor,
    CalibratorCandidateResult,
)
from ai_core.neural.calibration.calibrators import (
    BaseNeuralCalibrator,
    InvalidCalibrationStateError,
    IsotonicRegressionCalibrator,
    ModelVersionMismatchError,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
    UnfittedCalibratorError,
)
from ai_core.neural.calibration.metrics import (
    CalibrationMetricsReport,
    ReliabilityBinData,
    audit_calibration_metrics,
    compute_brier_score,
    compute_nll,
    compute_reliability_diagram,
    evaluate_calibration_by_category,
    evaluate_calibration_under_distribution_shift,
)
from ai_core.neural.calibration.taxonomy import (
    DEFAULT_FORENSIC_DISCLAIMER,
    PROBABILITY_INTERPRETATION_LIMITS,
    AbstentionReason,
    CalibratedPrediction,
    CalibratorMethod,
    OODStatus,
    OutputNature,
    SafeDecisionState,
    UncertaintyDecomposition,
    validate_conceptual_separation,
)

__all__ = [
    # Taxonomy
    "OutputNature",
    "SafeDecisionState",
    "AbstentionReason",
    "CalibratorMethod",
    "UncertaintyDecomposition",
    "OODStatus",
    "CalibratedPrediction",
    "validate_conceptual_separation",
    "DEFAULT_FORENSIC_DISCLAIMER",
    "PROBABILITY_INTERPRETATION_LIMITS",
    # Calibrators
    "BaseNeuralCalibrator",
    "TemperatureScalingCalibrator",
    "PlattScalingCalibrator",
    "IsotonicRegressionCalibrator",
    "ModelVersionMismatchError",
    "UnfittedCalibratorError",
    "InvalidCalibrationStateError",
    # Metrics
    "ReliabilityBinData",
    "CalibrationMetricsReport",
    "compute_brier_score",
    "compute_nll",
    "compute_reliability_diagram",
    "audit_calibration_metrics",
    "evaluate_calibration_by_category",
    "evaluate_calibration_under_distribution_shift",
    # Abstention
    "AbstentionPolicyConfig",
    "CalibratedAbstentionEngine",
    # Auditor
    "CalibrationAuditor",
    "CalibratorCandidateResult",
]
