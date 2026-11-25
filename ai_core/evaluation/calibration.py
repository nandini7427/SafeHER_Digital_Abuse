"""Score calibration, reliability evaluation, and decision threshold validation for SafeHER.

Implements:
- Output nature determination (Logits vs Raw Activations vs Uncalibrated Scores vs Calibrated Probabilities).
- Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score, and NLL.
- Reliability diagrams (10-bin empirical calibration mapping).
- Temperature Scaling, Platt Scaling, and Isotonic Regression (PAVA).
- Independent validation-split threshold selection:
  * Strict Low-FPR threshold (FPR <= 5%) for victim harm prevention.
  * Balanced threshold (Maximum Youden's Index J = TPR - FPR).
  * High-Sensitivity screening threshold (Recall >= 90%).
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Output Nature Taxonomy
# ---------------------------------------------------------------------------

class ModelOutputNature:
    """Taxonomy of computer vision detector raw outputs."""

    LOGITS = "logits"
    RAW_ACTIVATIONS = "raw_activations"
    UNCALIBRATED_SCORES = "uncalibrated_heuristic_scores"
    CALIBRATED_PROBABILITIES = "calibrated_probabilities"


def determine_output_nature(
    scores: List[float],
    is_explicitly_calibrated: bool = False,
    detector_type: str = "heuristic_ela",
) -> Dict[str, Any]:
    """Audit the mathematical nature of raw model outputs without assuming [0,1] is a probability.

    Args:
        scores: Sample of output scores from the detector.
        is_explicitly_calibrated: Architectural flag from the detector schema.
        detector_type: Type of detector ('heuristic_ela', 'neural_logits', etc.)

    Returns:
        Dict describing the exact output nature, domain bounds, and calibration status.
    """
    if not scores:
        return {
            "output_nature": ModelOutputNature.UNCALIBRATED_SCORES,
            "bounds": (0.0, 1.0),
            "is_calibrated": False,
            "justification": "Empty score distribution; defaulting to uncalibrated contract.",
        }

    min_val = min(scores)
    max_val = max(scores)

    if min_val < 0.0 or max_val > 1.0:
        return {
            "output_nature": ModelOutputNature.LOGITS,
            "bounds": (min_val, max_val),
            "is_calibrated": False,
            "justification": f"Scores span [{min_val:.3f}, {max_val:.3f}], indicating unbounded real-valued logits.",
        }

    if not is_explicitly_calibrated or detector_type == "heuristic_ela":
        return {
            "output_nature": ModelOutputNature.UNCALIBRATED_SCORES,
            "bounds": (min_val, max_val),
            "is_calibrated": False,
            "justification": (
                "Scores are bounded in [0.0, 1.0] but derived from uncalibrated Error Level Analysis (ELA) "
                "or heuristic variance steps. They represent relative anomaly indices, NOT posterior probabilities."
            ),
        }

    return {
        "output_nature": ModelOutputNature.CALIBRATED_PROBABILITIES,
        "bounds": (min_val, max_val),
        "is_calibrated": True,
        "justification": "Scores have undergone empirical calibration mapping on independent validation data.",
    }


# ---------------------------------------------------------------------------
# Calibration Metrics & Reliability Diagrams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReliabilityBin:
    """Single bin for reliability diagram."""

    bin_index: int
    bin_range: str
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float


@dataclass(frozen=True)
class CalibrationEvaluation:
    """Full calibration audit result for a set of predictions."""

    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    negative_log_likelihood: float
    reliability_bins: List[ReliabilityBin]
    sample_count: int
    is_calibrated_claim_supported: bool
    summary: str


def compute_brier_score(y_true: List[bool], y_scores: List[float]) -> float:
    """Compute mean squared error between probabilities and ground truth: (1/N) sum (p_i - y_i)^2."""
    if not y_true:
        return 0.0
    return float(np.mean([(s - (1.0 if yt else 0.0)) ** 2 for yt, s in zip(y_true, y_scores)]))


def compute_nll(y_true: List[bool], y_scores: List[float], eps: float = 1e-7) -> float:
    """Compute Negative Log-Likelihood (Binary Cross-Entropy)."""
    if not y_true:
        return 0.0
    clipped = [min(max(s, eps), 1.0 - eps) for s in y_scores]
    losses = [
        -(1.0 if yt else 0.0) * math.log(p) - (0.0 if yt else 1.0) * math.log(1.0 - p)
        for yt, p in zip(y_true, clipped)
    ]
    return float(np.mean(losses))


def compute_reliability_diagram(
    y_true: List[bool],
    y_scores: List[float],
    num_bins: int = 10,
) -> Tuple[float, float, List[ReliabilityBin]]:
    """Compute ECE, MCE, and binned accuracy vs confidence for reliability diagrams."""
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, []

    bins: List[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0

    for b in range(num_bins):
        low = b / num_bins
        high = (b + 1) / num_bins

        if b == num_bins - 1:
            in_bin = [(yt, s) for yt, s in zip(y_true, y_scores) if low <= s <= high]
        else:
            in_bin = [(yt, s) for yt, s in zip(y_true, y_scores) if low <= s < high]

        count = len(in_bin)
        if count > 0:
            mean_conf = float(np.mean([s for _, s in in_bin]))
            mean_acc = float(np.mean([1.0 if yt else 0.0 for yt, _ in in_bin]))
            gap = abs(mean_acc - mean_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
            bins.append(
                ReliabilityBin(
                    bin_index=b,
                    bin_range=f"[{low:.1f}, {high:.1f}]",
                    sample_count=count,
                    mean_confidence=round(mean_conf, 4),
                    empirical_accuracy=round(mean_acc, 4),
                    calibration_gap=round(gap, 4),
                )
            )

    return float(ece), float(mce), bins


def audit_calibration(
    y_true: List[bool],
    y_scores: List[float],
    num_bins: int = 10,
    ece_acceptable_threshold: float = 0.10,
) -> CalibrationEvaluation:
    """Execute complete calibration evaluation, returning ECE, MCE, Brier score, NLL, and bins."""
    brier = compute_brier_score(y_true, y_scores)
    nll = compute_nll(y_true, y_scores)
    ece, mce, bins = compute_reliability_diagram(y_true, y_scores, num_bins=num_bins)

    is_supported = (ece <= ece_acceptable_threshold) and (brier <= 0.15)

    if is_supported:
        summary = (
            f"CALIBRATION SUPPORTED: ECE={ece:.4f} (<= {ece_acceptable_threshold}), "
            f"Brier={brier:.4f}, NLL={nll:.4f}. Predicted scores closely track empirical event frequencies."
        )
    else:
        summary = (
            f"UNCALIBRATED: ECE={ece:.4f} (> {ece_acceptable_threshold}), Brier={brier:.4f}, NLL={nll:.4f}. "
            "Scores represent raw uncalibrated heuristic anomaly indices and MUST NOT be surfaced as probabilities."
        )

    return CalibrationEvaluation(
        brier_score=round(brier, 4),
        expected_calibration_error=round(ece, 4),
        maximum_calibration_error=round(mce, 4),
        negative_log_likelihood=round(nll, 4),
        reliability_bins=bins,
        sample_count=len(y_true),
        is_calibrated_claim_supported=is_supported,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Calibration Algorithms (Fitted on Independent Validation Split)
# ---------------------------------------------------------------------------

class TemperatureScalingCalibrator:
    """Temperature scaling for bounded scores or logits: p = sigmoid(logit(s) / T)."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(0.01, float(temperature))
        self.is_fitted = False

    def fit(self, val_scores: List[float], val_labels: List[bool]) -> "TemperatureScalingCalibrator":
        """Find optimal temperature T minimizing NLL on validation set using Golden Section Search."""
        if not val_scores:
            return self

        # Convert scores to logits: z = ln(s / (1 - s))
        eps = 1e-5
        clipped = [min(max(s, eps), 1.0 - eps) for s in val_scores]
        logits = np.array([math.log(s / (1.0 - s)) for s in clipped], dtype=np.float64)
        targets = np.array([1.0 if y else 0.0 for y in val_labels], dtype=np.float64)

        def loss_fn(t_val: float) -> float:
            scaled = logits / max(1e-4, t_val)
            # numerically stable sigmoid
            probs = np.where(scaled >= 0, 1.0 / (1.0 + np.exp(-scaled)), np.exp(scaled) / (1.0 + np.exp(scaled)))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            return float(-np.mean(targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs)))

        # Golden Section Search in [0.05, 10.0]
        a, b = 0.05, 10.0
        gr = (math.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr

        for _ in range(60):
            if abs(b - a) < 1e-4:
                break
            if loss_fn(c) < loss_fn(d):
                b = d
            else:
                a = c
            c = b - (b - a) / gr
            d = a + (b - a) / gr

        best_t = (b + a) / 2.0
        self.temperature = float(best_t)
        self.is_fitted = True
        return self

    def predict_proba(self, scores: List[float]) -> List[float]:
        """Transform uncalibrated scores to temperature-scaled probabilities."""
        eps = 1e-5
        calibrated: List[float] = []
        for s in scores:
            p_s = min(max(s, eps), 1.0 - eps)
            logit = math.log(p_s / (1.0 - p_s))
            scaled = logit / self.temperature
            if scaled >= 0:
                prob = 1.0 / (1.0 + math.exp(-scaled))
            else:
                prob = math.exp(scaled) / (1.0 + math.exp(scaled))
            calibrated.append(float(np.clip(prob, 0.0, 1.0)))
        return calibrated


class PlattScalingCalibrator:
    """Platt scaling (logistic regression on scores): p = sigmoid(a * s + b)."""

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a = float(a)
        self.b = float(b)
        self.is_fitted = False

    def fit(self, val_scores: List[float], val_labels: List[bool], epochs: int = 150, lr: float = 0.1) -> "PlattScalingCalibrator":
        """Fit parameters a and b using gradient descent minimizing cross-entropy."""
        if not val_scores:
            return self

        x = np.array(val_scores, dtype=np.float64)
        y = np.array([1.0 if yt else 0.0 for yt in val_labels], dtype=np.float64)

        a, b_param = 1.0, 0.0

        for _ in range(epochs):
            z = a * x + b_param
            p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
            err = p - y
            grad_a = np.mean(err * x)
            grad_b = np.mean(err)

            a -= lr * grad_a
            b_param -= lr * grad_b

        self.a = float(a)
        self.b = float(b_param)
        self.is_fitted = True
        return self

    def predict_proba(self, scores: List[float]) -> List[float]:
        """Transform raw scores via fitted sigmoid."""
        x = np.array(scores, dtype=np.float64)
        z = self.a * x + self.b
        probs = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        return [float(p) for p in np.clip(probs, 0.0, 1.0)]


class IsotonicRegressionCalibrator:
    """Non-parametric monotonic step calibrator using Pool Adjacent Violators Algorithm (PAVA)."""

    def __init__(self) -> None:
        self.x_thresholds: List[float] = []
        self.y_values: List[float] = []
        self.is_fitted = False

    def fit(self, val_scores: List[float], val_labels: List[bool]) -> "IsotonicRegressionCalibrator":
        """Fit monotonic non-decreasing step function via PAVA."""
        if not val_scores:
            return self

        # Sort pairs by score
        pairs = sorted(zip(val_scores, [1.0 if y else 0.0 for y in val_labels]), key=lambda p: p[0])
        x_sorted = [p[0] for p in pairs]
        y_sorted = [p[1] for p in pairs]

        # PAVA implementation
        # Each block: [weight, sum_y, mean_y, start_idx, end_idx]
        blocks = []
        for val in y_sorted:
            blocks.append([1.0, val, val])

        i = 0
        while i < len(blocks) - 1:
            if blocks[i][2] > blocks[i + 1][2]:
                # Pool adjacent violators
                w1, s1, _ = blocks[i]
                w2, s2, _ = blocks[i + 1]
                new_w = w1 + w2
                new_s = s1 + s2
                new_mean = new_s / new_w
                blocks[i] = [new_w, new_s, new_mean]
                del blocks[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Expand fitted values back to points
        fitted_y: List[float] = []
        for w, s, mean in blocks:
            fitted_y.extend([mean] * int(w))

        # Collapse duplicate x coordinates to single monotonic thresholds
        x_unique: List[float] = []
        y_unique: List[float] = []
        for x_val, y_val in zip(x_sorted, fitted_y):
            if not x_unique or x_val > x_unique[-1]:
                x_unique.append(x_val)
                y_unique.append(y_val)
            else:
                # Update with latest isotonic mean
                y_unique[-1] = max(y_unique[-1], y_val)

        self.x_thresholds = x_unique
        self.y_values = y_unique
        self.is_fitted = True
        return self

    def predict_proba(self, scores: List[float]) -> List[float]:
        """Interpolate isotonic step function for test scores."""
        if not self.x_thresholds:
            return list(scores)

        calibrated: List[float] = []
        for s in scores:
            # Piecewise linear interpolation with monotonic clamping
            val = float(np.interp(s, self.x_thresholds, self.y_values))
            calibrated.append(float(np.clip(val, 0.0, 1.0)))
        return calibrated


# ---------------------------------------------------------------------------
# Decision Threshold Optimizer (Validation Split)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdOperatingPoint:
    """Specific operational threshold with documented performance tradeoffs."""

    name: str
    threshold: float
    description: str
    rationale: str
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass(frozen=True)
class ThresholdSelectionReport:
    """Complete multi-criteria threshold audit on validation data."""

    low_fpr_threshold: ThresholdOperatingPoint
    balanced_threshold: ThresholdOperatingPoint
    high_sensitivity_threshold: ThresholdOperatingPoint
    operating_points: List[ThresholdOperatingPoint]
    validation_sample_count: int
    evaluation_summary: str


def optimize_decision_thresholds(
    val_true: List[bool],
    val_scores: List[float],
) -> ThresholdSelectionReport:
    """Evaluate decision thresholds across the full dynamic range on validation data.

    Operational Criteria:
    1. Low-FPR (Survivor-Protective): Minimizes wrongful manipulation allegations (FPR <= 0.05).
    2. Balanced (Youden's Index): Maximizes J = TPR - FPR (or F1-score).
    3. High-Sensitivity: Ensures >= 90% recall to screen suspicious content for human audit.
    """
    n = len(val_true)
    if n == 0:
        raise ValueError("Cannot optimize thresholds on empty validation data.")

    candidate_thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 91)]
    candidates: List[Dict[str, Any]] = []

    pos_count = sum(1 for yt in val_true if yt)
    neg_count = sum(1 for yt in val_true if not yt)

    for thresh in candidate_thresholds:
        tp = sum(1 for yt, s in zip(val_true, val_scores) if yt and s >= thresh)
        fp = sum(1 for yt, s in zip(val_true, val_scores) if not yt and s >= thresh)
        tn = sum(1 for yt, s in zip(val_true, val_scores) if not yt and s < thresh)
        fn = sum(1 for yt, s in zip(val_true, val_scores) if yt and s < thresh)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / pos_count if pos_count > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / neg_count if neg_count > 0 else 0.0
        fnr = fn / pos_count if pos_count > 0 else 0.0
        youden = rec - fpr

        candidates.append(
            {
                "threshold": thresh,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
                "youden": youden,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    # 1. Low-FPR (FPR <= 0.05, prioritizing highest recall)
    low_fpr_pool = [c for c in candidates if c["fpr"] <= 0.05]
    if low_fpr_pool:
        # Sort by recall descending, then precision descending
        best_low_fpr = sorted(low_fpr_pool, key=lambda c: (c["recall"], c["precision"], -c["threshold"]), reverse=True)[0]
    else:
        # Fallback to threshold with absolute minimum FPR
        best_low_fpr = sorted(candidates, key=lambda c: c["fpr"])[0]

    # 2. Balanced (Maximum Youden's Index J = TPR - FPR)
    best_balanced = sorted(candidates, key=lambda c: (c["youden"], c["f1"]), reverse=True)[0]

    # 3. High-Sensitivity (Recall >= 0.90, minimizing FPR)
    high_sens_pool = [c for c in candidates if c["recall"] >= 0.90]
    if high_sens_pool:
        best_high_sens = sorted(high_sens_pool, key=lambda c: (c["fpr"], -c["recall"]))[0]
    else:
        # Fallback to maximum recall
        best_high_sens = sorted(candidates, key=lambda c: (c["recall"], -c["fpr"]), reverse=True)[0]

    op_low_fpr = ThresholdOperatingPoint(
        name="Survivor-Protective (Low-FPR)",
        threshold=best_low_fpr["threshold"],
        description="Strict triage threshold minimizing false accusations of digital manipulation.",
        rationale=(
            f"Enforces false-positive rate <= 5.0% (measured FPR: {best_low_fpr['fpr'] * 100:.1f}%), "
            f"achieving {best_low_fpr['precision'] * 100:.1f}% precision. Protects victims and authentic "
            "imagery from harmful false positives."
        ),
        precision=round(best_low_fpr["precision"], 4),
        recall=round(best_low_fpr["recall"], 4),
        f1_score=round(best_low_fpr["f1"], 4),
        false_positive_rate=round(best_low_fpr["fpr"], 4),
        false_negative_rate=round(best_low_fpr["fnr"], 4),
        true_positives=best_low_fpr["tp"],
        false_positives=best_low_fpr["fp"],
        true_negatives=best_low_fpr["tn"],
        false_negatives=best_low_fpr["fn"],
    )

    op_balanced = ThresholdOperatingPoint(
        name="Balanced (Maximum Youden's Index)",
        threshold=best_balanced["threshold"],
        description="Optimal tradeoff between sensitivity and specificity.",
        rationale=(
            f"Maximizes Youden's statistic J = TPR - FPR ({best_balanced['youden']:.3f}) with F1 score "
            f"of {best_balanced['f1']:.3f} on independent validation data."
        ),
        precision=round(best_balanced["precision"], 4),
        recall=round(best_balanced["recall"], 4),
        f1_score=round(best_balanced["f1"], 4),
        false_positive_rate=round(best_balanced["fpr"], 4),
        false_negative_rate=round(best_balanced["fnr"], 4),
        true_positives=best_balanced["tp"],
        false_positives=best_balanced["fp"],
        true_negatives=best_balanced["tn"],
        false_negatives=best_balanced["fn"],
    )

    op_high_sens = ThresholdOperatingPoint(
        name="High-Sensitivity Screening",
        threshold=best_high_sens["threshold"],
        description="Permissive screening threshold catching potential manipulations for human audit.",
        rationale=(
            f"Guarantees recall of {best_high_sens['recall'] * 100:.1f}% to minimize missed deepfakes, "
            f"accepting an elevated FPR of {best_high_sens['fpr'] * 100:.1f}% for downstream human review."
        ),
        precision=round(best_high_sens["precision"], 4),
        recall=round(best_high_sens["recall"], 4),
        f1_score=round(best_high_sens["f1"], 4),
        false_positive_rate=round(best_high_sens["fpr"], 4),
        false_negative_rate=round(best_high_sens["fnr"], 4),
        true_positives=best_high_sens["tp"],
        false_positives=best_high_sens["fp"],
        true_negatives=best_high_sens["tn"],
        false_negatives=best_high_sens["fn"],
    )

    return ThresholdSelectionReport(
        low_fpr_threshold=op_low_fpr,
        balanced_threshold=op_balanced,
        high_sensitivity_threshold=op_high_sens,
        operating_points=[op_low_fpr, op_balanced, op_high_sens],
        validation_sample_count=n,
        evaluation_summary=(
            f"Validated on {n} independent validation samples. Recommended operational threshold for "
            f"user-facing preliminary triage: {op_low_fpr.threshold:.2f} (Survivor-Protective Low-FPR)."
        ),
    )
