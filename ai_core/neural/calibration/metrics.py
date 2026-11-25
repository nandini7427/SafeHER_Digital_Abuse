"""Calibration evaluation metrics, reliability diagrams, and distribution shift audits.

Implements:
- Brier score: Mean squared error between predicted probabilities and binary ground truth.
- Negative Log-Likelihood (NLL / Log Loss): Binary cross-entropy with numerical epsilon bounds.
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE): Binned confidence vs empirical accuracy.
- Reliability diagrams: Detailed 10-bin empirical calibration mapping.
- Stratified calibration by category: Calibration quality across generator families and manipulation types.
- Calibration under distribution shift: Evaluating calibration degradation under JPEG compression and perturbations.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ReliabilityBinData:
    """Single empirical bin in a reliability diagram."""

    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float


@dataclass(frozen=True)
class CalibrationMetricsReport:
    """Complete calibration audit report for a set of predictions."""

    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    negative_log_likelihood: float
    reliability_bins: List[ReliabilityBinData]
    sample_count: int
    is_well_calibrated: bool
    summary: str


def compute_brier_score(
    y_true: Union[List[bool], List[int], np.ndarray],
    y_probs: Union[List[float], np.ndarray],
) -> float:
    """Compute Brier Score: (1/N) * sum((p_i - y_i)^2)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_probs, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def compute_nll(
    y_true: Union[List[bool], List[int], np.ndarray],
    y_probs: Union[List[float], np.ndarray],
    eps: float = 1e-7,
) -> float:
    """Compute Negative Log-Likelihood (Binary Cross-Entropy)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_probs, dtype=np.float64), eps, 1.0 - eps)
    if len(y) == 0:
        return 0.0
    losses = - (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(losses))


def compute_reliability_diagram(
    y_true: Union[List[bool], List[int], np.ndarray],
    y_probs: Union[List[float], np.ndarray],
    num_bins: int = 10,
) -> Tuple[float, float, List[ReliabilityBinData]]:
    """Compute ECE, MCE, and reliability bin data across equal-width probability bins."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_probs, dtype=np.float64)
    n = len(y)
    if n == 0:
        return 0.0, 0.0, []

    bins: List[ReliabilityBinData] = []
    ece = 0.0
    mce = 0.0

    for b in range(num_bins):
        low = b / num_bins
        high = (b + 1) / num_bins

        if b == num_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)

        count = int(np.sum(mask))
        if count > 0:
            bin_conf = float(np.mean(p[mask]))
            bin_acc = float(np.mean(y[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)

            bins.append(
                ReliabilityBinData(
                    bin_index=b,
                    bin_lower=round(low, 2),
                    bin_upper=round(high, 2),
                    sample_count=count,
                    mean_confidence=round(bin_conf, 4),
                    empirical_accuracy=round(bin_acc, 4),
                    calibration_gap=round(gap, 4),
                )
            )

    return float(ece), float(mce), bins


def audit_calibration_metrics(
    y_true: Union[List[bool], List[int], np.ndarray],
    y_probs: Union[List[float], np.ndarray],
    num_bins: int = 10,
    ece_target: float = 0.08,
    brier_target: float = 0.12,
) -> CalibrationMetricsReport:
    """Audit comprehensive calibration metrics against production target thresholds."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_probs, dtype=np.float64)

    brier = compute_brier_score(y, p)
    nll = compute_nll(y, p)
    ece, mce, bins = compute_reliability_diagram(y, p, num_bins=num_bins)

    is_well_calibrated = bool((ece <= ece_target) and (brier <= brier_target))

    if is_well_calibrated:
        summary = (
            f"WELL-CALIBRATED: ECE={ece:.4f} (<= {ece_target}), Brier={brier:.4f} (<= {brier_target}), "
            f"MCE={mce:.4f}, NLL={nll:.4f}. Model confidence reliably reflects empirical event frequency."
        )
    else:
        summary = (
            f"MISCALIBRATED: ECE={ece:.4f} (> {ece_target}) or Brier={brier:.4f} (> {brier_target}). "
            "Model confidence diverges from empirical frequencies; abstention or recalibration required."
        )

    return CalibrationMetricsReport(
        brier_score=round(brier, 4),
        expected_calibration_error=round(ece, 4),
        maximum_calibration_error=round(mce, 4),
        negative_log_likelihood=round(nll, 4),
        reliability_bins=bins,
        sample_count=len(y),
        is_well_calibrated=is_well_calibrated,
        summary=summary,
    )


def evaluate_calibration_by_category(
    categories: List[str],
    y_true: Union[List[bool], List[int], np.ndarray],
    y_probs: Union[List[float], np.ndarray],
    num_bins: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Compute calibration metrics stratified by category (e.g. generator family or manipulation type)."""
    cats = list(categories)
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_probs, dtype=np.float64)

    if len(cats) != len(y):
        raise ValueError(f"Category count ({len(cats)}) != sample count ({len(y)})")

    unique_categories = sorted(set(cats))
    results: Dict[str, Dict[str, Any]] = {}

    for cat in unique_categories:
        indices = [i for i, c in enumerate(cats) if c == cat]
        cat_y = y[indices]
        cat_p = p[indices]

        if len(cat_y) == 0:
            continue

        brier = compute_brier_score(cat_y, cat_p)
        ece, mce, _ = compute_reliability_diagram(cat_y, cat_p, num_bins=num_bins)

        results[cat] = {
            "sample_count": len(cat_y),
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
            "maximum_calibration_error": round(mce, 4),
            "mean_confidence": round(float(np.mean(cat_p)), 4),
            "empirical_accuracy": round(float(np.mean(cat_y)), 4),
        }

    return results


def evaluate_calibration_under_distribution_shift(
    y_true: Union[List[bool], List[int], np.ndarray],
    clean_probs: Union[List[float], np.ndarray],
    shifted_probs_by_condition: Dict[str, Union[List[float], np.ndarray]],
    num_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate calibration degradation under distribution shifts (e.g. JPEG compression, noise)."""
    y = np.asarray(y_true, dtype=np.float64)
    clean_p = np.asarray(clean_probs, dtype=np.float64)

    clean_brier = compute_brier_score(y, clean_p)
    clean_ece, clean_mce, _ = compute_reliability_diagram(y, clean_p, num_bins=num_bins)

    shift_results: Dict[str, Dict[str, Any]] = {
        "baseline_clean": {
            "brier_score": round(clean_brier, 4),
            "expected_calibration_error": round(clean_ece, 4),
            "maximum_calibration_error": round(clean_mce, 4),
            "delta_ece": 0.0,
            "delta_brier": 0.0,
        }
    }

    for condition_name, shifted_p in shifted_probs_by_condition.items():
        sp = np.asarray(shifted_p, dtype=np.float64)
        if len(sp) != len(y):
            continue

        s_brier = compute_brier_score(y, sp)
        s_ece, s_mce, _ = compute_reliability_diagram(y, sp, num_bins=num_bins)

        shift_results[condition_name] = {
            "brier_score": round(s_brier, 4),
            "expected_calibration_error": round(s_ece, 4),
            "maximum_calibration_error": round(s_mce, 4),
            "delta_ece": round(s_ece - clean_ece, 4),
            "delta_brier": round(s_brier - clean_brier, 4),
        }

    return shift_results
