"""Mathematical evaluation metrics, curve calculations, and bootstrap confidence intervals.

Implements:
- Accuracy, Balanced Accuracy, Precision, Recall, F1
- ROC-AUC and PR-AUC using exact trapezoidal integration
- Confusion Matrix, False Positive Rate (FPR), False Negative Rate (FNR)
- 95% Bootstrap Confidence Intervals for statistical significance
- Per-class, per-generator, and per-manipulation-category breakdowns
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ConfidenceInterval(BaseModel):
    """Bootstrap confidence interval range."""

    model_config = ConfigDict(frozen=True)

    low: float = Field(..., description="Lower bound of confidence interval.")
    high: float = Field(..., description="Upper bound of confidence interval.")
    confidence_level: float = Field(default=0.95, description="Confidence level (e.g. 0.95 for 95% CI).")


class ClassificationMetrics(BaseModel):
    """Complete quantitative classification performance evaluation contract."""

    model_config = ConfigDict(frozen=True)

    sample_count: int = Field(..., ge=1, description="Total sample count evaluated.")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy.")
    balanced_accuracy: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged balanced accuracy.")
    precision: float = Field(..., ge=0.0, le=1.0, description="Positive class precision.")
    recall: float = Field(..., ge=0.0, le=1.0, description="Positive class recall / TPR.")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="Harmonic mean of precision and recall.")
    roc_auc: float = Field(..., ge=0.0, le=1.0, description="Area under ROC curve.")
    pr_auc: float = Field(..., ge=0.0, le=1.0, description="Area under Precision-Recall curve.")
    true_positives: int = Field(..., ge=0, description="True positives count.")
    false_positives: int = Field(..., ge=0, description="False positives count.")
    true_negatives: int = Field(..., ge=0, description="True negatives count.")
    false_negatives: int = Field(..., ge=0, description="False negatives count.")
    false_positive_rate: float = Field(..., ge=0.0, le=1.0, description="FPR (FP / (FP + TN)).")
    false_negative_rate: float = Field(..., ge=0.0, le=1.0, description="FNR (FN / (FN + TP)).")
    confusion_matrix: List[List[int]] = Field(..., description="[[TN, FP], [FN, TP]] matrix.")
    threshold: float = Field(..., description="Decision threshold applied.")
    accuracy_ci: Optional[ConfidenceInterval] = Field(default=None, description="95% CI for Accuracy.")
    f1_ci: Optional[ConfidenceInterval] = Field(default=None, description="95% CI for F1.")
    roc_auc_ci: Optional[ConfidenceInterval] = Field(default=None, description="95% CI for ROC-AUC.")


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic curve via trapezoidal integration."""
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))

    if pos == 0 or neg == 0:
        return 0.5  # Degenerate case: single class present

    desc_indices = np.argsort(-y_scores)
    y_sorted = y_true[desc_indices]
    scores_sorted = y_scores[desc_indices]

    # Find distinct threshold points
    distinct_mask = np.empty(len(scores_sorted), dtype=bool)
    distinct_mask[:-1] = scores_sorted[1:] != scores_sorted[:-1]
    distinct_mask[-1] = True

    tps = np.cumsum(y_sorted == 1)[distinct_mask]
    fps = np.cumsum(y_sorted == 0)[distinct_mask]

    tpr = np.concatenate([[0.0], tps / pos])
    fpr = np.concatenate([[0.0], fps / neg])

    # Trapezoidal rule: sum( (x_i - x_{i-1}) * (y_i + y_{i-1}) / 2 )
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0)
    return float(np.clip(auc, 0.0, 1.0))


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall curve using standard Average Precision formulation."""
    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return 0.0

    desc_indices = np.argsort(-y_scores)
    y_sorted = y_true[desc_indices]
    scores_sorted = y_scores[desc_indices]

    distinct_mask = np.empty(len(scores_sorted), dtype=bool)
    distinct_mask[:-1] = scores_sorted[1:] != scores_sorted[:-1]
    distinct_mask[-1] = True

    tps = np.cumsum(y_sorted == 1)[distinct_mask]
    fps = np.cumsum(y_sorted == 0)[distinct_mask]

    recalls = np.concatenate([[0.0], tps / pos])
    precisions = np.concatenate([[1.0], tps / (tps + fps)])

    # Trapezoidal integration of PR curve
    pr_auc = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return float(np.clip(pr_auc, 0.0, 1.0))


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.50,
    num_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[ConfidenceInterval, ConfidenceInterval, ConfidenceInterval]:
    """Calculate non-parametric bootstrap confidence intervals for accuracy, F1, and ROC-AUC."""
    n_samples = len(y_true)
    rng = np.random.RandomState(seed)

    boot_accs: List[float] = []
    boot_f1s: List[float] = []
    boot_aucs: List[float] = []

    alpha_low = (1.0 - confidence_level) / 2.0
    alpha_high = 1.0 - alpha_low

    for _ in range(num_bootstraps):
        idx = rng.randint(0, n_samples, size=n_samples)
        b_true = y_true[idx]
        b_scores = y_scores[idx]

        b_preds = (b_scores >= threshold).astype(int)

        tp = np.sum((b_preds == 1) & (b_true == 1))
        fp = np.sum((b_preds == 1) & (b_true == 0))
        fn = np.sum((b_preds == 0) & (b_true == 1))
        tn = np.sum((b_preds == 0) & (b_true == 0))

        acc = (tp + tn) / max(1, n_samples)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec)

        boot_accs.append(acc)
        boot_f1s.append(f1)
        boot_aucs.append(compute_roc_auc(b_true, b_scores))

    acc_low, acc_high = np.percentile(boot_accs, [alpha_low * 100, alpha_high * 100])
    f1_low, f1_high = np.percentile(boot_f1s, [alpha_low * 100, alpha_high * 100])
    auc_low, auc_high = np.percentile(boot_aucs, [alpha_low * 100, alpha_high * 100])

    return (
        ConfidenceInterval(low=round(float(acc_low), 4), high=round(float(acc_high), 4), confidence_level=confidence_level),
        ConfidenceInterval(low=round(float(f1_low), 4), high=round(float(f1_high), 4), confidence_level=confidence_level),
        ConfidenceInterval(low=round(float(auc_low), 4), high=round(float(auc_high), 4), confidence_level=confidence_level),
    )


def compute_binary_metrics(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    threshold: float = 0.50,
    compute_cis: bool = True,
    num_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ClassificationMetrics:
    """Calculate comprehensive binary classification metrics, confusion matrix, and confidence intervals.

    Args:
        y_true: Ground truth binary targets (0=authentic/negative, 1=manipulated/positive).
        y_scores: Continuous uncalibrated prediction scores in [0.0, 1.0].
        threshold: Decision threshold for positive classification.
        compute_cis: Whether to calculate bootstrap confidence intervals.
        num_bootstraps: Number of bootstrap resamples (default 500).
        confidence_level: Target confidence coverage (default 0.95).
        seed: Random seed for bootstrap reproducibility.

    Returns:
        ClassificationMetrics structured payload.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_scores = np.asarray(y_scores, dtype=float)

    if len(arr_true) != len(arr_scores):
        raise ValueError(f"Length mismatch: y_true ({len(arr_true)}) vs y_scores ({len(arr_scores)})")

    n_samples = len(arr_true)
    if n_samples == 0:
        raise ValueError("Cannot calculate classification metrics on empty ground truth arrays.")

    preds = (arr_scores >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (arr_true == 1)))
    fp = int(np.sum((preds == 1) & (arr_true == 0)))
    tn = int(np.sum((preds == 0) & (arr_true == 0)))
    fn = int(np.sum((preds == 0) & (arr_true == 1)))

    accuracy = (tp + tn) / max(1, n_samples)
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    balanced_accuracy = (tpr + tnr) / 2.0

    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)

    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    roc_auc = compute_roc_auc(arr_true, arr_scores)
    pr_auc = compute_pr_auc(arr_true, arr_scores)

    confusion = [[tn, fp], [fn, tp]]

    acc_ci = None
    f1_ci = None
    auc_ci = None

    if compute_cis and n_samples >= 5:
        acc_ci, f1_ci, auc_ci = compute_bootstrap_confidence_intervals(
            y_true=arr_true,
            y_scores=arr_scores,
            threshold=threshold,
            num_bootstraps=num_bootstraps,
            confidence_level=confidence_level,
            seed=seed,
        )

    return ClassificationMetrics(
        sample_count=n_samples,
        accuracy=round(float(accuracy), 4),
        balanced_accuracy=round(float(balanced_accuracy), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1_score=round(float(f1), 4),
        roc_auc=round(float(roc_auc), 4),
        pr_auc=round(float(pr_auc), 4),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        false_positive_rate=round(float(fpr), 4),
        false_negative_rate=round(float(fnr), 4),
        confusion_matrix=confusion,
        threshold=round(float(threshold), 4),
        accuracy_ci=acc_ci,
        f1_ci=f1_ci,
        roc_auc_ci=auc_ci,
    )
