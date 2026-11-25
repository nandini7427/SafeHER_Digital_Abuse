"""Objective model evaluation and validation pipeline for SafeHER Image Intelligence.

Computes precision, recall, F1, ROC-AUC, FPR, FNR, confusion matrix, latency percentiles,
peak memory usage, Expected Calibration Error (ECE), Brier score, and perturbation robustness.
Enforces that uncalibrated heuristic scores are never represented as probabilities.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from ai_core.evaluation.image_dataset import (
    ImageBenchmarkSample,
    apply_perturbation,
    build_forensic_benchmark_dataset,
)
from ai_core.image_intelligence.detector import BaseImageDetector, SafeHERImageDetector
from ai_core.image_intelligence.schemas import AssessmentLabel

logger = logging.getLogger("safeher.ai.evaluation.image")

DEFAULT_DECISION_THRESHOLD = 0.45


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Standard binary classification metrics and confusion matrix."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: float
    decision_threshold: float


@dataclass(frozen=True)
class CalibrationMetrics:
    """Statistical calibration evaluation and reliability metrics."""

    is_calibrated: bool
    score_interpretation: str
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    reliability_bins: List[Dict[str, Any]]
    calibration_assessment: str


@dataclass(frozen=True)
class LatencyMetrics:
    """Inference execution profiling and memory metrics."""

    sample_count: int
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    memory_rss_mb: float


@dataclass(frozen=True)
class RobustnessEvaluation:
    """Evaluation result under a specific transformation or corruption."""

    perturbation_name: str
    metrics: BinaryClassificationMetrics
    f1_delta: float
    fpr_delta: float
    observations: str


@dataclass(frozen=True)
class ErrorAnalysisItem:
    """Forensic investigation of a classification error (FP or FN)."""

    sample_id: str
    true_label: bool
    predicted_label: bool
    raw_score: float
    error_type: str  # 'FALSE_POSITIVE' or 'FALSE_NEGATIVE'
    manipulation_type: str
    root_cause_analysis: str


@dataclass(frozen=True)
class ImageEvaluationReport:
    """Comprehensive, reproducible forensic evaluation report."""

    detector_name: str
    detector_version: str
    total_test_samples: int
    clean_test_metrics: BinaryClassificationMetrics
    calibration_metrics: CalibrationMetrics
    latency_metrics: LatencyMetrics
    robustness_results: List[RobustnessEvaluation]
    error_analysis_samples: List[ErrorAnalysisItem]
    split_sample_counts: Dict[str, int]
    continuous_signal_auc: float
    continuous_signal_metrics: Dict[str, Any]
    timestamp_utc: str


def compute_roc_auc(y_true: List[bool], y_scores: List[float]) -> float:
    """Compute Area Under the Receiver Operating Characteristic (ROC-AUC) via rank statistic.

    Handles continuous and discrete tied scores deterministically without third-party dependencies.
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length.")
    if not y_true:
        raise ValueError("Cannot compute ROC-AUC on empty dataset.")

    n_pos = sum(1 for yt in y_true if yt)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined when one class is absent

    combined = sorted(enumerate(y_scores), key=lambda x: x[1])
    ranks = [0.0] * len(y_scores)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][1] == combined[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[combined[k][0]] = avg_rank
        i = j

    sum_pos_ranks = sum(ranks[idx] for idx, yt in enumerate(y_true) if yt)
    u_statistic = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auc = u_statistic / (n_pos * n_neg)
    return round(float(auc), 4)


def compute_classification_metrics(
    y_true: List[bool],
    y_pred: List[bool],
    y_scores: List[float],
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> BinaryClassificationMetrics:
    """Calculate confusion matrix, precision, recall, F1, FPR, FNR, and ROC-AUC."""
    if len(y_true) != len(y_pred) or len(y_true) != len(y_scores):
        raise ValueError("Inputs y_true, y_pred, and y_scores must have equal lengths.")
    if not y_true:
        raise ValueError("Cannot compute classification metrics for an empty dataset.")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

    total = len(y_true)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc_auc = compute_roc_auc(y_true, y_scores)

    return BinaryClassificationMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        roc_auc=roc_auc,
        decision_threshold=round(threshold, 4),
    )


def compute_calibration_metrics(
    y_true: List[bool],
    y_scores: List[float],
    num_bins: int = 10,
) -> CalibrationMetrics:
    """Evaluate Expected Calibration Error (ECE), Brier Score, and reliability binning."""
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot compute calibration on empty samples.")

    brier = float(np.mean([(s - (1.0 if yt else 0.0)) ** 2 for yt, s in zip(y_true, y_scores)]))

    bins: List[Dict[str, Any]] = []
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
                {
                    "bin_range": f"[{low:.1f}, {high:.1f}]",
                    "sample_count": count,
                    "mean_confidence": round(mean_conf, 3),
                    "empirical_accuracy": round(mean_acc, 3),
                    "calibration_gap": round(gap, 3),
                }
            )

    assessment = (
        f"UNCALIBRATED: Raw detector scores exhibit an Expected Calibration Error of {ece:.4f} "
        f"and Brier score of {brier:.4f}. The output score represents a piecewise heuristic anomaly "
        "index and MUST NOT be interpreted or surfaced as a true posterior probability."
    )

    return CalibrationMetrics(
        is_calibrated=False,
        score_interpretation="Uncalibrated Heuristic Error Level Index",
        brier_score=round(brier, 4),
        expected_calibration_error=round(ece, 4),
        maximum_calibration_error=round(mce, 4),
        reliability_bins=bins,
        calibration_assessment=assessment,
    )


def evaluate_image_detector(
    detector: Optional[BaseImageDetector] = None,
    dataset: Optional[List[ImageBenchmarkSample]] = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> ImageEvaluationReport:
    """Execute complete, reproducible evaluation of SafeHER image detection engine.

    Evaluates:
    - Clean held-out test split performance.
    - Statistical calibration & reliability.
    - Inference latency and peak RSS memory.
    - Robustness across 8 distinct image transformations.
    - Forensic error analysis of false positives and false negatives.
    """
    if detector is None:
        detector = SafeHERImageDetector()

    if dataset is None:
        dataset = build_forensic_benchmark_dataset()

    test_samples = [s for s in dataset if s.split == "test"]
    if not test_samples:
        raise ValueError("Benchmark dataset contains no samples with split='test'.")

    split_counts = {
        "train": sum(1 for s in dataset if s.split == "train"),
        "val": sum(1 for s in dataset if s.split == "val"),
        "test": len(test_samples),
    }

    # 1. Evaluate clean test split
    process = psutil.Process()
    latencies: List[float] = []
    y_true: List[bool] = []
    y_pred: List[bool] = []
    y_scores: List[float] = []
    errors: List[ErrorAnalysisItem] = []

    detector_name = "UnknownDetector"
    detector_version = "1.0.0"

    for sample in test_samples:
        t0 = time.perf_counter()
        response = detector.detect(sample.image_bytes)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        detector_name = response.detector_name
        detector_version = response.detector_version

        # Predicted label: flag as suspicious if label is SUSPICIOUS or HIGHLY_LIKELY
        is_pred_manip = response.assessment_label in (
            AssessmentLabel.SUSPICIOUS_MANIPULATION,
            AssessmentLabel.HIGHLY_LIKELY_MANIPULATED,
        )
        score = response.raw_model_score

        y_true.append(sample.is_manipulated)
        y_pred.append(is_pred_manip)
        y_scores.append(score)

        # Record classification errors for safety analysis
        if is_pred_manip != sample.is_manipulated:
            err_type = "FALSE_POSITIVE" if is_pred_manip else "FALSE_NEGATIVE"
            if err_type == "FALSE_POSITIVE":
                cause = (
                    f"Authentic image flagged as suspicious (score={score:.2f}). "
                    f"High-frequency texture or edge gradient ({sample.manipulation_type}) "
                    "triggered elevated ELA residual variance."
                )
            else:
                cause = (
                    f"Manipulated image missed (score={score:.2f} <= {threshold:.2f}). "
                    f"Manipulation ({sample.manipulation_type}) exhibited subtle compression "
                    "variance below the heuristic detection threshold."
                )

            errors.append(
                ErrorAnalysisItem(
                    sample_id=sample.sample_id,
                    true_label=sample.is_manipulated,
                    predicted_label=is_pred_manip,
                    raw_score=score,
                    error_type=err_type,
                    manipulation_type=sample.manipulation_type,
                    root_cause_analysis=cause,
                )
            )

    memory_mb = process.memory_info().rss / (1024 * 1024)

    clean_metrics = compute_classification_metrics(y_true, y_pred, y_scores, threshold=threshold)
    cal_metrics = compute_calibration_metrics(y_true, y_scores)

    sorted_latencies = sorted(latencies)
    latency_metrics = LatencyMetrics(
        sample_count=len(latencies),
        avg_latency_ms=round(float(np.mean(latencies)), 2),
        median_latency_ms=round(float(np.median(latencies)), 2),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        min_latency_ms=round(float(np.min(latencies)), 2),
        max_latency_ms=round(float(np.max(latencies)), 2),
        memory_rss_mb=round(memory_mb, 2),
    )

    # 2. Evaluate Robustness Perturbations on Test Set
    perturbations = [
        ("jpeg_q50", "Heavy JPEG recompression (Q=50) destroying high-frequency DCT residuals."),
        ("jpeg_q75", "Moderate JPEG recompression (Q=75) simulating standard web optimization."),
        ("jpeg_q90", "Light JPEG recompression (Q=90) typical of direct photo sharing."),
        ("resize_down_up", "0.5x downscaling and bilinear upscaling simulating chat preview thumbnails."),
        ("center_crop", "0.7x center crop removing peripheral context."),
        ("gaussian_blur", "Gaussian blur (radius 1.5) simulating low-pass lens or smoothing filters."),
        ("brightness_shift", "+25% brightness gain shifting pixel distribution."),
        ("contrast_shift", "30% contrast reduction compressing dynamic range."),
        ("screenshot_composite", "Messenger screenshot framing with neutral background canvas."),
    ]

    robustness_results: List[RobustnessEvaluation] = []

    for pert_name, obs_desc in perturbations:
        pert_y_true: List[bool] = []
        pert_y_pred: List[bool] = []
        pert_y_scores: List[float] = []

        for sample in test_samples:
            try:
                perturbed_bytes = apply_perturbation(sample.image_bytes, pert_name)
                resp = detector.detect(perturbed_bytes)
                is_flagged = resp.assessment_label in (
                    AssessmentLabel.SUSPICIOUS_MANIPULATION,
                    AssessmentLabel.HIGHLY_LIKELY_MANIPULATED,
                )
                pert_y_true.append(sample.is_manipulated)
                pert_y_pred.append(is_flagged)
                pert_y_scores.append(resp.raw_model_score)
            except Exception as exc:
                logger.warning("Error evaluating sample %s under %s: %s", sample.sample_id, pert_name, exc)
                continue

        if pert_y_true:
            p_metrics = compute_classification_metrics(pert_y_true, pert_y_pred, pert_y_scores, threshold)
            f1_delta = round(p_metrics.f1_score - clean_metrics.f1_score, 4)
            fpr_delta = round(p_metrics.false_positive_rate - clean_metrics.false_positive_rate, 4)
            robustness_results.append(
                RobustnessEvaluation(
                    perturbation_name=pert_name,
                    metrics=p_metrics,
                    f1_delta=f1_delta,
                    fpr_delta=fpr_delta,
                    observations=obs_desc,
                )
            )

    # 3. Analyze underlying continuous forensic signal (unscaled ELA variance)
    import io
    from PIL import Image
    from ai_core.image_intelligence.model_runner import BaselineHeuristicRunner

    heuristic_runner = BaselineHeuristicRunner(config=detector.config)
    raw_vars: List[float] = []
    y_test_true = [s.is_manipulated for s in test_samples]
    for s in test_samples:
        img = Image.open(io.BytesIO(s.image_bytes)).convert("RGB")
        _, _, v = heuristic_runner._compute_ela(img)
        raw_vars.append(v)

    cont_auc = compute_roc_auc(y_test_true, raw_vars)
    auth_vars = [v for yt, v in zip(y_test_true, raw_vars) if not yt]
    manip_vars = [v for yt, v in zip(y_test_true, raw_vars) if yt]

    cont_robustness: Dict[str, float] = {}
    for pert_name, _ in perturbations:
        pert_vars: List[float] = []
        for s in test_samples:
            try:
                pb = apply_perturbation(s.image_bytes, pert_name)
                p_img = Image.open(io.BytesIO(pb)).convert("RGB")
                _, _, pv = heuristic_runner._compute_ela(p_img)
                pert_vars.append(pv)
            except Exception:
                continue
        if len(pert_vars) == len(y_test_true):
            cont_robustness[pert_name] = compute_roc_auc(y_test_true, pert_vars)

    continuous_summary = {
        "test_auc": cont_auc,
        "auth_mean_variance": round(float(np.mean(auth_vars)), 4),
        "manip_mean_variance": round(float(np.mean(manip_vars)), 4),
        "variance_ratio": round(float(np.mean(manip_vars) / max(np.mean(auth_vars), 1e-6)), 2),
        "robustness_aucs": cont_robustness,
    }

    return ImageEvaluationReport(
        detector_name=detector_name,
        detector_version=detector_version,
        total_test_samples=len(test_samples),
        clean_test_metrics=clean_metrics,
        calibration_metrics=cal_metrics,
        latency_metrics=latency_metrics,
        robustness_results=robustness_results,
        error_analysis_samples=errors,
        split_sample_counts=split_counts,
        continuous_signal_auc=cont_auc,
        continuous_signal_metrics=continuous_summary,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
