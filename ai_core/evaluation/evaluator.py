"""Evaluation runner computing precision, recall, F1, and system metrics.

Separates threat-intent evaluation from toxicity evaluation. Toxicity metrics
use an explicitly documented raw-score threshold for development benchmarking;
they are not calibrated probabilities or production decision thresholds.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import psutil

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_core.evaluation.dataset import BENCHMARK_DATASET, BenchmarkSample
from ai_core.threat.analyzer import BaseThreatAnalyzer
from ai_core.threat.schemas import ThreatLevel

logger = logging.getLogger("safeher.ai.evaluation")

# Development-benchmark threshold only. Toxic-BERT sigmoid outputs are
# uncalibrated scores and must not be presented as probabilities.
TOXICITY_SCORE_THRESHOLD = 0.5


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard binary classification metrics and confusion-matrix counts."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationReport:
    """Comprehensive benchmark evaluation report."""

    analyzer_name: str
    sample_count: int
    threat_intent_metrics: ClassificationMetrics
    toxicity_metrics: Optional[ClassificationMetrics]
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    memory_rss_mb: float


def _compute_metrics(y_true: List[bool], y_pred: List[bool]) -> ClassificationMetrics:
    """Calculate confusion-matrix counts, precision, recall, F1, and accuracy.

    Raises:
        ValueError: If the inputs have different lengths or contain no samples.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of items.")
    if not y_true:
        raise ValueError("Cannot compute classification metrics for an empty dataset.")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    return ClassificationMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        accuracy=round(accuracy, 4),
    )


def evaluate_analyzer(
    analyzer: BaseThreatAnalyzer,
    samples: Optional[List[BenchmarkSample]] = None,
) -> EvaluationReport:
    """Evaluate threat intent and, when fully available, toxicity signals.

    Args:
        analyzer: BaseThreatAnalyzer implementation to evaluate.
        samples: Optional custom benchmark. ``None`` selects the bundled
            development dataset; an explicitly empty list is rejected.

    Returns:
        EvaluationReport containing classification, latency, and memory metrics.
        Toxicity metrics are ``None`` unless a valid Toxic-BERT toxicity score
        is available for every sample, preventing silent partial-sample metrics.

    Raises:
        ValueError: If the selected benchmark contains no samples.
    """
    dataset = BENCHMARK_DATASET if samples is None else list(samples)
    if not dataset:
        raise ValueError("Evaluation samples must contain at least one sample.")

    process = psutil.Process()
    latencies: List[float] = []
    y_threat_true: List[bool] = []
    y_threat_pred: List[bool] = []
    y_toxic_true: List[bool] = []
    y_toxic_pred: List[bool] = []

    for sample in dataset:
        t0 = time.perf_counter()
        response = analyzer.analyze(sample.text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # LOW / INSUFFICIENT_CONTEXT is not counted as a confirmed threat.
        is_predicted_threat = response.overall_threat_level in (
            ThreatLevel.MEDIUM,
            ThreatLevel.HIGH,
            ThreatLevel.SEVERE,
        )
        y_threat_true.append(sample.is_threat)
        y_threat_pred.append(is_predicted_threat)

        # Toxicity is evaluated only from the model's raw auxiliary score.
        # The 0.5 threshold is a development benchmark convention, not a
        # calibrated probability cutoff or a production decision threshold.
        toxic_score = response.category_scores.get("toxic_bert:toxic")
        if toxic_score is not None:
            try:
                score_value = float(toxic_score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score_value):
                y_toxic_true.append(sample.is_toxic)
                y_toxic_pred.append(score_value > TOXICITY_SCORE_THRESHOLD)

    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    threat_metrics = _compute_metrics(y_threat_true, y_threat_pred)
    if len(y_toxic_true) == len(dataset):
        toxic_metrics: Optional[ClassificationMetrics] = _compute_metrics(
            y_toxic_true, y_toxic_pred
        )
    else:
        toxic_metrics = None
        logger.warning(
            "Toxicity metrics omitted: valid toxic_bert:toxic scores were available "
            "for %d of %d benchmark samples.",
            len(y_toxic_true),
            len(dataset),
        )

    return EvaluationReport(
        analyzer_name=analyzer.__class__.__name__,
        sample_count=len(dataset),
        threat_intent_metrics=threat_metrics,
        toxicity_metrics=toxic_metrics,
        avg_latency_ms=round(sum(latencies) / len(latencies), 2),
        min_latency_ms=round(min(latencies), 2),
        max_latency_ms=round(max(latencies), 2),
        memory_rss_mb=round(mem_after_mb, 2),
    )
