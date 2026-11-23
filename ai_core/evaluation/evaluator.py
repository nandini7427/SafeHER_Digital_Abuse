"""Evaluation runner computing precision, recall, F1, and system metrics.

Separates threat intent evaluation from toxicity evaluation.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil

from ai_core.evaluation.dataset import BENCHMARK_DATASET, BenchmarkSample
from ai_core.threat.analyzer import BaseThreatAnalyzer
from ai_core.threat.schemas import ThreatLevel


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard evaluation metrics container."""

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
    """Calculate confusion matrix elements, precision, recall, F1, and accuracy."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

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
    """Run evaluation benchmark across threat intent and toxicity tasks.

    Args:
        analyzer: BaseThreatAnalyzer implementation to evaluate.
        samples: Optional custom benchmark sample list (defaults to BENCHMARK_DATASET).

    Returns:
        EvaluationReport containing precision, recall, F1, latency, and memory metrics.
    """
    dataset = samples or BENCHMARK_DATASET
    process = psutil.Process()

    latencies: List[float] = []
    y_threat_true: List[bool] = []
    y_threat_pred: List[bool] = []

    y_toxic_true: List[bool] = []
    y_toxic_pred: List[bool] = []
    has_toxic_signals = False

    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    for sample in dataset:
        t0 = time.perf_counter()
        response = analyzer.analyze(sample.text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # Threat intent decision: flagged if level is MEDIUM, HIGH, or SEVERE
        is_predicted_threat = response.overall_threat_level in [
            ThreatLevel.MEDIUM,
            ThreatLevel.HIGH,
            ThreatLevel.SEVERE,
        ]
        y_threat_true.append(sample.is_threat)
        y_threat_pred.append(is_predicted_threat)

        # Check for toxicity score if provided by model
        toxic_score = response.category_scores.get("toxic_bert:toxic")
        if toxic_score is not None:
            has_toxic_signals = True
            is_predicted_toxic = toxic_score > 0.5
            y_toxic_true.append(sample.is_toxic)
            y_toxic_pred.append(is_predicted_toxic)

    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    threat_metrics = _compute_metrics(y_threat_true, y_threat_pred)
    toxic_metrics = _compute_metrics(y_toxic_true, y_toxic_pred) if has_toxic_signals else None

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
