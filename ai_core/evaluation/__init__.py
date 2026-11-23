"""Model Evaluation and Benchmark Submodule

Contains evaluation protocols, benchmark datasets, and metric computation utilities.
"""

from ai_core.evaluation.dataset import BENCHMARK_DATASET, BenchmarkSample
from ai_core.evaluation.evaluator import (
    ClassificationMetrics,
    EvaluationReport,
    evaluate_analyzer,
)

__all__ = [
    "BENCHMARK_DATASET",
    "BenchmarkSample",
    "ClassificationMetrics",
    "EvaluationReport",
    "evaluate_analyzer",
]
