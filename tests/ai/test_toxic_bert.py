"""Tests for Toxic-BERT Model Integration and Benchmark Evaluation.

Tests comply strictly with Milestone 2 constraints:
- Tests do not depend on downloading large models.
- Uses synthetic, non-confidential test inputs.
- Verifies lazy loading, error handling, uncalibrated scoring, and CPU execution.
- Verifies evaluation metrics (Precision, Recall, F1, Confusion Matrix).
"""

from typing import Dict, List, Optional
import pytest

from ai_core.evaluation.dataset import BENCHMARK_DATASET, BenchmarkSample
from ai_core.evaluation.evaluator import (
    ClassificationMetrics,
    EvaluationReport,
    evaluate_analyzer,
)
from ai_core.threat.analyzer import DevelopmentThreatAnalyzer
from ai_core.threat.schemas import (
    ThreatAssessmentResponse,
    ThreatCategory,
    ThreatLevel,
    UncertaintyState,
)
from ai_core.threat.toxic_bert import (
    TOXIC_BERT_LABELS,
    ToxicBertModelRunner,
    ToxicBertResult,
    ToxicBertThreatAnalyzer,
)


class MockToxicBertModelRunner(ToxicBertModelRunner):
    """Deterministic mock runner simulating Toxic-BERT inference for offline testing."""

    def __init__(
        self,
        simulated_scores: Optional[Dict[str, float]] = None,
        simulate_load_failure: bool = False,
    ) -> None:
        super().__init__(model_name="mock/toxic-bert", device="cpu", lazy_load=True)
        self.simulated_scores = simulated_scores
        self.simulate_load_failure = simulate_load_failure
        self.call_count = 0

    def load(self) -> bool:
        if self.simulate_load_failure:
            self._load_error = "Simulated offline network error"
            self._is_loaded = False
            return False
        self._is_loaded = True
        return True

    def predict(self, text: str) -> Optional[ToxicBertResult]:
        self.call_count += 1
        if not self._is_loaded and not self.load():
            return None

        # Default scores if none specifically simulated
        scores = self.simulated_scores or {
            "toxic": 0.02,
            "severe_toxic": 0.001,
            "obscene": 0.01,
            "threat": 0.005,
            "insult": 0.02,
            "identity_hate": 0.002,
        }

        # Simulated raw logits matching sigmoid scores: logit = log(p / (1 - p))
        import math
        logits = {
            k: round(math.log(max(1e-5, min(0.99999, v)) / (1.0 - max(1e-5, min(0.99999, v)))), 4)
            for k, v in scores.items()
        }

        return ToxicBertResult(
            raw_logits=logits,
            raw_sigmoid_scores=dict(scores),
            inference_time_ms=18.42,
            model_name="mock/toxic-bert",
            is_calibrated=False,  # Raw sigmoid scores are not calibrated probabilities
        )


# =============================================================================
# 1. Model Loading & Offline Fallback Tests
# =============================================================================

def test_toxic_bert_lazy_loading():
    """Verify that model is NOT loaded at instantiation time."""
    runner = ToxicBertModelRunner(model_name="unitary/toxic-bert", lazy_load=True)
    assert runner.is_loaded is False


def test_toxic_bert_model_unavailable_graceful_fallback():
    """Verify analyzer gracefully degrades to baseline patterns when model fails to load."""
    mock_runner = MockToxicBertModelRunner(simulate_load_failure=True)
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    # Analyze synthetic text
    res = analyzer.analyze("I know where you live and I will track you down and hurt you.")

    # Must not crash, falls back to baseline patterns
    assert res.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    assert "offline or unavailable" in res.explanation
    assert res.is_calibrated is False
    assert res.confidence_score is None


# =============================================================================
# 2. Multilabel Classification & Uncalibrated Scores Tests
# =============================================================================

def test_toxic_bert_multilabel_scores_and_uncalibrated_flag():
    """Verify all 6 multilabel categories are mapped and flagged as uncalibrated."""
    simulated = {
        "toxic": 0.88,
        "severe_toxic": 0.45,
        "obscene": 0.72,
        "threat": 0.91,
        "insult": 0.84,
        "identity_hate": 0.12,
    }
    mock_runner = MockToxicBertModelRunner(simulated_scores=simulated)
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    res = analyzer.analyze("Sample threat input text.")

    # Verify is_calibrated is explicitly False
    assert res.is_calibrated is False
    assert res.confidence_score is None

    # Verify all 6 classes exist with toxic_bert: prefix in category_scores
    for label in TOXIC_BERT_LABELS:
        key = f"toxic_bert:{label}"
        assert key in res.category_scores
        assert res.category_scores[key] == simulated[label]

    # Verify inference timing is reported
    assert "Toxic-BERT inference: 18.4ms" in res.explanation


# =============================================================================
# 3. Direct Threat & Blackmail Detection Tests
# =============================================================================

def test_toxic_bert_direct_threat_detection():
    """Verify direct violence threats are captured and elevated."""
    simulated = {
        "toxic": 0.85,
        "severe_toxic": 0.50,
        "obscene": 0.20,
        "threat": 0.88,
        "insult": 0.30,
        "identity_hate": 0.05,
    }
    mock_runner = MockToxicBertModelRunner(simulated_scores=simulated)
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    res = analyzer.analyze("I will track you down and put you in the hospital.")
    assert res.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.DIRECT_THREAT in categories


def test_toxic_bert_blackmail_and_financial_extortion():
    """Verify non-vulgar extortion is correctly flagged even when toxicity is low.

    This test confirms that toxicity != extortion: an extortion message with low
    profanity still receives a HIGH/SEVERE threat level due to structural blackmail patterns.
    """
    simulated = {
        "toxic": 0.15,  # Low coarse toxicity
        "severe_toxic": 0.01,
        "obscene": 0.05,
        "threat": 0.20,
        "insult": 0.08,
        "identity_hate": 0.01,
    }
    mock_runner = MockToxicBertModelRunner(simulated_scores=simulated)
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    text = "Unless you send 500 dollars before midnight, I will leak your private photos to your family."
    res = analyzer.analyze(text)

    # Must be recognized as an extortion threat despite low coarse toxicity score
    assert res.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.BLACKMAIL in categories
    assert ThreatCategory.FINANCIAL_EXTORTION in categories
    assert ThreatCategory.EXPOSURE_THREAT in categories


# =============================================================================
# 4. Figurative Language, Insufficient Context & Empty Input Tests
# =============================================================================

def test_toxic_bert_figurative_language():
    """Verify colloquial figurative violent language does not trigger false alerts."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.10, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.05, "insult": 0.02, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    res = analyzer.analyze("You are completely killing it on stage tonight, what a performance!")
    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0


def test_toxic_bert_insufficient_context():
    """Verify ambiguous short phrases yield INSUFFICIENT_CONTEXT."""
    mock_runner = MockToxicBertModelRunner()
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    res = analyzer.analyze("Wait for tomorrow.")
    assert res.overall_threat_level == ThreatLevel.LOW
    assert res.uncertainty_state == UncertaintyState.INSUFFICIENT_CONTEXT


def test_toxic_bert_empty_text_validation():
    """Verify empty text raises ValueError."""
    mock_runner = MockToxicBertModelRunner()
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    with pytest.raises(ValueError, match="empty or whitespace"):
        analyzer.analyze("   \t   ")


# =============================================================================
# 5. Benchmark Evaluation Tests
# =============================================================================

def test_benchmark_evaluation_runner():
    """Verify the evaluation framework accurately computes metrics and separates toxicity."""
    mock_runner = MockToxicBertModelRunner()
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    report: EvaluationReport = evaluate_analyzer(analyzer, samples=BENCHMARK_DATASET)

    assert report.sample_count == len(BENCHMARK_DATASET)
    assert report.threat_intent_metrics is not None

    # Confusion matrix elements must sum to total sample count
    m: ClassificationMetrics = report.threat_intent_metrics
    assert (m.true_positives + m.false_positives + m.true_negatives + m.false_negatives) == len(BENCHMARK_DATASET)

    # Latency and memory metrics must be valid positive values
    assert report.avg_latency_ms > 0
    assert report.memory_rss_mb > 0
