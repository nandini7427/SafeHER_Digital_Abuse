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


# =============================================================================
# 6. Extortion Gap Remediation Tests for ToxicBertThreatAnalyzer
# =============================================================================

@pytest.mark.parametrize(
    "text,expected_categories",
    [
        (
            "Send me 500 dollars by midnight or I will publish your private photos.",
            {
                ThreatCategory.BLACKMAIL,
                ThreatCategory.EXPOSURE_THREAT,
                ThreatCategory.FINANCIAL_EXTORTION,
                ThreatCategory.URGENCY_PRESSURE,
            },
        ),
        (
            "Unless you transfer 1000 dollars, I will forward your private photos to your employer.",
            {
                ThreatCategory.BLACKMAIL,
                ThreatCategory.EXPOSURE_THREAT,
                ThreatCategory.FINANCIAL_EXTORTION,
            },
        ),
        (
            "Pay me ₹500 or I will leak your intimate videos.",
            {
                ThreatCategory.BLACKMAIL,
                ThreatCategory.EXPOSURE_THREAT,
                ThreatCategory.FINANCIAL_EXTORTION,
            },
        ),
    ],
)
def test_toxic_bert_remediated_extortion_positive_cases(text, expected_categories):
    """Verify ToxicBertThreatAnalyzer recognizes remediated extortion and exposure patterns."""
    # Simulate low coarse toxicity to prove structural extortion detection functions independently
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.08, "severe_toxic": 0.01, "obscene": 0.02, "threat": 0.15, "insult": 0.05, "identity_hate": 0.01}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    res = analyzer.analyze(text)

    assert res.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    detected = {span.category for span in res.flagged_spans}
    for cat in expected_categories:
        assert cat in detected, f"Expected category {cat} missing from detected: {detected}"


@pytest.mark.parametrize(
    "harmless_text",
    [
        "Please send me 500 dollars for the concert tickets.",
        "I will publish my new photography collection tomorrow.",
        "Please transfer the money for our group dinner.",
    ],
)
def test_toxic_bert_ordinary_requests_and_harmless_photos_negative_cases(harmless_text):
    """Verify ToxicBertThreatAnalyzer does not elevate harmless payment/photo requests."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.01, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.01, "insult": 0.01, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    res = analyzer.analyze(harmless_text)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0


# =============================================================================
# 7. Milestone 2 Final Review Regression Tests for ToxicBertThreatAnalyzer
# =============================================================================

def test_toxic_bert_unrelated_payment_and_separate_exposure_threat():
    """Verify ToxicBertThreatAnalyzer handles unrelated concert payment and separate exposure threat."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.05, "severe_toxic": 0.0, "obscene": 0.01, "threat": 0.08, "insult": 0.02, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    text = "Please send me 500 dollars for the concert tickets. If you report me, I will send your private photos to your family."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.HIGH
    detected_categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.BLACKMAIL in detected_categories
    assert ThreatCategory.EXPOSURE_THREAT in detected_categories
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected_categories


def test_toxic_bert_direct_threat_combined_with_extortion():
    """Verify ToxicBertThreatAnalyzer handles direct threat combined with extortion at SEVERE level."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.75, "severe_toxic": 0.30, "obscene": 0.20, "threat": 0.85, "insult": 0.40, "identity_hate": 0.05}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    text = "Pay me 1000 dollars or I will track you down and hurt you."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.SEVERE
    detected_categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.DIRECT_THREAT in detected_categories
    assert ThreatCategory.FINANCIAL_EXTORTION in detected_categories
    assert ThreatCategory.BLACKMAIL in detected_categories


def test_toxic_bert_exposure_threat_without_financial_demand():
    """Verify ToxicBertThreatAnalyzer handles standalone exposure threat at MEDIUM level."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.12, "severe_toxic": 0.02, "obscene": 0.05, "threat": 0.18, "insult": 0.04, "identity_hate": 0.01}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    text = "I will leak your intimate videos to everyone."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.MEDIUM
    detected_categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.EXPOSURE_THREAT in detected_categories
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected_categories


# =============================================================================
# 8. Final Edge-Case Remediation Tests for ToxicBertThreatAnalyzer
# =============================================================================

@pytest.mark.parametrize(
    "text",
    [
        "Please pay me before midnight for the rent.",
        "Please complete the wire transfer before Friday for invoice 204.",
        "Please transfer the requested cryptocurrency to our registered vendor wallet.",
    ],
)
def test_toxic_bert_deadlines_alone_not_extortion_negative_cases(text):
    """Issue 1: Verify routine payment requests with deadlines alone are NOT classified as extortion."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.01, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.01, "insult": 0.01, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected
    assert ThreatCategory.URGENCY_PRESSURE not in detected


@pytest.mark.parametrize(
    "text",
    [
        "Pay the ransom of 5000 dollars.",
        "Pay 1000 dollars to stop the leak.",
    ],
)
def test_toxic_bert_genuine_extortion_demands_using_ransom_preserved(text):
    """Issue 1: Verify genuine extortion demands using ransom or duress language are preserved."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.25, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.35, "insult": 0.05, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    res = analyzer.analyze(text)

    assert res.overall_threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.SEVERE]
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION in detected


def test_toxic_bert_scoped_ultimatum_matching_unrelated_publication():
    """Issue 2: Verify forward consequence search does not match unrelated statement far later or benign publication."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.02, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.01, "insult": 0.01, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    text = "Send me 500 dollars for concert tickets. Or I will publish my private photography collection tomorrow."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected
    assert ThreatCategory.EXPOSURE_THREAT not in detected


def test_toxic_bert_scoped_ultimatum_matching_direct_consequence():
    """Issue 2: Verify forward consequence search detects direct conditional extortion consequence."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.30, "severe_toxic": 0.10, "obscene": 0.10, "threat": 0.45, "insult": 0.15, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    text = "Send me 500 dollars, or I will publish your private photos."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.SEVERE
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION in detected
    assert ThreatCategory.BLACKMAIL in detected
    assert ThreatCategory.EXPOSURE_THREAT in detected


def test_toxic_bert_scoped_ultimatum_matching_separate_sentence_threat():
    """Issue 2: Verify routine dinner payment is NOT extortion when threat in subsequent sentence is condition on reporting."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.15, "severe_toxic": 0.05, "obscene": 0.02, "threat": 0.25, "insult": 0.05, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)
    text = "Please transfer 500 dollars for dinner. If you report me, I will share your private photos."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.HIGH
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.BLACKMAIL in detected
    assert ThreatCategory.EXPOSURE_THREAT in detected
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected
