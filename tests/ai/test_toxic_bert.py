"""Tests for Toxic-BERT Model Integration and Benchmark Evaluation.

Tests comply strictly with Milestone 2 constraints:
- Tests do not depend on downloading large models.
- Uses synthetic, non-confidential test inputs.
- Verifies lazy loading, error handling, uncalibrated scoring, and CPU execution.
- Verifies evaluation metrics (Precision, Recall, F1, Confusion Matrix).
"""

import json
import threading
import time
import shutil
import subprocess
from typing import Dict, List, Optional
import pytest

from ai_core.evaluation.dataset import BENCHMARK_DATASET, BenchmarkSample
from ai_core.config import ThreatAnalyzerConfig
from ai_core.evaluation.evaluator import (
    ClassificationMetrics,
    EvaluationReport,
    _compute_metrics,
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


def test_toxic_bert_concurrent_load_thread_safety(monkeypatch):
    """Verify concurrent threads calling load() do not duplicate model initialization."""
    import sys
    from types import ModuleType

    load_calls = 0
    lock = threading.Lock()

    class FakeTokenizer:
        pass

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_from_pretrained(*args, **kwargs):
        nonlocal load_calls
        with lock:
            load_calls += 1
        time.sleep(0.01)
        return FakeModel()

    def fake_tok_from_pretrained(*args, **kwargs):
        return FakeTokenizer()

    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(fake_tok_from_pretrained)}
    )
    transformers_stub.AutoModelForSequenceClassification = type(
        "AutoModelForSequenceClassification", (), {"from_pretrained": staticmethod(fake_from_pretrained)}
    )

    torch_stub = ModuleType("torch")
    torch_stub.set_num_threads = lambda n: None
    torch_stub.device = lambda d: d

    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    runner = ToxicBertModelRunner(model_name="mock/thread-test", lazy_load=True)

    threads = []
    for _ in range(10):
        t = threading.Thread(target=runner.load)
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert runner.is_loaded is True
    assert load_calls == 1


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



def test_toxic_bert_predict_handles_tokenization_failure(monkeypatch):
    """Verify tokenizer/runtime errors return None instead of crashing inference."""
    import sys
    from types import ModuleType

    # Stub optional ML imports so this test does not require PyTorch or model weights.
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))

    runner = ToxicBertModelRunner(model_name="mock/toxic-bert", lazy_load=True)
    runner._is_loaded = True

    def failing_tokenizer(*args, **kwargs):
        raise RuntimeError("simulated tokenizer failure")

    runner._tokenizer = failing_tokenizer
    runner._model = object()

    assert runner.predict("A harmless test message") is None


def test_toxic_bert_analyzer_falls_back_when_runner_raises():
    """Verify an unexpected runner exception does not interrupt baseline assessment."""

    class FailingPredictionRunner:
        def predict(self, text: str) -> Optional[ToxicBertResult]:
            raise RuntimeError("simulated inference failure")

    analyzer = ToxicBertThreatAnalyzer(model_runner=FailingPredictionRunner())
    result = analyzer.analyze("I will track you down and hurt you.")

    assert result.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    assert any(
        indicator.category == ThreatCategory.DIRECT_THREAT
        for indicator in result.threat_indicators
    )
    assert result.category_scores == {}
    assert "offline or unavailable" in result.explanation
    assert result.is_calibrated is False
    assert result.confidence_score is None


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


def test_toxic_bert_high_neural_threat_score_does_not_escalate_harmless_text():
    """A high uncalibrated Toxic-BERT score must not independently create a threat."""
    simulated = {
        "toxic": 0.05,
        "severe_toxic": 0.01,
        "obscene": 0.02,
        "threat": 0.95,
        "insult": 0.03,
        "identity_hate": 0.01,
    }
    mock_runner = MockToxicBertModelRunner(simulated_scores=simulated)
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    res = analyzer.analyze(
        "The team completed the project successfully and shared the report."
    )

    # The rule-based assessment remains authoritative for threat category/severity.
    assert res.overall_threat_level == ThreatLevel.NONE
    assert all(
        indicator.category != ThreatCategory.DIRECT_THREAT
        for indicator in res.threat_indicators
    )

    # Keep the neural score as an auxiliary, explicitly uncalibrated signal.
    assert res.category_scores["toxic_bert:threat"] == pytest.approx(0.95)
    assert res.is_calibrated is False
    assert res.confidence_score is None
    assert "Raw uncalibrated" in res.explanation


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


# =============================================================================
# 9. Evaluation Metric Correctness and Input Validation Tests
# =============================================================================

def test_compute_metrics_returns_expected_confusion_matrix_and_scores():
    """Verify precision, recall, F1, and accuracy against a known example."""
    metrics = _compute_metrics(
        y_true=[True, True, False, False],
        y_pred=[True, False, True, False],
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1_score == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)


def test_compute_metrics_rejects_different_input_lengths():
    """Avoid silently truncating metric inputs through zip()."""
    with pytest.raises(ValueError, match="same number of items"):
        _compute_metrics(y_true=[True, False], y_pred=[True])


def test_compute_metrics_rejects_empty_inputs():
    """Avoid returning apparently valid zero metrics for no samples."""
    with pytest.raises(ValueError, match="empty dataset"):
        _compute_metrics(y_true=[], y_pred=[])


def test_evaluate_analyzer_rejects_explicitly_empty_dataset():
    """An empty custom list must not silently select the bundled benchmark."""
    analyzer = DevelopmentThreatAnalyzer()

    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_analyzer(analyzer, samples=[])


def test_evaluate_analyzer_omits_toxicity_metrics_when_model_scores_are_partial():
    """Do not report toxicity metrics when the model omitted scores for a sample."""

    class MissingOneScoreRunner(MockToxicBertModelRunner):
        def predict(self, text: str) -> Optional[ToxicBertResult]:
            if text == BENCHMARK_DATASET[0].text:
                self.call_count += 1
                return None
            return super().predict(text)

    analyzer = ToxicBertThreatAnalyzer(model_runner=MissingOneScoreRunner())
    samples = BENCHMARK_DATASET[:2]

    report = evaluate_analyzer(analyzer, samples=samples)

    assert report.sample_count == 2
    assert report.toxicity_metrics is None


def test_evaluate_analyzer_reports_toxicity_metrics_when_all_scores_exist():
    """Verify toxicity metrics are available when every sample has a score."""
    analyzer = ToxicBertThreatAnalyzer(model_runner=MockToxicBertModelRunner())
    report = evaluate_analyzer(analyzer, samples=BENCHMARK_DATASET[:2])

    assert report.toxicity_metrics is not None
    metrics = report.toxicity_metrics
    assert (
        metrics.true_positives
        + metrics.false_positives
        + metrics.true_negatives
        + metrics.false_negatives
    ) == report.sample_count


# =============================================================================
# 10. Hybrid Analyzer Preprocessing Metadata Propagation Tests
# =============================================================================

def test_toxic_bert_analyzer_propagates_preprocessing_warnings():
    """The hybrid response must expose sanitization warnings from the baseline."""
    analyzer = ToxicBertThreatAnalyzer(model_runner=MockToxicBertModelRunner())
    original_text = "Please review this message\u200b carefully."

    result = analyzer.analyze(original_text)

    assert result.preprocessing_warnings
    assert any("zero-width" in warning.lower() for warning in result.preprocessing_warnings)
    assert result.input_truncated is False
    assert "preprocessing_warnings" in result.explanation


def test_toxic_bert_analyzer_propagates_truncation_metadata():
    """The hybrid response must disclose when the baseline truncated the input."""
    config = ThreatAnalyzerConfig(max_input_length=20, min_input_length=1)
    analyzer = ToxicBertThreatAnalyzer(
        config=config,
        model_runner=MockToxicBertModelRunner(),
    )
    original_text = "This is a much longer harmless message than twenty characters."

    result = analyzer.analyze(original_text)

    assert result.input_truncated is True
    assert any("exceeded 20 characters" in warning for warning in result.preprocessing_warnings)
    assert "complete original message was not analyzed" in result.explanation


# =============================================================================
# Model revision pinning tests
# =============================================================================


def test_default_toxic_bert_loads_tokenizer_and_model_from_pinned_revision(monkeypatch):
    """The default public model and tokenizer must use the same immutable revision."""
    import sys
    from types import ModuleType

    from ai_core.threat.toxic_bert import TOXIC_BERT_MODEL_REVISION

    calls = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["tokenizer"] = {"model_name": model_name, **kwargs}
            return object()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model"] = {"model_name": model_name, **kwargs}
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return None

    fake_torch = ModuleType("torch")
    fake_torch.set_num_threads = lambda count: None
    fake_torch.device = lambda device: device

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForSequenceClassification = FakeModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    runner = ToxicBertModelRunner(lazy_load=True)
    assert runner.load() is True

    assert runner.model_name == "unitary/toxic-bert"
    assert runner.model_revision == TOXIC_BERT_MODEL_REVISION
    assert calls["tokenizer"] == {
        "model_name": "unitary/toxic-bert",
        "revision": TOXIC_BERT_MODEL_REVISION,
    }
    assert calls["model"] == {
        "model_name": "unitary/toxic-bert",
        "revision": TOXIC_BERT_MODEL_REVISION,
    }


def test_custom_toxic_bert_model_does_not_receive_unrelated_default_revision():
    """A custom model ID must not inherit unitary/toxic-bert's revision."""
    runner = ToxicBertModelRunner(
        model_name="example/custom-toxic-model",
        lazy_load=True,
    )

    assert runner.model_revision is None


def test_custom_toxic_bert_model_can_receive_explicit_revision():
    """Custom model IDs can be pinned by their caller to a matching revision."""
    runner = ToxicBertModelRunner(
        model_name="example/custom-toxic-model",
        lazy_load=True,
        model_revision="0123456789abcdef0123456789abcdef01234567",
    )

    assert runner.model_revision == "0123456789abcdef0123456789abcdef01234567"


# =============================================================================
# 14. Analyzed Text and UTF-16 Offset Consistency for ToxicBertThreatAnalyzer
# =============================================================================

def test_toxic_bert_analyzed_text_field_and_utf16_offsets():
    """Verify ToxicBertThreatAnalyzer sets analyzed_text and consistent UTF-16 span offsets."""
    mock_runner = MockToxicBertModelRunner(
        simulated_scores={"toxic": 0.20, "severe_toxic": 0.05, "obscene": 0.05, "threat": 0.30, "insult": 0.10, "identity_hate": 0.0}
    )
    analyzer = ToxicBertThreatAnalyzer(model_runner=mock_runner)

    text = "Look 📸🚨 Send me 500 dollars or I will publish your private photos."
    res = analyzer.analyze(text)

    # analyzed_text must be populated and match working copy
    assert res.analyzed_text
    assert "Send me 500 dollars" in res.analyzed_text
    assert len(res.flagged_spans) > 0

    node_bin = shutil.which("node")

    for span in res.flagged_spans:
        # Python code-point slicing
        assert res.analyzed_text[span.start : span.end] == span.text

        # UTF-16 offsets validation
        assert span.start_utf16 is not None
        assert span.end_utf16 is not None
        assert span.start_utf16 >= 0
        assert span.end_utf16 >= span.start_utf16

        # UTF-16 byte-level slicing in Python
        utf16_bytes = res.analyzed_text.encode("utf-16-le")
        py_utf16_slice = utf16_bytes[span.start_utf16 * 2 : span.end_utf16 * 2].decode("utf-16-le")
        assert py_utf16_slice == span.text

        # Live Node.js verification if available
        if node_bin:
            js_code = (
                f"const text = {json.dumps(res.analyzed_text)};\n"
                f"const slice = text.slice({span.start_utf16}, {span.end_utf16});\n"
                f"if (slice !== {json.dumps(span.text)}) {{\n"
                f"  console.error('JS mismatch: expected', {json.dumps(span.text)}, 'got', slice);\n"
                f"  process.exit(1);\n"
                f"}}\n"
            )
            proc = subprocess.run([node_bin, "-e", js_code], capture_output=True, text=True)
            assert proc.returncode == 0, f"Node.js slice mismatch: {proc.stderr}"


def test_toxic_bert_truncated_input_analyzed_text():
    """Verify ToxicBertThreatAnalyzer correctly preserves input_truncated metadata and analyzed_text."""
    short_config = ThreatAnalyzerConfig(max_input_length=50)
    mock_runner = MockToxicBertModelRunner()
    analyzer = ToxicBertThreatAnalyzer(config=short_config, model_runner=mock_runner)

    long_text = "A" * 80 + " Send me 500 dollars or I will expose you."
    res = analyzer.analyze(long_text)

    assert res.input_truncated is True
    assert len(res.analyzed_text) <= 50
    assert any("truncated" in w.lower() for w in res.preprocessing_warnings)
