"""Additional regression tests for SafeHER AI safety and evidence integrity."""

import pytest
from pydantic import ValidationError

from ai_core.config import MANDATORY_AI_DISCLAIMER, ThreatAnalyzerConfig
from ai_core.threat.analyzer import DevelopmentThreatAnalyzer
from ai_core.threat.highlighting import partition_spans, slice_utf16
from ai_core.threat.preprocessing import preprocess_text
from ai_core.threat.schemas import (
    FlaggedSpan,
    ThreatAssessmentResponse,
    ThreatCategory,
    ThreatLevel,
    UncertaintyState,
)
from ai_core.threat.toxic_bert import (
    ToxicBertResult,
    ToxicBertThreatAnalyzer,
)


def _span_for(
    text: str,
    start: int,
    end: int,
    category: ThreatCategory = ThreatCategory.DIRECT_THREAT,
) -> FlaggedSpan:
    start_u16 = len(text[:start].encode("utf-16-le")) // 2
    end_u16 = len(text[:end].encode("utf-16-le")) // 2
    return FlaggedSpan(
        start=start,
        end=end,
        start_utf16=start_u16,
        end_utf16=end_u16,
        text=text[start:end],
        category=category,
        explanation="Synthetic test span",
    )


def _response(text: str, spans=None, **overrides) -> ThreatAssessmentResponse:
    values = {
        "analyzed_text": text,
        "overall_threat_level": ThreatLevel.NONE,
        "uncertainty_state": UncertaintyState.LOW,
        "explanation": "Synthetic test response.",
        "flagged_spans": spans or [],
    }
    values.update(overrides)
    return ThreatAssessmentResponse(**values)


def test_config_rejects_invalid_lengths_and_batch_size():
    with pytest.raises(ValueError, match="max_input_length"):
        ThreatAnalyzerConfig(max_input_length=0)
    with pytest.raises(ValueError, match="min_input_length"):
        ThreatAnalyzerConfig(min_input_length=-1)
    with pytest.raises(ValueError, match="cannot exceed"):
        ThreatAnalyzerConfig(min_input_length=20, max_input_length=10)
    with pytest.raises(ValueError, match="batch_size"):
        ThreatAnalyzerConfig(batch_size=0)


def test_config_is_immutable_and_normalizes_device():
    config = ThreatAnalyzerConfig(device=" CPU ")
    assert config.device == "cpu"
    with pytest.raises((AttributeError, TypeError)):
        config.device = "cuda"  # type: ignore[misc]


def test_config_rejects_invalid_device_and_boolean_values():
    with pytest.raises(ValueError, match="device"):
        ThreatAnalyzerConfig(device="auto-maybe")
    with pytest.raises(ValueError, match="enable_toxicity_check"):
        ThreatAnalyzerConfig(enable_toxicity_check=1)  # type: ignore[arg-type]


def test_preprocessing_rejects_invalid_length_settings():
    with pytest.raises(ValueError, match="max_length"):
        preprocess_text("hello", max_length=0)
    with pytest.raises(ValueError, match="min_length"):
        preprocess_text("hello", min_length=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        preprocess_text("hello", max_length=3, min_length=4)


def test_preprocessing_rejects_isolated_surrogates():
    with pytest.raises(ValueError, match="surrogate"):
        preprocess_text("normal text \ud800 end")


def test_preprocessing_discloses_nfkc_and_whitespace_changes():
    result = preprocess_text("ＡＢＣ     Send me money")
    assert result.working_copy == "ABC Send me money"
    warning_text = " ".join(result.warnings).lower()
    assert "nfkc" in warning_text
    assert "whitespace" in warning_text
    assert result.original_text == "ＡＢＣ     Send me money"


def test_preprocessing_normalizes_windows_and_classic_mac_line_endings():
    result = preprocess_text("first\r\nsecond\rthird")
    assert result.working_copy == "first\nsecond\nthird"
    assert result.warnings


def test_truncation_does_not_split_emoji_grapheme_cluster():
    family = "👨‍👩‍👧‍👦"
    raw = "abcd" + family + " more text"
    result = preprocess_text(raw, max_length=8)
    assert result.is_truncated is True
    assert result.working_copy == "abcd"
    assert len(result.working_copy) <= 8
    assert any("grapheme boundary" in warning for warning in result.warnings)


def test_truncation_preserves_complete_grapheme_when_it_fits():
    cluster = "👨‍👩‍👧‍👦"
    raw = "A" + cluster + " trailing words"
    result = preprocess_text(raw, max_length=8)
    assert result.working_copy == "A" + cluster


def test_flagged_span_rejects_zero_length_and_inverted_offsets():
    common = {
        "text": "x",
        "category": ThreatCategory.DIRECT_THREAT,
        "explanation": "test",
    }
    with pytest.raises(ValidationError, match="end greater than start"):
        FlaggedSpan(start=2, end=2, start_utf16=2, end_utf16=3, **common)
    with pytest.raises(ValidationError, match="end greater than start"):
        FlaggedSpan(start=3, end=2, start_utf16=3, end_utf16=4, **common)
    with pytest.raises(ValidationError, match="UTF-16 end"):
        FlaggedSpan(start=0, end=1, start_utf16=4, end_utf16=4, **common)


def test_assessment_rejects_span_outside_analyzed_text():
    span = _span_for("hello", 0, 5)
    with pytest.raises(ValidationError, match="beyond analyzed_text"):
        _response("hi", [span])


def test_assessment_rejects_span_text_mismatch():
    span = _span_for("hello", 0, 5)
    with pytest.raises(ValidationError, match="does not match"):
        _response("jello", [span])


def test_assessment_rejects_incorrect_utf16_offsets():
    """Reject UTF-16 offsets that do not match the analyzed text."""

    span = FlaggedSpan(
        start=1,          # Python index after the emoji
        end=3,            # Python index after AB
        start_utf16=1,    # Intentionally incorrect; should be 2
        end_utf16=3,      # Intentionally incorrect; should be 4
        text="AB",
        category=ThreatCategory.DIRECT_THREAT,
        explanation="test",
    )

    with pytest.raises(ValidationError, match="UTF-16 offsets"):
        _response("😀AB", [span])

def test_analyzed_text_is_required_and_nonblank():
    base = {
        "overall_threat_level": ThreatLevel.NONE,
        "uncertainty_state": UncertaintyState.LOW,
        "explanation": "No threats.",
    }
    with pytest.raises(ValidationError, match="analyzed_text"):
        ThreatAssessmentResponse(**base)
    with pytest.raises(ValidationError, match="substantive"):
        _response("   ")


def test_disclaimer_must_match_exact_configured_text():
    with pytest.raises(ValidationError, match="exactly match"):
        _response("hello", disclaimer="A different but non-empty disclaimer")
    assert _response("hello").disclaimer == MANDATORY_AI_DISCLAIMER


def test_slice_utf16_rejects_out_of_range_offsets():
    with pytest.raises(ValueError, match="exceeds"):
        slice_utf16("hello", 0, 6)
    with pytest.raises(ValueError, match="non-negative"):
        slice_utf16("hello", -1, 2)
    with pytest.raises(ValueError, match="ordered"):
        slice_utf16("hello", 4, 2)


def test_slice_utf16_rejects_surrogate_pair_split():
    with pytest.raises(ValueError, match="surrogate pair"):
        slice_utf16("A😀B", 2, 3)
    with pytest.raises(ValueError, match="surrogate pair"):
        slice_utf16("A😀B", 1, 2)


def test_partition_spans_rejects_out_of_range_and_mismatched_spans():
    valid = _span_for("hello", 0, 5)
    with pytest.raises(ValueError, match="out-of-range"):
        partition_spans("hi", [valid])

    wrong_text = FlaggedSpan(
        start=0,
        end=2,
        start_utf16=0,
        end_utf16=2,
        text="no",
        category=ThreatCategory.DIRECT_THREAT,
        explanation="test",
    )
    with pytest.raises(ValueError, match="does not match"):
        partition_spans("ok", [wrong_text])


def test_partition_spans_preserves_valid_overlapping_unicode_text():
    text = "😀 ABCDEFG"
    first = _span_for(text, 2, 7)
    second = _span_for(text, 5, 9, ThreatCategory.BLACKMAIL)
    segments = partition_spans(text, [first, second])
    assert "".join(segment.text for segment in segments) == text
    assert any(
        segment.is_flagged
        and ThreatCategory.DIRECT_THREAT in segment.categories
        and ThreatCategory.BLACKMAIL in segment.categories
        for segment in segments
    )


def test_partition_spans_rejects_empty_text_with_spans():
    span = _span_for("x", 0, 1)
    with pytest.raises(ValueError, match="empty analyzed_text"):
        partition_spans("", [span])


class _RecordingRunner:
    def __init__(self, result: ToxicBertResult):
        self.result = result
        self.seen_text = None
        self.calls = 0

    def predict(self, text: str):
        self.calls += 1
        self.seen_text = text
        return self.result


def _fake_result(*, input_truncated=False, token_count=0, max_tokens=512):
    scores = {
        "toxic": 0.1,
        "severe_toxic": 0.01,
        "obscene": 0.02,
        "threat": 0.03,
        "insult": 0.04,
        "identity_hate": 0.01,
    }
    return ToxicBertResult(
        raw_logits={key: 0.0 for key in scores},
        raw_sigmoid_scores=scores,
        inference_time_ms=1.0,
        model_name="mock/toxic-bert",
        input_token_count=token_count,
        max_input_tokens=max_tokens,
        input_truncated=input_truncated,
    )


def test_hybrid_analyzer_uses_exact_baseline_analyzed_text():
    runner = _RecordingRunner(_fake_result())
    analyzer = ToxicBertThreatAnalyzer(
        config=ThreatAnalyzerConfig(enable_toxicity_check=True),
        model_runner=runner,  # type: ignore[arg-type]
    )
    raw = "   Look 📸\u200b send me 500 dollars or I will publish your private photos.   "
    response = analyzer.analyze(raw)
    assert runner.seen_text == response.analyzed_text
    assert response.analyzed_text == preprocess_text(raw).working_copy


def test_hybrid_analyzer_discloses_neural_token_truncation():
    runner = _RecordingRunner(_fake_result(input_truncated=True, token_count=700, max_tokens=512))
    analyzer = ToxicBertThreatAnalyzer(
        config=ThreatAnalyzerConfig(enable_toxicity_check=True),
        model_runner=runner,  # type: ignore[arg-type]
    )
    response = analyzer.analyze("A harmless message for token coverage testing.")
    assert response.toxic_bert_input_truncated is True
    assert response.toxic_bert_input_token_count == 700
    assert response.toxic_bert_max_input_tokens == 512
    assert "first 512 tokens" in response.explanation
    assert "baseline pattern analyzer evaluated the full analyzed_text" in response.explanation


def test_hybrid_analyzer_respects_disabled_toxicity_flag():
    runner = _RecordingRunner(_fake_result())
    analyzer = ToxicBertThreatAnalyzer(
        config=ThreatAnalyzerConfig(enable_toxicity_check=False),
        model_runner=runner,  # type: ignore[arg-type]
    )
    response = analyzer.analyze("This is an ordinary harmless message.")
    assert runner.calls == 0
    assert response.category_scores == {}
    assert response.toxic_bert_input_truncated is None
    assert "disabled by configuration" in response.explanation
