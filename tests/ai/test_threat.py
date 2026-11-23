"""Comprehensive Unit Tests for SafeHER AI/ML Threat Assessment Foundation.

Tests do NOT depend on downloaded external models or network access.
All test scenarios use synthetic, un-fabricated data examples.
"""

import uuid
import pytest

from ai_core.threat.analyzer import DevelopmentThreatAnalyzer
from ai_core.threat.preprocessing import PreprocessedText, preprocess_text
from ai_core.threat.schemas import (
    FlaggedSpan,
    ThreatAssessmentResponse,
    ThreatCategory,
    ThreatIndicator,
    ThreatLevel,
    UncertaintyState,
)


@pytest.fixture
def analyzer() -> DevelopmentThreatAnalyzer:
    """Provides an instance of the baseline DevelopmentThreatAnalyzer."""
    return DevelopmentThreatAnalyzer()


# =============================================================================
# 1. Empty and Invalid Text Tests
# =============================================================================

def test_empty_text_raises_validation_error(analyzer):
    """Verify that empty, None, or whitespace-only inputs are rejected."""
    with pytest.raises(ValueError, match="empty or whitespace"):
        analyzer.analyze("")

    with pytest.raises(ValueError, match="empty or whitespace"):
        analyzer.analyze("    \n\t   ")

    with pytest.raises(ValueError, match="valid non-null string"):
        analyzer.analyze(None)  # type: ignore


def test_invisible_only_text_raises_validation_error(analyzer):
    """Verify text composed entirely of invisible zero-width characters is rejected."""
    with pytest.raises(ValueError, match="only non-printable or invisible"):
        analyzer.analyze("\u200B\u200C\u200D\uFEFF")


# =============================================================================
# 2. Normal Harmless Text Tests
# =============================================================================

def test_normal_harmless_text(analyzer):
    """Verify that standard benign communication yields threat level NONE with no flagged spans."""
    text = (
        "Hi Alex, thank you for sending over the project schedule. "
        "Let's review the presentation slides together at 3 PM tomorrow. Have a great day!"
    )
    res: ThreatAssessmentResponse = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    assert len(res.threat_indicators) == 0
    assert res.is_calibrated is False
    assert res.confidence_score is None
    assert res.uncertainty_state == UncertaintyState.LOW
    assert "No threat indicators" in res.explanation


# =============================================================================
# 3. Synthetic Blackmail & Extortion Tests
# =============================================================================

def test_synthetic_blackmail_and_extortion(analyzer):
    """Verify detection of synthetic blackmail, financial extortion, and exposure threats."""
    synthetic_blackmail = (
        "Unless you pay 500 dollars before the timer runs out, I will leak your private photos "
        "to your family and employer. You have 24 hours. Last chance."
    )
    res: ThreatAssessmentResponse = analyzer.analyze(synthetic_blackmail)

    assert res.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    assert res.is_calibrated is False
    assert res.confidence_score is None

    categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.BLACKMAIL in categories
    assert ThreatCategory.FINANCIAL_EXTORTION in categories
    assert ThreatCategory.EXPOSURE_THREAT in categories
    assert ThreatCategory.URGENCY_PRESSURE in categories
    assert len(res.flagged_spans) >= 3


# =============================================================================
# 4. Direct Threat Tests
# =============================================================================

def test_direct_threat_example(analyzer):
    """Verify detection of direct threats of physical violence."""
    direct_threat = (
        "I know where you live. I will track you down and hurt you. You are going to die."
    )
    res: ThreatAssessmentResponse = analyzer.analyze(direct_threat)

    assert res.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.DIRECT_THREAT in categories
    assert ThreatCategory.HARASSMENT_INTIMIDATION in categories

    # Verify character offsets match exact substrings in working copy
    for span in res.flagged_spans:
        assert direct_threat[span.start:span.end] == span.text


# =============================================================================
# 5. Figurative Language Tests
# =============================================================================

@pytest.mark.parametrize(
    "figurative_sentence",
    [
        "You are killing it on stage tonight, what an incredible presentation!",
        "I was dead tired yesterday and dying of laughter watching comedy with friends.",
        "Our debate team destroyed the competition in the finals and slayed the performance.",
        "I bombed the test yesterday so I'll have to bite the bullet and study harder.",
    ],
)
def test_figurative_language_does_not_trigger_threat(analyzer, figurative_sentence):
    """Verify that common violent-sounding idioms do NOT trigger false threat alerts."""
    res: ThreatAssessmentResponse = analyzer.analyze(figurative_sentence)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    assert "figurative" in res.explanation.lower() or "idiomatic" in res.explanation.lower()


# =============================================================================
# 6. Insufficient Context Tests
# =============================================================================

@pytest.mark.parametrize(
    "ambiguous_text",
    [
        "Wait for tomorrow.",
        "You will see.",
        "Just wait.",
        "Watch out.",
    ],
)
def test_insufficient_context_flagged(analyzer, ambiguous_text):
    """Verify that ambiguous short statements are categorized with INSUFFICIENT_CONTEXT."""
    res: ThreatAssessmentResponse = analyzer.analyze(ambiguous_text)

    assert res.overall_threat_level == ThreatLevel.LOW
    assert res.uncertainty_state == UncertaintyState.INSUFFICIENT_CONTEXT
    assert "insufficient" in res.explanation.lower()


# =============================================================================
# 7. Long Text Input & Truncation Tests
# =============================================================================

def test_long_text_input_preprocessing():
    """Verify that input exceeding max_length is cleanly truncated while preserving the original."""
    long_original = "word " * 3000  # ~15,000 characters
    max_len = 5000

    preprocessed = preprocess_text(long_original, max_length=max_len)

    # Working copy must be constrained to max_len
    assert len(preprocessed.working_copy) <= max_len
    assert preprocessed.is_truncated is True
    assert any("truncated" in w for w in preprocessed.warnings)

    # Original evidence must remain completely pristine
    assert len(preprocessed.original_text) == len(long_original)
    assert preprocessed.original_text == long_original


# =============================================================================
# 8. Correct Structured Output & Disclaimer Tests
# =============================================================================

def test_structured_output_schema_and_disclaimer(analyzer):
    """Verify complete Pydantic v2 schema validation, calibration flag, and mandatory disclaimer."""
    sample_text = "Unless you do what I say I will post your pictures."
    res: ThreatAssessmentResponse = analyzer.analyze(sample_text)

    # Validate UUID
    assert isinstance(res.assessment_id, uuid.UUID)

    # Validate calibration rules: must not present uncalibrated heuristics as probabilities
    assert res.is_calibrated is False
    assert res.confidence_score is None
    assert res.category_scores == {}

    # Validate mandatory disclaimer
    assert res.disclaimer is not None
    assert "DISCLAIMER" in res.disclaimer
    assert "legal" in res.disclaimer.lower()
    assert "emergency" in res.disclaimer.lower()

    # Pydantic v2 serialization test
    json_output = res.model_dump_json()
    assert isinstance(json_output, str)
    assert "assessment_id" in json_output
    assert "is_calibrated" in json_output
