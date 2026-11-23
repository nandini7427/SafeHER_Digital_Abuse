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


# =============================================================================
# 9. Extortion Detection Remediation Tests (Positive & Negative Scenarios)
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
        (
            "Transfer the money, otherwise I will expose your private videos.",
            {
                ThreatCategory.BLACKMAIL,
                ThreatCategory.EXPOSURE_THREAT,
                ThreatCategory.FINANCIAL_EXTORTION,
            },
        ),
        (
            "If you report me, I will send your private photos to your family.",
            {
                ThreatCategory.BLACKMAIL,
                ThreatCategory.EXPOSURE_THREAT,
            },
        ),
    ],
)
def test_remediated_extortion_positive_cases(analyzer, text, expected_categories):
    """Verify detection of previously missed extortion, exposure, and conditional threat structures."""
    res: ThreatAssessmentResponse = analyzer.analyze(text)

    # Must be categorized as HIGH or SEVERE threat
    assert res.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.SEVERE]
    assert res.is_calibrated is False
    assert res.confidence_score is None

    detected = {span.category for span in res.flagged_spans}
    for cat in expected_categories:
        assert cat in detected, f"Expected category {cat} missing from detected: {detected}"

    # Character offsets must match exact substrings in working text copy
    preprocessed = preprocess_text(text)
    for span in res.flagged_spans:
        assert preprocessed.working_copy[span.start : span.end] == span.text


@pytest.mark.parametrize(
    "harmless_text",
    [
        "Please send me 500 dollars for the concert tickets.",
        "I will publish my new photography collection tomorrow.",
        "Please transfer the money for our group dinner.",
        "I sent 500 dollars for the rent yesterday. Feel free to publish your vacation photos.",
        "Could you please transfer 20 dollars to split the lunch bill?",
    ],
)
def test_ordinary_requests_and_harmless_photos_negative_cases(analyzer, harmless_text):
    """Verify ordinary payment requests and harmless photography statements are NOT elevated to threat alerts."""
    res: ThreatAssessmentResponse = analyzer.analyze(harmless_text)

    # Must NOT be classified as extortion or threat
    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    assert len(res.threat_indicators) == 0
    assert ThreatCategory.FINANCIAL_EXTORTION not in [s.category for s in res.flagged_spans]
    assert ThreatCategory.EXPOSURE_THREAT not in [s.category for s in res.flagged_spans]
    assert ThreatCategory.BLACKMAIL not in [s.category for s in res.flagged_spans]


# =============================================================================
# 10. Milestone 2 Final Review Regression Tests (Context Separation & Severity)
# =============================================================================

def test_unrelated_payment_and_separate_exposure_threat(analyzer):
    """Regression Test 1: Unrelated concert-ticket payment and separate exposure threat.

    Verifies that a benign concert ticket payment mention is NOT categorized as financial
    extortion, while the separate blackmail and exposure threat is correctly detected at HIGH severity.
    """
    text = "Please send me 500 dollars for the concert tickets. If you report me, I will send your private photos to your family."
    res = analyzer.analyze(text)

    # Severity must be HIGH (blackmail + exposure threat, but NO extortion)
    assert res.overall_threat_level == ThreatLevel.HIGH
    assert res.uncertainty_state == UncertaintyState.LOW
    assert res.is_calibrated is False

    detected_categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.BLACKMAIL in detected_categories
    assert ThreatCategory.EXPOSURE_THREAT in detected_categories
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected_categories

    flagged_texts = [s.text for s in res.flagged_spans]
    assert "If you report me" in flagged_texts
    assert "send your private photos to your family" in flagged_texts
    assert not any("500 dollars" in t for t in flagged_texts)


def test_conditional_financial_blackmail_ultimatum(analyzer):
    """Regression Test 2: Original conditional financial blackmail example.

    Verifies full detection of financial extortion, blackmail, exposure threat, and urgency
    at SEVERE threat level.
    """
    text = "Send me 500 dollars by midnight or I will publish your private photos."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.SEVERE
    assert res.uncertainty_state == UncertaintyState.LOW
    assert res.is_calibrated is False

    detected_categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION in detected_categories
    assert ThreatCategory.BLACKMAIL in detected_categories
    assert ThreatCategory.EXPOSURE_THREAT in detected_categories
    assert ThreatCategory.URGENCY_PRESSURE in detected_categories

    flagged_texts = [s.text for s in res.flagged_spans]
    assert any("500 dollars" in t for t in flagged_texts)
    assert any("publish your private photos" in t for t in flagged_texts)


def test_ordinary_payment_with_unrelated_harmless_publication(analyzer):
    """Regression Test 3: Ordinary payment with unrelated harmless publication statement.

    Verifies that combining dinner payment with photography publication does NOT trigger threat alerts.
    """
    text = "Please transfer 500 dollars for the dinner. I will publish my new photography collection tomorrow."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    assert len(res.threat_indicators) == 0
    assert res.uncertainty_state == UncertaintyState.LOW


def test_exposure_threat_without_financial_demand(analyzer):
    """Regression Test 4: Exposure threat without a financial demand.

    Verifies that an unconsented exposure threat alone is categorized at MEDIUM severity.
    """
    text = "I will leak your intimate videos to everyone."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.MEDIUM
    assert res.uncertainty_state == UncertaintyState.MODERATE

    detected_categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.EXPOSURE_THREAT in detected_categories
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected_categories
    assert ThreatCategory.BLACKMAIL not in detected_categories


def test_direct_threat_combined_with_extortion(analyzer):
    """Regression Test 5: Direct threat combined with extortion.

    Verifies that combining direct bodily harm with financial extortion yields SEVERE severity.
    """
    text = "Pay me 1000 dollars or I will track you down and hurt you."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.SEVERE
    assert res.uncertainty_state == UncertaintyState.LOW

    detected_categories = {span.category for span in res.flagged_spans}
    assert ThreatCategory.DIRECT_THREAT in detected_categories
    assert ThreatCategory.FINANCIAL_EXTORTION in detected_categories
    assert ThreatCategory.BLACKMAIL in detected_categories


# =============================================================================
# 11. Final Edge-Case Remediation Tests (Deadlines & Scoped Ultimatum Matching)
# =============================================================================

@pytest.mark.parametrize(
    "text",
    [
        "Please pay me before midnight for the rent.",
        "Please complete the wire transfer before Friday for invoice 204.",
        "Please transfer the requested cryptocurrency to our registered vendor wallet.",
    ],
)
def test_deadlines_alone_not_extortion_negative_cases(analyzer, text):
    """Issue 1: Verify routine payment requests with deadlines alone are NOT classified as extortion or urgency abuse."""
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    assert len(res.threat_indicators) == 0
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected
    assert ThreatCategory.URGENCY_PRESSURE not in detected
    assert ThreatCategory.BLACKMAIL not in detected


@pytest.mark.parametrize(
    "text",
    [
        "Pay the ransom of 5000 dollars.",
        "Pay 1000 dollars to stop the leak.",
    ],
)
def test_genuine_extortion_demands_using_ransom_preserved(analyzer, text):
    """Issue 1: Verify genuine extortion demands using ransom or duress language are preserved."""
    res = analyzer.analyze(text)

    assert res.overall_threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.SEVERE]
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION in detected


def test_scoped_ultimatum_matching_unrelated_publication(analyzer):
    """Issue 2: Verify forward consequence search does not match unrelated statement far later or benign publication."""
    text = "Send me 500 dollars for concert tickets. Or I will publish my private photography collection tomorrow."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.NONE
    assert len(res.flagged_spans) == 0
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected
    assert ThreatCategory.EXPOSURE_THREAT not in detected
    assert ThreatCategory.BLACKMAIL not in detected


def test_scoped_ultimatum_matching_direct_consequence(analyzer):
    """Issue 2: Verify forward consequence search detects direct conditional extortion consequence."""
    text = "Send me 500 dollars, or I will publish your private photos."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.SEVERE
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.FINANCIAL_EXTORTION in detected
    assert ThreatCategory.BLACKMAIL in detected
    assert ThreatCategory.EXPOSURE_THREAT in detected


def test_scoped_ultimatum_matching_separate_sentence_threat(analyzer):
    """Issue 2: Verify routine dinner payment is NOT extortion when threat in subsequent sentence is condition on reporting."""
    text = "Please transfer 500 dollars for dinner. If you report me, I will share your private photos."
    res = analyzer.analyze(text)

    assert res.overall_threat_level == ThreatLevel.HIGH
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.BLACKMAIL in detected
    assert ThreatCategory.EXPOSURE_THREAT in detected
    assert ThreatCategory.FINANCIAL_EXTORTION not in detected
