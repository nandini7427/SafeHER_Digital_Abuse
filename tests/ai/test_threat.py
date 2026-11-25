"""Comprehensive Unit Tests for SafeHER AI/ML Threat Assessment Foundation.

Tests do NOT depend on downloaded external models or network access.
All test scenarios use synthetic, un-fabricated data examples.
"""

import json
import shutil
import subprocess
import uuid
import pytest
from pydantic import ValidationError

from ai_core.config import MANDATORY_AI_DISCLAIMER, ThreatAnalyzerConfig
from ai_core.threat.analyzer import DevelopmentThreatAnalyzer
from ai_core.threat.highlighting import (
    HighlightSegment,
    partition_spans,
    render_highlighted_html,
)
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
    # ZWJ/ZWNJ are preserved when part of real text, but a string made only
    # of formatting characters is still rejected as non-substantive input.
    with pytest.raises(ValueError, match="only non-printable or invisible"):
        analyzer.analyze("\u200B\uFEFF\u202E\u2066\u2069")


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

# =============================================================================
# 12. Unicode and Indic-Script Preservation Tests
# =============================================================================

@pytest.mark.parametrize(
    "indic_text,joiner",
    [
        ("यह क्\u200Dषेत्र है।", "\u200D"),  # Devanagari with ZWJ
        ("यह क्\u200Cषेत्र है।", "\u200C"),  # Devanagari with ZWNJ
        ("ಇದು ಕ್\u200Dಷೇತ್ರವಾಗಿದೆ।", "\u200D"),  # Kannada with ZWJ
        ("ಇದು ಕ್\u200Cಷೇತ್ರವಾಗಿದೆ।", "\u200C"),  # Kannada with ZWNJ
    ],
)
def test_indic_script_joiners_are_preserved(indic_text, joiner):
    """Verify ZWJ/ZWNJ needed for Indic shaping survive preprocessing."""
    result = preprocess_text(indic_text)

    assert joiner in result.working_copy
    assert result.working_copy == indic_text
    assert result.original_text == indic_text


def test_zero_width_space_bom_and_bidi_controls_are_removed_from_working_copy():
    """Verify deceptive invisible spacing/BOM/bidi controls are removed only from working text."""
    original = "Hello\u200B\uFEFF\u202Eworld\u2066!\u2069"

    result = preprocess_text(original)

    assert result.working_copy == "Helloworld!"
    assert result.original_text == original
    assert any("zero-width" in warning.lower() or "bidirectional" in warning.lower()
               for warning in result.warnings)


def test_left_to_right_and_right_to_left_marks_are_preserved():
    """Verify LRM/RLM are retained rather than stripped from multilingual text."""
    original = "English \u200Eעברית\u200F text"

    result = preprocess_text(original)

    assert "\u200E" in result.working_copy
    assert "\u200F" in result.working_copy
    assert result.original_text == original


def test_original_evidence_is_unchanged_when_unicode_controls_are_sanitized():
    """Verify sanitization never modifies the pristine original evidence string."""
    original = "ಕನ್ನಡ\u200B ಮತ್ತು हिंदी\u202E"

    result = preprocess_text(original)

    assert result.original_text == original
    assert "\u200B" not in result.working_copy
    assert "\u202E" not in result.working_copy
    assert "ಕನ್ನಡ" in result.working_copy
    assert "हिंदी" in result.working_copy



# =============================================================================
# 13. Preprocessing Metadata Exposure Tests
# =============================================================================

def test_preprocessing_warnings_are_exposed_in_assessment(analyzer):
    """Verify preprocessing notices reach the response and explain the modified working copy."""
    original = "This is a harmless\u200B message."

    result = analyzer.analyze(original)

    assert result.preprocessing_warnings
    assert any("zero-width" in warning.lower() for warning in result.preprocessing_warnings)
    assert result.input_truncated is False
    assert "preprocessing_warnings" in result.explanation


def test_truncated_input_is_disclosed_in_assessment(analyzer):
    """Verify the response explicitly discloses when only a prefix was analyzed."""
    original = "This is harmless text. " * 700

    result = analyzer.analyze(original)

    assert result.input_truncated is True
    assert any("truncated" in warning.lower() for warning in result.preprocessing_warnings)
    assert "truncated" in result.explanation.lower()
    assert "complete original message was not analyzed" in result.explanation.lower()


def test_unmodified_input_has_no_preprocessing_warnings(analyzer):
    """Verify normal text does not receive spurious preprocessing warnings."""
    result = analyzer.analyze("This is a normal harmless message.")

    assert result.preprocessing_warnings == []
    assert result.input_truncated is False

# =============================================================================
# 14. Mandatory Disclaimer Validation Tests
# =============================================================================

def _minimal_assessment_response(**overrides):
    """Build a valid minimal assessment response for schema validation tests."""
    values = {
        "analyzed_text": "Sample clean analyzed text",
        "overall_threat_level": ThreatLevel.NONE,
        "uncertainty_state": UncertaintyState.LOW,
        "explanation": "No threat indicators detected.",
    }
    values.update(overrides)
    return ThreatAssessmentResponse(**values)


def test_mandatory_disclaimer_defaults_when_omitted():
    """Omitting the disclaimer must populate the configured mandatory text."""
    response = _minimal_assessment_response()

    assert response.disclaimer == MANDATORY_AI_DISCLAIMER
    assert response.disclaimer.strip()


@pytest.mark.parametrize("invalid_disclaimer", ["", "   ", "\n\t  "])
def test_mandatory_disclaimer_rejects_empty_or_whitespace(invalid_disclaimer):
    """Explicitly empty or whitespace-only disclaimer values must be rejected."""
    with pytest.raises(ValidationError, match="mandatory AI disclaimer"):
        _minimal_assessment_response(disclaimer=invalid_disclaimer)


def test_mandatory_disclaimer_cannot_be_cleared_after_creation():
    """The frozen disclaimer field cannot be accidentally cleared after validation."""
    response = _minimal_assessment_response()

    with pytest.raises(ValidationError):
        response.disclaimer = ""

    assert response.disclaimer == MANDATORY_AI_DISCLAIMER


# =============================================================================
# 15. Analyzed Text and UTF-16 Offset Consistency Tests
# =============================================================================

def _assert_span_offsets_and_js_slice(analyzed_text: str, span: FlaggedSpan) -> None:
    """Verify that Python code-point and UTF-16 offsets slice analyzed_text identically."""
    # 1. Python code-point slicing
    assert analyzed_text[span.start : span.end] == span.text

    # 2. UTF-16 code-unit offsets must be non-null integers
    assert span.start_utf16 is not None
    assert span.end_utf16 is not None
    assert span.start_utf16 >= 0
    assert span.end_utf16 >= span.start_utf16

    # 3. Exact UTF-16 byte-level slicing in Python
    utf16_bytes = analyzed_text.encode("utf-16-le")
    py_utf16_slice = utf16_bytes[span.start_utf16 * 2 : span.end_utf16 * 2].decode("utf-16-le")
    assert py_utf16_slice == span.text

    # 4. Live execution against Node.js if available in the runtime environment
    node_bin = shutil.which("node")
    if node_bin:
        js_code = (
            f"const text = {json.dumps(analyzed_text)};\n"
            f"const slice = text.slice({span.start_utf16}, {span.end_utf16});\n"
            f"if (slice !== {json.dumps(span.text)}) {{\n"
            f"  console.error('JS mismatch: expected', {json.dumps(span.text)}, 'got', slice);\n"
            f"  process.exit(1);\n"
            f"}}\n"
        )
        proc = subprocess.run([node_bin, "-e", js_code], capture_output=True, text=True)
        assert proc.returncode == 0, f"Node.js slice mismatch: {proc.stderr}"


@pytest.mark.parametrize(
    "raw_input,expected_category",
    [
        # Leading and trailing whitespace
        (
            "   Send me 500 dollars or I will publish your private photos.   ",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
        # Multiple spaces and newlines
        (
            "Hey!\n\n\n\nSend me 500 dollars     or I will publish your private photos.",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
        # Zero-width spaces and bidi controls
        (
            "Alert:\u200b\u202aSend me 500 dollars or I will publish your private photos.\u202c",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
        # Hindi text with threat
        (
            "नमस्ते! Send me 500 dollars or I will publish your private photos.",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
        # Kannada text with threat
        (
            "ಕನ್ನಡ ಸಂದೇಶ: Send me 500 dollars or I will publish your private photos.",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
        # Precomposed Indic nukta character
        (
            "\u095eाइल: Send me 500 dollars or I will publish your private photos.",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
        # Emoji before a flagged threat
        (
            "📸 Send me 500 dollars or I will publish your private photos.",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
        # Multiple supplementary Unicode characters (complex emojis)
        (
            "Look 📸🚨⚠️🔪 Send me 500 dollars or I will publish your private photos.",
            ThreatCategory.FINANCIAL_EXTORTION,
        ),
    ],
)
def test_analyzed_text_and_utf16_span_consistency(analyzer, raw_input, expected_category):
    """Verify that every span matches analyzed_text in Python and via UTF-16 in JavaScript."""
    # Ensure raw_input evidence is not modified
    input_copy = str(raw_input)
    res = analyzer.analyze(raw_input)
    assert raw_input == input_copy, "Original input text must not be mutated"

    # analyzed_text must be present, non-empty, and match working copy
    assert res.analyzed_text
    prep = preprocess_text(raw_input)
    assert res.analyzed_text == prep.working_copy

    # At least one span detected
    assert len(res.flagged_spans) > 0
    categories = {s.category for s in res.flagged_spans}
    assert expected_category in categories

    # Every single span must slice cleanly in Python and via UTF-16
    for span in res.flagged_spans:
        _assert_span_offsets_and_js_slice(res.analyzed_text, span)


def test_analyzed_text_truncated_input(analyzer):
    """Verify truncation sets input_truncated and span offsets correctly match truncated analyzed_text."""
    short_config = ThreatAnalyzerConfig(max_input_length=60)
    constrained_analyzer = DevelopmentThreatAnalyzer(config=short_config)

    # 40 chars prefix + threat that fits within 60 chars
    text = "Intro padding... Pay me 500 dollars or else I will expose you far far far later in the text."
    res = constrained_analyzer.analyze(text)

    assert res.input_truncated is True
    assert len(res.analyzed_text) <= 60
    assert any("truncated" in w.lower() for w in res.preprocessing_warnings)

    for span in res.flagged_spans:
        _assert_span_offsets_and_js_slice(res.analyzed_text, span)


def test_original_evidence_unmodified_by_analyzer(analyzer):
    """Verify that analyzer never modifies, normalizes, or overwrites original input evidence."""
    messy_evidence = "   \u200b\u202aDirty evidence with spaces   \n\n\nand controls\x00   "
    evidence_backup = str(messy_evidence)

    try:
        analyzer.analyze(messy_evidence)
    except Exception:
        pass

    assert messy_evidence == evidence_backup


# =============================================================================
# 16. Mandatory UTF-16 Offsets and Overlapping Span Highlighting Tests
# =============================================================================

def test_flagged_span_rejects_missing_or_null_utf16_offsets():
    """FlaggedSpan requires non-null, non-negative start_utf16 and end_utf16."""
    # 1. Missing start_utf16
    with pytest.raises(ValidationError, match="start_utf16"):
        FlaggedSpan(
            start=0,
            end=5,
            end_utf16=5,
            text="hello",
            category=ThreatCategory.DIRECT_THREAT,
            explanation="test",
        )

    # 2. Missing end_utf16
    with pytest.raises(ValidationError, match="end_utf16"):
        FlaggedSpan(
            start=0,
            end=5,
            start_utf16=0,
            text="hello",
            category=ThreatCategory.DIRECT_THREAT,
            explanation="test",
        )

    # 3. Explicit None for start_utf16
    with pytest.raises(ValidationError):
        FlaggedSpan(
            start=0,
            end=5,
            start_utf16=None,  # type: ignore
            end_utf16=5,
            text="hello",
            category=ThreatCategory.DIRECT_THREAT,
            explanation="test",
        )

    # 4. Negative offset rejected
    with pytest.raises(ValidationError):
        FlaggedSpan(
            start=0,
            end=5,
            start_utf16=0,
            end_utf16=-1,
            text="hello",
            category=ThreatCategory.DIRECT_THREAT,
            explanation="test",
        )

    # 5. Valid mandatory fields succeed and serialize properly
    span = FlaggedSpan(
        start=0,
        end=5,
        start_utf16=0,
        end_utf16=5,
        text="hello",
        category=ThreatCategory.DIRECT_THREAT,
        explanation="test",
    )
    dumped = span.model_dump()
    assert dumped["start_utf16"] == 0
    assert dumped["end_utf16"] == 5

    json_str = span.model_dump_json()
    assert '"start_utf16":0' in json_str or '"start_utf16": 0' in json_str
    assert '"end_utf16":5' in json_str or '"end_utf16": 5' in json_str


def test_analyzer_generated_spans_strictly_contain_mandatory_utf16_offsets(analyzer):
    """Every span created by the analyzer must contain valid, non-null integer UTF-16 offsets."""
    text = "Send me 500 dollars or I will track you down and hurt you."
    res = analyzer.analyze(text)

    assert len(res.flagged_spans) > 0
    for span in res.flagged_spans:
        assert isinstance(span.start_utf16, int)
        assert isinstance(span.end_utf16, int)
        assert span.start_utf16 >= 0
        assert span.end_utf16 >= span.start_utf16

        # Serialization includes mandatory fields
        dumped = span.model_dump()
        assert "start_utf16" in dumped
        assert "end_utf16" in dumped
        assert isinstance(dumped["start_utf16"], int)
        assert isinstance(dumped["end_utf16"], int)


def test_highlight_adjacent_spans():
    """Adjacent spans sharing a boundary must be partitioned without gaps or overlap."""
    text = "Pay money. Leave now."
    span1 = FlaggedSpan(
        start=0,
        end=9,
        start_utf16=0,
        end_utf16=9,
        text="Pay money",
        category=ThreatCategory.FINANCIAL_EXTORTION,
        explanation="Payment demand",
    )
    span2 = FlaggedSpan(
        start=9,
        end=20,
        start_utf16=9,
        end_utf16=20,
        text=". Leave now",
        category=ThreatCategory.COERCION,
        explanation="Forced exit",
    )

    segments = partition_spans(text, [span1, span2])
    # Must preserve exact text
    reconstructed = "".join(seg.text for seg in segments)
    assert reconstructed == text

    # Must produce 2 contiguous flagged segments and 1 trailing segment ('.')
    assert len(segments) == 3
    assert segments[0].text == "Pay money"
    assert segments[0].categories == [ThreatCategory.FINANCIAL_EXTORTION]
    assert segments[1].text == ". Leave now"
    assert segments[1].categories == [ThreatCategory.COERCION]
    assert segments[2].text == "."
    assert not segments[2].is_flagged

    html_out = render_highlighted_html(text, [span1, span2])
    assert "Pay money" in html_out
    assert ". Leave now" in html_out


def test_highlight_nested_spans():
    """Nested spans (one inside another) must be partitioned without duplicate text."""
    text = "or I will publish your private photos"
    # Outer span: full phrase (BLACKMAIL)
    outer_span = FlaggedSpan(
        start=0,
        end=37,
        start_utf16=0,
        end_utf16=37,
        text=text,
        category=ThreatCategory.BLACKMAIL,
        explanation="Conditional ultimatum",
    )
    # Inner span: 'publish your private photos' (EXPOSURE_THREAT)
    inner_span = FlaggedSpan(
        start=10,
        end=37,
        start_utf16=10,
        end_utf16=37,
        text="publish your private photos",
        category=ThreatCategory.EXPOSURE_THREAT,
        explanation="Media distribution threat",
    )

    segments = partition_spans(text, [outer_span, inner_span])
    reconstructed = "".join(seg.text for seg in segments)
    assert reconstructed == text, f"Character loss/duplication: {reconstructed!r} != {text!r}"

    assert len(segments) == 2
    # Prefix: 'or I will ' has only BLACKMAIL
    assert segments[0].text == "or I will "
    assert segments[0].categories == [ThreatCategory.BLACKMAIL]

    # Nested section: 'publish your private photos' has BOTH categories
    assert segments[1].text == "publish your private photos"
    assert ThreatCategory.BLACKMAIL in segments[1].categories
    assert ThreatCategory.EXPOSURE_THREAT in segments[1].categories

    html_out = render_highlighted_html(text, [outer_span, inner_span])
    assert "threat-blackmail" in html_out
    assert "threat-exposure_threat" in html_out
    # Text must appear exactly once in HTML content (excluding tags)
    assert html_out.count("publish your private photos") == 1


def test_highlight_partially_overlapping_spans():
    """Partially overlapping spans must split into 3 segments preserving every character."""
    text = "ABCD EFGH IJKL"
    # Span A covers 'ABCD EF' (0 to 7)
    span_a = FlaggedSpan(
        start=0,
        end=7,
        start_utf16=0,
        end_utf16=7,
        text="ABCD EF",
        category=ThreatCategory.HARASSMENT_INTIMIDATION,
        explanation="Intimidation prefix",
    )
    # Span B covers 'EFGH IJ' (5 to 12)
    span_b = FlaggedSpan(
        start=5,
        end=12,
        start_utf16=5,
        end_utf16=12,
        text="EFGH IJ",
        category=ThreatCategory.DIRECT_THREAT,
        explanation="Threat suffix",
    )

    segments = partition_spans(text, [span_a, span_b])
    reconstructed = "".join(seg.text for seg in segments)
    assert reconstructed == text

    # Intervals: [0, 5], [5, 7], [7, 12], [12, 14]
    assert len(segments) == 4
    assert segments[0].text == "ABCD "
    assert segments[0].categories == [ThreatCategory.HARASSMENT_INTIMIDATION]

    assert segments[1].text == "EF"
    assert ThreatCategory.HARASSMENT_INTIMIDATION in segments[1].categories
    assert ThreatCategory.DIRECT_THREAT in segments[1].categories

    assert segments[2].text == "GH IJ"
    assert segments[2].categories == [ThreatCategory.DIRECT_THREAT]

    assert segments[3].text == "KL"
    assert not segments[3].is_flagged


def test_highlight_multiple_categories_on_overlapping_text():
    """Multiple categories on the exact same text span are preserved in classes and data attributes."""
    text = "I will ruin you"
    span1 = FlaggedSpan(
        start=0,
        end=15,
        start_utf16=0,
        end_utf16=15,
        text=text,
        category=ThreatCategory.DIRECT_THREAT,
        explanation="Physical violence",
    )
    span2 = FlaggedSpan(
        start=0,
        end=15,
        start_utf16=0,
        end_utf16=15,
        text=text,
        category=ThreatCategory.BLACKMAIL,
        explanation="Extortion coercion",
    )

    html_out = render_highlighted_html(text, [span1, span2])
    assert "threat-direct_threat" in html_out
    assert "threat-blackmail" in html_out
    assert 'data-categories="direct_threat,blackmail"' in html_out
    assert "I will ruin you" in html_out
    # String must not be duplicated
    assert html_out.count("I will ruin you") == 1


def test_highlight_overlapping_spans_with_emojis_and_js_execution():
    """Overlapping spans preceded by emojis match identically in Python and JavaScript."""
    text = "Look 📸🚨 Send me 500 dollars or I will publish your private photos."
    # Emojis 📸🚨 consume 4 UTF-16 code units (2 surrogate pairs)
    # text[:7] is 'Look 📸🚨 ' (Python len 7, UTF-16 len 9)
    prefix_u16 = len(text[: text.find("Send me")].encode("utf-16-le")) // 2

    # Span 1: 'Send me 500 dollars'
    s1_u16 = prefix_u16
    e1_u16 = s1_u16 + len("Send me 500 dollars")
    # Span 2 (overlapping): '500 dollars or I will publish'
    s2_u16 = s1_u16 + len("Send me ")
    e2_u16 = s2_u16 + len("500 dollars or I will publish")

    span1 = FlaggedSpan(
        start=text.find("Send me 500 dollars"),
        end=text.find("Send me 500 dollars") + len("Send me 500 dollars"),
        start_utf16=s1_u16,
        end_utf16=e1_u16,
        text="Send me 500 dollars",
        category=ThreatCategory.FINANCIAL_EXTORTION,
        explanation="Money demand",
    )
    span2 = FlaggedSpan(
        start=text.find("500 dollars or I will publish"),
        end=text.find("500 dollars or I will publish") + len("500 dollars or I will publish"),
        start_utf16=s2_u16,
        end_utf16=e2_u16,
        text="500 dollars or I will publish",
        category=ThreatCategory.BLACKMAIL,
        explanation="Ultimatum link",
    )

    py_segments = partition_spans(text, [span1, span2])
    py_reconstructed = "".join(seg.text for seg in py_segments)
    assert py_reconstructed == text

    # Node.js live verification if available
    node_bin = shutil.which("node")
    if node_bin:
        js_script = f"""
const text = {json.dumps(text)};
const spans = [
    {{ start_utf16: {s1_u16}, end_utf16: {e1_u16}, category: "financial_extortion", explanation: "Money demand" }},
    {{ start_utf16: {s2_u16}, end_utf16: {e2_u16}, category: "blackmail", explanation: "Ultimatum link" }}
];

const boundarySet = new Set([0, text.length]);
for (const s of spans) {{
    boundarySet.add(s.start_utf16);
    boundarySet.add(s.end_utf16);
}}
const sorted = Array.from(boundarySet).sort((a, b) => a - b);
let reconstructed = "";
for (let i = 0; i < sorted.length - 1; i++) {{
    reconstructed += text.slice(sorted[i], sorted[i + 1]);
}}
if (reconstructed !== text) {{
    console.error("Mismatch in JS! Got:", reconstructed);
    process.exit(1);
}}
"""
        proc = subprocess.run([node_bin, "-e", js_script], capture_output=True, text=True)
        assert proc.returncode == 0, f"JS execution failed: {proc.stderr}"


def test_highlight_exact_character_preservation_and_html_escaping():
    """Text with HTML characters is safely escaped and completely reconstructed."""
    unsafe_text = "Look: <script>alert('xss')</script> & pay me $500 > now!"
    span = FlaggedSpan(
        start=unsafe_text.find("pay me $500"),
        end=unsafe_text.find("pay me $500") + len("pay me $500"),
        start_utf16=len(unsafe_text[: unsafe_text.find("pay me $500")].encode("utf-16-le")) // 2,
        end_utf16=(len(unsafe_text[: unsafe_text.find("pay me $500")].encode("utf-16-le")) // 2) + len("pay me $500"),
        text="pay me $500",
        category=ThreatCategory.FINANCIAL_EXTORTION,
        explanation="Extortion <script> dangerous payload",
    )

    segments = partition_spans(unsafe_text, [span])
    reconstructed = "".join(seg.text for seg in segments)
    assert reconstructed == unsafe_text

    html_out = render_highlighted_html(unsafe_text, [span])
    # No raw script tags
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "&gt;" in html_out
    assert "&amp;" in html_out
    assert "pay me $500" in html_out
    # Attribute payload escaped
    assert "<script>" not in html_out.split('title="')[1].split('"')[0]


# =============================================================================
# 17. Stalking and Harassment Threat Triage Regression Tests
# =============================================================================

@pytest.mark.parametrize(
    "stalking_phrase",
    [
        "I am watching you right now, you will never be safe anywhere you go.",  # INTIM_01
        "I know where you live.",
        "I found where you live.",
        "I see you right now.",
        "You can't hide from me.",
        "Nowhere is safe.",
    ],
)
def test_stalking_and_surveillance_intimidation_receives_medium_threat_level(analyzer, stalking_phrase):
    """Verify explicit stalking and surveillance intimidation receive at least MEDIUM threat level."""
    res: ThreatAssessmentResponse = analyzer.analyze(stalking_phrase)

    assert res.overall_threat_level == ThreatLevel.MEDIUM
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.HARASSMENT_INTIMIDATION in detected
    assert res.is_calibrated is False
    assert res.confidence_score is None
    assert "stalking/intimidation" in res.explanation


@pytest.mark.parametrize(
    "benign_phrase",
    [
        "I am watching a movie right now with friends.",
        "I see you tomorrow at the office for lunch.",
        "I found where you parked your car.",
        "Nowhere is as peaceful as this garden.",
        "You will never believe what a great time we had yesterday.",
    ],
)
def test_benign_surveillance_lookalike_phrases_not_flagged(analyzer, benign_phrase):
    """Verify benign everyday statements with similar words are not falsely flagged as harassment."""
    res: ThreatAssessmentResponse = analyzer.analyze(benign_phrase)

    assert res.overall_threat_level == ThreatLevel.NONE
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.HARASSMENT_INTIMIDATION not in detected


@pytest.mark.parametrize(
    "urgency_phrase",
    [
        "You have 24 hours to review the document.",
        "Last chance to submit your feedback before the release.",
        "Please respond before the timer runs out.",
    ],
)
def test_urgency_pressure_alone_remains_low_threat_level(analyzer, urgency_phrase):
    """Verify isolated urgency deadlines without threats or coercion remain at LOW threat level."""
    res: ThreatAssessmentResponse = analyzer.analyze(urgency_phrase)

    assert res.overall_threat_level == ThreatLevel.LOW
    detected = {span.category for span in res.flagged_spans}
    assert ThreatCategory.URGENCY_PRESSURE in detected
    assert ThreatCategory.HARASSMENT_INTIMIDATION not in detected


def test_evaluation_framework_recognizes_intim_01_as_confirmed_threat(analyzer):
    """Verify benchmark sample INTIM_01 is recognized as a confirmed threat (no false negative)."""
    from ai_core.evaluation.dataset import BENCHMARK_DATASET
    from ai_core.evaluation.evaluator import evaluate_analyzer

    report = evaluate_analyzer(analyzer, samples=BENCHMARK_DATASET)

    # All confirmed threats in benchmark must be detected with zero false negatives
    assert report.threat_intent_metrics.false_negatives == 0
    assert report.threat_intent_metrics.recall == 1.0

    # Specifically verify INTIM_01 sample
    intim_sample = next(s for s in BENCHMARK_DATASET if s.sample_id == "INTIM_01")
    intim_res = analyzer.analyze(intim_sample.text)
    assert intim_res.overall_threat_level in (
        ThreatLevel.MEDIUM,
        ThreatLevel.HIGH,
        ThreatLevel.SEVERE,
    )
