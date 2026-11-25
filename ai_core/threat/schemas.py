"""Pydantic v2 schemas for SafeHER threat assessment."""

from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_core.config import MANDATORY_AI_DISCLAIMER


def _utf16_code_units(value: str) -> int:
    """Return JavaScript-compatible UTF-16 code-unit length for valid Unicode text."""
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError as exc:
        raise ValueError("Text contains an isolated Unicode surrogate code point.") from exc


class ThreatCategory(str, Enum):
    """Documented taxonomy of digital abuse threat intentions."""

    DIRECT_THREAT = "direct_threat"
    BLACKMAIL = "blackmail"
    FINANCIAL_EXTORTION = "financial_extortion"
    EXPOSURE_THREAT = "exposure_threat"
    COERCION = "coercion"
    HARASSMENT_INTIMIDATION = "harassment_intimidation"
    URGENCY_PRESSURE = "urgency_pressure"


class AuxiliarySignal(str, Enum):
    """Auxiliary NLP signals kept strictly separate from threat intent."""

    TOXICITY = "toxicity"
    INSULT = "insult"
    PROFANITY = "profanity"


class ThreatLevel(str, Enum):
    """Qualitative threat assessment level."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class UncertaintyState(str, Enum):
    """Categorical estimation of analyzer confidence and contextual sufficiency."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class FlaggedSpan(BaseModel):
    """Specific substring flagged as supporting evidence for a threat indicator.

    ``start``/``end`` are Python Unicode code-point offsets. ``start_utf16``/
    ``end_utf16`` are JavaScript UTF-16 code-unit offsets. All four offsets refer
    to ``ThreatAssessmentResponse.analyzed_text``.
    """

    start: int = Field(..., ge=0, description="Start code-point offset in analyzed_text")
    end: int = Field(..., ge=0, description="End code-point offset in analyzed_text")
    start_utf16: int = Field(..., ge=0, description="Start UTF-16 code-unit offset in analyzed_text")
    end_utf16: int = Field(..., ge=0, description="End UTF-16 code-unit offset in analyzed_text")
    text: str = Field(..., min_length=1, description="Exact highlighted text span")
    category: ThreatCategory = Field(..., description="Threat category assigned to span")
    explanation: str = Field(..., min_length=1, description="Explanation for the flagged span")

    @model_validator(mode="after")
    def validate_span_lengths_and_order(self) -> "FlaggedSpan":
        if self.start >= self.end:
            raise ValueError("Flagged span must have end greater than start.")
        if self.start_utf16 >= self.end_utf16:
            raise ValueError("Flagged span UTF-16 end must be greater than start.")
        if self.end - self.start != len(self.text):
            raise ValueError("Code-point offset length must equal the flagged text length.")
        if self.end_utf16 - self.start_utf16 != _utf16_code_units(self.text):
            raise ValueError("UTF-16 offset length must equal the UTF-16 length of flagged text.")
        return self

    model_config = ConfigDict(frozen=True)


class ThreatIndicator(BaseModel):
    """Structured indicator item identifying a detected pattern or semantic signal."""

    category: ThreatCategory = Field(..., description="Threat category")
    indicator_type: str = Field(..., description="Type of indicator (e.g. pattern, lexical, syntactic)")
    text: str = Field(..., description="Matching text or cue")
    explanation: str = Field(..., description="Human-understandable description of the threat indicator")

    model_config = ConfigDict(frozen=True)


class ThreatAssessmentResponse(BaseModel):
    """Structured, explainable response schema for threat assessments."""

    assessment_id: UUID = Field(default_factory=uuid4, description="Unique assessment identifier")
    analyzed_text: str = Field(
        ...,
        min_length=1,
        description=(
            "The exact preprocessed, normalized, whitespace-collapsed, and truncated "
            "working text evaluated by the analyzer. All flagged span offsets refer "
            "strictly to this text. This is not the immutable original evidence."
        ),
    )
    threat_indicators: List[ThreatIndicator] = Field(default_factory=list)
    category_scores: Dict[str, float] = Field(default_factory=dict)
    overall_threat_level: ThreatLevel = Field(...)
    is_calibrated: bool = Field(
        False,
        description="False for uncalibrated heuristics; do not display scores as probabilities when false.",
    )
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainty_state: UncertaintyState = Field(...)
    flagged_spans: List[FlaggedSpan] = Field(default_factory=list)
    explanation: str = Field(..., min_length=1)
    preprocessing_warnings: List[str] = Field(default_factory=list)
    input_truncated: bool = Field(default=False)
    toxic_bert_input_truncated: Optional[bool] = Field(
        None,
        description="Whether Toxic-BERT truncated its own tokenized input; null when it did not run.",
    )
    toxic_bert_input_token_count: Optional[int] = Field(
        None, ge=0, description="Token count before Toxic-BERT's token limit; null when unavailable."
    )
    toxic_bert_max_input_tokens: Optional[int] = Field(
        None, gt=0, description="Toxic-BERT token limit used for this assessment; null when model did not run."
    )
    disclaimer: str = Field(
        default=MANDATORY_AI_DISCLAIMER,
        min_length=1,
        frozen=True,
        validate_default=True,
        description="Mandatory configured legal and safety disclaimer",
    )

    @field_validator("disclaimer", mode="before")
    @classmethod
    def validate_mandatory_disclaimer(cls, value: object) -> str:
        """Require the exact configured disclaimer, not merely any non-empty text."""
        if value != MANDATORY_AI_DISCLAIMER:
            raise ValueError("disclaimer must exactly match the configured mandatory AI disclaimer")
        return MANDATORY_AI_DISCLAIMER

    @model_validator(mode="after")
    def validate_spans_against_analyzed_text(self) -> "ThreatAssessmentResponse":
        if not self.analyzed_text.strip():
            raise ValueError("analyzed_text must contain substantive analyzed content.")

        try:
            self.analyzed_text.encode("utf-16-le")
        except UnicodeEncodeError as exc:
            raise ValueError("analyzed_text contains an isolated Unicode surrogate code point.") from exc

        text_length = len(self.analyzed_text)
        for index, span in enumerate(self.flagged_spans):
            if span.end > text_length:
                raise ValueError(f"flagged_spans[{index}] ends beyond analyzed_text length.")
            if self.analyzed_text[span.start : span.end] != span.text:
                raise ValueError(f"flagged_spans[{index}] text does not match analyzed_text at its offsets.")

            expected_start_utf16 = _utf16_code_units(self.analyzed_text[: span.start])
            expected_end_utf16 = _utf16_code_units(self.analyzed_text[: span.end])
            if span.start_utf16 != expected_start_utf16 or span.end_utf16 != expected_end_utf16:
                raise ValueError(f"flagged_spans[{index}] UTF-16 offsets do not match analyzed_text.")
        return self

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)
