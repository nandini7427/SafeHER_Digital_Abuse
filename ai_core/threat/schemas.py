"""Pydantic v2 schemas for Threat Assessment Foundation."""

from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from ai_core.config import MANDATORY_AI_DISCLAIMER


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
    """Specific substring flagged as supporting evidence for a threat indicator."""

    start: int = Field(..., ge=0, description="Start character offset in working text copy")
    end: int = Field(..., ge=0, description="End character offset in working text copy")
    text: str = Field(..., min_length=1, description="Exact highlighted text span")
    category: ThreatCategory = Field(..., description="Threat category assigned to span")
    explanation: str = Field(..., description="Explanation of why this span was flagged")

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
    threat_indicators: List[ThreatIndicator] = Field(
        default_factory=list,
        description="List of detected threat indicators",
    )
    category_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Category scores where genuinely available from calibrated models",
    )
    overall_threat_level: ThreatLevel = Field(
        ...,
        description="Overall qualitative threat level",
    )
    is_calibrated: bool = Field(
        False,
        description="False if outputs are uncalibrated heuristics; do not display as probabilities if False",
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score if genuinely computed; None otherwise",
    )
    uncertainty_state: UncertaintyState = Field(
        ...,
        description="Evaluation of contextual sufficiency and assessment uncertainty",
    )
    flagged_spans: List[FlaggedSpan] = Field(
        default_factory=list,
        description="Explainable text spans with character offsets for UI highlighting",
    )
    explanation: str = Field(
        ...,
        description="Transparent summary of analysis reasoning, limitations, and detected signals",
    )
    disclaimer: str = Field(
        default=MANDATORY_AI_DISCLAIMER,
        description="Mandatory legal and safety disclaimer",
    )

    model_config = ConfigDict(from_attributes=True)
