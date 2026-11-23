"""Threat Assessment Submodule

Contains taxonomy definitions, Pydantic v2 schemas, text preprocessing,
and analyzer interfaces for digital abuse threat assessment.
"""

from ai_core.threat.analyzer import BaseThreatAnalyzer, DevelopmentThreatAnalyzer
from ai_core.threat.preprocessing import PreprocessedText, preprocess_text
from ai_core.threat.schemas import (
    FlaggedSpan,
    ThreatAssessmentResponse,
    ThreatCategory,
    ThreatIndicator,
    ThreatLevel,
    UncertaintyState,
)

__all__ = [
    "ThreatCategory",
    "ThreatLevel",
    "UncertaintyState",
    "ThreatIndicator",
    "FlaggedSpan",
    "ThreatAssessmentResponse",
    "PreprocessedText",
    "preprocess_text",
    "BaseThreatAnalyzer",
    "DevelopmentThreatAnalyzer",
]
