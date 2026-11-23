"""Threat Assessment Submodule

Contains taxonomy definitions, Pydantic v2 schemas, text preprocessing,
analyzer interfaces, and Toxic-BERT model integration for SafeHER.
"""

from ai_core.threat.analyzer import BaseThreatAnalyzer, DevelopmentThreatAnalyzer
from ai_core.threat.preprocessing import PreprocessedText, preprocess_text
from ai_core.threat.schemas import (
    AuxiliarySignal,
    FlaggedSpan,
    ThreatAssessmentResponse,
    ThreatCategory,
    ThreatIndicator,
    ThreatLevel,
    UncertaintyState,
)
from ai_core.threat.toxic_bert import (
    TOXIC_BERT_LABELS,
    ToxicBertModelRunner,
    ToxicBertResult,
    ToxicBertThreatAnalyzer,
)

__all__ = [
    "ThreatCategory",
    "AuxiliarySignal",
    "ThreatLevel",
    "UncertaintyState",
    "ThreatIndicator",
    "FlaggedSpan",
    "ThreatAssessmentResponse",
    "PreprocessedText",
    "preprocess_text",
    "BaseThreatAnalyzer",
    "DevelopmentThreatAnalyzer",
    "ToxicBertThreatAnalyzer",
    "ToxicBertModelRunner",
    "ToxicBertResult",
    "TOXIC_BERT_LABELS",
]
