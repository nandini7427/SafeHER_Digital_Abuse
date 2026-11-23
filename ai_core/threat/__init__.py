"""SafeHER threat assessment submodule."""

from ai_core.threat.analyzer import BaseThreatAnalyzer, DevelopmentThreatAnalyzer
from ai_core.threat.highlighting import (
    HighlightSegment,
    partition_spans,
    render_highlighted_html,
    slice_utf16,
)
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
    TOXIC_BERT_MAX_INPUT_TOKENS,
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
    "TOXIC_BERT_MAX_INPUT_TOKENS",
    "HighlightSegment",
    "partition_spans",
    "render_highlighted_html",
    "slice_utf16",
]
