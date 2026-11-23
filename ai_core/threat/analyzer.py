"""Threat Assessment Analyzers for SafeHER.

Defines the BaseThreatAnalyzer interface and the DevelopmentThreatAnalyzer
baseline implementation (placeholder pattern-based analyzer with no fabricated AI predictions).
"""

import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Pattern, Tuple

from ai_core.config import ThreatAnalyzerConfig, default_ai_config
from ai_core.threat.preprocessing import PreprocessedText, preprocess_text
from ai_core.threat.schemas import (
    FlaggedSpan,
    ThreatAssessmentResponse,
    ThreatCategory,
    ThreatIndicator,
    ThreatLevel,
    UncertaintyState,
)


class BaseThreatAnalyzer(ABC):
    """Abstract base class for all SafeHER threat assessment analyzers."""

    def __init__(self, config: Optional[ThreatAnalyzerConfig] = None) -> None:
        self.config = config or default_ai_config

    @abstractmethod
    def analyze(self, text: str) -> ThreatAssessmentResponse:
        """Analyze text and generate an explainable threat assessment response.

        Args:
            text: Raw input text to assess.

        Returns:
            ThreatAssessmentResponse containing qualitative indicators and disclaimers.
        """
        pass


# Common colloquial and figurative expressions that contain violent/harsh words
# but do NOT represent genuine digital abuse threat intent.
FIGURATIVE_EXPRESSIONS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"\b(?:you(?:'re| are)|\bi(?:'m| am)|we(?:'re| are))\s+killing it\b", re.I), "killing it (colloquial compliment: doing great)"),
    (re.compile(r"\bdying of laughter\b", re.I), "dying of laughter (figurative expression)"),
    (re.compile(r"\bdying to (?:know|see|hear|tell|meet)\b", re.I), "dying to (idiomatic eagerness)"),
    (re.compile(r"\b(?:movie|test|exam|interview|show)\s+bombed\b", re.I), "bombed (colloquial expression: performed poorly)"),
    (re.compile(r"\b(?:i|we|they)\s+bombed the (?:test|exam|interview)\b", re.I), "bombed the test (idiom: failed)"),
    (re.compile(r"\b(?:destroyed|slayed|crushed)\s+(?:that|the)\s+(?:presentation|stage|performance|workout|game)\b", re.I), "destroyed/slayed (colloquial compliment: excelled)"),
    (re.compile(r"\bdead tired\b", re.I), "dead tired (idiomatic exhaustion)"),
    (re.compile(r"\bbite the bullet\b", re.I), "bite the bullet (idiom: face a tough situation)"),
    (re.compile(r"\bto die for\b", re.I), "to die for (colloquial superlative)"),
]

# Ambiguous phrases indicating insufficient context when presented alone
AMBIGUOUS_SHORT_PATTERNS: List[Pattern[str]] = [
    re.compile(r"^(?:just\s+)?wait (?:and see|for tomorrow|until tomorrow|for it)\.?$", re.I),
    re.compile(r"^(?:you(?:'ll| will)\s+see)\.?$", re.I),
    re.compile(r"^(?:watch out|be careful)\.?$", re.I),
    re.compile(r"^(?:soon|just wait)\.?$", re.I),
]


class DevelopmentThreatAnalyzer(BaseThreatAnalyzer):
    """Development baseline analyzer using transparent structural pattern matching.

    NOTE: This is a transparent development placeholder and heuristic triage engine.
    It does NOT run a trained deep learning model and therefore:
    - Never fabricates calibrated probability distributions or pseudo-AI metrics.
    - Explicitly sets `is_calibrated = False` and `confidence_score = None`.
    - Produces explainable highlighted text spans for human review.
    """

    def __init__(self, config: Optional[ThreatAnalyzerConfig] = None) -> None:
        super().__init__(config=config)
        self._init_taxonomy_patterns()

    def _init_taxonomy_patterns(self) -> None:
        """Compile regex patterns representing the 7 threat taxonomy categories."""
        self.category_patterns: List[Tuple[ThreatCategory, Pattern[str], str]] = [
            # 1. Direct Threats
            (
                ThreatCategory.DIRECT_THREAT,
                re.compile(r"\b(?:i(?:'ll| will| am going to|'m gonna)\s+(?:kill|hurt|destroy|beat|shoot|stab|slit your throat|end you)|(?:you(?:'re| will| are going to)\s+(?:die|get hurt)))\b", re.I),
                "Direct statement indicating intention to inflict physical harm or violence.",
            ),
            (
                ThreatCategory.DIRECT_THREAT,
                re.compile(r"\b(?:track you down and (?:hurt|kill|find) you|put you in the hospital)\b", re.I),
                "Explicit threat of physical tracking and assault.",
            ),

            # 2. Blackmail
            (
                ThreatCategory.BLACKMAIL,
                re.compile(r"\b(?:if you (?:tell|report|call the police)|unless you (?:do what i say|send|pay)|do (?:as|what) i (?:say|tell you) or else)\b", re.I),
                "Conditional coercion attempting to enforce silence or compliance under threat.",
            ),
            (
                ThreatCategory.BLACKMAIL,
                re.compile(r"\b(?:or i (?:will )?(?:leak|send|post|expose|ruin))\b", re.I),
                "Conditional threat to release damaging information upon non-compliance.",
            ),

            # 3. Financial Extortion
            (
                ThreatCategory.FINANCIAL_EXTORTION,
                re.compile(r"\b(?:(?:pay|send|transfer)\s+(?:\$?\d+|money|cash|bitcoin|crypto|gift cards?))\b", re.I),
                "Demand for monetary payment, cryptocurrency, or financial assets.",
            ),
            (
                ThreatCategory.FINANCIAL_EXTORTION,
                re.compile(r"\b(?:ransom|pay me before|wire transfer)\b", re.I),
                "Ransom or financial payment demand under duress.",
            ),

            # 4. Exposure Threats
            (
                ThreatCategory.EXPOSURE_THREAT,
                re.compile(r"\b(?:leak (?:your )?(?:private|intimate|nude|personal)?\s*(?:photos?|pics?|videos?|pictures?)|post your (?:nudes|photos?|pics?))\b", re.I),
                "Threat to circulate or leak private or intimate media.",
            ),
            (
                ThreatCategory.EXPOSURE_THREAT,
                re.compile(r"\b(?:send (?:your )?(?:photos?|pics?|details|video) to (?:your )?(?:family|friends|followers|parents|employer|boss|school))\b", re.I),
                "Threat to distribute compromising or private material to victim's personal or professional contacts.",
            ),
            (
                ThreatCategory.EXPOSURE_THREAT,
                re.compile(r"\b(?:doxx(?:ing)? you|post your address|leak your (?:phone number|address))\b", re.I),
                "Threat of non-consensual disclosure of personally identifiable information (doxxing).",
            ),

            # 5. Coercion
            (
                ThreatCategory.COERCION,
                re.compile(r"\b(?:send (?:more|another) (?:photo|picture|pic|video)|get on video call now|take off your)\b", re.I),
                "Coercive demand compelling victim into producing additional media or interaction.",
            ),
            (
                ThreatCategory.COERCION,
                re.compile(r"\b(?:don't (?:you )?dare (?:tell|block|report)|keep this between us or)\b", re.I),
                "Attempt to isolate victim by forbidding communication with support or authorities.",
            ),

            # 6. Harassment & Intimidation
            (
                ThreatCategory.HARASSMENT_INTIMIDATION,
                re.compile(r"\b(?:i (?:know|found) where you live|i am watching you|i see you right now)\b", re.I),
                "Intimidation asserting uninvited surveillance or tracking of victim's physical whereabouts.",
            ),
            (
                ThreatCategory.HARASSMENT_INTIMIDATION,
                re.compile(r"\b(?:you will never be safe|nowhere is safe|you can't hide from me)\b", re.I),
                "Persistent menacing and fear-inducing psychological intimidation.",
            ),

            # 7. Urgency & Pressure
            (
                ThreatCategory.URGENCY_PRESSURE,
                re.compile(r"\b(?:you have (?:\d+|one|two|24)\s+(?:hours?|mins?|minutes?)|before the timer runs out|countdown)\b", re.I),
                "Imposition of artificial temporal deadlines to induce panic.",
            ),
            (
                ThreatCategory.URGENCY_PRESSURE,
                re.compile(r"\b(?:last chance|final warning|do it (?:right )?now or regret it)\b", re.I),
                "High-pressure ultimatums designed to prevent victim from consulting others.",
            ),
        ]

    def analyze(self, text: str) -> ThreatAssessmentResponse:
        """Perform preliminary threat assessment on input text.

        Args:
            text: Raw input text.

        Returns:
            ThreatAssessmentResponse with detected spans, categories, and disclaimers.
        """
        # Step 1: Preprocessing & Working Copy Isolation
        preprocessed: PreprocessedText = preprocess_text(
            text,
            max_length=self.config.max_input_length,
            min_length=self.config.min_input_length,
        )
        working_text = preprocessed.working_copy

        # Step 2: Check for Figurative Speech / Colloquial Idioms
        figurative_matches: List[str] = []
        for pattern, explanation in FIGURATIVE_EXPRESSIONS:
            if pattern.search(working_text):
                figurative_matches.append(explanation)

        # Step 3: Check for Insufficient Context
        is_ambiguous_short = False
        if preprocessed.word_count < 6:
            for pattern in AMBIGUOUS_SHORT_PATTERNS:
                if pattern.search(working_text):
                    is_ambiguous_short = True
                    break

        # Step 4: Scan for Threat Categories and Flag Spans
        flagged_spans: List[FlaggedSpan] = []
        detected_categories: set[ThreatCategory] = set()
        indicators: List[ThreatIndicator] = []

        # Only scan threat patterns if text is not solely a benign figurative expression
        for category, pattern, description in self.category_patterns:
            for match in pattern.finditer(working_text):
                span_text = match.group(0)
                span = FlaggedSpan(
                    start=match.start(),
                    end=match.end(),
                    text=span_text,
                    category=category,
                    explanation=description,
                )
                flagged_spans.append(span)
                detected_categories.add(category)
                indicators.append(
                    ThreatIndicator(
                        category=category,
                        indicator_type="lexical_pattern",
                        text=span_text,
                        explanation=description,
                    )
                )

        # Step 5: Determine Qualitative Threat Level & Uncertainty
        threat_level, uncertainty, explanation = self._determine_threat_level(
            detected_categories=detected_categories,
            flagged_spans=flagged_spans,
            figurative_matches=figurative_matches,
            is_ambiguous_short=is_ambiguous_short,
            preprocessed=preprocessed,
        )

        return ThreatAssessmentResponse(
            assessment_id=uuid.uuid4(),
            threat_indicators=indicators,
            category_scores={},  # Intentionally empty: no fake calibrated model probabilities
            overall_threat_level=threat_level,
            is_calibrated=False,  # Explicitly uncalibrated development baseline
            confidence_score=None,  # No fabricated confidence percentages
            uncertainty_state=uncertainty,
            flagged_spans=flagged_spans,
            explanation=explanation,
        )

    def _determine_threat_level(
        self,
        detected_categories: set[ThreatCategory],
        flagged_spans: List[FlaggedSpan],
        figurative_matches: List[str],
        is_ambiguous_short: bool,
        preprocessed: PreprocessedText,
    ) -> Tuple[ThreatLevel, UncertaintyState, str]:
        """Determine qualitative threat level and construct transparent explanation."""
        # Case A: Ambiguous short statement with insufficient context
        if is_ambiguous_short and not detected_categories:
            return (
                ThreatLevel.LOW,
                UncertaintyState.INSUFFICIENT_CONTEXT,
                "The message contains short or ambiguous wording without explicit threats or extortion demands. "
                "Context is insufficient to conclusively determine intent. Continued monitoring is advised.",
            )

        # Case B: Figurative language recognized and no true threats
        if figurative_matches and not detected_categories:
            fig_desc = "; ".join(figurative_matches)
            return (
                ThreatLevel.NONE,
                UncertaintyState.LOW,
                f"No threat indicators detected. The text contains benign figurative or idiomatic expressions ({fig_desc}).",
            )

        # Case C: No indicators found in normal text
        if not detected_categories:
            uncertainty = UncertaintyState.INSUFFICIENT_CONTEXT if preprocessed.word_count < 3 else UncertaintyState.LOW
            return (
                ThreatLevel.NONE,
                uncertainty,
                "No threat indicators, blackmail language, or extortion patterns were detected in the text.",
            )

        # Case D: Severe or High Threat (Direct violence, or Blackmail combined with exposure/extortion)
        has_direct_threat = ThreatCategory.DIRECT_THREAT in detected_categories
        has_blackmail = ThreatCategory.BLACKMAIL in detected_categories
        has_exposure = ThreatCategory.EXPOSURE_THREAT in detected_categories
        has_extortion = ThreatCategory.FINANCIAL_EXTORTION in detected_categories
        has_coercion = ThreatCategory.COERCION in detected_categories

        categories_str = ", ".join(c.value.replace("_", " ").title() for c in detected_categories)

        if has_direct_threat and (has_blackmail or has_exposure or has_extortion):
            level = ThreatLevel.SEVERE
            uncertainty = UncertaintyState.LOW
            explanation = (
                f"High-priority alert: Identified multiple compounded threat patterns including Direct Threats and Extortion ({categories_str}). "
                f"Detected {len(flagged_spans)} flagged span(s). Immediate human triage and safety review strongly advised."
            )
        elif has_direct_threat or (has_blackmail and (has_exposure or has_extortion)):
            level = ThreatLevel.HIGH
            uncertainty = UncertaintyState.LOW
            explanation = (
                f"Elevated threat indicators detected ({categories_str}). "
                f"Found {len(flagged_spans)} flagged span(s) indicating potential extortion, non-consensual exposure, or violence."
            )
        elif has_coercion or has_exposure or has_extortion or has_blackmail:
            level = ThreatLevel.MEDIUM
            uncertainty = UncertaintyState.MODERATE
            explanation = (
                f"Moderate threat indicators detected ({categories_str}). "
                f"The text exhibits coercive demands or distribution threats across {len(flagged_spans)} flagged span(s)."
            )
        else:
            # Only urgency pressure or harassment without explicit demands
            level = ThreatLevel.LOW
            uncertainty = UncertaintyState.MODERATE
            explanation = (
                f"Low-level or isolated pressure signals detected ({categories_str}). "
                "No explicit physical threats or verified extortion ultimatums found."
            )

        return level, uncertainty, explanation
