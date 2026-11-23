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

# Benign payment patterns representing routine commercial or social expenses
BENIGN_PAYMENT_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\bfor\s+(?:the\s+)?(?:concert\s+)?tickets?\b", re.I),
    re.compile(r"\bfor\s+(?:our\s+)?(?:group\s+)?dinner\b", re.I),
    re.compile(r"\bfor\s+(?:lunch|breakfast|dinner|food|groceries|rent|coffee|drinks?|gas|uber|cab|taxi|flight|hotel|trip)\b", re.I),
    re.compile(r"\bfor\s+(?:the\s+)?(?:rent|utilities|wifi|electric(?:ity)?|gas|bill|bills|subscription)\b", re.I),
    re.compile(r"\bfor\s+(?:the\s+)?invoice(?:\s+\w+)?\b", re.I),
    re.compile(r"\b(?:registered\s+)?vendor\s+wallet\b", re.I),
    re.compile(r"\b(?:vendor|merchant|business)\s+(?:wallet|account)\b", re.I),
    re.compile(r"\bsplit\s+(?:the\s+)?(?:bill|check|tab|dinner|lunch|cost|rent)\b", re.I),
    re.compile(r"\b(?:reimburse|reimbursement)\b", re.I),
]

# Strong, locally meaningful payment/duress cues. A bare occurrence of the word
# "ransom" elsewhere in a message must not turn an unrelated payment into extortion.
INHERENT_EXTORTION_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\b(?:pay|send|transfer|give|wire)\s+(?:me\s+)?(?:the\s+)?ransom\b", re.I),
    re.compile(r"\b(?:ransom\s+(?:of|payment|demand|money)|(?:blackmail|extortion)\s+(?:payment|demand|money))\b", re.I),
    re.compile(
        r"\bpay\s+(?:me\s+)?(?:(?:\$|₹|€|£)?\s*\d+\s*(?:dollars?|rupees?|bucks?|inr|usd)?\s+)?to\s+(?:delete|stop|keep\s+quiet|remove|destroy|not\s+(?:leak|share|post|expose))\b",
        re.I,
    ),
    re.compile(r"\bpay\s+(?:me\s+)?(?:[\$₹€£¥]\s*\d[\d,.]*|\d[\d,.]*\s*(?:dollars?|rupees?|bucks?))\s+as\s+ransom\b", re.I),
]

# Consequence connectors linking a demand to a threat consequence:
# e.g. "... or I will publish your private photos", "... otherwise I will expose your videos"
ULTIMATUM_CONSEQUENCE_CONNECTOR = re.compile(
    r"\b(?:or\s+(?:else\s+)?|otherwise\s+)(?:i\s+(?:will|'ll|am\s+going\s+to|'m\s+gonna)\s+)?"
    r"(?:(?:publish|leak|send|post|expose|ruin|forward|share|distribute|upload|release)\s+(?:(?:your|the)\s+(?:private|intimate|nude|personal|compromising|explicit)\s+(?:photos?|pics?|pictures?|videos?|images?|materials?|clips?|recordings?)|your\s+(?:nudes|photos?|pics?|videos?|images?|personal\s+information|details)|(?:photos?|pics?|details|videos?)\s+to\s+(?:your\s+)?(?:family|friends|followers|parents|employer|boss|school|contacts|everyone|the\s+public)|you\b|your\s+(?:life|reputation|career|family))|"
    r"(?:track|hunt|find|hurt|kill|destroy|beat|shoot|stab|harm|put\s+you\s+in))\b",
    re.I,
)

# Prerequisite clauses governing a demand:
# e.g. "Unless you transfer 1000 dollars...", "If you don't pay 500 dollars..."
ULTIMATUM_PREREQUISITE_GOVERNOR = re.compile(
    r"\b(?:unless\s+you\s+(?:do\s+what\s+i\s+say|send(?:\s+me)?|pay(?:\s+me)?|transfer|give(?:\s+me)?|wire|deliver)|"
    r"if\s+you\s+(?:don't|do\s+not|refuse\s+to)\s+(?:pay|send|transfer|wire))\b",
    re.I,
)


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

    @staticmethod
    def _get_clause_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
        """Return sentence-like clause bounds around a character span.

        Periods, semicolons and newlines are boundaries. Commas are retained
        because they commonly join both sides of an ultimatum.
        """
        before = text[:start]
        after = text[end:]
        previous_boundaries = [before.rfind(mark) for mark in (".", ";", "\n")]
        clause_start = max(previous_boundaries) + 1
        next_boundaries = [after.find(mark) for mark in (".", ";", "\n") if after.find(mark) >= 0]
        next_boundary = min(next_boundaries) if next_boundaries else len(after)
        return clause_start, end + next_boundary

    def _has_category_match(
        self,
        text: str,
        categories: set[ThreatCategory],
    ) -> bool:
        """Check whether selected taxonomy patterns occur in a bounded text window."""
        return any(
            category in categories and pattern.search(text) is not None
            for category, pattern, _ in self.category_patterns
        )

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

            # 2. Blackmail & Conditional Coercion
            (
                ThreatCategory.BLACKMAIL,
                re.compile(
                    r"\b(?:unless\s+you\s+(?:do\s+what\s+i\s+say|send(?:\s+me)?|pay(?:\s+me)?|transfer|give(?:\s+me)?|wire|comply|cooperate)|"
                    r"if\s+you\s+(?:tell|report(?:\s+me)?|call\s+(?:the\s+)?police|block\s+me|don't\s+(?:pay|send|transfer|comply)|refuse\s+to\s+(?:pay|send|comply))|"
                    r"do\s+(?:as|what)\s+i\s+(?:say|tell\s+you)\s+or\s+else)\b",
                    re.I,
                ),
                "Conditional coercion attempting to enforce silence or compliance under threat.",
            ),
            (
                ThreatCategory.BLACKMAIL,
                re.compile(
                    r"\b(?:(?:or\s+(?:else\s+)?|otherwise\s+)(?:i\s+(?:will|'ll|am\s+going\s+to|'m\s+gonna)\s+)?"
                    r"(?:(?:publish|leak|send|post|expose|ruin|forward|share|distribute|upload|release)\s+(?:(?:your|the)\s+(?:private|intimate|nude|personal|compromising|explicit)\s+(?:photos?|pics?|pictures?|videos?|images?|materials?|clips?|recordings?)|your\s+(?:nudes|photos?|pics?|videos?|images?|personal\s+information|details)|(?:photos?|pics?|details|videos?)\s+to\s+(?:your\s+)?(?:family|friends|followers|parents|employer|boss|school|contacts|everyone|the\s+public)|you\b|your\s+(?:life|reputation|career|family))|"
                    r"(?:track|hunt|find|hurt|kill|destroy|beat|shoot|stab|harm|put\s+you\s+in)))\b",
                    re.I,
                ),
                "Conditional threat to release damaging material or inflict harm upon non-compliance.",
            ),

            # 3. Financial Extortion
            (
                ThreatCategory.FINANCIAL_EXTORTION,
                re.compile(
                    r"\b(?:(?:pay|send|transfer|wire|give)(?:\s+me)?\s+(?:[\$₹€£¥]\s*\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|bucks?|rupees?|inr|usd|eur|gbp|cash)|(?:the\s+)?(?:money|cash|bitcoin|crypto|cryptocurrency|funds)|the\s+requested\s+(?:crypto|cryptocurrency|bitcoin|money|funds|ransom|amount)))\b",
                    re.I,
                ),
                "Demand for monetary payment, cryptocurrency, or financial assets.",
            ),
            (
                ThreatCategory.FINANCIAL_EXTORTION,
                re.compile(
                    r"\b(?:(?:pay|send|transfer|give|wire)\s+(?:me\s+)?(?:the\s+)?ransom(?:\s+of\s+(?:[\$₹€£¥]\s*)?\d[\d,.]*(?:\s*(?:dollars?|rupees?|bucks?))?)?|"
                    r"pay\s+me\s+before\s+(?:midnight|tomorrow|\d+\s*(?:am|pm)|the\s+timer)|"
                    r"wire\s+transfer\s+before)\b",
                    re.I,
                ),
                "Ransom or financial payment demand under duress.",
            ),

            # 4. Exposure Threats
            (
                ThreatCategory.EXPOSURE_THREAT,
                re.compile(
                    r"\b(?:(?:publish|leak|post|share|forward|distribute|expose|upload|release)\s+(?:your\s+)?(?:private|intimate|nude|personal|compromising|explicit)\s+(?:photos?|pics?|pictures?|videos?|images?|materials?|clips?|recordings?)|"
                    r"(?:publish|leak|post|share|forward|distribute|expose|upload)\s+your\s+(?:nudes|naked\s+photos?|private\s+photos?|intimate\s+(?:photos?|pics?|pictures?|videos?|images?))|"
                    r"(?:publish|leak|post|share|forward|distribute|expose|release)\s+your\s+personal\s+information)\b",
                    re.I,
                ),
                "Threat to circulate or leak private, intimate, or personal media/information.",
            ),
            (
                ThreatCategory.EXPOSURE_THREAT,
                re.compile(
                    r"\b(?:(?:send|forward|share|distribute|leak|publish)\s+(?:your\s+)?(?:private\s+|intimate\s+)?(?:photos?|pics?|details|videos?|images?|pictures?|personal\s+information)\s+to\s+(?:your\s+)?(?:family|friends|followers|parents|employer|boss|colleagues|school|contacts|relatives|everyone|the\s+public))\b",
                    re.I,
                ),
                "Threat to distribute compromising or private material to victim's personal or professional contacts.",
            ),
            (
                ThreatCategory.EXPOSURE_THREAT,
                re.compile(r"\b(?:doxx(?:ing)?\s+you|post\s+your\s+address|leak\s+your\s+(?:phone(?:\s+number)?|address))\b", re.I),
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
            (
                ThreatCategory.URGENCY_PRESSURE,
                re.compile(r"\b(?:(?:by|before)\s+midnight|you\s+have\s+until\s+midnight|before\s+the\s+clock\s+strikes)\b", re.I),
                "Artificial deadline imposed to force urgent victim compliance.",
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

        # Pass 1: Scan categories while separating financial and urgency matches for contextual triage
        financial_matches: List[Tuple[FlaggedSpan, ThreatIndicator]] = []
        urgency_matches: List[Tuple[FlaggedSpan, ThreatIndicator]] = []
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
                indicator = ThreatIndicator(
                    category=category,
                    indicator_type="lexical_pattern",
                    text=span_text,
                    explanation=description,
                )
                if category == ThreatCategory.FINANCIAL_EXTORTION:
                    financial_matches.append((span, indicator))
                elif category == ThreatCategory.URGENCY_PRESSURE:
                    urgency_matches.append((span, indicator))
                else:
                    flagged_spans.append(span)
                    detected_categories.add(category)
                    indicators.append(indicator)

        # Pass 2: Contextual validation of financial payment demands.
        # Do not use message-wide category presence to associate unrelated clauses.
        for span, indicator in financial_matches:
            if self._is_extortion_demand(span, working_text, detected_categories):
                flagged_spans.append(span)
                detected_categories.add(ThreatCategory.FINANCIAL_EXTORTION)
                indicators.append(indicator)

        # Evaluate each deadline in its own clause. A benign payment purpose in a
        # different sentence must not suppress a genuine pressure signal here.
        active_context_categories = {
            ThreatCategory.DIRECT_THREAT,
            ThreatCategory.EXPOSURE_THREAT,
            ThreatCategory.BLACKMAIL,
            ThreatCategory.COERCION,
            ThreatCategory.HARASSMENT_INTIMIDATION,
        }
        for span, indicator in urgency_matches:
            clause_start, clause_end = self._get_clause_bounds(
                working_text, span.start, span.end
            )
            local_clause = working_text[clause_start:clause_end]
            has_local_benign_purpose = any(
                pattern.search(local_clause) for pattern in BENIGN_PAYMENT_PATTERNS
            )
            has_local_threat_context = self._has_category_match(
                local_clause, active_context_categories
            )
            has_local_accepted_extortion = any(
                candidate.category == ThreatCategory.FINANCIAL_EXTORTION
                and clause_start <= candidate.start
                and candidate.end <= clause_end
                for candidate in flagged_spans
            )

            if has_local_benign_purpose and not has_local_threat_context and not has_local_accepted_extortion:
                continue

            flagged_spans.append(span)
            detected_categories.add(ThreatCategory.URGENCY_PRESSURE)
            indicators.append(indicator)

        # Deduplicate subsumed spans of the same category to preserve the most informative span
        deduped_spans: List[FlaggedSpan] = []
        for s in flagged_spans:
            if not any(
                other.category == s.category
                and other.start <= s.start
                and other.end >= s.end
                and (other.end - other.start) > (s.end - s.start)
                for other in flagged_spans
            ):
                deduped_spans.append(s)
        flagged_spans = deduped_spans

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

    def _is_extortion_demand(
        self,
        span: FlaggedSpan,
        working_text: str,
        detected_categories: set[ThreatCategory],
    ) -> bool:
        """Verify that a payment candidate is locally connected to coercion or a threat.

        The method intentionally does not treat a ransom/blackmail phrase or threat
        found elsewhere in the entire message as evidence against this payment span.
        """
        clause_start, clause_end = self._get_clause_bounds(
            working_text, span.start, span.end
        )
        local_clause = working_text[clause_start:clause_end]

        # Check strong inherent cues only when the cue overlaps the current
        # financial match. Merely mentioning a ransom in the same sentence as an
        # unrelated purchase must not convert that purchase into extortion.
        has_payment_bound_inherent_evidence = False
        for pattern in INHERENT_EXTORTION_PATTERNS:
            for match in pattern.finditer(local_clause):
                absolute_match_start = clause_start + match.start()
                absolute_match_end = clause_start + match.end()
                overlaps_payment = (
                    absolute_match_start < span.end
                    and absolute_match_end > span.start
                )
                if overlaps_payment:
                    has_payment_bound_inherent_evidence = True
                    break
            if has_payment_bound_inherent_evidence:
                break

        if has_payment_bound_inherent_evidence:
            return True

        has_local_benign_purpose = any(
            pattern.search(local_clause) is not None
            for pattern in BENIGN_PAYMENT_PATTERNS
        )

        # Search for a forward ultimatum only in the local clause, or (when no
        # benign payment purpose exists) a tightly bounded immediate continuation.
        forward_end = clause_end
        if not has_local_benign_purpose and clause_end < len(working_text):
            # Permit one immediately following sentence for constructions such as
            # "Pay me $500. Otherwise I will publish your private photos." Do not
            # scan beyond that next sentence for an unrelated threat.
            next_sentence_start = clause_end + 1  # include/consume the boundary
            while next_sentence_start < len(working_text) and working_text[next_sentence_start].isspace():
                next_sentence_start += 1

            following_text = working_text[next_sentence_start:]
            following_boundaries = [
                following_text.find(mark)
                for mark in (".", ";", "\n")
                if following_text.find(mark) >= 0
            ]
            if following_boundaries:
                forward_end = next_sentence_start + min(following_boundaries) + 1
            else:
                forward_end = len(working_text)

        forward_window = working_text[span.start:forward_end]
        consequence_match = ULTIMATUM_CONSEQUENCE_CONNECTOR.search(forward_window)
        has_forward_ultimatum = False
        if consequence_match is not None:
            gap = working_text[span.end:span.start + consequence_match.start()]
            sentence_break_count = sum(gap.count(mark) for mark in (".", ";", "\n"))
            has_local_consequence = self._has_category_match(
                forward_window,
                {
                    ThreatCategory.DIRECT_THREAT,
                    ThreatCategory.EXPOSURE_THREAT,
                    ThreatCategory.HARASSMENT_INTIMIDATION,
                },
            )
            has_forward_ultimatum = (
                has_local_consequence
                and sentence_break_count <= 1
                and len(gap.strip()) <= 60
            )

        # A prerequisite such as "Unless you transfer ..." counts only when a
        # concrete threat consequence also appears after the payment in this same
        # clause. The mere presence of an unrelated threat elsewhere is ignored.
        before_payment = working_text[clause_start:span.start]
        after_payment = working_text[span.end:clause_end]
        has_backward_prerequisite = (
            ULTIMATUM_PREREQUISITE_GOVERNOR.search(working_text[clause_start:span.end]) is not None
            and self._has_category_match(
                after_payment,
                {
                    ThreatCategory.DIRECT_THREAT,
                    ThreatCategory.EXPOSURE_THREAT,
                    ThreatCategory.HARASSMENT_INTIMIDATION,
                },
            )
        )

        if has_forward_ultimatum or has_backward_prerequisite:
            return True

        # A legitimate purpose in the same clause wins over an unrelated threat
        # or ransom mention elsewhere in the message.
        if has_local_benign_purpose:
            return False

        # No message-wide fallback here by design. If the payment itself is not
        # locally tied to a threat, it is not marked as financial extortion.
        return False

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
        elif has_blackmail and has_exposure and has_extortion:
            level = ThreatLevel.SEVERE
            uncertainty = UncertaintyState.LOW
            explanation = (
                f"High-priority alert: Identified multiple compounded threat patterns including Blackmail and Extortion ({categories_str}). "
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
