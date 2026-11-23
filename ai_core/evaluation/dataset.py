"""Documented validation benchmark dataset for SafeHER threat assessment.

Contains carefully curated, synthetic, non-confidential test examples
designed to evaluate threat intent detection, false-positive resistance (figurative language),
and the separation between toxicity and extortion.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BenchmarkSample:
    """Benchmark test sample with ground truth labels.

    Attributes:
        sample_id: Unique identifier.
        text: Synthetic sample text.
        is_threat: Ground truth for threat intent (violence, blackmail, extortion, coercion).
        is_toxic: Ground truth for coarse vulgarity or insults.
        primary_category: Categorical taxonomy label.
        rationale: Why this sample is labeled this way.
    """

    sample_id: str
    text: str
    is_threat: bool
    is_toxic: bool
    primary_category: str
    rationale: str


BENCHMARK_DATASET: List[BenchmarkSample] = [
    # --- Category: Direct Threats ---
    BenchmarkSample(
        sample_id="DT_01",
        text="I know where you live and I will track you down and hurt you.",
        is_threat=True,
        is_toxic=True,
        primary_category="direct_threat",
        rationale="Explicit threat of physical tracking and bodily injury.",
    ),
    BenchmarkSample(
        sample_id="DT_02",
        text="You are going to die tomorrow if you show up here.",
        is_threat=True,
        is_toxic=True,
        primary_category="direct_threat",
        rationale="Direct death threat specifying temporal condition.",
    ),

    # --- Category: Blackmail & Financial Extortion (Non-toxic vulgarity, pure extortion) ---
    BenchmarkSample(
        sample_id="EXT_01",
        text="Unless you send 500 dollars before midnight, I will forward your private photos to your employer.",
        is_threat=True,
        is_toxic=False,  # CRITICAL: No profanity or curse words, yet a severe extortion threat
        primary_category="financial_extortion",
        rationale="Conditional financial extortion with non-consensual image distribution threat.",
    ),
    BenchmarkSample(
        sample_id="EXT_02",
        text="Transfer 1000 in crypto right now or I will leak your pictures to your family and friends.",
        is_threat=True,
        is_toxic=False,
        primary_category="exposure_threat",
        rationale="Cryptocurrency ransom demand tied to social network exposure threat.",
    ),

    # --- Category: Coercive Control & Surveillance Intimidation ---
    BenchmarkSample(
        sample_id="COERC_01",
        text="Don't you dare call the police or tell your family, keep this between us or else.",
        is_threat=True,
        is_toxic=False,
        primary_category="coercion",
        rationale="Coercive isolation attempt forbidding reporting to law enforcement.",
    ),
    BenchmarkSample(
        sample_id="INTIM_01",
        text="I am watching you right now, you will never be safe anywhere you go.",
        is_threat=True,
        is_toxic=False,
        primary_category="harassment_intimidation",
        rationale="Stalking assertion and persistent menacing psychological intimidation.",
    ),

    # --- Category: Figurative Language / Idiomatic Expressions (False Positive Stress Tests) ---
    BenchmarkSample(
        sample_id="FIG_01",
        text="You are completely killing it on stage tonight, what a sensational performance!",
        is_threat=False,
        is_toxic=False,
        primary_category="figurative",
        rationale="Colloquial idiom 'killing it' expressing enthusiastic praise.",
    ),
    BenchmarkSample(
        sample_id="FIG_02",
        text="I bombed the calculus exam today, I was dead tired and dying of laughter with friends beforehand.",
        is_threat=False,
        is_toxic=False,
        primary_category="figurative",
        rationale="Multiple idioms ('bombed the exam', 'dead tired', 'dying of laughter') with zero threat intent.",
    ),
    BenchmarkSample(
        sample_id="FIG_03",
        text="Our debate team destroyed the competition and crushed the debate tournament.",
        is_threat=False,
        is_toxic=False,
        primary_category="figurative",
        rationale="Metaphorical sporting victory verbs with zero interpersonal violence.",
    ),

    # --- Category: Pure Toxicity / Insults (Toxicity != Extortion Threat) ---
    BenchmarkSample(
        sample_id="TOX_01",
        text="You are an annoying stupid idiot and the biggest loser I have ever seen.",
        is_threat=False,
        is_toxic=True,
        primary_category="insult",
        rationale="Insulting and toxic derogatory name-calling, but contains no threat of violence or extortion.",
    ),

    # --- Category: Harmless Communication ---
    BenchmarkSample(
        sample_id="HARM_01",
        text="Hi, could you please email me the updated spreadsheet before our 2 PM project sync?",
        is_threat=False,
        is_toxic=False,
        primary_category="harmless",
        rationale="Routine professional business communication.",
    ),
    BenchmarkSample(
        sample_id="HARM_02",
        text="Happy birthday! Hope you have a wonderful year ahead filled with joy and good health.",
        is_threat=False,
        is_toxic=False,
        primary_category="harmless",
        rationale="Standard friendly birthday greeting.",
    ),

    # --- Category: Ambiguous / Insufficient Context ---
    BenchmarkSample(
        sample_id="AMB_01",
        text="Wait for tomorrow.",
        is_threat=False,
        is_toxic=False,
        primary_category="insufficient_context",
        rationale="Short ambiguous fragment lacking adequate context to infer malicious intent.",
    ),
]
