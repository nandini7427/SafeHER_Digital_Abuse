"""Toxic-BERT Model Integration for SafeHER Threat Assessment.

Integrates unitary/toxic-bert for multilabel toxicity and aggressive threat detection.
Provides lazy model loading, raw uncalibrated sigmoid scoring, error handling,
and CPU execution optimization for AMD Ryzen 5 systems.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai_core.config import ThreatAnalyzerConfig, default_ai_config
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

logger = logging.getLogger("safeher.ai.toxic_bert")

# Standard 6 multilabel target classes for unitary/toxic-bert
TOXIC_BERT_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


@dataclass(frozen=True)
class ToxicBertResult:
    """Raw multilabel output from Toxic-BERT inference.

    NOTE: Raw sigmoid scores are mathematical activations, NOT calibrated probabilities.
    `is_calibrated` is explicitly False.
    """

    raw_logits: Dict[str, float]
    raw_sigmoid_scores: Dict[str, float]
    inference_time_ms: float
    model_name: str
    is_calibrated: bool = False
    labels: List[str] = field(default_factory=lambda: list(TOXIC_BERT_LABELS))


class ToxicBertModelRunner:
    """Lazy-loaded wrapper managing tokenization and inference for unitary/toxic-bert."""

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        device: str = "cpu",
        lazy_load: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

        if not lazy_load:
            self.load()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self) -> bool:
        """Lazily load tokenizer and sequence classification model on CPU."""
        if self._is_loaded:
            return True

        if self._load_error is not None:
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Hardware-aware optimization for Ryzen 5 3450U:
            # Constrain PyTorch thread count to prevent core saturation from degrading API latency
            torch.set_num_threads(min(4, os.cpu_count() or 4))

            logger.info("Loading Toxic-BERT tokenizer and model from '%s' on %s...", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(torch.device(self.device))
            self._model.eval()

            self._is_loaded = True
            logger.info("Toxic-BERT successfully initialized on %s.", self.device)
            return True

        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Toxic-BERT could not be loaded (%s). Running in heuristic fallback mode.", exc)
            return False

    def predict(self, text: str) -> Optional[ToxicBertResult]:
        """Execute CPU inference on normalized text copy.

        Returns:
            ToxicBertResult containing raw logits and uncalibrated sigmoid scores,
            or None if the model cannot be loaded.
        """
        if not self._is_loaded and not self.load():
            return None

        import numpy as np
        import torch

        start_time = time.perf_counter()

        # Tokenize input text (truncating at 512 tokens max)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        # Compute element-wise sigmoid: σ(x) = 1 / (1 + exp(-x))
        sigmoid_scores = 1.0 / (1.0 + np.exp(-logits))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        raw_logits_dict = {
            label: float(round(logits[i], 4))
            for i, label in enumerate(TOXIC_BERT_LABELS)
        }
        raw_sigmoid_dict = {
            label: float(round(sigmoid_scores[i], 4))
            for i, label in enumerate(TOXIC_BERT_LABELS)
        }

        return ToxicBertResult(
            raw_logits=raw_logits_dict,
            raw_sigmoid_scores=raw_sigmoid_dict,
            inference_time_ms=round(elapsed_ms, 2),
            model_name=self.model_name,
            is_calibrated=False,  # Raw sigmoid is uncalibrated
        )


class ToxicBertThreatAnalyzer(BaseThreatAnalyzer):
    """Hybrid threat analyzer combining SafeHER taxonomy patterns with real Toxic-BERT inference.

    - Uses pattern-based span highlighting for extortion, blackmail, and coercive control.
    - Uses Toxic-BERT for neural multi-label toxicity and aggressive threat scoring.
    - Gracefully handles missing model weights or offline environments.
    - Guarantees `is_calibrated=False` so raw sigmoid values are never shown as probabilities.
    """

    def __init__(
        self,
        config: Optional[ThreatAnalyzerConfig] = None,
        model_runner: Optional[ToxicBertModelRunner] = None,
    ) -> None:
        super().__init__(config=config)
        self.model_runner = model_runner or ToxicBertModelRunner(
            model_name=self.config.model_name_or_path or "unitary/toxic-bert",
            device=self.config.device,
            lazy_load=True,
        )
        # Use development baseline for initial pattern & span extraction
        self.baseline_analyzer = DevelopmentThreatAnalyzer(config=self.config)

    def analyze(self, text: str) -> ThreatAssessmentResponse:
        """Run complete hybrid assessment using preprocessing, patterns, and Toxic-BERT."""
        # 1. Run baseline assessment for 7-category taxonomy & explainable text spans
        base_response = self.baseline_analyzer.analyze(text)

        preprocessed = preprocess_text(
            text,
            max_length=self.config.max_input_length,
            min_length=self.config.min_input_length,
        )
        working_text = preprocessed.working_copy

        # 2. Run Toxic-BERT neural model inference
        toxic_result: Optional[ToxicBertResult] = self.model_runner.predict(working_text)

        # 3. Merge neural signals if model inference succeeded
        category_scores: Dict[str, float] = {}
        indicators = list(base_response.threat_indicators)
        explanation = base_response.explanation
        uncertainty = base_response.uncertainty_state
        level = base_response.overall_threat_level

        if toxic_result is not None:
            # Map raw sigmoid scores with explicit model prefix
            for label, score in toxic_result.raw_sigmoid_scores.items():
                category_scores[f"toxic_bert:{label}"] = score

            neural_threat = toxic_result.raw_sigmoid_scores.get("threat", 0.0)
            neural_toxic = toxic_result.raw_sigmoid_scores.get("toxic", 0.0)
            neural_insult = toxic_result.raw_sigmoid_scores.get("insult", 0.0)

            # Check if neural model identified high threat not captured by baseline lexical patterns
            if neural_threat > 0.6 and not any(ind.category == ThreatCategory.DIRECT_THREAT for ind in indicators):
                indicators.append(
                    ThreatIndicator(
                        category=ThreatCategory.DIRECT_THREAT,
                        indicator_type="neural_classification",
                        text=f"toxic_bert:threat score={neural_threat:.2f}",
                        explanation=f"Neural model identified elevated threat aggression (raw sigmoid: {neural_threat:.2f}).",
                    )
                )
                if level == ThreatLevel.NONE:
                    level = ThreatLevel.MEDIUM
                elif level == ThreatLevel.LOW:
                    level = ThreatLevel.HIGH

            explanation += (
                f" [Toxic-BERT inference: {toxic_result.inference_time_ms:.1f}ms (CPU). "
                f"Raw uncalibrated toxicity: {neural_toxic:.2f}, threat: {neural_threat:.2f}, insult: {neural_insult:.2f}]."
            )
        else:
            explanation += " [Toxic-BERT offline or unavailable; assessment generated via baseline pattern rules]."

        return ThreatAssessmentResponse(
            assessment_id=uuid.uuid4(),
            threat_indicators=indicators,
            category_scores=category_scores,
            overall_threat_level=level,
            is_calibrated=False,  # Explicitly uncalibrated
            confidence_score=None,
            uncertainty_state=uncertainty,
            flagged_spans=base_response.flagged_spans,
            explanation=explanation,
        )
