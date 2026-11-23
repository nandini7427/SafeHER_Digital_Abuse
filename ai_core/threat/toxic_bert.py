"""Toxic-BERT model integration for SafeHER threat assessment.

Toxic-BERT supplies auxiliary, uncalibrated toxicity signals only. SafeHER
threat intent and severity continue to come from the baseline taxonomy analyzer.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ai_core.config import ThreatAnalyzerConfig
from ai_core.threat.analyzer import BaseThreatAnalyzer, DevelopmentThreatAnalyzer
from ai_core.threat.schemas import ThreatAssessmentResponse

logger = logging.getLogger("safeher.ai.toxic_bert")

DEFAULT_TOXIC_BERT_MODEL_NAME = "unitary/toxic-bert"
TOXIC_BERT_MODEL_REVISION = "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"
TOXIC_BERT_MAX_INPUT_TOKENS = 512

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
    """Raw multilabel output and token-coverage metadata from Toxic-BERT.

    Raw sigmoid values are activations, not calibrated probabilities.
    """

    raw_logits: Dict[str, float]
    raw_sigmoid_scores: Dict[str, float]
    inference_time_ms: float
    model_name: str
    is_calibrated: bool = False
    labels: List[str] = field(default_factory=lambda: list(TOXIC_BERT_LABELS))
    input_token_count: Optional[int] = None
    max_input_tokens: int = TOXIC_BERT_MAX_INPUT_TOKENS
    input_truncated: Optional[bool] = None


class ToxicBertModelRunner:
    """Lazy-loaded wrapper for the unitary/toxic-bert sequence classifier."""

    def __init__(
        self,
        model_name: str = DEFAULT_TOXIC_BERT_MODEL_NAME,
        device: str = "cpu",
        lazy_load: bool = True,
        model_revision: Optional[str] = None,
        max_input_tokens: int = TOXIC_BERT_MAX_INPUT_TOKENS,
    ) -> None:
        if type(max_input_tokens) is not int or max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be a positive integer.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name.strip()
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.model_revision = (
            model_revision
            if model_revision is not None
            else (TOXIC_BERT_MODEL_REVISION if self.model_name == DEFAULT_TOXIC_BERT_MODEL_NAME else None)
        )
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
        """Load tokenizer and model lazily; failures leave baseline analysis available."""
        if self._is_loaded:
            return True
        if self._load_error is not None:
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(min(4, os.cpu_count() or 4))
            logger.info(
                "Loading Toxic-BERT '%s' revision '%s' on %s.",
                self.model_name,
                self.model_revision or "library default",
                self.device,
            )
            load_kwargs = {"revision": self.model_revision} if self.model_revision else {}
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **load_kwargs
            )
            self._model.to(torch.device(self.device))
            self._model.eval()
            self._is_loaded = True
            logger.info("Toxic-BERT successfully initialized on %s.", self.device)
            return True
        except Exception as exc:  # Optional runtime dependency/model loading boundary.
            self._load_error = str(exc)
            logger.warning("Toxic-BERT unavailable; using baseline analysis: %s", exc)
            return False

    @staticmethod
    def _token_count(input_ids: object) -> int:
        """Count unbatched or single-item batched tokenizer input IDs."""
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if not isinstance(input_ids, (list, tuple)):
            raise TypeError("Tokenizer input_ids must be a list-like sequence.")
        return len(input_ids)

    def predict(self, text: str) -> Optional[ToxicBertResult]:
        """Run inference and disclose whether model tokenization truncated the text."""
        if not self._is_loaded and not self.load():
            return None

        started = time.perf_counter()
        try:
            import numpy as np
            import torch

            # Count tokens before truncation so the response can disclose coverage.
            full_encoding = self._tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            token_count = self._token_count(full_encoding["input_ids"])
            token_truncated = token_count > self.max_input_tokens

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                padding=False,
            )
            inputs = {key: value.to(torch.device(self.device)) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0].detach().cpu().numpy()

            if len(logits) != len(TOXIC_BERT_LABELS):
                raise ValueError(
                    f"Expected {len(TOXIC_BERT_LABELS)} Toxic-BERT logits; received {len(logits)}."
                )

            with np.errstate(over="ignore"):
                sigmoid_scores = 1.0 / (1.0 + np.exp(-logits))

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logits_by_label = {
                label: float(round(float(logits[index]), 4))
                for index, label in enumerate(TOXIC_BERT_LABELS)
            }
            scores_by_label = {
                label: float(round(float(sigmoid_scores[index]), 4))
                for index, label in enumerate(TOXIC_BERT_LABELS)
            }

            return ToxicBertResult(
                raw_logits=logits_by_label,
                raw_sigmoid_scores=scores_by_label,
                inference_time_ms=round(elapsed_ms, 2),
                model_name=self.model_name,
                is_calibrated=False,
                input_token_count=token_count,
                max_input_tokens=self.max_input_tokens,
                input_truncated=token_truncated,
            )
        except Exception:
            logger.exception(
                "Toxic-BERT inference failed; returning no neural result so baseline analysis can be used."
            )
            return None


class ToxicBertThreatAnalyzer(BaseThreatAnalyzer):
    """Hybrid analyzer: rule-based threat intent plus auxiliary Toxic-BERT signals."""

    def __init__(
        self,
        config: Optional[ThreatAnalyzerConfig] = None,
        model_runner: Optional[ToxicBertModelRunner] = None,
    ) -> None:
        super().__init__(config=config)
        self.model_runner = model_runner or ToxicBertModelRunner(
            model_name=self.config.model_name_or_path or DEFAULT_TOXIC_BERT_MODEL_NAME,
            device=self.config.device,
            lazy_load=True,
        )
        self.baseline_analyzer = DevelopmentThreatAnalyzer(config=self.config)

    def analyze(self, text: str) -> ThreatAssessmentResponse:
        """Assess threat intent first, then optionally enrich with Toxic-BERT signals."""
        base_response = self.baseline_analyzer.analyze(text)

        if not self.config.enable_toxicity_check:
            return base_response.model_copy(
                update={
                    "explanation": (
                        base_response.explanation
                        + " Toxic-BERT auxiliary analysis was disabled by configuration."
                    )
                }
            )

        # Reuse the exact working copy already analyzed by the baseline; do not
        # preprocess raw evidence a second time or create divergent model inputs.
        analyzed_text = base_response.analyzed_text
        try:
            toxic_result: Optional[ToxicBertResult] = self.model_runner.predict(analyzed_text)
        except Exception:
            logger.exception("Toxic-BERT runner raised unexpectedly; retaining baseline assessment.")
            toxic_result = None

        if toxic_result is None:
            return base_response.model_copy(
                update={
                    "explanation": (
                        base_response.explanation
                        + " Toxic-BERT offline or unavailable, or inference failed; the assessment uses baseline pattern rules."
                    )
                }
            )

        category_scores = {
            f"toxic_bert:{label}": score
            for label, score in toxic_result.raw_sigmoid_scores.items()
        }
        neural_threat = toxic_result.raw_sigmoid_scores.get("threat", 0.0)
        neural_toxic = toxic_result.raw_sigmoid_scores.get("toxic", 0.0)
        neural_insult = toxic_result.raw_sigmoid_scores.get("insult", 0.0)

        explanation = (
            base_response.explanation
            + f" [Toxic-BERT inference: {toxic_result.inference_time_ms:.1f}ms (CPU). "
            + f"Raw uncalibrated toxicity: {neural_toxic:.2f}, "
            + f"threat: {neural_threat:.2f}, insult: {neural_insult:.2f}."
        )
        if toxic_result.input_truncated is True:
            if toxic_result.input_token_count is None:
                explanation += (
                    f" Toxic-BERT may have truncated its input at {toxic_result.max_input_tokens} tokens; "
                    "the runner did not report the full token count. The baseline pattern analyzer "
                    "evaluated the full analyzed_text."
                )
            else:
                explanation += (
                    f" Toxic-BERT evaluated only the first {toxic_result.max_input_tokens} tokens "
                    f"of {toxic_result.input_token_count}; later tokens were not evaluated by the neural model. "
                    "The baseline pattern analyzer evaluated the full analyzed_text."
                )
        elif toxic_result.input_token_count is not None:
            explanation += (
                f" Toxic-BERT tokenized {toxic_result.input_token_count} tokens "
                f"within its {toxic_result.max_input_tokens}-token limit."
            )
        explanation += "]"

        return ThreatAssessmentResponse(
            assessment_id=uuid.uuid4(),
            analyzed_text=base_response.analyzed_text,
            threat_indicators=list(base_response.threat_indicators),
            category_scores=category_scores,
            overall_threat_level=base_response.overall_threat_level,
            is_calibrated=False,
            confidence_score=None,
            uncertainty_state=base_response.uncertainty_state,
            flagged_spans=list(base_response.flagged_spans),
            explanation=explanation,
            preprocessing_warnings=list(base_response.preprocessing_warnings),
            input_truncated=base_response.input_truncated,
            toxic_bert_input_truncated=toxic_result.input_truncated,
            toxic_bert_input_token_count=toxic_result.input_token_count,
            toxic_bert_max_input_tokens=toxic_result.max_input_tokens,
        )
