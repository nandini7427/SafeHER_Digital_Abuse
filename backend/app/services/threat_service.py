"""Threat assessment service layer for SafeHER backend.

Encapsulates analyzer selection, singleton lifecycle, execution,
case ownership verification, privacy-preserving logging, and exception handling.
"""

import logging
import threading
import time
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_core.config import ThreatAnalyzerConfig
from ai_core.threat.analyzer import BaseThreatAnalyzer, DevelopmentThreatAnalyzer
from ai_core.threat.schemas import ThreatAssessmentResponse
from ai_core.threat.toxic_bert import ToxicBertThreatAnalyzer
from backend.app.config import settings
from backend.app.models import AuditLog, Case, User

logger = logging.getLogger("safeher.services.threat")

_analyzer_instance: Optional[BaseThreatAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_threat_analyzer() -> BaseThreatAnalyzer:
    """Return a thread-safe singleton analyzer instance configured by settings."""
    global _analyzer_instance
    if _analyzer_instance is None:
        with _analyzer_lock:
            if _analyzer_instance is None:
                ai_config = ThreatAnalyzerConfig(
                    max_input_length=settings.AI_MAX_INPUT_LENGTH,
                    device=settings.AI_DEVICE,
                    model_name_or_path=settings.AI_MODEL_PATH,
                    enable_toxicity_check=settings.AI_ENABLE_TOXIC_BERT,
                )
                if settings.AI_ENABLE_TOXIC_BERT:
                    logger.info("Initializing ToxicBertThreatAnalyzer (optional neural mode).")
                    _analyzer_instance = ToxicBertThreatAnalyzer(config=ai_config)
                else:
                    logger.info("Initializing DevelopmentThreatAnalyzer (default baseline mode).")
                    _analyzer_instance = DevelopmentThreatAnalyzer(config=ai_config)
    return _analyzer_instance


def set_threat_analyzer(analyzer: Optional[BaseThreatAnalyzer]) -> None:
    """Override or reset the singleton analyzer instance (primarily for testing)."""
    global _analyzer_instance
    with _analyzer_lock:
        _analyzer_instance = analyzer


class ThreatAssessmentService:
    """Service layer managing threat assessments and case links."""

    def __init__(self, analyzer: Optional[BaseThreatAnalyzer] = None) -> None:
        self._analyzer = analyzer

    @property
    def analyzer(self) -> BaseThreatAnalyzer:
        if self._analyzer is not None:
            return self._analyzer
        return get_threat_analyzer()

    def assess_threat(
        self,
        text: str,
        current_user: User,
        case_id: Optional[UUID] = None,
        db: Optional[Session] = None,
    ) -> ThreatAssessmentResponse:
        """Execute threat assessment with ownership verification and privacy safeguards.

        Args:
            text: Validated input text to analyze.
            current_user: Authenticated user requesting assessment.
            case_id: Optional case ID to link this assessment to.
            db: Optional database session for case verification.

        Returns:
            Structured ThreatAssessmentResponse with all metadata.

        Raises:
            HTTPException: 400 for input preprocessing failures (e.g. surrogates).
            HTTPException: 403 if user does not own the linked case.
            HTTPException: 404 if linked case does not exist.
            HTTPException: 500 for unexpected internal errors.
        """
        # 1. Enforce case ownership if case_id provided
        if case_id is not None:
            if db is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session required for case verification.",
                )

            case = db.scalar(select(Case).where(Case.id == case_id))
            if case is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Case not found.",
                )

            if case.owner_id != current_user.id:
                logger.warning(
                    "Unauthorized case link attempt: user_id=%s, case_id=%s",
                    current_user.id,
                    case_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this case.",
                )

        # 2. Run analysis with timing and exception boundary
        start_time = time.perf_counter()
        try:
            response = self.analyzer.analyze(text)
        except ValueError as exc:
            # Preprocessing rejection (e.g. isolated Unicode surrogates, invisible-only content)
            logger.warning(
                "Threat assessment input rejected: user_id=%s, reason=%s",
                current_user.id,
                str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input text: {str(exc)}",
            )
        except Exception as exc:
            # Catch unexpected internal runtime errors; NEVER log or leak sensitive evidence text
            logger.exception("Unexpected error during threat assessment for user_id=%s", current_user.id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during threat assessment. Please try again.",
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Privacy-safe telemetry logging: log ONLY non-sensitive metrics
        logger.info(
            "Threat assessment completed: assessment_id=%s, user_id=%s, threat_level=%s, indicators_count=%d, elapsed_ms=%.2f",
            response.assessment_id,
            current_user.id,
            response.overall_threat_level.value,
            len(response.threat_indicators),
            elapsed_ms,
        )

        # 4. Record audit log entry if linked to an existing case
        if case_id is not None and db is not None:
            audit = AuditLog(
                user_id=current_user.id,
                action="AI_THREAT_ASSESSMENT",
                entity_type="case",
                entity_id=str(case_id),
            )
            db.add(audit)
            db.commit()

        return response


# Default service instance for convenience
threat_service = ThreatAssessmentService()
