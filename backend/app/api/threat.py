"""FastAPI route handlers for SafeHER AI Threat Intelligence."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ai_core.threat.schemas import ThreatAssessmentResponse
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import ThreatAssessmentRequest
from backend.app.security.dependencies import get_current_user
from backend.app.services.threat_service import threat_service

router = APIRouter(prefix="/ai", tags=["AI Threat Intelligence"])


@router.post(
    "/threat-assessment",
    response_model=ThreatAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess text for digital abuse threats, extortion, and coercion",
    description=(
        "Analyzes raw text evidence to identify threat intentions across 7 primary categories "
        "(direct threats, blackmail, financial extortion, exposure threats, coercion, "
        "harassment/intimidation, urgency pressure). Returns qualitative threat levels, "
        "exact Python and UTF-16 highlighted span offsets, and an immutable legal disclaimer."
    ),
)
def assess_threat(
    payload: ThreatAssessmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ThreatAssessmentResponse:
    """Execute authenticated threat assessment on user-provided text."""
    return threat_service.assess_threat(
        text=payload.text,
        current_user=current_user,
        case_id=payload.case_id,
        db=db,
    )


@router.post(
    "/threat/assess",
    response_model=ThreatAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess text for digital abuse threats (alias route)",
    include_in_schema=True,
)
def assess_threat_alias(
    payload: ThreatAssessmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ThreatAssessmentResponse:
    """Alias endpoint for threat assessment."""
    return threat_service.assess_threat(
        text=payload.text,
        current_user=current_user,
        case_id=payload.case_id,
        db=db,
    )
