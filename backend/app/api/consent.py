
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import ConsentCreate, ConsentResponse
from backend.app.security.dependencies import get_current_user
from backend.app.services.consent import (
    grant_consent,
    get_user_consent,
    list_user_consents,
    withdraw_consent,
)


router = APIRouter(prefix="/consent", tags=["Consent"])


@router.post(
    "/",
    response_model=ConsentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_consent(
    consent_data: ConsentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Grant consent for the authenticated user."""

    return grant_consent(
        db=db,
        user_id=current_user.id,
        consent_version=consent_data.consent_version,
        purpose=consent_data.purpose,
    )


@router.get(
    "/",
    response_model=list[ConsentResponse],
)
def get_consent_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve consent history belonging to the authenticated user."""

    return list_user_consents(
        db=db,
        user_id=current_user.id,
    )


@router.post(
    "/{consent_id}/withdraw",
    response_model=ConsentResponse,
)
def withdraw_user_consent(
    consent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Withdraw consent only when the record belongs to the user."""

    consent = get_user_consent(
        db=db,
        consent_id=consent_id,
        user_id=current_user.id,
    )

    if consent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consent record not found.",
        )

    return withdraw_consent(
        db=db,
        consent=consent,
        user_id=current_user.id,
    )
