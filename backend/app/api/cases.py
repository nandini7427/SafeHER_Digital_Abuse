"""Authenticated case-management endpoints with strict owner isolation."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import Case, User
from backend.app.schemas import CaseCreate, CaseResponse
from backend.app.security.dependencies import get_current_user

router = APIRouter(prefix="/cases", tags=["Case Management"])


@router.get("", response_model=list[CaseResponse])
def list_owned_cases(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Case]:
    """List only cases belonging to the authenticated user."""
    statement = (
        select(Case)
        .where(Case.owner_id == current_user.id)
        .order_by(Case.created_at.desc(), Case.id.desc())
    )
    return list(db.scalars(statement).all())


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_owned_case(
    payload: CaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Case:
    """Create a case owned by the authenticated user."""
    case = Case(
        owner_id=current_user.id,
        title=payload.title,
        description=payload.description,
        status="open",
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


@router.get("/{case_id}", response_model=CaseResponse)
def get_owned_case(
    case_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Case:
    """Retrieve a case only when it belongs to the authenticated user."""
    case = db.scalar(
        select(Case).where(
            Case.id == case_id,
            Case.owner_id == current_user.id,
        )
    )
    if case is None:
        # Deliberately do not disclose whether another user's case exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found.")
    return case
