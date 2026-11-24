
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import Case, Evidence, User
from backend.app.schemas import CaseCreate, CaseUpdate, CaseResponse
from backend.app.security.dependencies import get_current_user
from backend.app.services.cases import (
    create_case,
    list_user_cases,
    get_user_case,
    update_case,
    delete_case,
)


router = APIRouter(prefix="/cases", tags=["Case Management"])


def get_owned_case(
    case_id: UUID,
    current_user: User,
    db: Session,
) -> Case:
    """Retrieve a case only if it belongs to the current user."""

    case = get_user_case(
        db=db,
        case_id=case_id,
        owner_id=current_user.id,
    )

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )

    return case


@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_case(
    case_data: CaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a case for the authenticated user."""

    return create_case(
        db=db,
        owner_id=current_user.id,
        title=case_data.title,
        description=case_data.description,
    )


@router.get(
    "",
    response_model=list[CaseResponse],
)
def read_user_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all cases belonging to the authenticated user."""

    return list_user_cases(
        db=db,
        owner_id=current_user.id,
    )


@router.get(
    "/{case_id}",
    response_model=CaseResponse,
)
def read_case(
    case_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve one case belonging to the authenticated user."""

    return get_owned_case(
        case_id=case_id,
        current_user=current_user,
        db=db,
    )


@router.patch(
    "/{case_id}",
    response_model=CaseResponse,
)
def update_existing_case(
    case_id: UUID,
    case_data: CaseUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an existing case belonging to the authenticated user."""

    case = get_owned_case(
        case_id=case_id,
        current_user=current_user,
        db=db,
    )

    updates = case_data.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field must be provided.",
        )

    if "title" in updates and updates["title"] is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Case title cannot be null.",
        )

    if "status" in updates and updates["status"] is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Case status cannot be null.",
        )

    return update_case(
        db=db,
        case=case,
        updates=updates,
    )


@router.delete(
    "/{case_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_existing_case(
    case_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a case only when it has no attached evidence."""

    case = get_owned_case(
        case_id=case_id,
        current_user=current_user,
        db=db,
    )

    existing_evidence = db.scalar(
        select(Evidence.id)
        .where(Evidence.case_id == case.id)
        .limit(1)
    )

    if existing_evidence is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete a case that contains evidence.",
        )

    delete_case(
        db=db,
        case=case,
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
