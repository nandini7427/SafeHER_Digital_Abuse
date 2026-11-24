
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Case


def create_case(
    db: Session,
    owner_id: UUID,
    title: str,
    description: str | None = None,
) -> Case:
    """Create a new case for the authenticated user."""

    new_case = Case(
        owner_id=owner_id,
        title=title,
        description=description,
    )

    db.add(new_case)
    db.commit()
    db.refresh(new_case)

    return new_case


def list_user_cases(
    db: Session,
    owner_id: UUID,
) -> list[Case]:
    """Return all cases belonging to the authenticated user."""

    statement = (
        select(Case)
        .where(Case.owner_id == owner_id)
        .order_by(Case.created_at.desc())
    )

    return list(db.scalars(statement).all())


def get_user_case(
    db: Session,
    case_id: UUID,
    owner_id: UUID,
) -> Case | None:
    """Find a case only if it belongs to the authenticated user."""

    statement = select(Case).where(
        Case.id == case_id,
        Case.owner_id == owner_id,
    )

    return db.scalar(statement)


def update_case(
    db: Session,
    case: Case,
    updates: dict,
) -> Case:
    """Update the supplied fields of an existing case."""

    for field, value in updates.items():
        setattr(case, field, value)

    db.commit()
    db.refresh(case)

    return case


def delete_case(
    db: Session,
    case: Case,
) -> None:
    """Delete an existing case."""

    db.delete(case)
    db.commit()
