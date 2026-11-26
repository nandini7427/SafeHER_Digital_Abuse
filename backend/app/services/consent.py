from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import AuditLog, Consent, current_time


def grant_consent(
    db: Session,
    user_id: UUID,
    consent_version: str,
    purpose: str,
) -> Consent:
    """Create a new consent record for the authenticated user."""

    consent = Consent(
        user_id=user_id,
        consent_version=consent_version,
        purpose=purpose,
        status="granted",
    )

    try:
        db.add(consent)
        db.flush()

        audit_entry = AuditLog(
            user_id=user_id,
            action="consent_granted",
            entity_type="consent",
            entity_id=str(consent.id),
        )

        db.add(audit_entry)
        db.commit()
        db.refresh(consent)

        return consent

    except Exception:
        db.rollback()
        raise


def list_user_consents(
    db: Session,
    user_id: UUID,
) -> list[Consent]:
    """Return consent history belonging only to the authenticated user."""

    statement = (
        select(Consent)
        .where(Consent.user_id == user_id)
        .order_by(Consent.granted_at.desc())
    )

    return list(db.scalars(statement).all())


def withdraw_consent(
    db: Session,
    consent: Consent,
    user_id: UUID,
) -> Consent:
    """Withdraw consent while preserving the original grant timestamp."""

    if consent.status == "withdrawn":
        return consent

    try:
        consent.status = "withdrawn"
        consent.withdrawn_at = current_time()

        audit_entry = AuditLog(
            user_id=user_id,
            action="consent_withdrawn",
            entity_type="consent",
            entity_id=str(consent.id),
        )

        db.add(audit_entry)
        db.commit()
        db.refresh(consent)

        return consent

    except Exception:
        db.rollback()
        raise


def get_user_consent(
    db: Session,
    consent_id: UUID,
    user_id: UUID,
) -> Consent | None:
    """Retrieve a consent record only if it belongs to the user."""

    statement = select(Consent).where(
        Consent.id == consent_id,
        Consent.user_id == user_id,
    )

    return db.scalar(statement)


def has_active_consent(
    db: Session,
    user_id: UUID,
    purpose: str,
) -> bool:
    """Check whether the user has active consent for a specific purpose."""

    statement = select(Consent.id).where(
        Consent.user_id == user_id,
        Consent.purpose == purpose,
        Consent.status == "granted",
    )

    return db.scalar(statement) is not None