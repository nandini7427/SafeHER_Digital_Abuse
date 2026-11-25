
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database import Base


def current_time():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    role: Mapped[str] = mapped_column(
        String(30),
        default="user",
        nullable=False
    )

    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=current_time,
        nullable=False
    )


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="open",
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=current_time,
        nullable=False
    )


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4
    )

    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("cases.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    storage_key: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False
    )

    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=current_time,
        nullable=False
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    entity_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=current_time,
        nullable=False
    )


class ImageReview(Base):
    __tablename__ = "image_reviews"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )

    evidence_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("evidence.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("cases.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    requester_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
        nullable=False,
    )

    ai_assessment_snapshot: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    request_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    human_verdict: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    reviewer_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    disagreement_rationale: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    is_disagreement_with_ai: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )

    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=current_time,
        nullable=False,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class EvidenceCleanupLease(Base):
    """Database-backed cleanup claim and lease record for orphan evidence reconciliation.

    Tracks files claimed for cleanup, preventing concurrent deletion of active uploads,
    enabling quarantine audits, crash recovery, and state transitions.
    """
    __tablename__ = "evidence_cleanup_leases"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4
    )

    storage_key: Mapped[str] = mapped_column(
        String(255),
        index=True,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="CLAIMED",
        nullable=False
    )  # CLAIMED, QUARANTINED, RESTORED, PURGED, ABORTED

    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )  # "quarantine" or "purge"

    claimed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=current_time,
        nullable=False
    )

    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    notes: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )
