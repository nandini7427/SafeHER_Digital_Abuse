"""add image review and evidence cleanup lease tables

Revision ID: b61f0a2c9d71
Revises: aae125f74655
Create Date: 2026-11-24

The ORM defines ImageReview and EvidenceCleanupLease, but the original schema
revision did not create either table. This additive migration supports both
fresh installations and databases that have already applied the initial revision.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b61f0a2c9d71"
down_revision: Union[str, Sequence[str], None] = "aae125f74655"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "image_reviews",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("evidence_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("requester_id", sa.Uuid(), nullable=False),
        sa.Column("reviewer_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=30), server_default="pending", nullable=False),
        sa.Column("ai_assessment_snapshot", sa.Text(), nullable=False),
        sa.Column("request_reason", sa.Text(), nullable=True),
        sa.Column("human_verdict", sa.String(length=50), nullable=True),
        sa.Column("reviewer_notes", sa.Text(), nullable=True),
        sa.Column("disagreement_rationale", sa.Text(), nullable=True),
        sa.Column("is_disagreement_with_ai", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["evidence_id"], ["evidence.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["requester_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reviewer_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_image_reviews_evidence_id"), "image_reviews", ["evidence_id"], unique=False)
    op.create_index(op.f("ix_image_reviews_case_id"), "image_reviews", ["case_id"], unique=False)
    op.create_index(op.f("ix_image_reviews_requester_id"), "image_reviews", ["requester_id"], unique=False)
    op.create_index(op.f("ix_image_reviews_reviewer_id"), "image_reviews", ["reviewer_id"], unique=False)

    op.create_table(
        "evidence_cleanup_leases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), server_default="CLAIMED", nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_evidence_cleanup_leases_storage_key"),
        "evidence_cleanup_leases",
        ["storage_key"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_evidence_cleanup_leases_storage_key"), table_name="evidence_cleanup_leases")
    op.drop_table("evidence_cleanup_leases")
    op.drop_index(op.f("ix_image_reviews_reviewer_id"), table_name="image_reviews")
    op.drop_index(op.f("ix_image_reviews_requester_id"), table_name="image_reviews")
    op.drop_index(op.f("ix_image_reviews_case_id"), table_name="image_reviews")
    op.drop_index(op.f("ix_image_reviews_evidence_id"), table_name="image_reviews")
    op.drop_table("image_reviews")
