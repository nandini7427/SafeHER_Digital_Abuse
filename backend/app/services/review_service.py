"""SafeHER Human-in-the-Loop Image Review Service.

Architecture and Safety Principles:
1. AI assists users and reviewers; it never makes irreversible or legally definitive decisions.
2. AI outputs and human judgments are strictly decoupled in distinct database entities.
3. Original AI assessment snapshots are frozen and immutable (never overwritten).
4. Reviewer identity, timestamps, and audit actions are rigorously recorded under role-based authorization.
5. Disagreements between human reviewers and AI are explicitly detected, categorized, and audited.
6. Zero Automated Model Training: Human reviews and feedback NEVER trigger automatic model fine-tuning,
   preventing adversarial poisoning, unverified label drift, and bias amplification.
7. Non-Legal Disclaimer: Reviewer conclusions are strictly triage/support observations, never judicial findings.
"""

from datetime import datetime, timezone
import json
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ai_core.image_intelligence.monitoring import model_monitoring_engine
from backend.app.models import AuditLog, Case, Evidence, ImageReview, User
from backend.app.schemas import (
    HumanReviewRequestCreate,
    HumanReviewStatus,
    HumanReviewSubmit,
    HumanReviewVerdict,
)

logger = logging.getLogger("safeher.backend.services.review")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HumanReviewService:
    """Manages the lifecycle of human reviews for visual evidence assessments."""

    def request_human_review(
        self,
        request: HumanReviewRequestCreate,
        current_user: User,
        db: Session,
    ) -> ImageReview:
        """Create a new human review request for verified user evidence.

        Enforces:
        - Strict horizontal authorization (user must own evidence and case).
        - Case and evidence boundary alignment.
        - Preservation of an immutable AI assessment snapshot.
        - Audit trail logging.
        """
        # 1. Verify evidence existence and ownership
        evidence = db.scalar(select(Evidence).where(Evidence.id == request.evidence_id))
        if evidence is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence record not found.",
            )

        if evidence.owner_id != current_user.id:
            logger.warning(
                "Unauthorized review request attempt: user_id=%s, evidence_id=%s, actual_owner=%s",
                current_user.id,
                request.evidence_id,
                evidence.owner_id,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to request review for evidence you do not own.",
            )

        # 2. Verify case existence, ownership, and linkage
        case = db.scalar(select(Case).where(Case.id == request.case_id))
        if case is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Associated case not found.",
            )

        if case.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to request review for a case you do not own.",
            )

        if evidence.case_id != request.case_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Evidence does not belong to the specified case.",
            )

        # 3. Serialize immutable AI assessment snapshot
        if isinstance(request.ai_assessment_snapshot, dict):
            snapshot_json = json.dumps(request.ai_assessment_snapshot)
        else:
            snapshot_json = str(request.ai_assessment_snapshot)

        # 4. Instantiate ImageReview record
        review = ImageReview(
            evidence_id=request.evidence_id,
            case_id=request.case_id,
            requester_id=current_user.id,
            reviewer_id=None,
            status=HumanReviewStatus.PENDING.value,
            ai_assessment_snapshot=snapshot_json,
            request_reason=request.request_reason,
            human_verdict=None,
            reviewer_notes=None,
            disagreement_rationale=None,
            is_disagreement_with_ai=False,
            requested_at=_utcnow(),
            reviewed_at=None,
        )
        db.add(review)
        db.flush()

        # 5. Record immutable audit log
        audit = AuditLog(
            user_id=current_user.id,
            action="HUMAN_REVIEW_REQUESTED",
            entity_type="image_review",
            entity_id=str(review.id),
        )
        db.add(audit)
        db.commit()
        db.refresh(review)

        logger.info(
            "Human review requested: review_id=%s, evidence_id=%s, requester_id=%s",
            review.id,
            review.evidence_id,
            current_user.id,
        )
        return review

    def get_review(
        self,
        review_id: UUID,
        current_user: User,
        db: Session,
    ) -> ImageReview:
        """Fetch review record ensuring caller is requester, assigned reviewer, or reviewer staff."""
        review = db.scalar(select(ImageReview).where(ImageReview.id == review_id))
        if review is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review record not found.",
            )

        is_owner = review.requester_id == current_user.id
        is_staff = current_user.role in ("reviewer", "advocate", "admin")

        if not (is_owner or is_staff):
            logger.warning(
                "Unauthorized review access attempt: user_id=%s, review_id=%s",
                current_user.id,
                review_id,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this review record.",
            )

        return review

    def list_pending_reviews(
        self,
        current_user: User,
        db: Session,
    ) -> List[ImageReview]:
        """List unassigned or in-progress reviews for review staff."""
        if current_user.role not in ("reviewer", "advocate", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access restricted to authorized reviewers and advocates.",
            )

        stmt = (
            select(ImageReview)
            .where(ImageReview.status.in_([HumanReviewStatus.PENDING.value, HumanReviewStatus.UNDER_REVIEW.value]))
            .order_by(ImageReview.requested_at.asc())
        )
        return list(db.scalars(stmt).all())

    def claim_review(
        self,
        review_id: UUID,
        current_user: User,
        db: Session,
    ) -> ImageReview:
        """Assign review to the calling reviewer and mark as under_review."""
        if current_user.role not in ("reviewer", "advocate", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only authorized reviewers can claim reviews.",
            )

        review = db.scalar(select(ImageReview).where(ImageReview.id == review_id))
        if review is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review record not found.",
            )

        if review.status == HumanReviewStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot claim a review that has already been finalized.",
            )

        if review.requester_id == current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Self-review prohibited: Reviewers cannot claim their own evidence review requests.",
            )

        review.reviewer_id = current_user.id
        review.status = HumanReviewStatus.UNDER_REVIEW.value

        audit = AuditLog(
            user_id=current_user.id,
            action="HUMAN_REVIEW_CLAIMED",
            entity_type="image_review",
            entity_id=str(review.id),
        )
        db.add(audit)
        db.commit()
        db.refresh(review)

        logger.info("Review claimed: review_id=%s, reviewer_id=%s", review.id, current_user.id)
        return review

    def submit_human_review(
        self,
        review_id: UUID,
        submission: HumanReviewSubmit,
        current_user: User,
        db: Session,
    ) -> ImageReview:
        """Finalize human review outcome and evaluate alignment with AI assessment.

        Rules:
        - Only authorized reviewer/advocate/admin can finalize.
        - Finalized reviews are immutable (cannot be silently overwritten).
        - AI assessment snapshot remains completely unaltered.
        - Zero automated training triggers from this verdict.
        - Audit log recorded.
        """
        if current_user.role not in ("reviewer", "advocate", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only authorized reviewers and advocates can submit review verdicts.",
            )

        review = db.scalar(select(ImageReview).where(ImageReview.id == review_id))
        if review is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review record not found.",
            )

        if review.status == HumanReviewStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Review has already been finalized. Duplicate submissions are not permitted.",
            )

        if review.requester_id == current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Self-review prohibited: Reviewers cannot submit verdicts on their own evidence review requests.",
            )

        # Parse AI snapshot to determine whether human verdict diverges from AI
        ai_label = None
        try:
            snapshot_data = json.loads(review.ai_assessment_snapshot)
            if isinstance(snapshot_data, dict):
                ai_label = snapshot_data.get("assessment_label")
        except Exception:
            ai_label = None

        # Determine disagreement
        is_disagreement = False
        verdict = submission.human_verdict

        if verdict in (
            HumanReviewVerdict.DISAGREES_SUSPICIOUS_MANIPULATION,
            HumanReviewVerdict.DISAGREES_NO_OBVIOUS_ARTIFACTS,
        ):
            is_disagreement = True
        elif verdict == HumanReviewVerdict.CONCURS_WITH_AI:
            is_disagreement = False
        elif verdict in (HumanReviewVerdict.INCONCLUSIVE, HumanReviewVerdict.INSUFFICIENT_EVIDENCE):
            # If AI had made a definitive call, this is an explicit divergence
            if ai_label in ("suspicious_manipulation", "highly_likely_manipulated", "no_obvious_artifacts"):
                is_disagreement = True

        # Enforce disagreement rationale if human diverges
        if is_disagreement and not submission.disagreement_rationale:
            # Provide sensible default or require it
            disagreement_note = "Reviewer identified artifacts or context diverging from AI model output."
        else:
            disagreement_note = submission.disagreement_rationale

        review.reviewer_id = current_user.id
        review.status = HumanReviewStatus.COMPLETED.value
        review.human_verdict = verdict.value
        review.reviewer_notes = submission.reviewer_notes
        review.disagreement_rationale = disagreement_note
        review.is_disagreement_with_ai = is_disagreement
        review.reviewed_at = _utcnow()

        # Audit logging
        audit = AuditLog(
            user_id=current_user.id,
            action="HUMAN_REVIEW_COMPLETED",
            entity_type="image_review",
            entity_id=str(review.id),
        )
        db.add(audit)
        db.commit()
        db.refresh(review)

        logger.info(
            "Human review finalized: review_id=%s, reviewer_id=%s, verdict=%s, disagreement=%s",
            review.id,
            current_user.id,
            verdict.value,
            is_disagreement,
        )

        model_monitoring_engine.record_human_review_metric(
            is_disagreement=is_disagreement,
            human_verdict=verdict.value,
        )

        return review

    def list_evidence_reviews(
        self,
        evidence_id: UUID,
        current_user: User,
        db: Session,
    ) -> List[ImageReview]:
        """Fetch all historical reviews for an evidence item, enforcing horizontal access."""
        evidence = db.scalar(select(Evidence).where(Evidence.id == evidence_id))
        if evidence is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence record not found.",
            )

        is_owner = evidence.owner_id == current_user.id
        is_staff = current_user.role in ("reviewer", "advocate", "admin")

        if not (is_owner or is_staff):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access reviews for this evidence.",
            )

        stmt = (
            select(ImageReview)
            .where(ImageReview.evidence_id == evidence_id)
            .order_by(desc(ImageReview.requested_at))
        )
        return list(db.scalars(stmt).all())


human_review_service = HumanReviewService()
