"""FastAPI route handlers for SafeHER Human-in-the-Loop Image Review Workflow."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import (
    HumanReviewRequestCreate,
    HumanReviewResponse,
    HumanReviewSubmit,
)
from backend.app.security.dependencies import get_current_user
from backend.app.security.permissions import require_roles
from backend.app.services.review_service import human_review_service

router = APIRouter(prefix="/image", tags=["Human-in-the-Loop Image Review"])


@router.post(
    "/review/request",
    response_model=HumanReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request a human review for an AI image assessment",
    description=(
        "Enables users to request human expert or advocate triage for uncertain, ambiguous, "
        "or high-stakes visual manipulation assessments. Preserves an immutable snapshot "
        "of the original AI model output."
    ),
)
def request_human_review(
    request: HumanReviewRequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HumanReviewResponse:
    """Create a human review request for verified user-owned evidence."""
    review = human_review_service.request_human_review(request, current_user, db)
    return HumanReviewResponse.model_validate(review)


@router.get(
    "/review/{review_id}",
    response_model=HumanReviewResponse,
    summary="Retrieve details of a specific human review",
    description="Accessible by the requesting user or authorized review staff.",
)
def get_human_review(
    review_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HumanReviewResponse:
    """Fetch review details and current status."""
    review = human_review_service.get_review(review_id, current_user, db)
    return HumanReviewResponse.model_validate(review)


@router.get(
    "/reviews/pending",
    response_model=List[HumanReviewResponse],
    summary="List pending and active reviews for triage",
    description="Restricted to authorized reviewers, advocates, and administrators.",
)
def list_pending_reviews(
    current_reviewer: User = Depends(require_roles("reviewer", "advocate", "admin")),
    db: Session = Depends(get_db),
) -> List[HumanReviewResponse]:
    """List pending triage queue items for review staff."""
    reviews = human_review_service.list_pending_reviews(current_reviewer, db)
    return [HumanReviewResponse.model_validate(r) for r in reviews]


@router.post(
    "/review/{review_id}/claim",
    response_model=HumanReviewResponse,
    summary="Claim a review item for active inspection",
    description="Marks the review as under_review and assigns reviewer identity.",
)
def claim_human_review(
    review_id: UUID,
    current_reviewer: User = Depends(require_roles("reviewer", "advocate", "admin")),
    db: Session = Depends(get_db),
) -> HumanReviewResponse:
    """Assign review to current reviewer."""
    review = human_review_service.claim_review(review_id, current_reviewer, db)
    return HumanReviewResponse.model_validate(review)


@router.post(
    "/review/{review_id}/submit",
    response_model=HumanReviewResponse,
    summary="Submit final human review verdict and observations",
    description=(
        "Records the reviewer's independent findings, checks for disagreement with AI, "
        "and logs an immutable audit trail. Original AI outputs are never overwritten."
    ),
)
def submit_human_review(
    review_id: UUID,
    submission: HumanReviewSubmit,
    current_reviewer: User = Depends(require_roles("reviewer", "advocate", "admin")),
    db: Session = Depends(get_db),
) -> HumanReviewResponse:
    """Finalize human review outcome."""
    review = human_review_service.submit_human_review(
        review_id=review_id,
        submission=submission,
        current_user=current_reviewer,
        db=db,
    )
    return HumanReviewResponse.model_validate(review)


@router.get(
    "/evidence/{evidence_id}/reviews",
    response_model=List[HumanReviewResponse],
    summary="Get complete review history for an evidence item",
    description="Returns all historical reviews associated with an evidence record.",
)
def list_evidence_reviews(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[HumanReviewResponse]:
    """Fetch review history for an evidence record."""
    reviews = human_review_service.list_evidence_reviews(evidence_id, current_user, db)
    return [HumanReviewResponse.model_validate(r) for r in reviews]
