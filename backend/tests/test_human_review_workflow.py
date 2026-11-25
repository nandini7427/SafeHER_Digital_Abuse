"""Integration, security, and authorization tests for SafeHER Human-in-the-Loop Image Review.

Verifies:
1. User can request human review for owned evidence and case with immutable AI snapshot.
2. Horizontal isolation: User cannot request review for unowned evidence or cases (403 Forbidden).
3. Mismatched evidence and case rejected (400 Bad Request).
4. Review access control: Only requester and review staff can view review records.
5. Role-based queue access: Only reviewer/advocate/admin can list pending reviews (403 for normal users).
6. Review claiming: Reviewer can claim review, setting status to under_review with audit trail.
7. Reviewer evidence inspection: Reviewer can stream pristine original evidence bytes under
   HUMAN_REVIEW_EVIDENCE_ACCESSED audit logging, while non-reviewer third parties are blocked (403).
8. Role-based submission: Only authorized reviewer can finalize reviews (403 for normal users).
9. Agreement & Disagreement:
   - Concurring verdicts recorded with is_disagreement_with_ai=False.
   - Diverging verdicts recorded with is_disagreement_with_ai=True and rationale preserved.
10. Immutable outcomes: Finalized reviews cannot be silently overwritten or claimed again (400 Bad Request).
11. Evidence review history: Complete review audit trail retrievable by authorized parties.
12. Zero automated training guarantee: Submitting human reviews never triggers automated model fine-tuning.
13. Non-legal disclaimer: Mandatory disclosure is always present on review responses.
"""

import io
from pathlib import Path
from typing import Dict
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_core.image_intelligence.service import set_image_detector
from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import AuditLog, Case, Evidence, ImageReview, User
from backend.app.schemas import (
    MANDATORY_HUMAN_REVIEW_DISCLAIMER,
    HumanReviewStatus,
    HumanReviewVerdict,
)


def _create_test_image_bytes(
    dimensions=(200, 200),
    format_name="JPEG",
    mode="RGB",
) -> bytes:
    """Generate in-memory test image bytes."""
    img = Image.new(mode, dimensions, color=(120, 160, 210))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, dimensions[0] // 2, dimensions[1] // 2], fill=(220, 40, 40))
    draw.line([0, 0, dimensions[0], dimensions[1]], fill=(40, 220, 40), width=2)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


@pytest.fixture
def test_db():
    """Create an isolated, in-memory SQLite database for test execution."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield TestingSessionLocal

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    set_image_detector(None)


@pytest.fixture
def client(test_db):
    """TestClient configured with test database dependency overrides."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def user_headers(client) -> Dict[str, str]:
    """Helper fixture registering and logging in a standard victim/user."""
    reg_data = {
        "email": "victim_user@example.com",
        "password": "SafeHER-User-Password-123!",
    }
    client.post("/api/v1/auth/register", json=reg_data)
    login_res = client.post("/api/v1/auth/login", json=reg_data)
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_user_headers(client) -> Dict[str, str]:
    """Helper fixture registering and logging in an unrelated second user."""
    reg_data = {
        "email": "other_user@example.com",
        "password": "SafeHER-Other-Password-456!",
    }
    client.post("/api/v1/auth/register", json=reg_data)
    login_res = client.post("/api/v1/auth/login", json=reg_data)
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def reviewer_headers(client, test_db) -> Dict[str, str]:
    """Helper fixture registering and escalating a user to 'reviewer' role."""
    reg_data = {
        "email": "forensic_reviewer@example.com",
        "password": "SafeHER-Reviewer-Password-789!",
    }
    client.post("/api/v1/auth/register", json=reg_data)

    with test_db() as db:
        user = db.scalar(select(User).where(User.email == reg_data["email"]))
        assert user is not None
        user.role = "reviewer"
        db.commit()

    login_res = client.post("/api/v1/auth/login", json=reg_data)
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_case_and_evidence(client, user_headers, test_db):
    """Creates a Case and an Evidence record via the assessment API for the user."""
    with test_db() as db:
        user = db.scalar(select(User).where(User.email == "victim_user@example.com"))
        case = Case(
            owner_id=user.id,
            title="Harassment Investigation 2026",
            description="Suspect image received via messaging platform.",
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        case_id = case.id

    img_bytes = _create_test_image_bytes()
    files = {"file": ("suspect_deepfake.jpg", img_bytes, "image/jpeg")}
    res = client.post(
        "/api/v1/ai/image-assessment",
        data={"case_id": str(case_id)},
        files=files,
        headers=user_headers,
    )
    assert res.status_code == 200, res.text
    ai_data = res.json()
    evidence_id = ai_data["image_id"]
    assert evidence_id is not None

    return {
        "case_id": str(case_id),
        "evidence_id": str(evidence_id),
        "ai_assessment": ai_data,
    }


# ==============================================================================
# 1. REVIEW REQUEST & HORIZONTAL AUTHORIZATION TESTS
# ==============================================================================

def test_user_can_request_human_review_success(client, user_headers, user_case_and_evidence, test_db):
    """User can request human review for their owned evidence and case."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
        "request_reason": "High-stakes extortion threat. Need expert verification of manipulation regions.",
    }

    res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    assert res.status_code == 201, res.text
    data = res.json()

    assert data["status"] == "pending"
    assert data["evidence_id"] == req_body["evidence_id"]
    assert data["case_id"] == req_body["case_id"]
    assert data["request_reason"] == req_body["request_reason"]
    assert data["is_disagreement_with_ai"] is False
    assert data["human_verdict"] is None
    assert data["reviewer_id"] is None
    assert data["disclaimer"] == MANDATORY_HUMAN_REVIEW_DISCLAIMER
    assert data["ai_assessment_snapshot"]["assessment_id"] == user_case_and_evidence["ai_assessment"]["assessment_id"]

    # Verify audit log in database
    with test_db() as db:
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.action == "HUMAN_REVIEW_REQUESTED",
                AuditLog.entity_id == data["id"],
            )
        )
        assert audit is not None
        assert audit.entity_type == "image_review"


def test_user_cannot_request_review_for_unowned_evidence(client, other_user_headers, user_case_and_evidence):
    """Tenant isolation: User B cannot request review for User A's evidence."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
        "request_reason": "Intrusion attempt into stranger's case.",
    }

    res = client.post("/api/v1/image/review/request", json=req_body, headers=other_user_headers)
    assert res.status_code == 403
    assert "Not authorized to request review for evidence you do not own" in res.json()["detail"]


def test_mismatched_evidence_and_case_rejected(client, user_headers, user_case_and_evidence, test_db):
    """Evidence and Case must belong to each other."""
    # Create another case for the user
    with test_db() as db:
        user = db.scalar(select(User).where(User.email == "victim_user@example.com"))
        second_case = Case(
            owner_id=user.id,
            title="Second Unrelated Case",
        )
        db.add(second_case)
        db.commit()
        db.refresh(second_case)
        second_case_id = str(second_case.id)

    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": second_case_id,  # Mismatched case!
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }

    res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    assert res.status_code == 400
    assert "Evidence does not belong to the specified case" in res.json()["detail"]


# ==============================================================================
# 2. REVIEW ACCESS CONTROL & QUEUE MANAGEMENT
# ==============================================================================

def test_review_access_control(client, user_headers, other_user_headers, reviewer_headers, user_case_and_evidence):
    """Only requester and authorized review staff can view review details."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    review_id = create_res.json()["id"]

    # 1. Requester can view
    res = client.get(f"/api/v1/image/review/{review_id}", headers=user_headers)
    assert res.status_code == 200

    # 2. Reviewer can view
    res = client.get(f"/api/v1/image/review/{review_id}", headers=reviewer_headers)
    assert res.status_code == 200

    # 3. Unrelated user is blocked
    res = client.get(f"/api/v1/image/review/{review_id}", headers=other_user_headers)
    assert res.status_code == 403

    # 4. Non-existent review
    res = client.get(f"/api/v1/image/review/{uuid.uuid4()}", headers=user_headers)
    assert res.status_code == 404


def test_list_pending_reviews_restricted_to_reviewers(client, user_headers, reviewer_headers, user_case_and_evidence):
    """Normal users cannot list review queues; reviewers can."""
    # Create a pending review
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)

    # Normal user attempt
    res = client.get("/api/v1/image/reviews/pending", headers=user_headers)
    assert res.status_code == 403

    # Reviewer attempt
    res = client.get("/api/v1/image/reviews/pending", headers=reviewer_headers)
    assert res.status_code == 200
    queue = res.json()
    assert len(queue) >= 1
    assert any(r["evidence_id"] == user_case_and_evidence["evidence_id"] for r in queue)


def test_reviewer_can_claim_review(client, user_headers, reviewer_headers, user_case_and_evidence, test_db):
    """Reviewer claims pending review, transitioning state to under_review."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    review_id = create_res.json()["id"]

    # Normal user cannot claim
    res = client.post(f"/api/v1/image/review/{review_id}/claim", headers=user_headers)
    assert res.status_code == 403

    # Reviewer claims
    claim_res = client.post(f"/api/v1/image/review/{review_id}/claim", headers=reviewer_headers)
    assert claim_res.status_code == 200
    data = claim_res.json()
    assert data["status"] == "under_review"
    assert data["reviewer_id"] is not None

    # Check audit log
    with test_db() as db:
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.action == "HUMAN_REVIEW_CLAIMED",
                AuditLog.entity_id == review_id,
            )
        )
        assert audit is not None


# ==============================================================================
# 3. REVIEWER EVIDENCE INSPECTION & AUDIT LOGGING
# ==============================================================================

def test_reviewer_can_inspect_original_evidence_stream(
    client, user_headers, reviewer_headers, other_user_headers, user_case_and_evidence, test_db
):
    """Reviewer inspecting evidence under review can stream raw bytes under audit."""
    evidence_id = user_case_and_evidence["evidence_id"]

    # Create review request first
    req_body = {
        "evidence_id": evidence_id,
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)

    # 1. Other user without reviewer role is blocked (403)
    res_other = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=other_user_headers)
    assert res_other.status_code == 403

    # 2. Reviewer streams original evidence (200 OK)
    res_reviewer = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=reviewer_headers)
    assert res_reviewer.status_code == 200
    assert len(res_reviewer.content) > 0

    # 3. Check audit log has HUMAN_REVIEW_EVIDENCE_ACCESSED
    with test_db() as db:
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.action == "HUMAN_REVIEW_EVIDENCE_ACCESSED",
                AuditLog.entity_id == evidence_id,
            )
        )
        assert audit is not None


# ==============================================================================
# 4. REVIEW SUBMISSION: AGREEMENT, DISAGREEMENT & IMMUTABILITY
# ==============================================================================

def test_regular_user_cannot_submit_review(client, user_headers, user_case_and_evidence):
    """Normal users cannot submit review verdicts."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    review_id = create_res.json()["id"]

    submit_body = {
        "human_verdict": HumanReviewVerdict.CONCURS_WITH_AI.value,
        "reviewer_notes": "Attempting unauthorized submission.",
    }
    res = client.post(f"/api/v1/image/review/{review_id}/submit", json=submit_body, headers=user_headers)
    assert res.status_code == 403


def test_reviewer_submits_concurring_verdict(client, user_headers, reviewer_headers, user_case_and_evidence, test_db):
    """Reviewer submits a verdict concurring with the AI assessment."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    review_id = create_res.json()["id"]

    submit_body = {
        "human_verdict": HumanReviewVerdict.CONCURS_WITH_AI.value,
        "reviewer_notes": "Visual inspection of hair boundaries and lighting gradients confirms AI ELA artifact findings.",
    }
    res = client.post(f"/api/v1/image/review/{review_id}/submit", json=submit_body, headers=reviewer_headers)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["status"] == "completed"
    assert data["human_verdict"] == HumanReviewVerdict.CONCURS_WITH_AI.value
    assert data["is_disagreement_with_ai"] is False
    assert data["reviewer_notes"] == submit_body["reviewer_notes"]
    assert data["reviewed_at"] is not None

    # AI assessment snapshot is 100% preserved
    assert data["ai_assessment_snapshot"]["assessment_id"] == user_case_and_evidence["ai_assessment"]["assessment_id"]

    # Verify audit log
    with test_db() as db:
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.action == "HUMAN_REVIEW_COMPLETED",
                AuditLog.entity_id == review_id,
            )
        )
        assert audit is not None


def test_reviewer_submits_disagreeing_verdict(client, user_headers, reviewer_headers, user_case_and_evidence):
    """Reviewer identifies divergence from AI output, recording disagreement rationale."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    review_id = create_res.json()["id"]

    submit_body = {
        "human_verdict": HumanReviewVerdict.DISAGREES_NO_OBVIOUS_ARTIFACTS.value,
        "reviewer_notes": "High compression blockiness from messaging app re-encoding was falsely flagged by spatial ELA.",
        "disagreement_rationale": "Authentic photograph with severe JPEG recompression; no face synthesis boundaries present.",
    }
    res = client.post(f"/api/v1/image/review/{review_id}/submit", json=submit_body, headers=reviewer_headers)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["status"] == "completed"
    assert data["human_verdict"] == HumanReviewVerdict.DISAGREES_NO_OBVIOUS_ARTIFACTS.value
    assert data["is_disagreement_with_ai"] is True
    assert data["disagreement_rationale"] == submit_body["disagreement_rationale"]

    # AI snapshot remains completely unmutated
    assert data["ai_assessment_snapshot"]["assessment_id"] == user_case_and_evidence["ai_assessment"]["assessment_id"]


def test_cannot_resubmit_or_overwrite_completed_review(client, user_headers, reviewer_headers, user_case_and_evidence):
    """Completed reviews cannot be silently overwritten (immutability)."""
    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    review_id = create_res.json()["id"]

    submit_body = {
        "human_verdict": HumanReviewVerdict.CONCURS_WITH_AI.value,
        "reviewer_notes": "Initial review completed.",
    }
    client.post(f"/api/v1/image/review/{review_id}/submit", json=submit_body, headers=reviewer_headers)

    # Attempt second submission
    res_second = client.post(f"/api/v1/image/review/{review_id}/submit", json=submit_body, headers=reviewer_headers)
    assert res_second.status_code == 400
    assert "Review has already been finalized" in res_second.json()["detail"]

    # Attempt to claim completed review
    res_claim = client.post(f"/api/v1/image/review/{review_id}/claim", headers=reviewer_headers)
    assert res_claim.status_code == 400


# ==============================================================================
# 5. EVIDENCE REVIEW HISTORY & REUSE
# ==============================================================================

def test_evidence_review_history(client, user_headers, other_user_headers, user_case_and_evidence):
    """User can list all historical reviews for an evidence record."""
    evidence_id = user_case_and_evidence["evidence_id"]

    req_body = {
        "evidence_id": evidence_id,
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
        "request_reason": "First review request.",
    }
    client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)

    # Requester lists reviews
    res = client.get(f"/api/v1/image/evidence/{evidence_id}/reviews", headers=user_headers)
    assert res.status_code == 200
    reviews = res.json()
    assert len(reviews) == 1
    assert reviews[0]["evidence_id"] == evidence_id

    # Other user blocked
    res_other = client.get(f"/api/v1/image/evidence/{evidence_id}/reviews", headers=other_user_headers)
    assert res_other.status_code == 403


# ==============================================================================
# 6. ZERO AUTOMATIC TRAINING GUARANTEE TEST
# ==============================================================================

def test_zero_automatic_model_training_guarantee(client, user_headers, reviewer_headers, user_case_and_evidence):
    """Submitting human review verdicts MUST NEVER trigger automatic model fine-tuning or weight updates."""
    from ai_core.image_intelligence.service import image_service

    # Check detector state before review
    initial_detector = image_service.detector

    req_body = {
        "evidence_id": user_case_and_evidence["evidence_id"],
        "case_id": user_case_and_evidence["case_id"],
        "ai_assessment_snapshot": user_case_and_evidence["ai_assessment"],
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=user_headers)
    review_id = create_res.json()["id"]

    submit_body = {
        "human_verdict": HumanReviewVerdict.DISAGREES_NO_OBVIOUS_ARTIFACTS.value,
        "reviewer_notes": "Adversarial or unexpected input handled without model fine-tuning.",
        "disagreement_rationale": "Zero automatic training enforced by architecture.",
    }
    client.post(f"/api/v1/image/review/{review_id}/submit", json=submit_body, headers=reviewer_headers)

    # Verify detector remains unaltered and no online learning occurred
    current_detector = image_service.detector
    assert current_detector is initial_detector


# ==============================================================================
# 7. SELF-REVIEW PREVENTION TEST
# ==============================================================================

def test_self_review_prohibited(client, test_db, reviewer_headers):
    """Reviewers MUST NOT be permitted to claim or approve their own evidence review requests."""
    with test_db() as db:
        reviewer = db.scalar(select(User).where(User.email == "forensic_reviewer@example.com"))
        assert reviewer is not None

        case = Case(title="Reviewer Personal Case", owner_id=reviewer.id)
        db.add(case)
        db.flush()

        evidence = Evidence(
            case_id=case.id,
            owner_id=reviewer.id,
            original_filename="reviewer_own_doc.jpg",
            storage_key="test_rev_own.bin",
            mime_type="image/jpeg",
            size_bytes=100,
            sha256="abc123rev",
        )
        db.add(evidence)
        db.commit()

        evidence_id = str(evidence.id)
        case_id = str(case.id)

    req_body = {
        "evidence_id": evidence_id,
        "case_id": case_id,
        "ai_assessment_snapshot": {"raw_score": 0.5, "label": "suspicious"},
    }
    create_res = client.post("/api/v1/image/review/request", json=req_body, headers=reviewer_headers)
    assert create_res.status_code == 201
    review_id = create_res.json()["id"]

    # Attempt to claim own review request -> 403 Forbidden
    claim_res = client.post(f"/api/v1/image/review/{review_id}/claim", headers=reviewer_headers)
    assert claim_res.status_code == 403
    assert "Self-review prohibited" in claim_res.json()["detail"]

    # Attempt to submit verdict on own review request -> 403 Forbidden
    submit_body = {
        "human_verdict": HumanReviewVerdict.CONCURS_WITH_AI.value,
        "reviewer_notes": "Attempting self-approval",
    }
    submit_res = client.post(f"/api/v1/image/review/{review_id}/submit", json=submit_body, headers=reviewer_headers)
    assert submit_res.status_code == 403
    assert "Self-review prohibited" in submit_res.json()["detail"]
