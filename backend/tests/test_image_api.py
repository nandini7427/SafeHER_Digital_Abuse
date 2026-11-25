"""Integration and security tests for SafeHER AI Image Intelligence Backend API.

Verifies:
- Authentication and authorization enforcement (missing token, invalid token, inactive user).
- Strict input validation (invalid format, corrupt image, oversized image, empty file).
- Case ownership verification and horizontal tenant isolation (404 missing, 403 unowned).
- Case association and secure evidence preservation (Evidence record, storage key, SHA-256).
- Audit log persistence with privacy safeguards.
- Fallback mechanics on model unavailability or runner errors.
- Privacy-safe logging (zero raw image bytes logged).
"""

import io
import logging
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_core.image_intelligence.config import (
    MANDATORY_IMAGE_AI_DISCLAIMER,
    ImageDetectorConfig,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import BaseImageModelRunner
from ai_core.image_intelligence.service import set_image_detector
from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import AuditLog, Case, Evidence, User


def _create_test_image_bytes(
    dimensions=(200, 200),
    format_name="JPEG",
    mode="RGB",
) -> bytes:
    """Generate in-memory test image bytes."""
    img = Image.new(mode, dimensions, color=(100, 150, 200))
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
def auth_headers(client) -> Dict[str, str]:
    """Helper fixture registering and logging in a standard test user."""
    reg_data = {
        "email": "image_victim@example.com",
        "password": "SafeHER-Secure-Password-123!",
    }
    reg_res = client.post("/api/v1/auth/register", json=reg_data)
    assert reg_res.status_code == 201

    login_res = client.post("/api/v1/auth/login", json=reg_data)
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def second_user_headers(client) -> Dict[str, str]:
    """Helper fixture registering a second user for multi-tenant isolation testing."""
    reg_data = {
        "email": "image_other_user@example.com",
        "password": "SafeHER-Other-Password-456!",
    }
    reg_res = client.post("/api/v1/auth/register", json=reg_data)
    assert reg_res.status_code == 201

    login_res = client.post("/api/v1/auth/login", json=reg_data)
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# 1. AUTHENTICATION & AUTHORIZATION TESTS
# ============================================================================


def test_image_assessment_unauthenticated_rejected(client):
    """Reject unauthenticated image assessment requests with 401."""
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/v1/ai/image-assessment",
        files={"file": ("evidence.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_image_assessment_invalid_token_rejected(client):
    """Reject invalid bearer tokens with 401."""
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers={"Authorization": "Bearer invalid.fake.token"},
        files={"file": ("evidence.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired access token."


def test_image_assessment_inactive_user_rejected(client, auth_headers, test_db):
    """Reject requests from inactive user accounts."""
    db = test_db()
    try:
        user = db.scalar(select(User).where(User.email == "image_victim@example.com"))
        assert user is not None
        user.is_active = False
        db.commit()
    finally:
        db.close()

    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("evidence.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "User not found or inactive."


# ============================================================================
# 2. VALID IMAGE ASSESSMENT & CONTRACT TESTS
# ============================================================================


def test_image_assessment_valid_image(client, auth_headers):
    """Verify successful image assessment with structured response."""
    img_bytes = _create_test_image_bytes(dimensions=(256, 256), format_name="JPEG")
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert "assessment_id" in data
    assert data["detector_name"] == "SafeHER-Baseline-ELA-Heuristics"
    assert data["detection_task"] == "hybrid_manipulation"
    assert 0.0 <= data["raw_model_score"] <= 1.0
    assert data["is_calibrated"] is False
    assert data["confidence_score"] is None
    # Multi-detector disagreement is an allowed safe outcome; never force an
    # inconclusive result into a fake positive or negative authenticity decision.
    assert data["assessment_label"] in (
        "no_obvious_artifacts",
        "suspicious_manipulation",
        "highly_likely_manipulated",
        "inconclusive",
    )
    assert data["disclaimer"] == MANDATORY_IMAGE_AI_DISCLAIMER
    assert len(data["model_limitations"]) >= 4
    assert data["metadata_summary"]["format"] == "JPEG"
    assert data["metadata_summary"]["dimensions"] == [256, 256]


def test_image_assessment_alias_endpoint(client, auth_headers):
    """Verify alias route /api/v1/ai/image/assess works identically."""
    img_bytes = _create_test_image_bytes(dimensions=(200, 200), format_name="PNG")
    response = client.post(
        "/api/v1/ai/image/assess",
        headers=auth_headers,
        files={"file": ("photo.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["metadata_summary"]["format"] == "PNG"


# ============================================================================
# 3. INPUT VALIDATION & SECURITY REJECTION TESTS
# ============================================================================


def test_image_assessment_invalid_file_rejected(client, auth_headers):
    """Reject non-image files (e.g. HTML/text masquerading as an image)."""
    fake_bytes = b"<html><body>Malicious payload</body></html>"
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("test.jpg", fake_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported or invalid image file format" in response.json()["detail"]


def test_image_assessment_corrupt_image_rejected(client, auth_headers):
    """Reject truncated / corrupt image streams."""
    valid_bytes = _create_test_image_bytes()
    corrupt_bytes = valid_bytes[:30]  # severely truncated
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Corrupt or unreadable image stream" in response.json()["detail"]


def test_image_assessment_empty_image_rejected(client, auth_headers):
    """Reject zero-byte upload."""
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_image_assessment_oversized_image_rejected(client, auth_headers, monkeypatch):
    """Reject images exceeding MAX_UPLOAD_SIZE_MB."""
    # Temporarily set max upload size to 0.001 MB (~1024 bytes)
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0.001)
    large_bytes = _create_test_image_bytes(dimensions=(300, 300))
    assert len(large_bytes) > 1024

    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("large.jpg", large_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "exceeds the maximum allowable size" in response.json()["detail"]


# ============================================================================
# 4. CASE ASSOCIATION & EVIDENCE PRESERVATION TESTS
# ============================================================================


def test_image_assessment_missing_case_not_found(client, auth_headers):
    """Reject assessment with non-existent case_id with 404."""
    img_bytes = _create_test_image_bytes()
    random_case_id = str(uuid.uuid4())
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        data={"case_id": random_case_id},
        files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found."


def test_image_assessment_unowned_case_forbidden(client, auth_headers, second_user_headers, test_db):
    """Reject linking an image assessment to a case owned by another user with 403."""
    db = test_db()
    try:
        user2 = db.scalar(select(User).where(User.email == "image_other_user@example.com"))
        unowned_case = Case(
            owner_id=user2.id,
            title="Victim 2 Confidential Case",
            description="Private evidence",
            status="open",
        )
        db.add(unowned_case)
        db.commit()
        unowned_case_id = str(unowned_case.id)
    finally:
        db.close()

    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,  # User 1 attempting to link to User 2's case
        data={"case_id": unowned_case_id},
        files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized to access this case."


def test_image_assessment_with_owned_case_succeeds(client, auth_headers, test_db, tmp_path, monkeypatch):
    """Verify linking an owned case persists Evidence, saves file, and writes AuditLog."""
    # Point private storage to a temporary directory for clean isolation
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "private_evidence"))

    db = test_db()
    try:
        user1 = db.scalar(select(User).where(User.email == "image_victim@example.com"))
        user1_id = user1.id
        owned_case = Case(
            owner_id=user1_id,
            title="Harassment Incident Evidence Case",
            description="Suspect sent manipulated photo",
            status="open",
        )
        db.add(owned_case)
        db.commit()
        owned_case_id = str(owned_case.id)
    finally:
        db.close()

    img_bytes = _create_test_image_bytes(dimensions=(256, 256), format_name="JPEG")
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        data={"case_id": owned_case_id},
        files={"file": ("suspect_photo.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] is not None
    evidence_id = uuid.UUID(data["image_id"])

    # Verify Evidence record in database
    db = test_db()
    try:
        evidence = db.scalar(select(Evidence).where(Evidence.id == evidence_id))
        assert evidence is not None
        assert evidence.case_id == uuid.UUID(owned_case_id)
        assert evidence.owner_id == user1_id
        assert evidence.original_filename == "suspect_photo.jpg"
        assert evidence.size_bytes == len(img_bytes)
        assert evidence.sha256 == data["metadata_summary"]["sha256"]

        # Verify pristine evidence file saved on disk
        saved_file = Path(settings.PRIVATE_STORAGE_PATH) / evidence.storage_key
        assert saved_file.exists()
        assert saved_file.read_bytes() == img_bytes

        # Verify AuditLog created
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.action == "AI_IMAGE_ASSESSMENT",
                AuditLog.entity_id == owned_case_id,
            )
        )
        assert audit is not None
        assert audit.user_id == user1_id
    finally:
        db.close()


# ============================================================================
# 5. RESILIENT FALLBACK & PRIVACY LOGGING TESTS
# ============================================================================


def test_image_assessment_runner_failure_fallback(client, auth_headers):
    """Verify fallback to baseline when a model runner raises an unexpected error."""
    failing_runner = MagicMock(spec=BaseImageModelRunner)
    failing_runner.predict.side_effect = RuntimeError("Simulated runtime crash")

    detector = SafeHERImageDetector(runner=failing_runner)
    set_image_detector(detector)

    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
    )
    # Must succeed via fallback
    assert response.status_code == 200
    assert response.json()["detector_name"] == "SafeHER-Baseline-ELA-Heuristics"


def test_image_assessment_privacy_safe_logging(client, auth_headers, caplog):
    """Verify telemetry logs do not expose raw image bytes or base64 data."""
    caplog.set_level(logging.INFO)
    img_bytes = _create_test_image_bytes()

    response = client.post(
        "/api/v1/ai/image-assessment",
        headers=auth_headers,
        files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200

    log_text = caplog.text
    # Log must confirm completion
    assert "Image assessment completed" in log_text
    # Log must NEVER contain raw binary signatures or large hex chunks
    assert str(img_bytes[:20]) not in log_text
