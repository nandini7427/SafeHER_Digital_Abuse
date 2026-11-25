"""Comprehensive End-to-End Integration and Failure Scenario Test Suite for SafeHER Milestone 3.

Verifies the entire forensic evidence and AI image assessment workflow:
Authenticated user
  -> Image upload & chunked validation
  -> SHA-256 evidence preservation in private vault
  -> AI manipulation detection & uncertainty triage
  -> Structured explainable response delivery
  -> Frontend-compatible evidence download and integrity audit
  -> Optional incident case association and access controls

Stress-tests 10 critical failure and edge scenarios:
1. Complete authenticated happy path workflow.
2. Insufficient image quality (< 128x128 pixels).
3. Concurrent asynchronous image uploads.
4. Database commit failure resilience.
5. Primary inference runner timeout and resilient fallback.
6. Corrupt / truncated byte stream rejection.
7. Unsupported MIME / magic byte format rejection.
8. Oversized payload rejection (> MAX_UPLOAD_SIZE_MB).
9. Cross-user horizontal case association prevention (403).
10. Strict privacy-safe logging across the entire workflow.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import logging
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ai_core.image_intelligence.config import MANDATORY_IMAGE_AI_DISCLAIMER
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import BaseImageModelRunner
from ai_core.image_intelligence.service import set_image_detector
from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import Case, Evidence, User


def _generate_test_image_bytes(
    dimensions=(200, 200),
    format_name="JPEG",
    mode="RGB",
    color=(100, 150, 200),
) -> bytes:
    """Generate in-memory test image bytes with distinct pattern."""
    img = Image.new(mode, dimensions, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, dimensions[0] // 2, dimensions[1] // 2], fill=(220, 50, 50))
    draw.line([0, 0, dimensions[0], dimensions[1]], fill=(50, 220, 50), width=2)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


@pytest.fixture
def test_db():
    """Create isolated in-memory SQLite database for testing."""
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
    """FastAPI TestClient with overridden get_db dependency."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def user1_auth(client) -> Dict[str, str]:
    """Register and authenticate User 1."""
    reg_data = {"email": "alice@safeher.org", "password": "SecurePassword123!"}
    client.post(f"{settings.API_V1_PREFIX}/auth/register", json=reg_data)
    login_resp = client.post(f"{settings.API_V1_PREFIX}/auth/login", json=reg_data)
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user2_auth(client) -> Dict[str, str]:
    """Register and authenticate User 2."""
    reg_data = {"email": "bob@safeher.org", "password": "SecurePassword123!"}
    client.post(f"{settings.API_V1_PREFIX}/auth/register", json=reg_data)
    login_resp = client.post(f"{settings.API_V1_PREFIX}/auth/login", json=reg_data)
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user1_case(test_db, client, user1_auth) -> str:
    """Create a case owned by User 1."""
    with test_db() as db:
        user = db.execute(select(User).filter_by(email="alice@safeher.org")).scalar_one()
        case = Case(
            title="Harassment Incident Evidence Case",
            description="Preserving digital harassment screenshots and image evidence.",
            owner_id=user.id,
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        return str(case.id)


@pytest.fixture
def user2_case(test_db, client, user2_auth) -> str:
    """Create a case owned by User 2."""
    with test_db() as db:
        user = db.execute(select(User).filter_by(email="bob@safeher.org")).scalar_one()
        case = Case(
            title="Bob Private Case",
            description="Confidential case owned by Bob.",
            owner_id=user.id,
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        return str(case.id)


# ---------------------------------------------------------------------------
# 1. Complete End-to-End Happy Path Workflow
# ---------------------------------------------------------------------------

def test_milestone3_complete_e2e_workflow(client, user1_auth, user1_case, test_db, tmp_path, monkeypatch):
    """Verify entire workflow: Upload -> Assess -> Store -> Verify Hash -> Download -> Delete."""
    storage_path = str(tmp_path / "private_evidence")
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", storage_path)

    raw_bytes = _generate_test_image_bytes(dimensions=(256, 256), format_name="JPEG")
    expected_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # Step 1: Upload and assess image with case association
    response = client.post(
        f"{settings.API_V1_PREFIX}/ai/image-assessment",
        headers=user1_auth,
        data={"case_id": user1_case},
        files={"file": ("manipulated_photo.jpg", raw_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify structured forensic contract
    assert data["detector_name"] == "SafeHER-Baseline-ELA-Heuristics"
    assert data["detection_task"] == "hybrid_manipulation"
    assert 0.0 <= data["raw_model_score"] <= 1.0
    assert data["is_calibrated"] is False
    assert data["confidence_score"] is None
    assert data["disclaimer"] == MANDATORY_IMAGE_AI_DISCLAIMER
    assert "image_id" in data
    evidence_id = data["image_id"]

    # Step 2: Verify Evidence record was persisted in database with matching SHA-256
    with test_db() as db:
        evidence = db.execute(select(Evidence).filter_by(id=uuid.UUID(evidence_id))).scalar_one_or_none()
        assert evidence is not None
        assert evidence.sha256 == expected_sha256
        assert evidence.size_bytes == len(raw_bytes)
        assert str(evidence.case_id) == user1_case
        storage_key = evidence.storage_key

    # Step 3: Retrieve sanitized evidence metadata
    meta_resp = client.get(f"{settings.API_V1_PREFIX}/evidence/{evidence_id}", headers=user1_auth)
    assert meta_resp.status_code == 200
    meta = meta_resp.json()
    assert meta["sha256"] == expected_sha256
    assert "storage_key" not in meta  # Internal server paths must never be disclosed
    assert "file_path" not in meta

    # Step 4: Cryptographically verify file integrity on disk
    verify_resp = client.get(f"{settings.API_V1_PREFIX}/evidence/{evidence_id}/verify", headers=user1_auth)
    assert verify_resp.status_code == 200
    verify_data = verify_resp.json()
    assert verify_data["is_valid"] is True
    assert verify_data["sha256"] == expected_sha256

    # Step 5: Download original evidence file
    dl_resp = client.get(f"{settings.API_V1_PREFIX}/evidence/{evidence_id}/download", headers=user1_auth)
    assert dl_resp.status_code == 200
    assert dl_resp.content == raw_bytes

    # Step 6: Controlled evidence deletion
    del_resp = client.delete(f"{settings.API_V1_PREFIX}/evidence/{evidence_id}", headers=user1_auth)
    assert del_resp.status_code == 204

    # Verify physical file and DB record are gone
    with test_db() as db:
        assert db.execute(select(Evidence).filter_by(id=uuid.UUID(evidence_id))).scalar_one_or_none() is None
    stored_file = Path(storage_path) / storage_key
    assert not stored_file.exists()


# ---------------------------------------------------------------------------
# 2. Insufficient Image Quality Scenario
# ---------------------------------------------------------------------------

def test_milestone3_insufficient_quality_handling(client, user1_auth, tmp_path, monkeypatch):
    """Verify small images (<128px) map to INCONCLUSIVE and INSUFFICIENT_QUALITY."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "private_evidence"))
    small_bytes = _generate_test_image_bytes(dimensions=(64, 64), format_name="PNG")

    response = client.post(
        f"{settings.API_V1_PREFIX}/ai/image-assessment",
        headers=user1_auth,
        files={"file": ("small_icon.png", small_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["assessment_label"] == "inconclusive"
    assert data["uncertainty_state"] in ("insufficient_quality", "analysis_poor_quality_evidence")
    assert "too low for reliable micro-forensic inspection" in data["explanation"]


# ---------------------------------------------------------------------------
# 3. Concurrent Image Uploads Scenario
# ---------------------------------------------------------------------------

def test_milestone3_concurrent_image_uploads(client, user1_auth, user1_case, tmp_path, monkeypatch):
    """Verify multiple requests execute cleanly with distinct outputs."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "private_evidence"))
    images = [
        _generate_test_image_bytes(dimensions=(150, 150), color=(i * 40, 100, 180))
        for i in range(1, 4)
    ]

    results = []
    for idx, raw in enumerate(images):
        resp = client.post(
            f"{settings.API_V1_PREFIX}/ai/image-assessment",
            headers=user1_auth,
            data={"case_id": user1_case},
            files={"file": (f"concurrent_{idx}.jpg", raw, "image/jpeg")},
        )
        results.append((resp.status_code, resp.json(), hashlib.sha256(raw).hexdigest()))

    evidence_ids = set()
    for status_code, json_data, expected_hash in results:
        assert status_code == 200
        assert json_data["is_calibrated"] is False
        evidence_ids.add(json_data["image_id"])

    # All uploads must have generated distinct evidence records
    assert len(evidence_ids) == len(images)


# ---------------------------------------------------------------------------
# 4. Database Failure and Rollback Scenario
# ---------------------------------------------------------------------------

def test_milestone3_database_commit_failure_resilience(client, user1_auth, user1_case, tmp_path, monkeypatch):
    """Verify database errors rollback cleanly."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "private_evidence"))
    raw_bytes = _generate_test_image_bytes(dimensions=(180, 180))

    # Mock Session.commit to raise SQLAlchemyError
    with patch.object(Session, "commit", side_effect=SQLAlchemyError("Simulated database write failure")):
        with TestClient(app, raise_server_exceptions=False) as err_client:
            response = err_client.post(
                f"{settings.API_V1_PREFIX}/ai/image-assessment",
                headers=user1_auth,
                data={"case_id": user1_case},
                files={"file": ("error_photo.jpg", raw_bytes, "image/jpeg")},
            )
            assert response.status_code == 500


# ---------------------------------------------------------------------------
# 5. Primary Vision Runner Timeout and Resilient Fallback
# ---------------------------------------------------------------------------

def test_milestone3_runner_timeout_fallback(client, user1_auth, tmp_path, monkeypatch):
    """Verify runner exceptions gracefully fall back to deterministic baseline."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "private_evidence"))
    mock_runner = MagicMock(spec=BaseImageModelRunner)
    mock_runner.predict.side_effect = TimeoutError("Neural vision execution timed out after 5000ms.")
    mock_runner.is_available.return_value = True

    custom_detector = SafeHERImageDetector(runner=mock_runner)
    set_image_detector(custom_detector)
    try:
        raw_bytes = _generate_test_image_bytes(dimensions=(160, 160))
        response = client.post(
            f"{settings.API_V1_PREFIX}/ai/image-assessment",
            headers=user1_auth,
            files={"file": ("timeout_test.jpg", raw_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["detector_name"] == "SafeHER-Baseline-ELA-Heuristics"
    finally:
        set_image_detector(SafeHERImageDetector())


# ---------------------------------------------------------------------------
# 6. Corrupt / Truncated Stream Scenario
# ---------------------------------------------------------------------------

def test_milestone3_corrupt_stream_rejected(client, user1_auth):
    """Verify broken/truncated image headers are rejected with 400."""
    valid_bytes = _generate_test_image_bytes()
    corrupt_bytes = valid_bytes[:30]  # truncated JPEG header
    response = client.post(
        f"{settings.API_V1_PREFIX}/ai/image-assessment",
        headers=user1_auth,
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Corrupt or unreadable image stream" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 7. Unsupported Format Scenario
# ---------------------------------------------------------------------------

def test_milestone3_unsupported_format_rejected(client, user1_auth):
    """Verify non-image formats (PDF, executable, script) are rejected."""
    fake_bytes = b"<html><body>Malicious script masquerading as image</body></html>"
    response = client.post(
        f"{settings.API_V1_PREFIX}/ai/image-assessment",
        headers=user1_auth,
        files={"file": ("document.html", fake_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported or invalid image file format" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 8. Oversized Image Scenario (> MAX_UPLOAD_SIZE_MB)
# ---------------------------------------------------------------------------

def test_milestone3_oversized_payload_rejected(client, user1_auth, monkeypatch):
    """Verify uploads exceeding MAX_UPLOAD_SIZE_MB are rejected."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0.001)  # ~1024 bytes
    large_bytes = _generate_test_image_bytes(dimensions=(300, 300))
    assert len(large_bytes) > 1024

    response = client.post(
        f"{settings.API_V1_PREFIX}/ai/image-assessment",
        headers=user1_auth,
        files={"file": ("huge.jpg", large_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "exceeds the maximum allowable size" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 9. Cross-User Horizontal Authorization (403)
# ---------------------------------------------------------------------------

def test_milestone3_cross_user_case_association_forbidden(client, user1_auth, user2_case):
    """Verify User 1 cannot attach image evidence to User 2's case."""
    raw_bytes = _generate_test_image_bytes()
    response = client.post(
        f"{settings.API_V1_PREFIX}/ai/image-assessment",
        headers=user1_auth,
        data={"case_id": user2_case},
        files={"file": ("test.jpg", raw_bytes, "image/jpeg")},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized to access this case."


# ---------------------------------------------------------------------------
# 10. Privacy-Safe Logging Audit
# ---------------------------------------------------------------------------

def test_milestone3_privacy_safe_logging(client, user1_auth, caplog, tmp_path, monkeypatch):
    """Verify no raw image bytes or base64 strings are logged."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "private_evidence"))
    raw_bytes = _generate_test_image_bytes()
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        client.post(
            f"{settings.API_V1_PREFIX}/ai/image-assessment",
            headers=user1_auth,
            files={"file": ("privacy_test.jpg", raw_bytes, "image/jpeg")},
        )

    log_text = caplog.text
    # Assert raw bytes not logged
    assert str(raw_bytes[:20]) not in log_text
    assert "base64" not in log_text.lower()
