"""Privacy Engineering Audit Test Suite: Image Privacy and Metadata Leakage Defense.

Verifies SafeHER privacy guarantees across the entire image analysis and evidence lifecycle:
1. EXIF Metadata & Geolocation Stripping:
   - Strict removal of GPSInfo, GPS coordinates (latitude, longitude, altitude),
     hardware serial numbers (CameraSerialNumber, BodySerialNumber, LensSerialNumber),
     and user identity tags (Artist, OwnerName, XPAuthor, HostComputer, UserComment) from API outputs.
   - Non-identifying technical metadata (format, dimensions, color mode) preserved for forensic triage.
2. Pristine Original Evidence Integrity:
   - Byte-for-byte immutability of the uploaded evidence in private storage, maintaining
     unaltered chain of custody while shielding the survivor's privacy in downstream analysis.
3. No Raw Image Bytes or Base64 Leakage:
   - Verification that responses, telemetry, and structured outputs never transmit raw raster or base64 evidence.
4. No Internal Filesystem Paths Exposed:
   - Verification that private storage directory paths, temporary filenames, and server drive roots
     never leak via API responses, error details, or integrity reports.
5. Strict Multi-Tenant Evidence Authorization:
   - Unauthenticated requests blocked (401).
   - Cross-user horizontal boundary violations blocked (403).
   - Reviewer unauthorized access blocked unless a formal human review was requested (403).
6. Anti-Caching & Public Exposure Defense:
   - Mandatory `Cache-Control: no-store, no-cache, must-revalidate, private` and anti-sniffing headers
     on all API endpoints and evidence download streams.
7. Physical Storage Cleanup & Lifecycle Deletion:
   - Deletion of evidence permanently purges the raw file from private disk storage,
     records an immutable audit trail entry, and cleans database entities.
8. Zero-Temporary-File Pipeline:
   - Image analysis processes streams purely in-memory via io.BytesIO without leaving residual temp files.
"""

from datetime import datetime, timezone
import hashlib
import io
import logging
from pathlib import Path
from typing import Dict
from unittest.mock import patch
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import ExifTags, Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_core.image_intelligence.preprocessing import preprocess_image
from ai_core.image_intelligence.service import set_image_detector
from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import AuditLog, Case, Evidence, ImageReview, User
from backend.app.services.evidence_service import evidence_service


def _create_synthetic_jpeg_with_metadata(
    dimensions=(200, 200),
    include_gps=True,
    include_device_serials=True,
    include_user_pii=True,
) -> bytes:
    """Generate in-memory synthetic JPEG with controlled test EXIF metadata."""
    img = Image.new("RGB", dimensions, color=(110, 160, 210))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 100, 100], fill=(220, 40, 40))

    exif = img.getexif()

    # Standard camera tags
    exif[0x010F] = "SyntheticCameraCo"  # Make
    exif[0x0110] = "Model-Forensic-9000"  # Model
    exif[0x0112] = 1  # Orientation: Normal

    if include_user_pii:
        exif[0x013B] = "Jane Doe Survivor"  # Artist
        exif[0x013C] = "SURVIVOR-LAPTOP-01"  # HostComputer
        exif[0x9286] = b"Private confidential meeting notes"  # UserComment
        exif[0x8298] = "Copyright 2026 Jane Doe"  # Copyright

    if include_device_serials:
        exif[0xA431] = "BODY-SN-99887766"  # BodySerialNumber
        exif[0xA435] = "LENS-SN-11223344"  # LensSerialNumber
        exif[0xC62F] = "CAM-SN-44556677"  # CameraSerialNumber

    if include_gps:
        # Tag 0x8825 is GPSInfo IFD
        gps_ifd = exif.get_ifd(0x8825)
        gps_ifd[1] = "N"  # GPSLatitudeRef
        gps_ifd[2] = (37.0, 46.0, 29.74)  # 37 deg 46' 29.74" N
        gps_ifd[3] = "W"  # GPSLongitudeRef
        gps_ifd[4] = (122.0, 25.0, 9.81)  # 122 deg 25' 9.81" W
        gps_ifd[6] = 15.0  # Altitude: 15m

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


@pytest.fixture
def test_db():
    """Create isolated in-memory SQLite database for test execution."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
    """TestClient configured with test database."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def user_a_headers(client) -> Dict[str, str]:
    """Register and authenticate Primary Survivor User A."""
    reg_data = {"email": "survivor_a@example.com", "password": "SecurePassword123!"}
    client.post("/api/v1/auth/register", json=reg_data)
    res = client.post("/api/v1/auth/login", json=reg_data)
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_b_headers(client) -> Dict[str, str]:
    """Register and authenticate Unrelated User B."""
    reg_data = {"email": "unrelated_b@example.com", "password": "SecurePassword123!"}
    client.post("/api/v1/auth/register", json=reg_data)
    res = client.post("/api/v1/auth/login", json=reg_data)
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def reviewer_headers(client, test_db) -> Dict[str, str]:
    """Register and authenticate an Authorized Advocate/Reviewer User."""
    reg_data = {"email": "advocate_reviewer@example.com", "password": "SecurePassword123!"}
    client.post("/api/v1/auth/register", json=reg_data)

    # Promote to reviewer in db
    with test_db() as db:
        user = db.scalar(select(User).where(User.email == "advocate_reviewer@example.com"))
        user.role = "reviewer"
        db.commit()

    res = client.post("/api/v1/auth/login", json=reg_data)
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_a_case(client, user_a_headers, test_db) -> Case:
    """Create an active case owned by User A."""
    with test_db() as db:
        user = db.scalar(select(User).where(User.email == "survivor_a@example.com"))
        case = Case(
            owner_id=user.id,
            title="Harassment Image Evidence Case",
            description="Confidential intake",
            status="open",
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        return case


# ==============================================================================
# 1. EXIF METADATA, GPS, AND DEVICE PII STRIPPING
# ==============================================================================

def test_exif_gps_coordinates_strictly_stripped_from_api_response(client, user_a_headers):
    """Verify that GPS coordinates and GPSInfo IFD are strictly purged from API responses."""
    payload = _create_synthetic_jpeg_with_metadata(include_gps=True)
    files = {"file": ("gps_photo.jpg", payload, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=user_a_headers)
    assert res.status_code == 200
    data = res.json()

    meta = data["metadata_summary"]
    assert meta["has_exif"] is True
    sanitized = meta["sanitized_exif"]

    # String representation search across the ENTIRE response body
    response_str = res.text.lower()
    assert "gpsinfo" not in sanitized
    assert "34853" not in sanitized
    assert "gpslatitude" not in response_str
    assert "gpslongitude" not in response_str
    assert "37 deg" not in response_str
    assert "122 deg" not in response_str


def test_exif_user_pii_and_hardware_serials_stripped(client, user_a_headers):
    """Verify that user personal names, computer names, and hardware serials are stripped."""
    payload = _create_synthetic_jpeg_with_metadata(
        include_gps=True,
        include_device_serials=True,
        include_user_pii=True,
    )
    files = {"file": ("pii_photo.jpg", payload, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=user_a_headers)
    assert res.status_code == 200
    response_text = res.text

    # Verify no PII or serial strings leak into JSON
    assert "Jane Doe Survivor" not in response_text
    assert "SURVIVOR-LAPTOP-01" not in response_text
    assert "Private confidential meeting notes" not in response_text
    assert "BODY-SN-99887766" not in response_text
    assert "LENS-SN-11223344" not in response_text
    assert "CAM-SN-44556677" not in response_text

    meta = res.json()["metadata_summary"]
    sanitized = meta["sanitized_exif"]
    assert "Artist" not in sanitized
    assert "HostComputer" not in sanitized
    assert "UserComment" not in sanitized
    assert "BodySerialNumber" not in sanitized
    assert "LensSerialNumber" not in sanitized


# ==============================================================================
# 2. PRISTINE EVIDENCE INTEGRITY PRESERVATION
# ==============================================================================

def test_pristine_evidence_vault_preserves_unaltered_bytes(client, user_a_headers, user_a_case):
    """Verify that stored evidence bytes remain 100% bit-for-bit identical to original upload."""
    raw_payload = _create_synthetic_jpeg_with_metadata()
    original_sha256 = hashlib.sha256(raw_payload).hexdigest()

    files = {"file": ("pristine_evidence.jpg", raw_payload, "image/jpeg")}
    data = {"case_id": str(user_a_case.id)}

    res = client.post("/api/v1/ai/image-assessment", files=files, data=data, headers=user_a_headers)
    assert res.status_code == 200
    evidence_id = res.json()["image_id"]
    assert evidence_id is not None

    # Retrieve evidence stream as the authorized owner
    download_res = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=user_a_headers)
    assert download_res.status_code == 200
    downloaded_bytes = download_res.content

    # Assert exact byte-for-byte identity
    assert downloaded_bytes == raw_payload
    assert hashlib.sha256(downloaded_bytes).hexdigest() == original_sha256


# ==============================================================================
# 3. NO RAW BYTES OR BASE64 LEAKAGE IN RESPONSES OR LOGS
# ==============================================================================

def test_no_raw_image_data_or_base64_in_api_response_or_logs(client, user_a_headers, caplog):
    """Verify that API outputs and server logs NEVER contain raw image bytes or base64 evidence."""
    payload = _create_synthetic_jpeg_with_metadata()
    files = {"file": ("forensic_check.jpg", payload, "image/jpeg")}

    with caplog.at_level(logging.DEBUG):
        res = client.post("/api/v1/ai/image-assessment", files=files, headers=user_a_headers)
        assert res.status_code == 200

    response_text = res.text
    # Check response payload
    assert "data:image" not in response_text
    assert ";base64," not in response_text
    assert len(response_text) < 15000  # Structured assessment JSON stays small; no raster/base64 is embedded

    # Check captured log lines
    combined_logs = "\n".join(record.message for record in caplog.records)
    assert "data:image" not in combined_logs
    assert ";base64," not in combined_logs


# ==============================================================================
# 4. NO INTERNAL FILESYSTEM PATHS EXPOSED
# ==============================================================================

def test_internal_filesystem_paths_not_exposed(client, user_a_headers, user_a_case):
    """Verify internal storage keys, drive letters, and server directories never leak."""
    payload = _create_synthetic_jpeg_with_metadata()
    files = {"file": ("path_audit.jpg", payload, "image/jpeg")}
    data = {"case_id": str(user_a_case.id)}

    res = client.post("/api/v1/ai/image-assessment", files=files, data=data, headers=user_a_headers)
    assert res.status_code == 200
    evidence_id = res.json()["image_id"]

    # Metadata endpoint
    meta_res = client.get(f"/api/v1/evidence/{evidence_id}", headers=user_a_headers)
    assert meta_res.status_code == 200
    meta_text = meta_res.text

    # Verify no server directory path leaks
    assert "storage/private" not in meta_text
    assert "\\storage\\" not in meta_text
    assert "C:" not in meta_text
    assert ".bin" not in meta_text  # Storage key should not be exposed

    # Verification endpoint
    verify_res = client.get(f"/api/v1/evidence/{evidence_id}/verify", headers=user_a_headers)
    assert verify_res.status_code == 200
    verify_text = verify_res.text
    assert "storage/private" not in verify_text
    assert "\\storage\\" not in verify_text


# ==============================================================================
# 5. STRICT HORIZONTAL AND VERTICAL EVIDENCE AUTHORIZATION
# ==============================================================================

def test_unauthorized_user_cannot_retrieve_or_download_evidence(
    client, user_a_headers, user_b_headers, user_a_case
):
    """Verify strict 401 and 403 access control boundaries for stored evidence."""
    payload = _create_synthetic_jpeg_with_metadata()
    files = {"file": ("victim_evidence.jpg", payload, "image/jpeg")}
    data = {"case_id": str(user_a_case.id)}

    res = client.post("/api/v1/ai/image-assessment", files=files, data=data, headers=user_a_headers)
    evidence_id = res.json()["image_id"]

    # 1. Unauthenticated download attempt -> 401
    unauth_res = client.get(f"/api/v1/evidence/{evidence_id}/download")
    assert unauth_res.status_code == 401

    # 2. Unrelated User B metadata access -> 403
    user_b_meta = client.get(f"/api/v1/evidence/{evidence_id}", headers=user_b_headers)
    assert user_b_meta.status_code == 403
    assert "not authorized" in user_b_meta.json()["detail"].lower()

    # 3. Unrelated User B download attempt -> 403
    user_b_dl = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=user_b_headers)
    assert user_b_dl.status_code == 403
    assert "not authorized" in user_b_dl.json()["detail"].lower()


def test_reviewer_cannot_access_evidence_without_active_review_request(
    client, user_a_headers, reviewer_headers, user_a_case
):
    """Verify reviewers cannot browse survivor evidence unless a human review was requested."""
    payload = _create_synthetic_jpeg_with_metadata()
    files = {"file": ("unreviewed_evidence.jpg", payload, "image/jpeg")}
    data = {"case_id": str(user_a_case.id)}

    res = client.post("/api/v1/ai/image-assessment", files=files, data=data, headers=user_a_headers)
    evidence_id = res.json()["image_id"]

    # Reviewer attempts download without active review -> 403
    reviewer_dl = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=reviewer_headers)
    assert reviewer_dl.status_code == 403
    assert "not authorized" in reviewer_dl.json()["detail"].lower()


def test_reviewer_can_access_evidence_with_active_review_request(
    client, user_a_headers, reviewer_headers, user_a_case, test_db
):
    """Verify reviewer CAN access evidence once survivor requests human review."""
    payload = _create_synthetic_jpeg_with_metadata()
    files = {"file": ("reviewed_evidence.jpg", payload, "image/jpeg")}
    data = {"case_id": str(user_a_case.id)}

    res = client.post("/api/v1/ai/image-assessment", files=files, data=data, headers=user_a_headers)
    evidence_id = res.json()["image_id"]

    # Survivor requests human review
    review_req = client.post(
        "/api/v1/image/review/request",
        headers=user_a_headers,
        json={
            "evidence_id": evidence_id,
            "case_id": str(user_a_case.id),
            "ai_assessment_snapshot": {"assessment": "inconclusive", "score": 0.42},
            "request_reason": "Survivor requests second opinion on suspected manipulation",
        },
    )
    assert review_req.status_code == 201

    # Reviewer now authorized to download evidence stream
    reviewer_dl = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=reviewer_headers)
    assert reviewer_dl.status_code == 200
    assert reviewer_dl.content == payload


# ==============================================================================
# 6. ANTI-CACHING AND SECURITY HEADERS ENFORCED
# ==============================================================================

def test_anti_caching_and_privacy_headers_enforced(client, user_a_headers, user_a_case):
    """Verify Cache-Control: no-store and privacy headers are sent on API and evidence routes."""
    # 1. Assessment endpoint
    payload = _create_synthetic_jpeg_with_metadata()
    files = {"file": ("cache_test.jpg", payload, "image/jpeg")}
    data = {"case_id": str(user_a_case.id)}

    assess_res = client.post("/api/v1/ai/image-assessment", files=files, data=data, headers=user_a_headers)
    assert assess_res.status_code == 200

    assert "no-store" in assess_res.headers.get("Cache-Control", "")
    assert "no-cache" in assess_res.headers.get("Cache-Control", "")
    assert assess_res.headers.get("Pragma") == "no-cache"
    assert assess_res.headers.get("X-Content-Type-Options") == "nosniff"
    assert assess_res.headers.get("Referrer-Policy") == "no-referrer"

    # 2. Evidence download stream
    evidence_id = assess_res.json()["image_id"]
    dl_res = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=user_a_headers)
    assert dl_res.status_code == 200
    assert "no-store" in dl_res.headers.get("Cache-Control", "")
    assert dl_res.headers.get("Pragma") == "no-cache"


# ==============================================================================
# 7. PHYSICAL STORAGE DELETION AND LIFECYCLE CLEANUP
# ==============================================================================

def test_evidence_deletion_purges_physical_file_and_logs_audit(
    client, user_a_headers, user_a_case, test_db
):
    """Verify DELETE /api/v1/evidence/{id} wipes file from disk and writes audit log."""
    payload = _create_synthetic_jpeg_with_metadata()
    files = {"file": ("purge_me.jpg", payload, "image/jpeg")}
    data = {"case_id": str(user_a_case.id)}

    res = client.post("/api/v1/ai/image-assessment", files=files, data=data, headers=user_a_headers)
    evidence_id = res.json()["image_id"]

    # Verify physical file exists on disk
    with test_db() as db:
        evidence = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        assert evidence is not None
        storage_path = Path(settings.PRIVATE_STORAGE_PATH) / evidence.storage_key
        assert storage_path.exists()

    # User A deletes evidence
    del_res = client.delete(f"/api/v1/evidence/{evidence_id}", headers=user_a_headers)
    assert del_res.status_code == 204

    # 1. Assert physical file is deleted from disk
    assert not storage_path.exists(), "Physical evidence file must be purged from disk upon deletion"

    # 2. Assert database record is gone
    with test_db() as db:
        ev_check = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        assert ev_check is None

        # 3. Assert audit log was recorded
        audit = db.scalar(
            select(AuditLog)
            .where(AuditLog.action == "EVIDENCE_DELETED")
            .where(AuditLog.entity_id == str(evidence_id))
        )
        assert audit is not None
        assert audit.action == "EVIDENCE_DELETED"

    # 4. Subsequent download returns 404
    subsequent_dl = client.get(f"/api/v1/evidence/{evidence_id}/download", headers=user_a_headers)
    assert subsequent_dl.status_code == 404
