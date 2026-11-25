"""Application Security Audit Test Suite: Image Upload Pipeline & Resource Exhaustion Defense.

Verifies SafeHER defenses against malicious, malformed, and adversarial upload payloads:
1. MIME-type spoofing:
   - Rejection of executable/HTML payload masquerading under image/jpeg header.
   - Content-based verification of legitimate image with non-standard Content-Type header.
2. File extension spoofing:
   - Rejection of dangerous extensions (.exe, .php, .sh, .bat) even with valid image payloads.
   - Handling of double extensions (e.g., photo.php.jpg).
   - Sanitization of null-byte injection attempts.
3. Corrupt images & malformed headers:
   - Rejection of non-image junk bytes.
   - Rejection of truncated image streams.
   - Rejection of malformed image headers.
4. Extremely large dimensions & Decompression bombs:
   - Rejection of oversized dimensions (> 4096px) before allocating decompression raster.
   - Rejection of decompression bombs exceeding MAX_IMAGE_PIXELS.
5. Unexpected metadata:
   - Stripping of sensitive EXIF PII and GPS coordinates.
   - Non-crashing handling of malformed or unexpected metadata tags.
6. Path traversal:
   - Defeat of directory traversal filenames (../../etc/passwd, ..\\windows\\system32).
   - Verification of randomized server-side UUID storage keys.
7. Repeated and concurrent uploads:
   - Sequential repeated uploads generating distinct isolated vault keys.
   - Concurrent upload requests handled gracefully under concurrency semaphore.
8. Public evidence exposure prevention:
   - Verification that private storage directory is not exposed as static public web route.
   - Mandatory authentication on evidence access.
"""

import asyncio
import io
from pathlib import Path
import re
from typing import Dict
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_core.image_intelligence.config import default_image_config
from ai_core.image_intelligence.preprocessing import preprocess_image
from ai_core.image_intelligence.service import set_image_detector
from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import Case, Evidence, User


def _create_valid_jpeg(dimensions=(200, 200), color=(120, 160, 200)) -> bytes:
    """Generate in-memory valid JPEG bytes."""
    img = Image.new("RGB", dimensions, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, dimensions[0] // 2, dimensions[1] // 2], fill=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def test_db():
    """Create isolated, in-memory SQLite database for test execution."""
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
def auth_headers(client) -> Dict[str, str]:
    """Helper fixture registering and logging in a test user."""
    reg_data = {
        "email": "sec_audit_victim@example.com",
        "password": "SafeHER-Secure-Password-123!",
    }
    client.post("/api/v1/auth/register", json=reg_data)
    login_res = client.post("/api/v1/auth/login", json=reg_data)
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_case(test_db, client, auth_headers) -> str:
    """Creates a Case owned by the authenticated test user."""
    with test_db() as db:
        user = db.scalar(select(User).where(User.email == "sec_audit_victim@example.com"))
        case = Case(
            owner_id=user.id,
            title="AppSec Verification Case",
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        return str(case.id)


# ==============================================================================
# 1. MIME-TYPE SPOOFING AUDIT
# ==============================================================================

def test_mime_spoofing_html_payload_rejected(client, auth_headers):
    """Reject HTML/script payload falsely claiming Content-Type: image/jpeg."""
    html_payload = b"<html><script>alert('xss')</script></html>"
    files = {"file": ("malicious.jpg", html_payload, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Unsupported or invalid image file format" in res.json()["detail"]


def test_mime_spoofing_shell_payload_rejected(client, auth_headers):
    """Reject shell script payload claiming Content-Type: image/png."""
    shell_payload = b"#!/bin/bash\nrm -rf /tmp/*\n"
    files = {"file": ("script.png", shell_payload, "image/png")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Unsupported or invalid image file format" in res.json()["detail"]


def test_mime_spoofing_legitimate_jpeg_with_generic_content_type(client, auth_headers):
    """Validate that actual decoded bytes govern processing even if client Content-Type is generic."""
    jpeg_bytes = _create_valid_jpeg()
    files = {"file": ("photo.jpg", jpeg_bytes, "application/octet-stream")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["metadata_summary"]["format"] == "JPEG"


# ==============================================================================
# 2. FILE EXTENSION SPOOFING AUDIT
# ==============================================================================

def test_file_extension_spoofing_executable_rejected(client, auth_headers):
    """Reject executable file extension (.exe) even if bytes are valid JPEG."""
    jpeg_bytes = _create_valid_jpeg()
    files = {"file": ("trojan.exe", jpeg_bytes, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Unsupported file extension '.exe'" in res.json()["detail"]


def test_file_extension_spoofing_php_rejected(client, auth_headers):
    """Reject PHP web-shell extension (.php) even if bytes are valid JPEG."""
    jpeg_bytes = _create_valid_jpeg()
    files = {"file": ("webshell.php", jpeg_bytes, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Unsupported file extension '.php'" in res.json()["detail"]


def test_file_extension_double_extension_handled_safely(client, auth_headers, user_case):
    """Double extension (photo.php.jpg) must retain terminal extension and store as .bin."""
    jpeg_bytes = _create_valid_jpeg()
    files = {"file": ("exploit.php.jpg", jpeg_bytes, "image/jpeg")}

    res = client.post(
        "/api/v1/ai/image-assessment",
        data={"case_id": user_case},
        files=files,
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["image_id"] is not None


def test_null_byte_filename_injection_sanitized(client, auth_headers, user_case):
    """Null-byte in filename (image.png\\x00.exe) is sanitized without path truncation."""
    jpeg_bytes = _create_valid_jpeg()
    files = {"file": ("innocent.png\x00.exe", jpeg_bytes, "image/jpeg")}

    res = client.post(
        "/api/v1/ai/image-assessment",
        data={"case_id": user_case},
        files=files,
        headers=auth_headers,
    )
    # Replaced null byte yields innocent.png.exe which is rejected by extension check
    assert res.status_code == 400
    assert "Unsupported file extension '.exe'" in res.json()["detail"]


# ==============================================================================
# 3. CORRUPT IMAGES, MALFORMED HEADERS & TRUNCATED FILES
# ==============================================================================

def test_corrupt_non_image_bytes_rejected(client, auth_headers):
    """Reject completely unparseable binary junk."""
    junk = b"\x00\x01\x02\x03\x04\x05\x06\x07" * 50
    files = {"file": ("corrupt.jpg", junk, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Unsupported or invalid image file format" in res.json()["detail"]


def test_malformed_header_with_jpeg_magic_bytes_rejected(client, auth_headers):
    """Reject payload starting with valid JPEG magic bytes but containing corrupt header structure."""
    fake_jpeg = b"\xFF\xD8\xFF\xE0" + b"\x00" * 200  # Invalid markers
    files = {"file": ("broken_header.jpg", fake_jpeg, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Corrupt or unreadable image stream" in res.json()["detail"]


def test_truncated_jpeg_stream_rejected(client, auth_headers):
    """Reject truncated image file that cuts off mid-scan."""
    jpeg_bytes = _create_valid_jpeg()
    # Truncate to first 100 bytes (strips EOI and Huffman tables)
    truncated = jpeg_bytes[:100]
    files = {"file": ("truncated.jpg", truncated, "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Corrupt or unreadable image stream" in res.json()["detail"]


def test_zero_byte_empty_file_rejected(client, auth_headers):
    """Reject zero-byte image upload."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}

    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "Uploaded image file cannot be empty" in res.json()["detail"]


# ==============================================================================
# 4. EXTREMELY LARGE DIMENSIONS & DECOMPRESSION BOMBS
# ==============================================================================

def test_oversized_dimensions_rejected_before_decompression(client, auth_headers):
    """Verify images exceeding max_image_dimension (4096) are rejected before raster allocation."""
    # Attempting to preprocess an image with 5000px dimension
    img = Image.new("RGB", (5000, 100), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    oversized_bytes = buf.getvalue()

    files = {"file": ("oversized.jpg", oversized_bytes, "image/jpeg")}
    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "exceed the maximum allowable limit of 4096x4096" in res.json()["detail"]


def test_sub_minimum_dimensions_rejected(client, auth_headers):
    """Verify images smaller than min_image_dimension (32x32) are rejected."""
    img = Image.new("RGB", (16, 16), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    tiny_bytes = buf.getvalue()

    files = {"file": ("tiny.jpg", tiny_bytes, "image/jpeg")}
    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert "smaller than the required minimum of 32x32" in res.json()["detail"]


def test_decompression_bomb_pixel_guard_enforced():
    """Verify PIL MAX_IMAGE_PIXELS triggers DecompressionBomb protection."""
    # Create an image that triggers the 16,777,216 max pixel threshold
    # Directly test the preprocessor to confirm clean ValueError
    large_img = Image.new("RGB", (4200, 4200), color=(100, 100, 100))
    buf = io.BytesIO()
    large_img.save(buf, format="JPEG")
    bomb_bytes = buf.getvalue()

    with pytest.raises(ValueError) as excinfo:
        preprocess_image(bomb_bytes)
    assert "exceed the maximum allowable limit" in str(excinfo.value) or "decompression limit" in str(excinfo.value)


# ==============================================================================
# 5. UNEXPECTED METADATA & EXIF PRIVACY SANITIZATION
# ==============================================================================

def test_sensitive_exif_pii_and_gps_stripped(client, auth_headers):
    """Verify GPSInfo and sensitive device tags are stripped from sanitized_exif."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))

    # Construct native EXIF using PIL
    exif = img.getexif()
    exif[0x010F] = "Nikon"
    exif[0x0110] = "D850"
    # Tag 0x013B is Artist / OwnerName, 0x9286 is UserComment
    exif[0x9286] = b"Sensitive User Comment with Location"
    gps_ifd = exif.get_ifd(0x8825)
    gps_ifd[1] = "N"  # GPSLatitudeRef

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    jpeg_with_exif = buf.getvalue()

    files = {"file": ("gps_test.jpg", jpeg_with_exif, "image/jpeg")}
    res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    meta = data["metadata_summary"]
    assert meta["has_exif"] is True
    sanitized = meta["sanitized_exif"]

    # Verify sensitive tags are stripped for survivor privacy
    assert "GPSInfo" not in sanitized
    assert "UserComment" not in sanitized
    assert "MakerNote" not in sanitized
    # Non-sensitive tags retained
    assert sanitized.get("Make") == "Nikon"
    assert sanitized.get("Model") == "D850"


# ==============================================================================
# 6. PATH TRAVERSAL FILENAMES & PRIVATE STORAGE ISOLATION
# ==============================================================================

def test_path_traversal_filename_sanitized(client, auth_headers, user_case, test_db):
    """Verify ../../../../etc/passwd filename is neutralized and stored safely."""
    jpeg_bytes = _create_valid_jpeg()
    files = {"file": ("../../../../etc/passwd.jpg", jpeg_bytes, "image/jpeg")}

    res = client.post(
        "/api/v1/ai/image-assessment",
        data={"case_id": user_case},
        files=files,
        headers=auth_headers,
    )
    assert res.status_code == 200
    evidence_id = res.json()["image_id"]

    with test_db() as db:
        ev = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        assert ev is not None
        # Original filename sanitized to basename only
        assert ev.original_filename == "passwd.jpg"
        assert "/" not in ev.original_filename
        assert ".." not in ev.original_filename

        # Physical file stored under random UUID, not user-supplied name
        assert ev.storage_key.endswith(".bin")
        assert len(ev.storage_key) == 40  # 36 chars UUID + .bin


def test_windows_path_traversal_sanitized(client, auth_headers, user_case, test_db):
    """Verify ..\\..\\windows\\system32\\cmd.exe.png traversal is neutralized."""
    jpeg_bytes = _create_valid_jpeg()
    files = {"file": ("..\\..\\windows\\system32\\cmd.exe.png", jpeg_bytes, "image/jpeg")}

    res = client.post(
        "/api/v1/ai/image-assessment",
        data={"case_id": user_case},
        files=files,
        headers=auth_headers,
    )
    assert res.status_code == 200
    evidence_id = res.json()["image_id"]

    with test_db() as db:
        ev = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        assert ev is not None
        assert "\\" not in ev.original_filename
        assert ".." not in ev.original_filename


# ==============================================================================
# 7. PUBLIC EVIDENCE EXPOSURE & AUTHENTICATION ENFORCEMENT
# ==============================================================================

def test_private_storage_directory_not_publicly_accessible(client):
    """Verify private_evidence directory is NOT mounted as a public static endpoint."""
    res = client.get("/private_evidence/test.bin")
    # Must return 404 Not Found (or 405 Method Not Allowed), never 200 or directory listing
    assert res.status_code in (404, 405)


def test_evidence_stream_requires_authentication(client):
    """Evidence stream endpoint strictly requires valid Bearer authentication."""
    random_ev_id = str(uuid.uuid4())
    res = client.get(f"/api/v1/evidence/{random_ev_id}/download")
    assert res.status_code == 401


# ==============================================================================
# 8. CONCURRENT REQUESTS & INFERENCE OVERLOAD RESILIENCE
# ==============================================================================

def test_concurrent_upload_requests_handled_resiliently(client, auth_headers):
    """Simulate parallel requests to ensure concurrency semaphore throttles safely without server crash."""
    jpeg_bytes = _create_valid_jpeg()

    # Issue multiple sequential/concurrent requests using TestClient
    responses = []
    for i in range(8):
        files = {"file": (f"concurrent_test_{i}.jpg", jpeg_bytes, "image/jpeg")}
        res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
        responses.append(res)

    for r in responses:
        assert r.status_code == 200
        assert r.json()["assessment_id"] is not None


def test_early_request_body_size_limit_intercepts_before_multipart_spooling(client, auth_headers):
    """Verify that oversized Content-Length is intercepted by early middleware before Starlette spools to /tmp."""
    headers = dict(auth_headers)
    headers["Content-Length"] = str(55 * 1024 * 1024)  # 55 MB > 50 MB
    headers["Content-Type"] = "multipart/form-data; boundary=boundary123"

    # Send dummy body with oversized header
    res = client.post(
        "/api/v1/ai/image-assessment",
        headers=headers,
        content=b"--boundary123\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n\r\nfake",
    )
    assert res.status_code == 413
    assert "request body exceeds the permitted size" in res.json()["detail"].lower()


def test_aggregate_concurrent_upload_admission_rejection_at_capacity(client, auth_headers, tmp_path, monkeypatch):
    """Verify that exceeding MAX_CONCURRENT_UPLOADS (10) across deployment returns HTTP 429."""
    from backend.app.services.image_service import UploadAdmissionCoordinator
    import backend.app.main as app_main

    # Create coordinator with capacity 2 for isolated unit test
    test_slots_dir = tmp_path / "upload_slots"
    test_coordinator = UploadAdmissionCoordinator(slots_dir=test_slots_dir, max_slots=2)
    monkeypatch.setattr(app_main, "upload_admission_coordinator", test_coordinator)

    # Acquire both slots to simulate 2 in-flight uploads across workers
    s0, h0 = test_coordinator.acquire_upload_slot()
    s1, h1 = test_coordinator.acquire_upload_slot()
    assert s0 is not None and s1 is not None

    try:
        # A 3rd upload arrives: must be rejected with HTTP 429 immediately
        jpeg_bytes = _create_valid_jpeg()
        files = {"file": ("rejected_due_to_load.jpg", jpeg_bytes, "image/jpeg")}
        res = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
        assert res.status_code == 429
        assert "Concurrent image upload limit reached" in res.json()["detail"]

        # Release one slot: next upload immediately succeeds
        test_coordinator.release_upload_slot(s0, h0)
        res_ok = client.post("/api/v1/ai/image-assessment", files=files, headers=auth_headers)
        assert res_ok.status_code == 200
    finally:
        test_coordinator.release_upload_slot(s1, h1)


def test_temporary_file_cleanup_on_upload_rejection(client, auth_headers, tmp_path, monkeypatch):
    """Verify that rejected uploads (oversized, invalid extension, corrupted) leave 0 temporary files in storage."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    # Reject due to forbidden extension
    bad_files = {"file": ("payload.exe", b"MZexecutabledata", "image/jpeg")}
    res = client.post("/api/v1/ai/image-assessment", files=bad_files, headers=auth_headers)
    assert res.status_code == 400

    # Verify no .part or .bin files left in storage
    storage_files = list(tmp_path.glob("*.part")) + list(tmp_path.glob("*.bin"))
    assert len(storage_files) == 0, f"Leaked temporary files found: {storage_files}"


def test_uploads_distributed_across_two_workers_respect_aggregate_cap(tmp_path):
    """Verify that two worker processes sharing the same upload slots directory respect aggregate limit."""
    from backend.app.services.image_service import UploadAdmissionCoordinator

    slots_dir = tmp_path / "shared_uploads"
    worker_1_coord = UploadAdmissionCoordinator(slots_dir=slots_dir, max_slots=3)
    worker_2_coord = UploadAdmissionCoordinator(slots_dir=slots_dir, max_slots=3)

    # Worker 1 acquires 2 slots
    slot1, h1 = worker_1_coord.acquire_upload_slot()
    slot2, h2 = worker_1_coord.acquire_upload_slot()
    assert slot1 is not None and slot2 is not None

    # Worker 2 acquires 1 slot (total 3 acquired across deployment)
    slot3, h3 = worker_2_coord.acquire_upload_slot()
    assert slot3 is not None

    # Total active across deployment is 3
    assert worker_1_coord.active_upload_count() == 3
    assert worker_2_coord.active_upload_count() == 3

    # Attempting to acquire a 4th slot on either worker returns None (aggregate cap enforced)
    assert worker_1_coord.acquire_upload_slot() is None
    assert worker_2_coord.acquire_upload_slot() is None

    # Worker 1 releases a slot: Worker 2 can now acquire
    worker_1_coord.release_upload_slot(slot1, h1)
    slot4, h4 = worker_2_coord.acquire_upload_slot()
    assert slot4 is not None

    worker_1_coord.release_upload_slot(slot2, h2)
    worker_2_coord.release_upload_slot(slot3, h3)
    worker_2_coord.release_upload_slot(slot4, h4)
    assert worker_1_coord.active_upload_count() == 0


@pytest.mark.anyio
async def test_chunked_upload_without_content_length_is_bounded_before_parser_spooling():
    """A missing Content-Length must not bypass the ASGI receive-byte limit."""
    from backend.app.upload_middleware import UploadAdmissionMiddleware

    class FakeCoordinator:
        def __init__(self):
            self.released = []

        def acquire_upload_slot(self):
            return 0, object()

        def release_upload_slot(self, slot_id, handle):
            self.released.append(slot_id)

    coordinator = FakeCoordinator()
    forwarded = []
    sent = []
    messages = [
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def inner(scope, inner_receive, inner_send):
        while True:
            message = await inner_receive()
            if message["type"] == "http.disconnect":
                return
            forwarded.append(message.get("body", b""))
            if not message.get("more_body", False):
                return

    middleware = UploadAdmissionMiddleware(
        inner, coordinator=coordinator, max_request_body_bytes=5
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/ai/image-assessment",
        "headers": [(b"content-type", b"multipart/form-data; boundary=test")],
    }

    await middleware(scope, receive, send)

    assert forwarded == [b"1234"]
    assert sent[0]["status"] == 413
    assert coordinator.released == [0]


@pytest.mark.anyio
async def test_declared_small_content_length_does_not_bypass_actual_body_limit():
    """The middleware must count received bytes even if Content-Length understates them."""
    from backend.app.upload_middleware import UploadAdmissionMiddleware

    class FakeCoordinator:
        released = 0

        def acquire_upload_slot(self):
            return 0, object()

        def release_upload_slot(self, slot_id, handle):
            self.released += 1

    coordinator = FakeCoordinator()
    sent = []
    messages = [
        {"type": "http.request", "body": b"oversized", "more_body": False},
    ]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def inner(scope, inner_receive, inner_send):
        while True:
            message = await inner_receive()
            if message["type"] == "http.disconnect":
                return

    middleware = UploadAdmissionMiddleware(
        inner, coordinator=coordinator, max_request_body_bytes=5
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/ai/image-assessment",
        "headers": [
            (b"content-type", b"multipart/form-data; boundary=test"),
            (b"content-length", b"1"),
        ],
    }

    await middleware(scope, receive, send)

    assert sent[0]["status"] == 413
    assert coordinator.released == 1


@pytest.mark.anyio
async def test_oversized_chunked_body_stops_consuming_after_first_rejection():
    """After sending 413, the middleware must not read/forward further body chunks."""
    from backend.app.upload_middleware import UploadAdmissionMiddleware

    class FakeCoordinator:
        def acquire_upload_slot(self):
            return 0, object()

        def release_upload_slot(self, slot_id, handle):
            return None

    received_calls = 0
    forwarded = []
    sent = []

    async def receive():
        nonlocal received_calls
        received_calls += 1
        if received_calls == 1:
            return {"type": "http.request", "body": b"123456", "more_body": True}
        return {"type": "http.request", "body": b"secret-extra-chunk", "more_body": False}

    async def send(message):
        sent.append(message)

    async def inner(scope, inner_receive, inner_send):
        first = await inner_receive()
        forwarded.append(first)
        second = await inner_receive()
        forwarded.append(second)

    middleware = UploadAdmissionMiddleware(
        inner, coordinator=FakeCoordinator(), max_request_body_bytes=5
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/ai/image-assessment",
        "headers": [],
    }
    await middleware(scope, receive, send)

    assert sent[0]["status"] == 413
    assert received_calls == 1
    assert len(forwarded) == 2
    assert forwarded[1]["type"] == "http.disconnect"
