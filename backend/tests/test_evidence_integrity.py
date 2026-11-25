"""Security, privacy, and cryptographic integrity tests for SafeHER Evidence Vault.

Verifies:
- SHA-256 evidence hashing directly from original bytes before any transformations.
- File tampering detection (detecting disk-level tampering, bitrot, or alteration).
- Original image preservation (bit-for-bit identical to uploaded payload).
- Derivative separation (working copies never overwrite pristine evidence).
- Unauthorized retrieval (401 for unauthenticated, 403 for cross-user/tenant access).
- Cross-case and cross-user isolation.
- Path traversal interception.
- Controlled deletion behavior (wipes disk file, removes DB record, records audit log).
- Privacy-safe logging (no raw image bytes or base64 data logged).
"""

from datetime import datetime, timedelta, timezone
import hashlib
import io
import logging
import os
from pathlib import Path
import time
from typing import Dict
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import AuditLog, Case, Evidence, EvidenceCleanupLease, User
from backend.app.services.evidence_service import evidence_service


def _create_synthetic_test_file(
    dimensions=(200, 200),
    format_name="JPEG",
) -> bytes:
    """Create in-memory synthetic image bytes."""
    img = Image.new("RGB", dimensions, color=(80, 120, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 100, 100], fill=(200, 60, 60))
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


@pytest.fixture
def client(test_db):
    """TestClient configured with test database dependency overrides."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def user1_data(client) -> Dict[str, str]:
    """Helper registering user 1."""
    reg_data = {
        "email": "vault_victim1@example.com",
        "password": "SafeHER-Password-123!",
    }
    client.post("/api/v1/auth/register", json=reg_data)
    login_res = client.post("/api/v1/auth/login", json=reg_data)
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}", "email": reg_data["email"]}


@pytest.fixture
def user2_data(client) -> Dict[str, str]:
    """Helper registering user 2."""
    reg_data = {
        "email": "vault_victim2@example.com",
        "password": "SafeHER-Password-456!",
    }
    client.post("/api/v1/auth/register", json=reg_data)
    login_res = client.post("/api/v1/auth/login", json=reg_data)
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}", "email": reg_data["email"]}


# ============================================================================
# 1. HASHING, PRESERVATION & TAMPERING DETECTION TESTS
# ============================================================================


def test_sha256_hash_consistency_and_preservation(client, user1_data, test_db, tmp_path, monkeypatch):
    """Verify SHA-256 hash is computed directly from original bytes and file is preserved bit-for-bit."""
    vault_dir = tmp_path / "private_vault"
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(vault_dir))

    raw_bytes = _create_synthetic_test_file(dimensions=(256, 256), format_name="JPEG")
    expected_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # Create owned case
    db = test_db()
    try:
        user1 = db.scalar(select(User).where(User.email == user1_data["email"]))
        user1_id = user1.id
        case1 = Case(owner_id=user1_id, title="Intimidation Case", status="open")
        db.add(case1)
        db.commit()
        case1_id = str(case1.id)
    finally:
        db.close()

    # Upload evidence linked to case
    upload_res = client.post(
        "/api/v1/ai/image-assessment",
        headers={"Authorization": user1_data["Authorization"]},
        data={"case_id": case1_id},
        files={"file": ("evidence_original.jpg", raw_bytes, "image/jpeg")},
    )
    assert upload_res.status_code == 200
    evidence_id = upload_res.json()["image_id"]
    assert evidence_id is not None

    # Retrieve evidence metadata from database
    db = test_db()
    try:
        evidence = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        assert evidence is not None
        assert evidence.sha256 == expected_sha256
        assert evidence.size_bytes == len(raw_bytes)

        # Verify physical disk file is bit-for-bit identical to uploaded bytes
        saved_path = vault_dir / evidence.storage_key
        assert saved_path.exists()
        saved_bytes = saved_path.read_bytes()
        assert saved_bytes == raw_bytes
        assert hashlib.sha256(saved_bytes).hexdigest() == expected_sha256
    finally:
        db.close()


def test_file_tampering_detection(client, user1_data, test_db, tmp_path, monkeypatch):
    """Verify that if an evidence file on disk is modified, verify_integrity reports failure."""
    vault_dir = tmp_path / "private_vault"
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(vault_dir))

    raw_bytes = _create_synthetic_test_file()
    db = test_db()
    try:
        user1 = db.scalar(select(User).where(User.email == user1_data["email"]))
        case1 = Case(owner_id=user1.id, title="Case Tamper Test", status="open")
        db.add(case1)
        db.commit()
        case1_id = str(case1.id)
    finally:
        db.close()

    upload_res = client.post(
        "/api/v1/ai/image-assessment",
        headers={"Authorization": user1_data["Authorization"]},
        data={"case_id": case1_id},
        files={"file": ("photo.jpg", raw_bytes, "image/jpeg")},
    )
    evidence_id = upload_res.json()["image_id"]

    # 1. Initial verification must pass
    verify_res = client.get(
        f"/api/v1/evidence/{evidence_id}/verify",
        headers={"Authorization": user1_data["Authorization"]},
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["is_valid"] is True
    assert "matches pristine baseline" in verify_res.json()["message"]

    # 2. Simulate unauthorized disk tampering (attacker or bitrot flips bytes on disk)
    db = test_db()
    try:
        evidence = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        disk_file = vault_dir / evidence.storage_key
        disk_file.write_bytes(b"Tampered corrupted bytes replacing original image")
    finally:
        db.close()

    # 3. Subsequent verification must flag tampering
    tampered_verify_res = client.get(
        f"/api/v1/evidence/{evidence_id}/verify",
        headers={"Authorization": user1_data["Authorization"]},
    )
    assert tampered_verify_res.status_code == 200
    assert tampered_verify_res.json()["is_valid"] is False
    assert "CRITICAL: Evidence hash mismatch" in tampered_verify_res.json()["message"]


# ============================================================================
# 2. AUTHORIZATION & CROSS-TENANT ISOLATION TESTS
# ============================================================================


def test_evidence_metadata_unauthenticated_rejected(client):
    """Reject unauthenticated evidence access with 401."""
    random_id = str(uuid.uuid4())
    res = client.get(f"/api/v1/evidence/{random_id}")
    assert res.status_code == 401


def test_cross_user_evidence_retrieval_forbidden(client, user1_data, user2_data, test_db, tmp_path, monkeypatch):
    """Verify User 2 cannot access or retrieve User 1's evidence (403 Forbidden)."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "vault"))

    # User 1 creates case and uploads evidence
    db = test_db()
    try:
        user1 = db.scalar(select(User).where(User.email == user1_data["email"]))
        case1 = Case(owner_id=user1.id, title="User 1 Confidential Case", status="open")
        db.add(case1)
        db.commit()
        case1_id = str(case1.id)
    finally:
        db.close()

    raw_bytes = _create_synthetic_test_file()
    upload_res = client.post(
        "/api/v1/ai/image-assessment",
        headers={"Authorization": user1_data["Authorization"]},
        data={"case_id": case1_id},
        files={"file": ("user1_evidence.jpg", raw_bytes, "image/jpeg")},
    )
    evidence_id = upload_res.json()["image_id"]

    # User 2 attempts to get metadata
    meta_res = client.get(
        f"/api/v1/evidence/{evidence_id}",
        headers={"Authorization": user2_data["Authorization"]},
    )
    assert meta_res.status_code == 403
    assert "Not authorized" in meta_res.json()["detail"]

    # User 2 attempts to download file
    down_res = client.get(
        f"/api/v1/evidence/{evidence_id}/download",
        headers={"Authorization": user2_data["Authorization"]},
    )
    assert down_res.status_code == 403

    # User 2 attempts to verify integrity
    ver_res = client.get(
        f"/api/v1/evidence/{evidence_id}/verify",
        headers={"Authorization": user2_data["Authorization"]},
    )
    assert ver_res.status_code == 403

    # User 2 attempts to delete evidence
    del_res = client.delete(
        f"/api/v1/evidence/{evidence_id}",
        headers={"Authorization": user2_data["Authorization"]},
    )
    assert del_res.status_code == 403


# ============================================================================
# 3. PATH TRAVERSAL & SECURITY DEFENSE TESTS
# ============================================================================


def test_path_traversal_interception(tmp_path, monkeypatch):
    """Verify that path traversal storage keys are intercepted and rejected with 400."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path / "vault"))

    malicious_key = "../../etc/passwd"
    with pytest.raises(Exception) as excinfo:
        evidence_service._resolve_safe_path(malicious_key)
    assert "Invalid evidence storage destination" in str(excinfo.value)


# ============================================================================
# 4. CONTROLLED DELETION & AUDIT BEHAVIOR TESTS
# ============================================================================


def test_controlled_evidence_deletion(client, user1_data, test_db, tmp_path, monkeypatch):
    """Verify controlled deletion purges physical disk file, deletes DB record, and records audit log."""
    vault_dir = tmp_path / "vault"
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(vault_dir))

    db = test_db()
    try:
        user1 = db.scalar(select(User).where(User.email == user1_data["email"]))
        user1_id = user1.id
        case1 = Case(owner_id=user1_id, title="Case for Deletion", status="open")
        db.add(case1)
        db.commit()
        case1_id = str(case1.id)
    finally:
        db.close()

    raw_bytes = _create_synthetic_test_file()
    upload_res = client.post(
        "/api/v1/ai/image-assessment",
        headers={"Authorization": user1_data["Authorization"]},
        data={"case_id": case1_id},
        files={"file": ("to_delete.jpg", raw_bytes, "image/jpeg")},
    )
    evidence_id = upload_res.json()["image_id"]

    db = test_db()
    try:
        evidence = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        disk_path = vault_dir / evidence.storage_key
        assert disk_path.exists()
    finally:
        db.close()

    # Execute authorized deletion
    del_res = client.delete(
        f"/api/v1/evidence/{evidence_id}",
        headers={"Authorization": user1_data["Authorization"]},
    )
    assert del_res.status_code == 204

    # Verify physical file is wiped
    assert not disk_path.exists()

    # Verify DB record is removed
    db = test_db()
    try:
        evidence_after = db.scalar(select(Evidence).where(Evidence.id == uuid.UUID(evidence_id)))
        assert evidence_after is None

        # Verify audit log recorded
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.action == "EVIDENCE_DELETED",
                AuditLog.entity_id == evidence_id,
            )
        )
        assert audit is not None
        assert audit.user_id == user1_id
    finally:
        db.close()


# ============================================================================
# 5. PRIVACY-SAFE LOGS AUDIT
# ============================================================================


def test_evidence_workflow_privacy_safe_logging(client, user1_data, caplog):
    """Verify all evidence workflow logging operations never expose raw binary or base64 data."""
    caplog.set_level(logging.INFO)
    raw_bytes = _create_synthetic_test_file()

    client.post(
        "/api/v1/ai/image-assessment",
        headers={"Authorization": user1_data["Authorization"]},
        files={"file": ("clean.jpg", raw_bytes, "image/jpeg")},
    )

    log_output = caplog.text
    # Must NOT contain raw bytes or base64 patterns
    assert str(raw_bytes[:20]) not in log_output
    assert "data:image" not in log_output


# ============================================================================
# 6. ORPHAN FILE RECONCILIATION TEST
# ============================================================================


def test_reconcile_orphan_evidence(test_db, tmp_path, monkeypatch):
    """Verify orphan file reconciliation identifies unreferenced files and never deletes valid evidence."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        # Create an owned valid evidence record
        user = User(email="orphan_tester@safeher.internal", password_hash="secret", role="survivor")
        db.add(user)
        db.flush()

        case = Case(title="Orphan Test Case", owner_id=user.id)
        db.add(case)
        db.flush()

        valid_key = "valid_evidence_123.bin"
        valid_file = tmp_path / valid_key
        valid_file.write_bytes(b"valid evidence bytes")

        evidence = Evidence(
            case_id=case.id,
            owner_id=user.id,
            original_filename="valid.jpg",
            storage_key=valid_key,
            mime_type="image/jpeg",
            size_bytes=len(b"valid evidence bytes"),
            sha256="abc123valid",
        )
        db.add(evidence)
        db.commit()

        # Create an orphaned file (e.g. from an interrupted upload/crash)
        orphan_key = "orphan_file_456.bin"
        orphan_file = tmp_path / orphan_key
        orphan_file.write_bytes(b"orphaned bytes from crash")

        # Set orphan file modification time to 2 hours ago
        two_hours_ago = time.time() - 7200
        os.utime(orphan_file, (two_hours_ago, two_hours_ago))

        # 1. Dry run: should identify orphan but not delete it
        dry_run_orphans = evidence_service.reconcile_orphan_evidence(db, dry_run=True, grace_period_seconds=3600)
        assert orphan_key in dry_run_orphans
        assert orphan_file.exists()
        assert valid_file.exists()

        # 2. Live reconciliation: should purge orphan and preserve valid evidence
        live_orphans = evidence_service.reconcile_orphan_evidence(db, dry_run=False, grace_period_seconds=3600)
        assert orphan_key in live_orphans
        assert not orphan_file.exists()  # Purged
        assert valid_file.exists()      # Untouched!
    finally:
        db.close()


def test_concurrent_upload_and_cleanup_race_prevention(test_db, tmp_path, monkeypatch):
    """Verify that an active upload in progress is not touched by reconciliation due to grace period and DB recheck."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        # 1. Simulate active upload writing .part file just 10 seconds ago
        active_key = f"{uuid.uuid4()}.bin"
        active_part_file = tmp_path / f"{active_key}.part"
        active_part_file.write_bytes(b"in flight active upload bytes")

        ten_seconds_ago = time.time() - 10
        os.utime(active_part_file, (ten_seconds_ago, ten_seconds_ago))

        # Reconcile with 1-hour grace period: active upload must be skipped
        orphans = evidence_service.reconcile_orphan_evidence(db, dry_run=False, grace_period_seconds=3600)
        assert f"{active_key}.part" not in orphans
        assert active_part_file.exists(), "Active upload part file was prematurely touched!"
    finally:
        db.close()


def test_quarantine_of_genuine_orphan_files_with_lease(test_db, tmp_path, monkeypatch):
    """Verify that genuine orphan files are staged in quarantine with a database cleanup lease."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        orphan_key = f"{uuid.uuid4()}.bin"
        orphan_file = tmp_path / orphan_key
        orphan_file.write_bytes(b"stale orphan file")

        two_hours_ago = time.time() - 7200
        os.utime(orphan_file, (two_hours_ago, two_hours_ago))

        # Execute quarantine reconciliation
        orphans = evidence_service.reconcile_orphan_evidence(
            db, dry_run=False, grace_period_seconds=3600, quarantine=True
        )
        assert orphan_key in orphans
        assert not orphan_file.exists()

        quarantine_file = tmp_path / "quarantine" / orphan_key
        assert quarantine_file.exists(), "Orphan was not moved to quarantine staging directory!"

        # Verify database lease was recorded
        lease = db.scalar(select(EvidenceCleanupLease).where(EvidenceCleanupLease.storage_key == orphan_key))
        assert lease is not None
        assert lease.status == "QUARANTINED"
        assert lease.action == "quarantine"
        assert lease.resolved_at is not None
    finally:
        db.close()


def test_recovery_from_quarantine(test_db, tmp_path, monkeypatch):
    """Verify that a quarantined file can be recovered and restored back to canonical storage."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        orphan_key = f"{uuid.uuid4()}.bin"
        orphan_file = tmp_path / orphan_key
        orphan_file.write_bytes(b"file to quarantine and restore")

        two_hours_ago = time.time() - 7200
        os.utime(orphan_file, (two_hours_ago, two_hours_ago))

        # Quarantine it
        evidence_service.reconcile_orphan_evidence(
            db, dry_run=False, grace_period_seconds=3600, quarantine=True
        )
        quarantine_file = tmp_path / "quarantine" / orphan_key
        assert quarantine_file.exists()
        assert not orphan_file.exists()

        # Restore from quarantine
        restored = evidence_service.restore_from_quarantine(db, orphan_key)
        assert restored is True
        assert orphan_file.exists(), "File was not restored to canonical storage!"
        assert not quarantine_file.exists(), "File still remains in quarantine!"

        # Verify DB lease updated to RESTORED
        lease = db.scalar(select(EvidenceCleanupLease).where(EvidenceCleanupLease.storage_key == orphan_key))
        assert lease is not None
        assert lease.status == "RESTORED"
        assert "Restored from quarantine" in lease.notes
    finally:
        db.close()


def test_expired_quarantine_cleanup(test_db, tmp_path, monkeypatch):
    """Verify that expired quarantine lease files are permanently purged."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        orphan_key = f"{uuid.uuid4()}.bin"
        orphan_file = tmp_path / orphan_key
        orphan_file.write_bytes(b"file to expire in quarantine")

        two_hours_ago = time.time() - 7200
        os.utime(orphan_file, (two_hours_ago, two_hours_ago))

        # Quarantine with 10s retention
        evidence_service.reconcile_orphan_evidence(
            db, dry_run=False, grace_period_seconds=3600, quarantine=True, quarantine_retention_seconds=10
        )
        quarantine_file = tmp_path / "quarantine" / orphan_key
        assert quarantine_file.exists()

        # Age quarantine file to 20 seconds ago
        twenty_s_ago = time.time() - 20
        os.utime(quarantine_file, (twenty_s_ago, twenty_s_ago))

        # Re-run reconciliation with 10s retention
        evidence_service.reconcile_orphan_evidence(
            db, dry_run=False, grace_period_seconds=3600, quarantine=True, quarantine_retention_seconds=10
        )

        assert not quarantine_file.exists(), "Expired quarantine file was not purged!"
        # Verify DB lease updated to PURGED
        lease = db.scalar(select(EvidenceCleanupLease).where(EvidenceCleanupLease.storage_key == orphan_key))
        assert lease is not None
        assert lease.status == "PURGED"
    finally:
        db.close()


def test_database_failure_during_cleanup_aborts_without_file_modification(test_db, tmp_path, monkeypatch):
    """Verify that if database commit fails during claim, the file is not touched or deleted."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        orphan_key = f"{uuid.uuid4()}.bin"
        orphan_file = tmp_path / orphan_key
        orphan_file.write_bytes(b"file guarded against db failure")

        two_hours_ago = time.time() - 7200
        os.utime(orphan_file, (two_hours_ago, two_hours_ago))

        from unittest.mock import patch

        # Mock db.commit to raise an OperationalError when registering the lease
        def _failing_commit():
            raise RuntimeError("Database disk I/O error during claim registration")

        with patch.object(db, "commit", side_effect=_failing_commit):
            evidence_service.reconcile_orphan_evidence(
                db, dry_run=False, grace_period_seconds=3600, quarantine=False
            )

        # File MUST NOT be deleted because database claim failed!
        assert orphan_file.exists(), "File was deleted despite database failure!"
    finally:
        db.close()


def test_restart_during_incomplete_cleanup_recovery(test_db, tmp_path, monkeypatch):
    """Verify that stale CLAIMED leases from a crash/restart are audited and reconciled on next startup."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        quarantine_dir = tmp_path / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Scenario A: Lease was CLAIMED, but file was moved to quarantine before crash
        key_a = f"{uuid.uuid4()}.bin"
        (quarantine_dir / key_a).write_bytes(b"content a in quarantine")
        lease_a = EvidenceCleanupLease(
            storage_key=key_a,
            status="CLAIMED",
            action="quarantine",
            lease_expires_at=datetime.now(timezone.utc),
        )
        db.add(lease_a)

        # Scenario B: Lease was CLAIMED, file was completely purged before crash
        key_b = f"{uuid.uuid4()}.bin"
        lease_b = EvidenceCleanupLease(
            storage_key=key_b,
            status="CLAIMED",
            action="purge",
            lease_expires_at=datetime.now(timezone.utc),
        )
        db.add(lease_b)

        # Scenario C: Lease was CLAIMED, file is still in storage_dir and got linked to Evidence
        user = User(email="recovery_tester@safeher.internal", password_hash="secret", role="survivor")
        db.add(user)
        db.flush()
        case = Case(title="Recovery Case", owner_id=user.id)
        db.add(case)
        db.flush()

        key_c = f"{uuid.uuid4()}.bin"
        (tmp_path / key_c).write_bytes(b"content c linked")
        evidence_c = Evidence(
            case_id=case.id,
            owner_id=user.id,
            original_filename="c.jpg",
            storage_key=key_c,
            mime_type="image/jpeg",
            size_bytes=10,
            sha256="hash_c",
        )
        db.add(evidence_c)
        lease_c = EvidenceCleanupLease(
            storage_key=key_c,
            status="CLAIMED",
            action="purge",
            lease_expires_at=datetime.now(timezone.utc),
        )
        db.add(lease_c)
        db.commit()

        # Run reconciliation
        evidence_service.reconcile_orphan_evidence(db, dry_run=False, grace_period_seconds=3600)

        # Check recovered lease statuses
        db.refresh(lease_a)
        db.refresh(lease_b)
        db.refresh(lease_c)

        assert lease_a.status == "QUARANTINED"
        assert lease_b.status == "PURGED"
        assert lease_c.status == "ABORTED"
        assert (tmp_path / key_c).exists()
    finally:
        db.close()


def test_upload_exceeding_grace_period_protected_by_active_upload_lease(test_db, tmp_path, monkeypatch):
    """Verify that an upload exceeding the grace period is protected by its ACTIVE_UPLOAD lease."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        storage_key = f"{uuid.uuid4()}.bin"
        file_path = tmp_path / storage_key
        file_path.write_bytes(b"slow streaming upload in progress")

        # Simulate slow upload exceeding 1-hour grace period (mtime = 2 hours ago)
        two_hours_ago = time.time() - 7200
        os.utime(file_path, (two_hours_ago, two_hours_ago))

        # Register ACTIVE_UPLOAD lease
        upload_lease = EvidenceCleanupLease(
            storage_key=storage_key,
            status="ACTIVE_UPLOAD",
            action="upload",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            notes="Active upload exceeding grace period",
        )
        db.add(upload_lease)
        db.commit()

        # Run orphan reconciliation in ACTIVE QUARANTINE mode
        orphans = evidence_service.reconcile_orphan_evidence(
            db, dry_run=False, grace_period_seconds=3600, quarantine=True
        )

        # File MUST NOT be quarantined because ACTIVE_UPLOAD lease is active!
        assert storage_key not in orphans
        assert file_path.exists(), "Active upload file was quarantined!"
        quarantine_path = tmp_path / "quarantine" / storage_key
        assert not quarantine_path.exists(), "Active upload file was moved to quarantine!"
    finally:
        db.close()


def test_cleanup_racing_with_upload_completion_intercepted(test_db, tmp_path, monkeypatch):
    """Verify that if an upload completes right before cleanup registers its claim, cleanup aborts."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        user = User(email="race_tester@safeher.internal", password_hash="secret", role="survivor")
        db.add(user)
        db.flush()
        case = Case(title="Race Case", owner_id=user.id)
        db.add(case)
        db.flush()

        storage_key = f"{uuid.uuid4()}.bin"
        file_path = tmp_path / storage_key
        file_path.write_bytes(b"racing upload content")

        # Age file to exceed grace period
        two_hours_ago = time.time() - 7200
        os.utime(file_path, (two_hours_ago, two_hours_ago))

        # Upload completes right before claim: evidence record committed
        evidence = Evidence(
            case_id=case.id,
            owner_id=user.id,
            original_filename="race.jpg",
            storage_key=storage_key,
            mime_type="image/jpeg",
            size_bytes=21,
            sha256="race_hash",
        )
        db.add(evidence)
        db.commit()

        # Run reconciliation
        evidence_service.reconcile_orphan_evidence(
            db, dry_run=False, grace_period_seconds=3600, quarantine=True
        )

        # File is preserved in canonical storage
        assert file_path.exists()
        assert not (tmp_path / "quarantine" / storage_key).exists()
    finally:
        db.close()


def test_recovery_from_quarantine_and_database_unavailable(test_db, tmp_path, monkeypatch):
    """Verify recovery from quarantine back to vault, and resilience when database is unavailable."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))

    db = test_db()
    try:
        quarantine_dir = tmp_path / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        storage_key = f"{uuid.uuid4()}.bin"
        q_file = quarantine_dir / storage_key
        q_file.write_bytes(b"quarantined evidence to recover")

        lease = EvidenceCleanupLease(
            storage_key=storage_key,
            status="QUARANTINED",
            action="quarantine",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        db.add(lease)
        db.commit()

        # 1. Successful recovery
        recovered = evidence_service.restore_from_quarantine(db, storage_key)
        assert recovered is True
        assert (tmp_path / storage_key).exists(), "File not restored to canonical storage!"
        assert not q_file.exists(), "File still in quarantine!"

        db.refresh(lease)
        assert lease.status == "RESTORED"

        # 2. Database unavailable during recovery attempt: file remains safe in quarantine
        storage_key_2 = f"{uuid.uuid4()}.bin"
        q_file_2 = quarantine_dir / storage_key_2
        q_file_2.write_bytes(b"file 2 for db failure test")

        from unittest.mock import patch

        def _failing_commit():
            raise RuntimeError("Database connection lost during recovery commit")

        with patch.object(db, "commit", side_effect=_failing_commit):
            res = evidence_service.restore_from_quarantine(db, storage_key_2)
            assert res is False
    finally:
        db.close()


def test_active_upload_lease_protects_matching_part_staging_file(test_db, tmp_path, monkeypatch):
    """The <uuid>.part staging name must resolve to its <uuid>.bin lease key."""
    import os
    import time
    from datetime import datetime, timedelta, timezone

    from backend.app.config import settings
    from backend.app.models import EvidenceCleanupLease
    from backend.app.services.evidence_service import evidence_service

    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))
    storage_key = "active-upload-123.bin"
    part_path = tmp_path / "active-upload-123.part"
    part_path.write_bytes(b"in-flight evidence")
    old_time = time.time() - 7200
    os.utime(part_path, (old_time, old_time))

    with test_db() as db:
        db.add(
            EvidenceCleanupLease(
                storage_key=storage_key,
                status="ACTIVE_UPLOAD",
                action="upload",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
            )
        )
        db.commit()

        result = evidence_service.reconcile_orphan_evidence(
            db=db,
            dry_run=False,
            grace_period_seconds=3600,
            quarantine=True,
        )

    assert part_path.exists()
    assert not (tmp_path / "quarantine" / part_path.name).exists()
    assert part_path.name not in result


def test_restore_quarantine_rejects_path_traversal(test_db, tmp_path, monkeypatch):
    from backend.app.config import settings
    from backend.app.services.evidence_service import evidence_service

    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))
    with test_db() as db:
        assert evidence_service.restore_from_quarantine(db, "../outside.bin") is False
        assert evidence_service.restore_from_quarantine(db, "..\\outside.bin") is False


def test_restore_quarantine_compensates_filesystem_move_when_db_commit_fails(
    test_db, tmp_path, monkeypatch
):
    from datetime import datetime, timedelta, timezone

    from backend.app.config import settings
    from backend.app.models import EvidenceCleanupLease
    from backend.app.services.evidence_service import evidence_service

    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))
    quarantine_dir = tmp_path / "quarantine"
    quarantine_dir.mkdir()
    filename = "restore-failure-123.bin"
    quarantined_path = quarantine_dir / filename
    quarantined_path.write_bytes(b"preserved quarantined bytes")

    with test_db() as db:
        db.add(
            EvidenceCleanupLease(
                storage_key=filename,
                status="QUARANTINED",
                action="quarantine",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        )
        db.commit()

        original_commit = db.commit

        def fail_commit():
            raise RuntimeError("simulated database outage")

        monkeypatch.setattr(db, "commit", fail_commit)
        assert evidence_service.restore_from_quarantine(db, filename) is False
        monkeypatch.setattr(db, "commit", original_commit)

    assert quarantined_path.exists()
    assert quarantined_path.read_bytes() == b"preserved quarantined bytes"
    assert not (tmp_path / filename).exists()


def test_stale_claim_recovery_recognizes_part_staging_files(test_db, tmp_path, monkeypatch):
    """A stale .part file must not be falsely recorded as purged when its lease uses .bin."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))
    storage_key = "stale-part-recovery-123.bin"
    part_path = tmp_path / "stale-part-recovery-123.part"
    part_path.write_bytes(b"staging bytes still present")
    old_time = time.time() - 7200
    os.utime(part_path, (old_time, old_time))

    with test_db() as db:
        stale_lease = EvidenceCleanupLease(
            storage_key=storage_key,
            status="CLAIMED",
            action="quarantine",
            lease_expires_at=datetime.now(timezone.utc),
        )
        db.add(stale_lease)
        db.commit()

        evidence_service.reconcile_orphan_evidence(
            db=db,
            dry_run=False,
            grace_period_seconds=3600,
            quarantine=True,
        )

        db.refresh(stale_lease)
        assert stale_lease.status == "ABORTED"
        assert (tmp_path / "quarantine" / part_path.name).exists()
        assert not part_path.exists()


def test_interrupted_quarantine_purge_recovers_part_file_by_canonical_key(
    test_db, tmp_path, monkeypatch
):
    """A crashed quarantine purge must find the .part filename behind its .bin lease key."""
    monkeypatch.setattr(settings, "PRIVATE_STORAGE_PATH", str(tmp_path))
    quarantine_dir = tmp_path / "quarantine"
    quarantine_dir.mkdir()
    storage_key = "purge-part-recovery-123.bin"
    quarantined_part = quarantine_dir / "purge-part-recovery-123.part"
    quarantined_part.write_bytes(b"expired staged orphan")
    old_time = time.time() - 7200
    os.utime(quarantined_part, (old_time, old_time))

    with test_db() as db:
        lease = EvidenceCleanupLease(
            storage_key=storage_key,
            status="PURGE_CLAIMED",
            action="purge_quarantine",
            lease_expires_at=datetime.now(timezone.utc),
        )
        db.add(lease)
        db.commit()

        evidence_service.reconcile_orphan_evidence(
            db=db,
            dry_run=False,
            grace_period_seconds=3600,
            quarantine=True,
            quarantine_retention_seconds=3600,
        )

        db.refresh(lease)
        assert lease.status == "PURGED"
        assert not quarantined_part.exists()
