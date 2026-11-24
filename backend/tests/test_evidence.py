
import hashlib
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import AuditLog, Case, Evidence


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def test_db():
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
    db = TestingSessionLocal()

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    yield db

    app.dependency_overrides.clear()
    db.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def client(test_db):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def private_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        settings,
        "PRIVATE_STORAGE_PATH",
        str(tmp_path),
    )
    return tmp_path


# --------------------------------------------------
# Helper: Register and login
# --------------------------------------------------

def register_and_login(client, email):
    password = "SafeHER-Test-Password123!"

    registration = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )

    assert registration.status_code == 201

    login = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )

    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"

    return (
        UUID(registration.json()["id"]),
        login.json()["access_token"],
    )


# --------------------------------------------------
# Helper: Create a test case
# --------------------------------------------------

def create_case(test_db, owner_id):
    case = Case(
        owner_id=UUID(str(owner_id)),
        title="Test Evidence Case",
        description="Case created for evidence testing",
    )

    test_db.add(case)
    test_db.commit()
    test_db.refresh(case)

    return str(case.id)


# --------------------------------------------------
# 1. Successful evidence upload
# --------------------------------------------------

def test_evidence_upload_success(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "evidence@example.com",
    )

    case_id = create_case(test_db, user_id)
    file_content = b"SafeHER evidence test content"

    response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert response.status_code == 201

    result = response.json()

    assert result["message"] == "Evidence uploaded successfully."
    assert result["original_filename"] == "proof.txt"
    assert result["size_bytes"] == len(file_content)
    assert result["sha256"] == hashlib.sha256(
        file_content
    ).hexdigest()

    storage_files = list(private_storage.iterdir())
    assert len(storage_files) == 1
    assert storage_files[0].exists()

    audit_entry = test_db.scalar(
        select(AuditLog).where(
            AuditLog.action == "evidence_uploaded"
        )
    )

    assert audit_entry is not None
    assert audit_entry.user_id == user_id


# --------------------------------------------------
# 2. Upload requires authentication
# --------------------------------------------------

def test_evidence_upload_requires_authentication(
    client,
    test_db,
):
    user_id, _ = register_and_login(
        client,
        "unauth-upload@example.com",
    )

    case_id = create_case(test_db, user_id)

    response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        files={
            "file": (
                "proof.txt",
                b"Test evidence",
                "text/plain",
            ),
        },
    )

    assert response.status_code == 401


# --------------------------------------------------
# 3. User cannot upload to another user's case
# --------------------------------------------------

def test_user_cannot_upload_to_another_users_case(
    client,
    test_db,
    private_storage,
):
    _, user_one_token = register_and_login(
        client,
        "user-one@example.com",
    )

    user_two_id, _ = register_and_login(
        client,
        "user-two@example.com",
    )

    user_two_case_id = create_case(
        test_db,
        user_two_id,
    )

    response = client.post(
        f"/api/v1/evidence/cases/{user_two_case_id}",
        headers={"Authorization": f"Bearer {user_one_token}"},
        files={
            "file": (
                "proof.txt",
                b"Unauthorized evidence",
                "text/plain",
            ),
        },
    )

    assert response.status_code == 404
    assert len(list(private_storage.iterdir())) == 0


# --------------------------------------------------
# 4. Unsupported file format is rejected
# --------------------------------------------------

def test_unsupported_file_format_rejected(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "unsupported@example.com",
    )

    case_id = create_case(test_db, user_id)

    response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "malware.exe",
                b"Test content",
                "application/octet-stream",
            ),
        },
    )

    assert response.status_code == 400
    assert len(list(private_storage.iterdir())) == 0


# --------------------------------------------------
# 5. File content must match its extension
# --------------------------------------------------

def test_file_content_must_match_extension(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "fake-file@example.com",
    )

    case_id = create_case(test_db, user_id)

    response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "fake.jpg",
                b"This is not a real JPEG image",
                "image/jpeg",
            ),
        },
    )

    assert response.status_code == 400
    assert len(list(private_storage.iterdir())) == 0


# --------------------------------------------------
# 6. User can list their own case evidence
# --------------------------------------------------

def test_user_can_list_own_case_evidence(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "list-evidence@example.com",
    )

    case_id = create_case(test_db, user_id)
    file_content = b"SafeHER listing test"

    upload_response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert upload_response.status_code == 201

    response = client.get(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["case_id"] == case_id
    assert result["total"] == 1
    assert len(result["evidence"]) == 1

    evidence = result["evidence"][0]

    assert evidence["original_filename"] == "proof.txt"
    assert evidence["size_bytes"] == len(file_content)
    assert evidence["sha256"] == hashlib.sha256(
        file_content
    ).hexdigest()

    assert "storage_key" not in evidence


# --------------------------------------------------
# 7. User cannot list another user's case evidence
# --------------------------------------------------

def test_user_cannot_list_another_users_case_evidence(
    client,
    test_db,
):
    _, user_one_token = register_and_login(
        client,
        "list-user-one@example.com",
    )

    user_two_id, _ = register_and_login(
        client,
        "list-user-two@example.com",
    )

    user_two_case_id = create_case(
        test_db,
        user_two_id,
    )

    response = client.get(
        f"/api/v1/evidence/cases/{user_two_case_id}",
        headers={"Authorization": f"Bearer {user_one_token}"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found."


# --------------------------------------------------
# 8. Evidence listing requires authentication
# --------------------------------------------------

def test_evidence_listing_requires_authentication(
    client,
    test_db,
):
    user_id, _ = register_and_login(
        client,
        "list-noauth@example.com",
    )

    case_id = create_case(test_db, user_id)

    response = client.get(
        f"/api/v1/evidence/cases/{case_id}",
    )

    assert response.status_code == 401


# --------------------------------------------------
# 9. User can download their own evidence
# --------------------------------------------------

def test_user_can_download_own_evidence(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "download-own@example.com",
    )

    case_id = create_case(test_db, user_id)
    file_content = b"Private SafeHER evidence"

    upload_response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert upload_response.status_code == 201
    evidence_id = upload_response.json()["id"]

    response = client.get(
        f"/api/v1/evidence/{evidence_id}/download",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.content == file_content
    assert "attachment" in response.headers["content-disposition"]
    assert "proof.txt" in response.headers["content-disposition"]


# --------------------------------------------------
# 10. Evidence download requires authentication
# --------------------------------------------------

def test_evidence_download_requires_authentication(
    client,
    test_db,
):
    user_id, token = register_and_login(
        client,
        "download-noauth@example.com",
    )

    case_id = create_case(test_db, user_id)

    upload_response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "proof.txt",
                b"Private evidence",
                "text/plain",
            ),
        },
    )

    assert upload_response.status_code == 201
    evidence_id = upload_response.json()["id"]

    response = client.get(
        f"/api/v1/evidence/{evidence_id}/download",
    )

    assert response.status_code == 401


# --------------------------------------------------
# 11. User cannot download another user's evidence
# --------------------------------------------------

def test_user_cannot_download_another_users_evidence(
    client,
    test_db,
    private_storage,
):
    _, user_one_token = register_and_login(
        client,
        "download-user-one@example.com",
    )

    user_two_id, user_two_token = register_and_login(
        client,
        "download-user-two@example.com",
    )

    user_two_case_id = create_case(
        test_db,
        user_two_id,
    )

    upload_response = client.post(
        f"/api/v1/evidence/cases/{user_two_case_id}",
        headers={"Authorization": f"Bearer {user_two_token}"},
        files={
            "file": (
                "private.txt",
                b"User two private evidence",
                "text/plain",
            ),
        },
    )

    assert upload_response.status_code == 201
    evidence_id = upload_response.json()["id"]

    response = client.get(
        f"/api/v1/evidence/{evidence_id}/download",
        headers={"Authorization": f"Bearer {user_one_token}"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Evidence not found."


# --------------------------------------------------
# 12. Missing evidence file returns 404
# --------------------------------------------------

def test_download_missing_evidence_file_returns_404(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "download-missing@example.com",
    )

    case_id = create_case(test_db, user_id)

    upload_response = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "proof.txt",
                b"Evidence whose file will be removed",
                "text/plain",
            ),
        },
    )

    assert upload_response.status_code == 201
    evidence_id = upload_response.json()["id"]

    # Simulate a missing file in private storage.
    for stored_file in private_storage.iterdir():
        stored_file.unlink()

    response = client.get(
        f"/api/v1/evidence/{evidence_id}/download",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Evidence file not found."


# --------------------------------------------------
# 13. Same user cannot upload identical evidence twice
# --------------------------------------------------

def test_duplicate_evidence_upload_is_rejected(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "duplicate-test@example.com",
    )

    case_id = create_case(test_db, user_id)
    file_content = b"SafeHER duplicate evidence test"
    headers = {"Authorization": f"Bearer {token}"}

    first_upload = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers=headers,
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert first_upload.status_code == 201

    second_upload = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers=headers,
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert second_upload.status_code == 409
    assert "Duplicate evidence detected" in second_upload.json()["detail"]

    # Only the original file should remain.
    assert len(list(private_storage.iterdir())) == 1

    evidence_records = test_db.scalars(
        select(Evidence).where(
            Evidence.owner_id == user_id
        )
    ).all()

    assert len(evidence_records) == 1

    audit_entries = test_db.scalars(
        select(AuditLog).where(
            AuditLog.user_id == user_id,
            AuditLog.action == "evidence_uploaded",
        )
    ).all()

    assert len(audit_entries) == 1


# --------------------------------------------------
# 14. Duplicate detection works across different filenames
# --------------------------------------------------

def test_duplicate_evidence_with_different_filename_is_rejected(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "duplicate-filename@example.com",
    )

    case_id = create_case(test_db, user_id)
    file_content = b"Identical content with different filenames"
    headers = {"Authorization": f"Bearer {token}"}

    first_upload = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers=headers,
        files={
            "file": (
                "original.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert first_upload.status_code == 201

    second_upload = client.post(
        f"/api/v1/evidence/cases/{case_id}",
        headers=headers,
        files={
            "file": (
                "renamed.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert second_upload.status_code == 409
    assert len(list(private_storage.iterdir())) == 1


# --------------------------------------------------
# 15. Duplicate detection works across different cases
# --------------------------------------------------

def test_duplicate_evidence_across_different_cases_is_rejected(
    client,
    test_db,
    private_storage,
):
    user_id, token = register_and_login(
        client,
        "duplicate-cases@example.com",
    )

    first_case_id = create_case(test_db, user_id)
    second_case_id = create_case(test_db, user_id)

    file_content = b"Same evidence uploaded to two cases"
    headers = {"Authorization": f"Bearer {token}"}

    first_upload = client.post(
        f"/api/v1/evidence/cases/{first_case_id}",
        headers=headers,
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert first_upload.status_code == 201

    second_upload = client.post(
        f"/api/v1/evidence/cases/{second_case_id}",
        headers=headers,
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert second_upload.status_code == 409
    assert len(list(private_storage.iterdir())) == 1


# --------------------------------------------------
# 16. Different users can upload identical evidence
# --------------------------------------------------

def test_different_users_can_upload_identical_evidence(
    client,
    test_db,
    private_storage,
):
    user_one_id, user_one_token = register_and_login(
        client,
        "duplicate-user-one@example.com",
    )

    user_two_id, user_two_token = register_and_login(
        client,
        "duplicate-user-two@example.com",
    )

    user_one_case = create_case(test_db, user_one_id)
    user_two_case = create_case(test_db, user_two_id)

    file_content = b"Identical evidence for different users"

    first_upload = client.post(
        f"/api/v1/evidence/cases/{user_one_case}",
        headers={"Authorization": f"Bearer {user_one_token}"},
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert first_upload.status_code == 201

    second_upload = client.post(
        f"/api/v1/evidence/cases/{user_two_case}",
        headers={"Authorization": f"Bearer {user_two_token}"},
        files={
            "file": (
                "proof.txt",
                file_content,
                "text/plain",
            ),
        },
    )

    assert second_upload.status_code == 201

    # Each user should have their own stored copy.
    assert len(list(private_storage.iterdir())) == 2

    evidence_records = test_db.scalars(
        select(Evidence)
    ).all()

    assert len(evidence_records) == 2
