
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import Case, Evidence


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

    return (
        UUID(registration.json()["id"]),
        login.json()["access_token"],
    )


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_create_case(client):
    _, token = register_and_login(client, "case1@example.com")

    response = client.post(
        "/api/v1/cases",
        json={
            "title": "Online Harassment",
            "description": "Harassment through social media.",
        },
        headers=auth_headers(token),
    )

    assert response.status_code == 201

    data = response.json()

    assert data["title"] == "Online Harassment"
    assert data["description"] == "Harassment through social media."
    assert data["status"] == "open"
    assert "id" in data
    assert "owner_id" in data


def test_list_user_cases(client):
    _, token = register_and_login(client, "case2@example.com")

    headers = auth_headers(token)

    client.post(
        "/api/v1/cases",
        json={"title": "First Case"},
        headers=headers,
    )

    client.post(
        "/api/v1/cases",
        json={"title": "Second Case"},
        headers=headers,
    )

    response = client.get(
        "/api/v1/cases",
        headers=headers,
    )

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_users_can_only_see_their_own_cases(client):
    _, token1 = register_and_login(client, "owner1@example.com")
    _, token2 = register_and_login(client, "owner2@example.com")

    create_response = client.post(
        "/api/v1/cases",
        json={"title": "Private Case"},
        headers=auth_headers(token1),
    )

    case_id = create_response.json()["id"]

    response = client.get(
        "/api/v1/cases",
        headers=auth_headers(token2),
    )

    assert response.status_code == 200
    assert response.json() == []

    response = client.get(
        f"/api/v1/cases/{case_id}",
        headers=auth_headers(token2),
    )

    assert response.status_code == 404


def test_get_single_case(client):
    _, token = register_and_login(client, "case3@example.com")

    create_response = client.post(
        "/api/v1/cases",
        json={"title": "Evidence Review"},
        headers=auth_headers(token),
    )

    case_id = create_response.json()["id"]

    response = client.get(
        f"/api/v1/cases/{case_id}",
        headers=auth_headers(token),
    )

    assert response.status_code == 200
    assert response.json()["id"] == case_id
    assert response.json()["title"] == "Evidence Review"


def test_update_case(client):
    _, token = register_and_login(client, "case4@example.com")

    headers = auth_headers(token)

    create_response = client.post(
        "/api/v1/cases",
        json={"title": "Original Title"},
        headers=headers,
    )

    case_id = create_response.json()["id"]

    response = client.patch(
        f"/api/v1/cases/{case_id}",
        json={
            "title": "Updated Title",
            "description": "Updated description.",
            "status": "under_review",
        },
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["title"] == "Updated Title"
    assert data["description"] == "Updated description."
    assert data["status"] == "under_review"


def test_empty_update_is_rejected(client):
    _, token = register_and_login(client, "case5@example.com")

    headers = auth_headers(token)

    create_response = client.post(
        "/api/v1/cases",
        json={"title": "Test Case"},
        headers=headers,
    )

    case_id = create_response.json()["id"]

    response = client.patch(
        f"/api/v1/cases/{case_id}",
        json={},
        headers=headers,
    )

    assert response.status_code == 400


def test_null_title_is_rejected(client):
    _, token = register_and_login(client, "case6@example.com")

    headers = auth_headers(token)

    create_response = client.post(
        "/api/v1/cases",
        json={"title": "Test Case"},
        headers=headers,
    )

    case_id = create_response.json()["id"]

    response = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"title": None},
        headers=headers,
    )

    assert response.status_code == 422


def test_delete_case_without_evidence(client, test_db):
    _, token = register_and_login(client, "case7@example.com")

    headers = auth_headers(token)

    create_response = client.post(
        "/api/v1/cases",
        json={"title": "Case to Delete"},
        headers=headers,
    )

    case_id = UUID(create_response.json()["id"])

    response = client.delete(
        f"/api/v1/cases/{case_id}",
        headers=headers,
    )

    assert response.status_code == 204

    deleted_case = test_db.scalar(
        select(Case).where(Case.id == case_id)
    )

    assert deleted_case is None


def test_delete_case_with_evidence_is_blocked(client, test_db):
    _, token = register_and_login(client, "case8@example.com")

    headers = auth_headers(token)

    create_response = client.post(
        "/api/v1/cases",
        json={"title": "Case With Evidence"},
        headers=headers,
    )

    case_id = UUID(create_response.json()["id"])

    evidence = Evidence(
        owner_id=UUID(create_response.json()["owner_id"]),
        case_id=case_id,
        original_filename="evidence.png",
        storage_key="test-storage-key",
        mime_type="image/png",
        size_bytes=100,
        sha256="a" * 64,
    )

    test_db.add(evidence)
    test_db.commit()

    response = client.delete(
        f"/api/v1/cases/{case_id}",
        headers=headers,
    )

    assert response.status_code == 409

    existing_case = test_db.scalar(
        select(Case).where(Case.id == case_id)
    )

    assert existing_case is not None


def test_case_endpoints_require_authentication(client):
    response = client.get("/api/v1/cases")

    assert response.status_code in (401, 403)


def test_invalid_case_id_is_rejected(client):
    _, token = register_and_login(client, "case9@example.com")

    response = client.get(
        "/api/v1/cases/not-a-valid-uuid",
        headers=auth_headers(token),
    )

    assert response.status_code == 422
