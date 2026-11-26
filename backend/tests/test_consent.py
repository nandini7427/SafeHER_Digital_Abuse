
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.main import app


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


def create_test_consent(client, token):
    return client.post(
        "/api/v1/consent/",
        json={
            "consent_version": "1.0",
            "purpose": "Evidence analysis",
        },
        headers=auth_headers(token),
    )


def test_create_consent(client):
    _, token = register_and_login(client, "consent1@example.com")

    response = create_test_consent(client, token)

    assert response.status_code == 201

    data = response.json()

    assert data["consent_version"] == "1.0"
    assert data["purpose"] == "Evidence analysis"
    assert data["status"] == "granted"
    assert data["withdrawn_at"] is None


def test_list_user_consents(client):
    _, token = register_and_login(client, "consent2@example.com")

    create_test_consent(client, token)

    response = client.get(
        "/api/v1/consent/",
        headers=auth_headers(token),
    )

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_users_can_only_see_their_own_consents(client):
    _, token1 = register_and_login(client, "consent3@example.com")
    _, token2 = register_and_login(client, "consent4@example.com")

    create_test_consent(client, token1)

    response = client.get(
        "/api/v1/consent/",
        headers=auth_headers(token2),
    )

    assert response.status_code == 200
    assert response.json() == []


def test_withdraw_consent(client):
    _, token = register_and_login(client, "consent5@example.com")

    create_response = create_test_consent(client, token)
    consent_id = create_response.json()["id"]

    response = client.post(
        f"/api/v1/consent/{consent_id}/withdraw",
        headers=auth_headers(token),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "withdrawn"
    assert response.json()["withdrawn_at"] is not None


def test_users_cannot_withdraw_other_users_consent(client):
    _, token1 = register_and_login(client, "consent6@example.com")
    _, token2 = register_and_login(client, "consent7@example.com")

    create_response = create_test_consent(client, token1)
    consent_id = create_response.json()["id"]

    response = client.post(
        f"/api/v1/consent/{consent_id}/withdraw",
        headers=auth_headers(token2),
    )

    assert response.status_code == 404


def test_consent_endpoints_require_authentication(client):
    response = client.get("/api/v1/consent/")

    assert response.status_code in (401, 403)
