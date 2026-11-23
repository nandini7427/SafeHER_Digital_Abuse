import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import User
from backend.app.security.token import decode_access_token


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
    with TestClient(app) as test_client:
        yield test_client


def test_user_registration(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )

    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
    assert "password_hash" not in response.json()


def test_duplicate_registration(client):
    data = {
        "email": "test@example.com",
        "password": "SafeHER-Test-Password-123!",
    }

    first_response = client.post("/api/v1/auth/register", json=data)
    second_response = client.post("/api/v1/auth/register", json=data)

    assert first_response.status_code == 201
    assert second_response.status_code == 409


def test_successful_login(client):
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )
    assert registration.status_code == 201

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"

    user_id = decode_access_token(response.json()["access_token"])
    assert str(user_id) == registration.json()["id"]


def test_incorrect_password_rejected(client):
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "Wrong-Password-123!",
        },
    )

    assert response.status_code == 401


def test_inactive_account_rejected(client, test_db):
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )

    with test_db() as db:
        user = db.scalar(
            select(User).where(User.email == "test@example.com")
        )
        user.is_active = False
        db.commit()

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )

    assert response.status_code == 403


from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.app.security.dependencies import get_current_user
from backend.app.security.token import create_access_token


def test_missing_token_rejected(test_db):
    with test_db() as db:
        with pytest.raises(HTTPException) as exc:
            get_current_user(None, db)

    assert exc.value.status_code == 401


def test_invalid_token_rejected(test_db):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="invalid.token.value",
    )

    with test_db() as db:
        with pytest.raises(HTTPException) as exc:
            get_current_user(credentials, db)

    assert exc.value.status_code == 401


def test_valid_token_returns_user(test_db):
    with test_db() as db:
        user = User(
            email="protected@example.com",
            password_hash="test-hash",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token(user.id)
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token,
        )

        authenticated_user = get_current_user(credentials, db)

        assert authenticated_user.id == user.id
        assert authenticated_user.email == "protected@example.com"


def test_inactive_user_token_rejected(test_db):
    with test_db() as db:
        user = User(
            email="inactive@example.com",
            password_hash="test-hash",
            is_active=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token(user.id)
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token,
        )

        with pytest.raises(HTTPException) as exc:
            get_current_user(credentials, db)

    assert exc.value.status_code == 401

def test_me_endpoint_requires_token(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_me_endpoint_rejects_invalid_token(client):
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid.token.value"},
    )
    assert response.status_code == 401


def test_me_endpoint_accepts_valid_token(client):
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "email": "profile@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )
    assert registration.status_code == 201

    login = client.post(
        "/api/v1/auth/login",
        json={
            "email": "profile@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )
    assert login.status_code == 200

    token = login.json()["access_token"]

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["email"] == "profile@example.com"
    assert response.json()["id"] == registration.json()["id"]
    assert "password_hash" not in response.json()

from backend.app.security.permissions import require_roles


def test_regular_user_denied_admin_permission(test_db):
    regular_user = User(
        email="regular@example.com",
        password_hash="test-hash",
        role="user",
    )

    with test_db() as db:
        db.add(regular_user)
        db.commit()
        db.refresh(regular_user)

        admin_checker = require_roles("admin")

        with pytest.raises(HTTPException) as exc:
            admin_checker(regular_user)

    assert exc.value.status_code == 403


def test_admin_allowed_admin_permission(test_db):
    admin_user = User(
        email="admin@example.com",
        password_hash="test-hash",
        role="admin",
    )

    with test_db() as db:
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)

        admin_checker = require_roles("admin")
        result = admin_checker(admin_user)

    assert result.id == admin_user.id
    assert result.role == "admin"

def test_regular_user_denied_admin_summary(client):
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "email": "ordinary@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )
    assert registration.status_code == 201

    login = client.post(
        "/api/v1/auth/login",
        json={
            "email": "ordinary@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )
    token = login.json()["access_token"]

    response = client.get(
        "/api/v1/admin/summary",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403


def test_admin_can_access_admin_summary(client, test_db):
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "email": "administrator@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )
    assert registration.status_code == 201

    with test_db() as db:
        admin_user = db.scalar(
            select(User).where(
                User.email == "administrator@example.com"
            )
        )
        admin_user.role = "admin"
        db.commit()

    login = client.post(
        "/api/v1/auth/login",
        json={
            "email": "administrator@example.com",
            "password": "SafeHER-Test-Password-123!",
        },
    )
    token = login.json()["access_token"]

    response = client.get(
        "/api/v1/admin/summary",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert "total_users" in response.json()
    assert "total_cases" in response.json()
    assert "total_evidence" in response.json()

def test_registration_cannot_assign_admin_role(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "attempted-admin@example.com",
            "password": "SafeHER-Test-Password-123!",
            "role": "admin",
        },
    )

    assert response.status_code == 201
    assert response.json()["role"] == "user"
