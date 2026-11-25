"""Authentication and owner-isolation tests for the case management API."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
import pytest

from backend.app.database import Base, get_db
from backend.app.main import app


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def register_and_login(client: TestClient, email: str) -> dict[str, str]:
    credentials = {"email": email, "password": "SafeHER-Case-Test-Password-123!"}
    assert client.post("/api/v1/auth/register", json=credentials).status_code == 201
    login = client.post("/api/v1/auth/login", json=credentials)
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_cases_require_authentication(client: TestClient):
    assert client.get("/api/v1/cases").status_code == 401
    assert client.post("/api/v1/cases", json={"title": "My incident"}).status_code == 401


def test_user_can_create_and_list_owned_cases(client: TestClient):
    headers = register_and_login(client, "owner@example.test")
    created = client.post(
        "/api/v1/cases",
        headers=headers,
        json={"title": "  Suspected image abuse  ", "description": "  Preserve original link and date. "},
    )
    assert created.status_code == 201
    data = created.json()
    assert data["title"] == "Suspected image abuse"
    assert data["description"] == "Preserve original link and date."
    assert data["status"] == "open"
    assert data["owner_id"]

    listed = client.get("/api/v1/cases", headers=headers)
    assert listed.status_code == 200
    assert [case["id"] for case in listed.json()] == [data["id"]]

    fetched = client.get(f"/api/v1/cases/{data['id']}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["id"] == data["id"]


def test_case_is_not_visible_to_another_user(client: TestClient):
    owner_headers = register_and_login(client, "owner2@example.test")
    other_headers = register_and_login(client, "other2@example.test")
    created = client.post(
        "/api/v1/cases",
        headers=owner_headers,
        json={"title": "Private incident"},
    )
    case_id = created.json()["id"]

    assert client.get("/api/v1/cases", headers=other_headers).json() == []
    assert client.get(f"/api/v1/cases/{case_id}", headers=other_headers).status_code == 404


def test_case_title_rejects_whitespace_only(client: TestClient):
    headers = register_and_login(client, "title@example.test")
    response = client.post("/api/v1/cases", headers=headers, json={"title": "   "})
    assert response.status_code == 422
