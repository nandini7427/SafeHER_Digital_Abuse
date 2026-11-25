"""Integration and security tests for SafeHER AI Threat Assessment API.

Verifies:
- Authentication and authorization enforcement.
- Strict input validation (empty, whitespace, oversized, invalid data types, surrogates).
- Baseline DevelopmentThreatAnalyzer execution and response schema integrity.
- Case ownership verification and tenant isolation.
- Privacy safeguards (zero plaintext evidence logging).
- Offline fallback and neural configuration behavior.
"""

import logging
from typing import Dict, Optional
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_core.config import MANDATORY_AI_DISCLAIMER, ThreatAnalyzerConfig
from ai_core.threat.analyzer import DevelopmentThreatAnalyzer
from ai_core.threat.toxic_bert import ToxicBertModelRunner, ToxicBertResult, ToxicBertThreatAnalyzer
from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import AuditLog, Case, User
from backend.app.services.threat_service import (
    get_threat_analyzer,
    set_threat_analyzer,
    threat_service,
)


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
    # Reset singleton analyzer after test run
    set_threat_analyzer(None)


@pytest.fixture
def client(test_db):
    """TestClient configured with test database dependency overrides."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client) -> Dict[str, str]:
    """Helper fixture registering and logging in a standard test user."""
    reg_data = {
        "email": "victim_test@example.com",
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
        "email": "other_victim@example.com",
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

def test_threat_assessment_unauthenticated_rejected(client):
    """Reject unauthenticated threat assessment requests with 401."""
    response = client.post(
        "/api/v1/ai/threat-assessment",
        json={"text": "I will hurt you."},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_threat_assessment_invalid_token_rejected(client):
    """Reject requests with forged or malformed Bearer tokens."""
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers={"Authorization": "Bearer forged.invalid.token.signature"},
        json={"text": "I will hurt you."},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired access token."


def test_threat_assessment_inactive_user_rejected(client, auth_headers, test_db):
    """Reject valid tokens if user account has been deactivated."""
    # Deactivate the user directly in database
    db = test_db()
    user = db.scalar(select(User).where(User.email == "victim_test@example.com"))
    assert user is not None
    user.is_active = False
    db.commit()
    db.close()

    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": "I will hurt you."},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "User not found or inactive."


def test_threat_assessment_valid_token_success(client, auth_headers):
    """Allow threat assessment when valid Bearer token is provided."""
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": "I will track you down and hurt you."},
    )
    assert response.status_code == 200


# ============================================================================
# 2. INPUT VALIDATION TESTS
# ============================================================================

def test_threat_assessment_empty_text_rejected(client, auth_headers):
    """Reject empty string text payloads with 422 Unprocessable Entity."""
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": ""},
    )
    assert response.status_code == 422


def test_threat_assessment_whitespace_only_rejected(client, auth_headers):
    """Reject whitespace-only strings with 422 Unprocessable Entity."""
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": "   \n\t  "},
    )
    assert response.status_code == 422


def test_threat_assessment_oversized_text_rejected(client, auth_headers):
    """Reject text exceeding 10,000 characters with 422 Unprocessable Entity."""
    oversized = "A" * 10001
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": oversized},
    )
    assert response.status_code == 422


def test_threat_assessment_missing_fields_rejected(client, auth_headers):
    """Reject request missing the required 'text' field."""
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={},
    )
    assert response.status_code == 422


def test_threat_assessment_invalid_data_types_rejected(client, auth_headers):
    """Reject non-string text types (e.g. integers, lists)."""
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": 123456789},
    )
    assert response.status_code == 422


def test_threat_assessment_isolated_surrogates_rejected(client, auth_headers):
    """Reject strings containing only invisible characters with 400 Bad Request."""
    invisible_text = "\u200B\u200B\u200B"
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": invisible_text},
    )
    # The service catches ValueError from preprocessing and returns 400
    assert response.status_code == 400
    assert "invisible" in response.json()["detail"].lower() or "non-printable" in response.json()["detail"].lower()


# ============================================================================
# 3. AI THREAT ASSESSMENT INTEGRATION & CONTRACT TESTS
# ============================================================================

def test_threat_assessment_direct_threat_contract(client, auth_headers):
    """Verify response structure, offsets, and disclaimer on direct physical threat."""
    text = "I know where you live and I will track you down and hurt you."
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": text},
    )
    assert response.status_code == 200
    data = response.json()

    # Core contract checks
    assert uuid.UUID(data["assessment_id"])
    assert data["analyzed_text"] == text
    assert data["overall_threat_level"] in ("high", "severe")
    assert data["is_calibrated"] is False
    assert data["confidence_score"] is None
    assert data["uncertainty_state"] == "low"
    assert data["disclaimer"] == MANDATORY_AI_DISCLAIMER

    # Indicators & Flagged Spans
    assert len(data["threat_indicators"]) >= 1
    direct_indicator = next(
        ind for ind in data["threat_indicators"] if ind["category"] == "direct_threat"
    )
    assert direct_indicator["indicator_type"] in ("lexical_pattern", "pattern")

    assert len(data["flagged_spans"]) >= 1
    span = data["flagged_spans"][0]
    assert span["category"] == "direct_threat"
    assert span["start"] >= 0
    assert span["end"] > span["start"]
    assert span["start_utf16"] >= 0
    assert span["end_utf16"] > span["start_utf16"]
    # Check that slice matches analyzed text
    assert data["analyzed_text"][span["start"]:span["end"]] == span["text"]


def test_threat_assessment_financial_extortion_remediated(client, auth_headers):
    """Verify remediated financial extortion combined with blackmail and exposure threat."""
    text = "Send me 500 dollars by midnight or I will publish your private photos."
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": text},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["overall_threat_level"] == "severe"
    categories = {ind["category"] for ind in data["threat_indicators"]}
    assert "financial_extortion" in categories
    assert "exposure_threat" in categories
    assert "blackmail" in categories
    assert "urgency_pressure" in categories


def test_threat_assessment_figurative_language_negative(client, auth_headers):
    """Verify figurative violent words do NOT trigger threat alerts."""
    text = "You are completely killing it on stage tonight, what a sensational performance!"
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={"text": text},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["overall_threat_level"] == "none"
    assert data["uncertainty_state"] == "low"
    assert len(data["flagged_spans"]) == 0
    assert "killing it" in data["explanation"].lower() or "no threat indicators" in data["explanation"].lower()


def test_threat_assessment_alias_endpoint(client, auth_headers):
    """Verify alias route /api/v1/ai/threat/assess returns identical results."""
    text = "Unless you pay me 1000 dollars, I will expose your private videos."
    response = client.post(
        "/api/v1/ai/threat/assess",
        headers=auth_headers,
        json={"text": text},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["overall_threat_level"] == "severe"


# ============================================================================
# 4. CASE ASSOCIATION & TENANT ISOLATION TESTS
# ============================================================================

def test_threat_assessment_with_owned_case_succeeds(client, auth_headers, test_db):
    """Associate threat assessment with an existing case owned by the authenticated user."""
    db = test_db()
    user = db.scalar(select(User).where(User.email == "victim_test@example.com"))
    assert user is not None
    user_id = user.id

    case = Case(
        owner_id=user_id,
        title="Harassment Incident #1",
        description="Repeated blackmail messages.",
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    case_id = str(case.id)
    db.close()

    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={
            "text": "Pay me 500 dollars or else.",
            "case_id": case_id,
        },
    )
    assert response.status_code == 200

    # Verify audit log was recorded
    db = test_db()
    audit = db.scalar(
        select(AuditLog).where(
            AuditLog.entity_id == case_id,
            AuditLog.action == "AI_THREAT_ASSESSMENT",
        )
    )
    assert audit is not None
    assert audit.user_id == user_id
    db.close()


def test_threat_assessment_with_unowned_case_forbidden(client, auth_headers, second_user_headers, test_db):
    """Reject attempt by User A to link assessment to a case owned by User B with 403 Forbidden."""
    db = test_db()
    user_b = db.scalar(select(User).where(User.email == "other_victim@example.com"))
    assert user_b is not None

    case_b = Case(
        owner_id=user_b.id,
        title="User B Incident",
        description="Confidential case.",
    )
    db.add(case_b)
    db.commit()
    db.refresh(case_b)
    case_b_id = str(case_b.id)
    db.close()

    # User A tries to assess text linking to User B's case
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,  # User A
        json={
            "text": "Pay me 500 dollars.",
            "case_id": case_b_id,
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized to access this case."


def test_threat_assessment_with_nonexistent_case_not_found(client, auth_headers):
    """Return 404 Not Found when referencing a non-existent case ID."""
    random_case_id = str(uuid.uuid4())
    response = client.post(
        "/api/v1/ai/threat-assessment",
        headers=auth_headers,
        json={
            "text": "Pay me 500 dollars.",
            "case_id": random_case_id,
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found."


# ============================================================================
# 5. PRIVACY & APPLICATION SECURITY TESTS
# ============================================================================

def test_threat_assessment_does_not_log_victim_text(client, auth_headers, caplog):
    """Verify raw victim evidence text NEVER appears in server application logs."""
    sensitive_evidence = "SECRET_VICTIM_INTIMATE_EVIDENCE_PAY_ME_NOW_OR_LEAK"

    with caplog.at_level(logging.INFO):
        response = client.post(
            "/api/v1/ai/threat-assessment",
            headers=auth_headers,
            json={"text": sensitive_evidence},
        )
        assert response.status_code == 200

    # Ensure sensitive string is strictly absent from all logged messages
    for record in caplog.records:
        assert sensitive_evidence not in record.message
        assert sensitive_evidence not in str(record.args)

    # Verify telemetry metadata was logged
    log_messages = [r.message for r in caplog.records]
    assert any("Threat assessment completed" in msg for msg in log_messages)


# ============================================================================
# 6. NEURAL BEHAVIOR & FALLBACK TESTS
# ============================================================================

def test_baseline_mode_does_not_load_toxic_bert(client, auth_headers):
    """Confirm default backend configuration uses DevelopmentThreatAnalyzer with no Toxic-BERT."""
    analyzer = get_threat_analyzer()
    assert isinstance(analyzer, DevelopmentThreatAnalyzer)
    assert not isinstance(analyzer, ToxicBertThreatAnalyzer)
    assert analyzer.config.enable_toxicity_check is False


class MockFailingToxicBertRunner(ToxicBertModelRunner):
    """Mock runner simulating neural inference crash."""
    def __init__(self):
        super().__init__(model_name="mock/failing-bert", device="cpu", lazy_load=True)

    def load(self) -> bool:
        return True

    def predict(self, text: str) -> Optional[ToxicBertResult]:
        raise RuntimeError("Simulated neural inference GPU/CPU failure.")


def test_neural_mode_fallback_on_failure(client, auth_headers):
    """Verify that if neural model inference fails, response safely falls back to baseline."""
    # Configure hybrid analyzer with mock failing runner
    try:
        failing_runner = MockFailingToxicBertRunner()
        hybrid_analyzer = ToxicBertThreatAnalyzer(model_runner=failing_runner)
        set_threat_analyzer(hybrid_analyzer)

        text = "Send me 500 dollars or I will leak your photos."
        response = client.post(
            "/api/v1/ai/threat-assessment",
            headers=auth_headers,
            json={"text": text},
        )
        assert response.status_code == 200
        data = response.json()

        # Baseline detection is preserved
        assert data["overall_threat_level"] in ("medium", "high", "severe")
        # Fallback explanation is disclosed
        assert "baseline pattern rules" in data["explanation"]
    finally:
        set_threat_analyzer(None)


def test_neural_mode_activation_via_config(client, auth_headers):
    """Verify neural mode is selected and auxiliary inference enabled when AI_ENABLE_TOXIC_BERT=True."""
    orig_flag = settings.AI_ENABLE_TOXIC_BERT
    try:
        settings.AI_ENABLE_TOXIC_BERT = True
        set_threat_analyzer(None)

        analyzer = get_threat_analyzer()
        assert isinstance(analyzer, ToxicBertThreatAnalyzer)
        assert analyzer.config.enable_toxicity_check is True

        from tests.ai.test_toxic_bert import MockToxicBertModelRunner
        mock_runner = MockToxicBertModelRunner(
            simulated_scores={
                "toxic": 0.85,
                "severe_toxic": 0.1,
                "obscene": 0.05,
                "threat": 0.9,
                "insult": 0.7,
                "identity_hate": 0.01,
            }
        )
        mock_hybrid = ToxicBertThreatAnalyzer(config=analyzer.config, model_runner=mock_runner)
        set_threat_analyzer(mock_hybrid)

        response = client.post(
            "/api/v1/ai/threat-assessment",
            headers=auth_headers,
            json={"text": "I will track you down and hurt you."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "toxic_bert:threat" in data["category_scores"]
        assert data["category_scores"]["toxic_bert:threat"] == 0.9
        assert "Toxic-BERT inference" in data["explanation"]
    finally:
        settings.AI_ENABLE_TOXIC_BERT = orig_flag
        set_threat_analyzer(None)


def test_neural_mode_disabled_explanation(client, auth_headers):
    """Verify that if ToxicBertThreatAnalyzer has enable_toxicity_check=False, neural check is bypassed."""
    orig_flag = settings.AI_ENABLE_TOXIC_BERT
    try:
        settings.AI_ENABLE_TOXIC_BERT = False
        set_threat_analyzer(None)

        analyzer = get_threat_analyzer()
        assert isinstance(analyzer, DevelopmentThreatAnalyzer)
        assert analyzer.config.enable_toxicity_check is False

        disabled_config = ThreatAnalyzerConfig(enable_toxicity_check=False)
        from tests.ai.test_toxic_bert import MockToxicBertModelRunner
        mock_runner = MockToxicBertModelRunner()
        disabled_hybrid = ToxicBertThreatAnalyzer(config=disabled_config, model_runner=mock_runner)
        set_threat_analyzer(disabled_hybrid)

        response = client.post(
            "/api/v1/ai/threat-assessment",
            headers=auth_headers,
            json={"text": "I will track you down and hurt you."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "disabled by configuration" in data["explanation"]
        assert mock_runner.call_count == 0
    finally:
        settings.AI_ENABLE_TOXIC_BERT = orig_flag
        set_threat_analyzer(None)
