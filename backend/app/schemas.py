from datetime import datetime
from enum import Enum
import json
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRegister(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()

        if "@" not in value or "." not in value.rsplit("@", 1)[-1]:
            raise ValueError("Enter a valid email address.")

        return value


class UserLogin(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime




class CaseCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(default=None, max_length=5000)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 3:
            raise ValueError("Case title must contain at least 3 non-whitespace characters.")
        return cleaned

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class CaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_id: UUID
    title: str
    description: Optional[str] = None
    status: str
    created_at: datetime


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    case_id: UUID
    owner_id: UUID
    original_filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    uploaded_at: datetime


class EvidenceIntegrityResponse(BaseModel):
    evidence_id: UUID
    sha256: str
    is_valid: bool
    verified_at: datetime
    message: str


class ThreatAssessmentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw text of the suspect communication to assess for threats and extortion.",
    )
    case_id: UUID | None = Field(
        default=None,
        description="Optional existing case ID to link this threat assessment with.",
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Input text cannot be empty or whitespace only.")
        return value


MANDATORY_HUMAN_REVIEW_DISCLAIMER: str = (
    "SafeHER human reviews represent independent expert or advocate observations for triage and "
    "support purposes only. They do not constitute formal legal findings, judicial evidence certifications, "
    "or definitive forensic guarantees. In legal proceedings, independent accredited forensic discovery "
    "and chain of custody must be established according to local rules of evidence."
)


class HumanReviewStatus(str, Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class HumanReviewVerdict(str, Enum):
    CONCURS_WITH_AI = "concurs_with_ai"
    DISAGREES_SUSPICIOUS_MANIPULATION = "disagrees_suspicious_manipulation"
    DISAGREES_NO_OBVIOUS_ARTIFACTS = "disagrees_no_obvious_artifacts"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class HumanReviewRequestCreate(BaseModel):
    evidence_id: UUID
    case_id: UUID
    ai_assessment_snapshot: Dict[str, Any]
    request_reason: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="User context, uncertainty concerns, or specific regions requesting human inspection.",
    )


class HumanReviewSubmit(BaseModel):
    human_verdict: HumanReviewVerdict
    reviewer_notes: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="Detailed visual inspection rationale, artifact examination notes, and observations.",
    )
    disagreement_rationale: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Mandatory or recommended reasoning when human verdict contradicts AI assessment.",
    )

    @field_validator("reviewer_notes")
    @classmethod
    def validate_reviewer_notes(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Reviewer notes cannot be empty or whitespace only.")
        return v.strip()


class HumanReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    evidence_id: UUID
    case_id: UUID
    requester_id: UUID
    reviewer_id: Optional[UUID] = None
    status: str
    ai_assessment_snapshot: Dict[str, Any]
    request_reason: Optional[str] = None
    human_verdict: Optional[str] = None
    reviewer_notes: Optional[str] = None
    disagreement_rationale: Optional[str] = None
    is_disagreement_with_ai: bool = False
    requested_at: datetime
    reviewed_at: Optional[datetime] = None
    disclaimer: str = MANDATORY_HUMAN_REVIEW_DISCLAIMER

    @field_validator("ai_assessment_snapshot", mode="before")
    @classmethod
    def parse_ai_snapshot(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {"raw": v}
        if isinstance(v, dict):
            return v
        return {}


# Neural inference schema re-exports for backend consumer transparency
from ai_core.neural.integration.schemas import (
    IntegratedForensicAssessment,
    NeuralInferenceMetadata,
)

__all__ = [
    "UserRegister",
    "UserLogin",
    "TokenResponse",
    "UserResponse",
    "CaseCreate",
    "CaseResponse",
    "EvidenceResponse",
    "EvidenceIntegrityResponse",
    "ThreatAssessmentRequest",
    "HumanReviewStatus",
    "HumanReviewVerdict",
    "HumanReviewRequestCreate",
    "HumanReviewSubmit",
    "HumanReviewResponse",
    "MANDATORY_HUMAN_REVIEW_DISCLAIMER",
    "IntegratedForensicAssessment",
    "NeuralInferenceMetadata",
]
