
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------
# Authentication Schemas
# --------------------------------------------------

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


# --------------------------------------------------
# Case Management Schemas
# --------------------------------------------------

class CaseCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Case title cannot be empty.")

        return value


class CaseUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    status: str | None = Field(default=None, min_length=1, max_length=30)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError("Case title cannot be empty.")

        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError("Case status cannot be empty.")

        return value


class CaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_id: UUID
    title: str
    description: str | None
    status: str
    created_at: datetime

# --------------------------------------------------
# Consent Management Schemas
# --------------------------------------------------

class ConsentCreate(BaseModel):
    consent_version: str = Field(min_length=1, max_length=50)
    purpose: str = Field(min_length=3, max_length=255)

    @field_validator("consent_version", "purpose")
    @classmethod
    def validate_consent_fields(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("This field cannot be empty.")

        return value


class ConsentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    consent_version: str
    purpose: str
    status: str
    granted_at: datetime
    withdrawn_at: datetime | None
