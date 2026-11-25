import secrets
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Known placeholder secrets that must never be used in staging or production.
KNOWN_PLACEHOLDER_SECRETS: set[str] = {
    "change-me",
    "secret",
    "password",
    "changethis",
    "placeholder",
    "change-this-secret",
    "your-secret-key",
    "your-secret-key-here",
    "replace-with-a-secure",
    "replace-with-a-secure-secret-key",
    "development-only-change-this-secret",
    "staging-only-change-this-secret",
    "production-only-change-this-secret",
    "default-secret-key-do-not-use-in-prod",
    "default-secret-key",
}


class Settings(BaseSettings):
    APP_NAME: str = "SafeHER Digital Abuse Intelligence Platform"
    ENVIRONMENT: Literal["development", "test", "staging", "production"] = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Database configuration. SQLite is for local development only.
    DATABASE_URL: str = "sqlite:///./safeher_dev.db"

    # Authentication configuration. Development and test receive an ephemeral secret if blank;
    # staging and production must provide an explicitly configured secret of at least 32 characters.
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: Literal["HS256"] = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=5, le=1440)

    # Evidence upload and resource bounds configuration.
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, ge=1, le=100)
    MAX_CONCURRENT_UPLOADS: int = Field(default=10, ge=1, le=100)
    CONTAINER_MAX_CONCURRENT_ANALYSES: int = Field(default=2, ge=1, le=32)
    WORKER_MAX_CONCURRENT_ANALYSES: int = Field(default=1, ge=1, le=16)
    IMAGE_TMPFS_LIMIT_MB: int = Field(default=512, ge=64)
    CAPACITY_SLOTS_DIR: str | None = None
    PRIVATE_STORAGE_PATH: str = "private_evidence"

    # AI Threat Assessment configuration.
    AI_ENABLE_TOXIC_BERT: bool = False
    AI_DEVICE: str = "cpu"
    AI_MAX_INPUT_LENGTH: int = Field(default=10000, ge=100, le=100000)
    AI_MODEL_PATH: str | None = None

    # AI Neural Image Intelligence configuration (Milestone 4).
    # Neural inference remains disabled until genuinely trained, approved artifacts exist.
    AI_ENABLE_NEURAL_DETECTOR: bool = False
    AI_NEURAL_MODEL_A_PATH: str | None = None
    AI_NEURAL_MODEL_B_PATH: str | None = None
    AI_NEURAL_MODEL_A_SHA256: str | None = None
    AI_NEURAL_MODEL_B_SHA256: str | None = None
    AI_NEURAL_EXPECTED_VERSION_A: str = "1.0.0"
    AI_NEURAL_EXPECTED_VERSION_B: str = "1.0.0"
    AI_NEURAL_INFERENCE_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0, le=120)
    AI_NEURAL_MAX_BATCH_SIZE: int = Field(default=1, ge=1, le=16)
    AI_NEURAL_ALLOWED_ARTIFACT_DIRS: list[str] = [
        "ai_core/neural/artifacts",
        "models/checkpoints",
    ]

    @model_validator(mode="after")
    def validate_environment_security(self):
        normalized_secret = (self.JWT_SECRET_KEY or "").strip()
        if self.ENVIRONMENT in ("staging", "production"):
            env_name = self.ENVIRONMENT.capitalize()
            if normalized_secret.lower() in KNOWN_PLACEHOLDER_SECRETS:
                raise ValueError(
                    f"{env_name} JWT_SECRET_KEY must not be a placeholder or default secret."
                )
            if len(normalized_secret) < 32:
                raise ValueError(
                    f"{env_name} requires JWT_SECRET_KEY with at least 32 characters."
                )
            self.JWT_SECRET_KEY = normalized_secret
            if self.ENVIRONMENT == "production":
                if self.DATABASE_URL.startswith("sqlite"):
                    raise ValueError(
                        "Production must use a managed database; SQLite is development-only."
                    )
                if self.DEBUG:
                    raise ValueError("DEBUG must be disabled in production.")
        elif self.ENVIRONMENT in ("development", "test"):
            if not normalized_secret:
                # A fresh per-process development or test key avoids a known, static fallback.
                self.JWT_SECRET_KEY = secrets.token_urlsafe(48)
            else:
                self.JWT_SECRET_KEY = normalized_secret
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
