
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "SafeHER Digital Abuse Intelligence Platform"
    API_V1_PREFIX: str = "/api/v1"

    # Database configuration
    DATABASE_URL: str = "sqlite:///./safeher_dev.db"

    # Authentication configuration
    JWT_SECRET_KEY: str = "development-only-change-this-secret"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Evidence upload configuration
    MAX_UPLOAD_SIZE_MB: int = 50
    PRIVATE_STORAGE_PATH: str = "private_evidence"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
