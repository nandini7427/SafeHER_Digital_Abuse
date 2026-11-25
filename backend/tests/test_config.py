import pytest
from pydantic import ValidationError

from backend.app.config import KNOWN_PLACEHOLDER_SECRETS, Settings

VALID_32_CHAR_SECRET = "a-secure-and-random-32-char-secret!"
VALID_STAGING_SECRET = "staging-secure-secret-key-for-test-32-chars"
VALID_PROD_SECRET = "production-secure-secret-key-at-least-32-chars"
VALID_PROD_DB = "postgresql://user:pass@db.internal:5432/safeher_prod"


class TestDevelopmentEnvironment:
    """Verifies development environment behavior and ephemeral key generation."""

    def test_development_generates_ephemeral_jwt_secret_when_blank(self):
        settings = Settings(_env_file=None, ENVIRONMENT="development", JWT_SECRET_KEY="")
        assert len(settings.JWT_SECRET_KEY) >= 32

    def test_development_generates_ephemeral_jwt_secret_when_whitespace(self):
        settings = Settings(_env_file=None, ENVIRONMENT="development", JWT_SECRET_KEY="   ")
        assert len(settings.JWT_SECRET_KEY) >= 32

    def test_development_ephemeral_secrets_are_unique_per_process(self):
        s1 = Settings(_env_file=None, ENVIRONMENT="development", JWT_SECRET_KEY="")
        s2 = Settings(_env_file=None, ENVIRONMENT="development", JWT_SECRET_KEY="")
        assert s1.JWT_SECRET_KEY != s2.JWT_SECRET_KEY

    def test_development_preserves_explicitly_configured_jwt_secret(self):
        custom_secret = "dev-custom-secret-key"
        settings = Settings(_env_file=None, ENVIRONMENT="development", JWT_SECRET_KEY=custom_secret)
        assert settings.JWT_SECRET_KEY == custom_secret

    def test_development_normalizes_secret_with_surrounding_whitespace(self):
        settings = Settings(
            _env_file=None,
            ENVIRONMENT="development",
            JWT_SECRET_KEY="   dev-custom-secret-with-spaces   ",
        )
        assert settings.JWT_SECRET_KEY == "dev-custom-secret-with-spaces"


class TestTestEnvironment:
    """Verifies test environment behavior and backwards compatibility."""

    def test_test_environment_generates_ephemeral_jwt_secret_when_blank(self):
        settings = Settings(_env_file=None, ENVIRONMENT="test", JWT_SECRET_KEY="")
        assert len(settings.JWT_SECRET_KEY) >= 32

    def test_test_environment_preserves_explicitly_configured_jwt_secret(self):
        custom_secret = "test-custom-secret-key"
        settings = Settings(_env_file=None, ENVIRONMENT="test", JWT_SECRET_KEY=custom_secret)
        assert settings.JWT_SECRET_KEY == custom_secret


class TestStagingEnvironment:
    """Verifies staging environment enforces strict security validation."""

    def test_staging_rejects_missing_or_blank_jwt_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, ENVIRONMENT="staging", JWT_SECRET_KEY="")
        assert "Staging requires JWT_SECRET_KEY with at least 32 characters" in str(exc_info.value)

    def test_staging_rejects_whitespace_jwt_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, ENVIRONMENT="staging", JWT_SECRET_KEY=" " * 35)
        assert "Staging requires JWT_SECRET_KEY with at least 32 characters" in str(exc_info.value)

    def test_staging_rejects_short_jwt_secret(self):
        short_secret = "short-staging-secret-key"  # 24 chars < 32
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, ENVIRONMENT="staging", JWT_SECRET_KEY=short_secret)
        assert "Staging requires JWT_SECRET_KEY with at least 32 characters" in str(exc_info.value)

    def test_staging_rejects_31_char_boundary_jwt_secret(self):
        secret_31 = "a" * 31
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, ENVIRONMENT="staging", JWT_SECRET_KEY=secret_31)
        assert "Staging requires JWT_SECRET_KEY with at least 32 characters" in str(exc_info.value)

    @pytest.mark.parametrize("placeholder", sorted(KNOWN_PLACEHOLDER_SECRETS))
    def test_staging_rejects_known_placeholder_secrets(self, placeholder):
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, ENVIRONMENT="staging", JWT_SECRET_KEY=placeholder)
        assert "Staging JWT_SECRET_KEY must not be a placeholder or default secret" in str(exc_info.value)

    def test_staging_rejects_placeholder_with_mixed_case_and_whitespace(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, ENVIRONMENT="staging", JWT_SECRET_KEY="  StAgInG-oNlY-cHaNgE-tHiS-sEcReT  ")
        assert "Staging JWT_SECRET_KEY must not be a placeholder or default secret" in str(exc_info.value)

    def test_staging_normalizes_valid_secret_with_surrounding_whitespace(self):
        settings = Settings(
            _env_file=None,
            ENVIRONMENT="staging",
            JWT_SECRET_KEY=f"   {VALID_STAGING_SECRET}   ",
        )
        assert settings.ENVIRONMENT == "staging"
        assert settings.JWT_SECRET_KEY == VALID_STAGING_SECRET

    def test_staging_accepts_valid_secret_containing_placeholder_substring(self):
        legitimate_key = "legitimate-high-entropy-secret-key-32-chars-long"
        settings = Settings(
            _env_file=None,
            ENVIRONMENT="staging",
            JWT_SECRET_KEY=legitimate_key,
        )
        assert settings.JWT_SECRET_KEY == legitimate_key

    def test_staging_accepts_valid_configured_secret(self):
        settings = Settings(
            _env_file=None,
            ENVIRONMENT="staging",
            JWT_SECRET_KEY=VALID_STAGING_SECRET,
        )
        assert settings.ENVIRONMENT == "staging"
        assert settings.JWT_SECRET_KEY == VALID_STAGING_SECRET


class TestProductionEnvironment:
    """Verifies production environment enforces strict security, database, and debug safeguards."""

    def test_production_rejects_missing_or_blank_jwt_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                _env_file=None,
                ENVIRONMENT="production",
                JWT_SECRET_KEY="",
                DATABASE_URL=VALID_PROD_DB,
                DEBUG=False,
            )
        assert "Production requires JWT_SECRET_KEY with at least 32 characters" in str(exc_info.value)

    def test_production_rejects_whitespace_jwt_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                _env_file=None,
                ENVIRONMENT="production",
                JWT_SECRET_KEY=" " * 40,
                DATABASE_URL=VALID_PROD_DB,
                DEBUG=False,
            )
        assert "Production requires JWT_SECRET_KEY with at least 32 characters" in str(exc_info.value)

    def test_production_rejects_short_jwt_secret(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                _env_file=None,
                ENVIRONMENT="production",
                JWT_SECRET_KEY="short-production-key",
                DATABASE_URL=VALID_PROD_DB,
                DEBUG=False,
            )
        assert "Production requires JWT_SECRET_KEY with at least 32 characters" in str(exc_info.value)

    @pytest.mark.parametrize("placeholder", sorted(KNOWN_PLACEHOLDER_SECRETS))
    def test_production_rejects_known_placeholder_secrets(self, placeholder):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                _env_file=None,
                ENVIRONMENT="production",
                JWT_SECRET_KEY=placeholder,
                DATABASE_URL=VALID_PROD_DB,
                DEBUG=False,
            )
        assert "Production JWT_SECRET_KEY must not be a placeholder or default secret" in str(exc_info.value)

    def test_production_preserves_sqlite_database_safeguard(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                _env_file=None,
                ENVIRONMENT="production",
                JWT_SECRET_KEY=VALID_PROD_SECRET,
                DATABASE_URL="sqlite:///./prod.db",
                DEBUG=False,
            )
        assert "Production must use a managed database; SQLite is development-only" in str(exc_info.value)

    def test_production_preserves_debug_safeguard(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                _env_file=None,
                ENVIRONMENT="production",
                JWT_SECRET_KEY=VALID_PROD_SECRET,
                DATABASE_URL=VALID_PROD_DB,
                DEBUG=True,
            )
        assert "DEBUG must be disabled in production" in str(exc_info.value)

    def test_production_accepts_valid_configuration(self):
        settings = Settings(
            _env_file=None,
            ENVIRONMENT="production",
            JWT_SECRET_KEY=VALID_PROD_SECRET,
            DATABASE_URL=VALID_PROD_DB,
            DEBUG=False,
        )
        assert settings.ENVIRONMENT == "production"
        assert settings.JWT_SECRET_KEY == VALID_PROD_SECRET
        assert settings.DEBUG is False

    def test_production_normalizes_valid_secret_with_surrounding_whitespace(self):
        settings = Settings(
            _env_file=None,
            ENVIRONMENT="production",
            JWT_SECRET_KEY=f"   {VALID_PROD_SECRET}   ",
            DATABASE_URL=VALID_PROD_DB,
            DEBUG=False,
        )
        assert settings.ENVIRONMENT == "production"
        assert settings.JWT_SECRET_KEY == VALID_PROD_SECRET
