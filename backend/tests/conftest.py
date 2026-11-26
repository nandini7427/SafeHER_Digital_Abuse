
import pytest

from backend.app.security.rate_limiter import rate_limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    # Reset request counts before each test.
    rate_limiter.reset()

    yield

    # Clean up after each test.
    rate_limiter.reset()
