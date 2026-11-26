import pytest
from fastapi import HTTPException

from backend.app.security.rate_limiter import RateLimiter


def test_login_allows_five_requests_and_blocks_sixth():
    limiter = RateLimiter()

    for _ in range(5):
        limiter.check("test-login", limit=5, window_seconds=60)

    with pytest.raises(HTTPException) as exc:
        limiter.check("test-login", limit=5, window_seconds=60)

    assert exc.value.status_code == 429


def test_registration_allows_three_requests_and_blocks_fourth():
    limiter = RateLimiter()

    for _ in range(3):
        limiter.check("test-register", limit=3, window_seconds=3600)

    with pytest.raises(HTTPException) as exc:
        limiter.check("test-register", limit=3, window_seconds=3600)

    assert exc.value.status_code == 429