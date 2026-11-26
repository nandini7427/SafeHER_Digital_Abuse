
import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status


class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int):
        current_time = time.monotonic()
        window_start = current_time - window_seconds

        with self.lock:
            # Remove expired request timestamps
            valid_requests = [
                timestamp
                for timestamp in self.requests[key]
                if timestamp > window_start
            ]

            self.requests[key] = valid_requests

            if len(valid_requests) >= limit:
                retry_after = max(
                    1,
                    int(valid_requests[0] + window_seconds - current_time) + 1,
                )

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

            self.requests[key].append(current_time)

    def reset(self):
        """Clear all stored request counts, primarily for automated tests."""
        with self.lock:
            self.requests.clear()


rate_limiter = RateLimiter()


def rate_limit(limit: int, window_seconds: int):
    def dependency(request: Request):
        client_host = (
            request.client.host if request.client else "unknown"
        )

        key = f"{client_host}:{request.url.path}"

        rate_limiter.check(
            key=key,
            limit=limit,
            window_seconds=window_seconds,
        )

    return dependency
