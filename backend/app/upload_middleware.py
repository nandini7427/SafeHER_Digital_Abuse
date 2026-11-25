"""ASGI upload admission and request-body size guard.

This middleware runs before FastAPI/Starlette multipart parsing, so it can limit
chunked requests and requests without a Content-Length header before they are
spooled to a temporary file.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class UploadAdmissionMiddleware:
    """Bound concurrent image uploads and cap actual request-body bytes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        coordinator: Any = None,
        coordinator_provider: Callable[[], Any] | None = None,
        max_request_body_bytes: int,
        upload_paths: tuple[str, ...] = (
            "/api/v1/ai/image-assessment",
            "/api/v1/ai/image/assess",
        ),
    ) -> None:
        if max_request_body_bytes < 1:
            raise ValueError("max_request_body_bytes must be positive")
        if coordinator is None and coordinator_provider is None:
            raise ValueError("A coordinator or coordinator_provider is required")
        self.app = app
        self.coordinator = coordinator
        self.coordinator_provider = coordinator_provider
        self.max_request_body_bytes = max_request_body_bytes
        self.upload_paths = upload_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not any(path.endswith(suffix) for suffix in self.upload_paths):
            await self.app(scope, receive, send)
            return

        content_lengths = [
            value for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]
        declared_length: int | None = None
        if content_lengths:
            # Duplicate or malformed length headers are rejected rather than
            # interpreted differently by the proxy and application server.
            if len(content_lengths) != 1:
                await self._send_json(send, 400, "Invalid Content-Length header.")
                return
            try:
                declared_length = int(content_lengths[0].decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                await self._send_json(send, 400, "Invalid Content-Length header.")
                return
            if declared_length < 0:
                await self._send_json(send, 400, "Invalid Content-Length header.")
                return
            if declared_length > self.max_request_body_bytes:
                await self._send_json(send, 413, "Image upload request body exceeds the permitted size.")
                return

        coordinator = (
            self.coordinator_provider()
            if self.coordinator_provider is not None
            else self.coordinator
        )
        slot_info = coordinator.acquire_upload_slot()
        if slot_info is None:
            await self._send_json(send, 429, "Concurrent image upload limit reached. Please retry shortly.")
            return

        slot_id, handle = slot_info
        received_bytes = 0
        rejected_for_size = False

        async def limited_receive() -> Message:
            nonlocal received_bytes, rejected_for_size
            # Once the limit is crossed, never consume or forward additional
            # request-body chunks to the multipart parser.
            if rejected_for_size:
                return {"type": "http.disconnect"}
            message = await receive()
            if message.get("type") == "http.request" and not rejected_for_size:
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_request_body_bytes:
                    rejected_for_size = True
                    await self._send_json(
                        send,
                        413,
                        "Image upload request body exceeds the permitted size.",
                    )
                    # Stop the multipart parser without passing the oversized
                    # chunk downstream. The response above is already complete.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if not rejected_for_size:
                await send(message)

        try:
            try:
                await self.app(scope, limited_receive, guarded_send)
            except Exception:
                # Multipart parsers commonly raise a disconnect exception when
                # the receive wrapper terminates an oversized request. The 413
                # response has already been sent; do not turn it into a 500.
                if not rejected_for_size:
                    raise
        finally:
            coordinator.release_upload_slot(slot_id, handle)

    @staticmethod
    async def _send_json(send: Send, status_code: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
