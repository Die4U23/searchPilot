"""request_id 中间件 + 访问日志 + 兜底 500 信封。

用纯 ASGI 实现，保证错误处理时 request_id 的 contextvar 仍然有效。
"""

from __future__ import annotations

import logging
import time

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from searchpilot.errors import error_response, exception_to_error
from searchpilot.observability import (
    bind_request_id,
    is_valid_request_id,
    new_request_id,
    reset_request_id,
)

REQUEST_ID_HEADER = "X-Request-Id"
access_logger = logging.getLogger("searchpilot.access")


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming if incoming and is_valid_request_id(incoming) else new_request_id()
        token = bind_request_id(request_id)
        scope.setdefault("state", {})["request_id"] = request_id

        started_at = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_header(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            try:
                await self.app(scope, receive, send_with_header)
            except Exception as exc:
                if response_started:
                    raise
                response = error_response(exception_to_error(exc))
                status_code = response.status_code
                await response(scope, receive, send_with_header)
        finally:
            access_logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            reset_request_id(token)
