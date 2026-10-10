"""统一错误信封：{"error": {"code", "message", "retryable", "request_id"}}。"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from searchpilot.observability import get_request_id, new_request_id
from searchpilot.ports import IdempotencyConflictError

logger = logging.getLogger(__name__)

INVALID_INPUT = "INVALID_INPUT"
ITEM_NOT_FOUND = "ITEM_NOT_FOUND"
IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
UNAUTHORIZED = "UNAUTHORIZED"
FORBIDDEN = "FORBIDDEN"
NOT_READY = "NOT_READY"
INTERNAL = "INTERNAL"

_MAX_REPORTED_VALIDATION_ERRORS = 5


class ApiError(Exception):
    """可预期的业务错误，携带信封所需的全部字段。"""

    def __init__(self, code: str, message: str, status_code: int, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


def is_not_ready_error(exc: BaseException) -> bool:
    """按类名识别 SearchNotReadyError / RecommendNotReadyError 等，避免 import 其他工作流的模块。"""
    return any(cls.__name__.endswith("NotReadyError") for cls in type(exc).__mro__)


def _validation_message(exc: RequestValidationError) -> str:
    parts: list[str] = []
    for item in exc.errors()[:_MAX_REPORTED_VALIDATION_ERRORS]:
        path = ".".join(str(p) for p in item.get("loc", ()))
        msg = str(item.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{path}: {msg}" if path else msg)
    extra = len(exc.errors()) - _MAX_REPORTED_VALIDATION_ERRORS
    if extra > 0:
        parts.append(f"(+{extra} more)")
    return "; ".join(parts) or "invalid request"


def _http_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase.lower()
    except ValueError:
        return ""


def exception_to_error(exc: Exception) -> ApiError:
    """把任意异常映射为 ApiError；未知异常只记日志，响应里不带细节。"""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, RequestValidationError):
        return ApiError(INVALID_INPUT, _validation_message(exc), 422)
    if isinstance(exc, IdempotencyConflictError):
        return ApiError(
            IDEMPOTENCY_CONFLICT, "idempotency key was already used with different content", 409
        )
    if isinstance(exc, StarletteHTTPException):
        phrase = _http_phrase(exc.status_code)
        if exc.status_code >= 500:
            return ApiError(INTERNAL, phrase or "internal server error", exc.status_code)
        return ApiError(INVALID_INPUT, phrase or "bad request", exc.status_code)
    if is_not_ready_error(exc):
        logger.warning("dependency not ready: %s", type(exc).__name__)
        return ApiError(NOT_READY, "service is not ready", 503, retryable=True)
    logger.error("unhandled exception", exc_info=exc)
    return ApiError(INTERNAL, "internal server error", 500)


def error_envelope(error: ApiError, request_id: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
            "request_id": request_id or get_request_id() or new_request_id(),
        }
    }


def error_response(error: ApiError, extra: dict[str, Any] | None = None) -> JSONResponse:
    content = error_envelope(error)
    if extra:
        content.update(extra)
    return JSONResponse(status_code=error.status_code, content=content)


async def _handle(request: Request, exc: Exception) -> JSONResponse:
    return error_response(exception_to_error(exc))


def register_exception_handlers(app: FastAPI) -> None:
    for exc_type in (
        ApiError,
        RequestValidationError,
        StarletteHTTPException,
        IdempotencyConflictError,
        Exception,
    ):
        app.add_exception_handler(exc_type, _handle)
