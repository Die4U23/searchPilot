from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from searchpilot import errors
from searchpilot.bootstrap import create_app
from searchpilot.errors import ApiError, exception_to_error
from searchpilot.ports import IdempotencyConflictError


def test_error_code_constants() -> None:
    assert errors.INVALID_INPUT == "INVALID_INPUT"
    assert errors.ITEM_NOT_FOUND == "ITEM_NOT_FOUND"
    assert errors.IDEMPOTENCY_CONFLICT == "IDEMPOTENCY_CONFLICT"
    assert errors.NOT_READY == "NOT_READY"
    assert errors.INTERNAL == "INTERNAL"


def test_api_error_fields() -> None:
    err = ApiError("NOT_READY", "later", 503, retryable=True)
    assert (err.code, err.message, err.status_code, err.retryable) == (
        "NOT_READY",
        "later",
        503,
        True,
    )
    assert ApiError("INVALID_INPUT", "bad", 422).retryable is False


def test_exception_to_error_mapping() -> None:
    conflict = exception_to_error(IdempotencyConflictError("k"))
    assert (conflict.code, conflict.status_code, conflict.retryable) == (
        "IDEMPOTENCY_CONFLICT",
        409,
        False,
    )

    class RecommendNotReadyError(Exception):
        pass

    not_ready = exception_to_error(RecommendNotReadyError("/secret/path"))
    assert (not_ready.code, not_ready.status_code, not_ready.retryable) == ("NOT_READY", 503, True)
    assert "secret" not in not_ready.message

    internal = exception_to_error(ValueError("SELECT 1"))
    assert (internal.code, internal.status_code, internal.retryable) == ("INTERNAL", 500, False)
    assert "SELECT" not in internal.message


def test_exception_to_error_passes_api_error_through() -> None:
    err = ApiError("ITEM_NOT_FOUND", "gone", 404)
    assert exception_to_error(err) is err


def _app_raising(exc: Exception) -> FastAPI:
    app = create_app()

    @app.get("/boom")
    def boom() -> None:
        raise exc

    return app


def test_handler_api_error_envelope(fakes: SimpleNamespace) -> None:
    client = TestClient(_app_raising(ApiError("ITEM_NOT_FOUND", "item does not exist", 404)))
    response = client.get("/boom")
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "ITEM_NOT_FOUND",
            "message": "item does not exist",
            "retryable": False,
            "request_id": response.headers["X-Request-Id"],
        }
    }


def test_handler_idempotency_conflict_409(fakes: SimpleNamespace) -> None:
    response = TestClient(_app_raising(IdempotencyConflictError("k"))).get("/boom")
    assert response.status_code == 409
    fakes.assert_error_envelope(response.json(), "IDEMPOTENCY_CONFLICT")


def test_handler_uncaught_exception_500_with_request_id(fakes: SimpleNamespace) -> None:
    client = TestClient(
        _app_raising(RuntimeError("password=hunter2")), raise_server_exceptions=False
    )
    response = client.get("/boom", headers={"X-Request-Id": "req_" + "b" * 32})
    assert response.status_code == 500
    error = fakes.assert_error_envelope(response.json(), "INTERNAL", retryable=False)
    assert error["request_id"] == "req_" + "b" * 32
    assert response.headers["X-Request-Id"] == "req_" + "b" * 32
    assert "hunter2" not in response.text


def test_handler_uncaught_exception_also_reraised_free_for_default_client(
    fakes: SimpleNamespace,
) -> None:
    # 默认 TestClient(raise_server_exceptions=True) 也不应看到异常：中间件已兜底成 500 信封
    response = TestClient(_app_raising(RuntimeError("x"))).get("/boom")
    assert response.status_code == 500
    fakes.assert_error_envelope(response.json(), "INTERNAL")


def test_registered_exception_handler_builds_envelope() -> None:
    # 兜底的 Exception handler（ServerErrorMiddleware 路径）同样输出信封
    response = asyncio.run(errors._handle(None, RuntimeError("x")))  # type: ignore[arg-type]
    assert response.status_code == 500
    body = json.loads(bytes(response.body))
    assert set(body["error"]) == {"code", "message", "retryable", "request_id"}


def test_validation_error_rewritten_to_envelope(
    make_client: Callable[..., TestClient], fakes: SimpleNamespace
) -> None:
    response = make_client(search=fakes.FakeSearch()).post(
        "/search", json={"query": 123, "limit": "many"}
    )
    assert response.status_code == 422
    assert "detail" not in response.json()
    error = fakes.assert_error_envelope(response.json(), "INVALID_INPUT", retryable=False)
    assert "body.query" in error["message"]
    assert "body.limit" in error["message"]


def test_validation_error_does_not_echo_input(
    make_client: Callable[..., TestClient], fakes: SimpleNamespace
) -> None:
    response = make_client(feedback=fakes.FakeFeedbackStore()).post(
        "/feedback", json=fakes.feedback_payload(item_id="secret token!")
    )
    assert response.status_code == 422
    assert "secret token" not in response.text


def test_method_not_allowed_envelope(
    make_client: Callable[..., TestClient], fakes: SimpleNamespace
) -> None:
    response = make_client().get("/search")
    assert response.status_code == 405
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT")


@pytest.mark.parametrize("count", [8])
def test_validation_message_is_truncated(count: int) -> None:
    from fastapi.exceptions import RequestValidationError

    exc = RequestValidationError(
        [{"loc": ("body", f"f{i}"), "msg": "bad", "type": "x"} for i in range(count)]
    )
    err = exception_to_error(exc)
    assert err.status_code == 422
    assert "(+3 more)" in err.message
