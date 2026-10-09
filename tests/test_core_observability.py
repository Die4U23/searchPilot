from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from searchpilot.observability import (
    JsonFormatter,
    bind_request_id,
    configure_logging,
    get_request_id,
    is_valid_request_id,
    new_request_id,
    reset_request_id,
)


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("searchpilot.access")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield stream
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def test_new_request_id_format() -> None:
    rid = new_request_id()
    assert is_valid_request_id(rid)
    assert rid != new_request_id()
    assert not is_valid_request_id("req_" + "g" * 32)


def test_contextvar_bind_and_reset() -> None:
    assert get_request_id() is None
    token = bind_request_id("req_" + "1" * 32)
    assert get_request_id() == "req_" + "1" * 32
    reset_request_id(token)
    assert get_request_id() is None


def test_json_formatter_fields_and_extras() -> None:
    token = bind_request_id("req_" + "2" * 32)
    try:
        record = logging.LogRecord("x.y", logging.WARNING, __file__, 1, "hello %s", ("世界",), None)
        record.status = 200
        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_id(token)
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "x.y"
    assert payload["msg"] == "hello 世界"
    assert payload["request_id"] == "req_" + "2" * 32
    assert payload["status"] == 200
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_without_request_id_and_with_exception() -> None:
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        record = logging.LogRecord("l", logging.ERROR, __file__, 1, "oops", (), sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] is None
    assert "ValueError: bad" in payload["exc"]


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    old_level = root.level
    try:
        configure_logging("INFO")
        configure_logging("DEBUG")
        configure_logging("INFO")
        installed = [h for h in root.handlers if h not in before]
        assert len(installed) <= 1
        assert sum(isinstance(h.formatter, JsonFormatter) for h in root.handlers) == 1
        assert root.level == logging.INFO
    finally:
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
        root.setLevel(old_level)


def test_access_log_contains_request_id_and_fields(
    make_client: Callable[..., TestClient], log_stream: io.StringIO
) -> None:
    rid = "req_" + "c" * 32
    response = make_client().get("/health/live", headers={"X-Request-Id": rid})
    assert response.status_code == 200
    lines = [json.loads(line) for line in log_stream.getvalue().splitlines()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["request_id"] == rid
    assert entry["method"] == "GET"
    assert entry["path"] == "/health/live"
    assert entry["status"] == 200
    assert isinstance(entry["duration_ms"], float)
    assert entry["msg"] == "request"


def test_access_log_has_generated_request_id_and_error_status(
    make_client: Callable[..., TestClient], log_stream: io.StringIO, fakes: SimpleNamespace
) -> None:
    response = make_client().post("/search", json={"query": "x"})
    assert response.status_code == 503
    entry = json.loads(log_stream.getvalue().splitlines()[-1])
    assert entry["request_id"] == response.headers["X-Request-Id"]
    assert entry["status"] == 503
    assert entry["path"] == "/search"


def test_access_log_does_not_include_query_string(
    make_client: Callable[..., TestClient], log_stream: io.StringIO
) -> None:
    make_client().get("/health/live?token=secret")
    assert "secret" not in log_stream.getvalue()
