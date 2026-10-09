from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from searchpilot import bootstrap
from searchpilot.config import get_settings


def test_build_default_app_never_fails_when_components_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken_import(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(bootstrap.importlib, "import_module", broken_import)
    with caplog.at_level(logging.WARNING, logger="searchpilot.bootstrap"):
        app = bootstrap.build_default_app()
    assert app.state.search is None
    assert app.state.recommend is None
    assert app.state.feedback is None
    assert any("unavailable" in r.getMessage() for r in caplog.records)

    client = TestClient(app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.post("/search", json={"query": "x"}).status_code == 503
    assert client.post("/recommend", json={"user_id": "u"}).status_code == 503


def test_build_default_app_with_database_url_but_no_db_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARCHPILOT_DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    get_settings.cache_clear()

    def broken_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(bootstrap.importlib, "import_module", broken_import)
    app = bootstrap.build_default_app()
    # 配了数据库却加载失败：反馈接口 503，而不是悄悄退回内存实现
    assert app.state.feedback is None
    assert app.state.probes == ()


def test_build_default_app_with_empty_dirs_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCHPILOT_DATA_DIR", "/nonexistent/searchpilot/data")
    monkeypatch.setenv("SEARCHPILOT_ARTIFACT_DIR", "/nonexistent/searchpilot/artifacts")
    get_settings.cache_clear()
    app = bootstrap.build_default_app()
    client = TestClient(app)
    assert client.get("/health/live").status_code == 200
    # 没有索引产物：搜索不可用，且不能 200 + 空数组
    assert client.post("/search", json={"query": "x"}).status_code == 503


def test_lifespan_runs_closers() -> None:
    app = bootstrap.create_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
    assert closed == [True]


def test_create_app_signature_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        bootstrap.create_app(None)  # type: ignore[misc]
