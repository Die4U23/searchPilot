from __future__ import annotations

import pytest

from searchpilot.db.connection import create_pool


@pytest.fixture
def _pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """最小池（0 连接不立即建连），避免测试触碰真实数据库。"""
    monkeypatch.setenv("SEARCHPILOT_DB_POOL_MIN", "0")
    monkeypatch.setenv("SEARCHPILOT_DB_POOL_MAX", "2")
    monkeypatch.setenv("SEARCHPILOT_DB_STATEMENT_TIMEOUT_MS", "5000")


def test_create_pool_preserves_url_options_and_appends_statement_timeout(
    _pool_env: None,
) -> None:
    """URL 中显式的 options（如 search_path）必须原样保留，statement_timeout 追加其后。"""
    url = "postgresql://u:p@localhost:5432/db?options=-c+search_path%3Dtest_schema"
    pool = create_pool(url)
    try:
        conninfo = pool.conninfo
    finally:
        pool.close()
    assert "search_path=test_schema" in conninfo
    assert "statement_timeout=5000" in conninfo


def test_create_pool_does_not_duplicate_explicit_statement_timeout(_pool_env: None) -> None:
    """URL 已显式给出 statement_timeout 时不得再追加，避免重复 -c 选项。"""
    url = "postgresql://u:p@localhost:5432/db?options=-c+statement_timeout%3D1200"
    pool = create_pool(url)
    try:
        conninfo = pool.conninfo
    finally:
        pool.close()
    assert conninfo.count("statement_timeout=1200") == 1
    assert "statement_timeout=5000" not in conninfo


def test_create_pool_rejects_invalid_timeouts(_pool_env: None) -> None:
    import os

    os.environ["SEARCHPILOT_DB_STATEMENT_TIMEOUT_MS"] = "0"
    with pytest.raises(ValueError, match="TIMEOUT"):
        create_pool("postgresql://u:p@localhost:5432/db")
