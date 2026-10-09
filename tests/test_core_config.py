from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from searchpilot.config import Settings, clear_settings_cache, get_settings


def test_defaults() -> None:
    s = Settings()
    assert s.database_url is None
    assert s.data_dir == Path("./data")
    assert s.artifact_dir == Path("./artifacts")
    assert s.data_version is None
    assert s.max_query_chars == 256
    assert s.max_limit == 100
    assert s.log_level == "INFO"
    assert (s.db_pool_min, s.db_pool_max) == (1, 4)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCHPILOT_DATABASE_URL", "postgresql://u:p@h:5432/d")
    monkeypatch.setenv("SEARCHPILOT_DATA_DIR", "/srv/data")
    monkeypatch.setenv("SEARCHPILOT_ARTIFACT_DIR", "/srv/artifacts")
    monkeypatch.setenv("SEARCHPILOT_DATA_VERSION", "abc123def456")
    monkeypatch.setenv("SEARCHPILOT_MAX_QUERY_CHARS", "64")
    monkeypatch.setenv("SEARCHPILOT_MAX_LIMIT", "20")
    monkeypatch.setenv("SEARCHPILOT_LOG_LEVEL", "debug")
    monkeypatch.setenv("SEARCHPILOT_DB_POOL_MIN", "2")
    monkeypatch.setenv("SEARCHPILOT_DB_POOL_MAX", "8")
    s = Settings()
    assert s.database_url == "postgresql://u:p@h:5432/d"
    assert s.data_dir == Path("/srv/data")
    assert s.artifact_dir == Path("/srv/artifacts")
    assert s.data_version == "abc123def456"
    assert s.max_query_chars == 64
    assert s.max_limit == 20
    assert s.log_level == "DEBUG"
    assert (s.db_pool_min, s.db_pool_max) == (2, 8)


def test_blank_optional_values_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCHPILOT_DATABASE_URL", "")
    monkeypatch.setenv("SEARCHPILOT_DATA_VERSION", "  ")
    s = Settings()
    assert s.database_url is None
    assert s.data_version is None


def test_unprefixed_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_LIMIT", "5")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    s = Settings()
    assert s.max_limit == 100
    assert s.database_url is None


def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCHPILOT_MAX_LIMIT", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached_and_clearable(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("SEARCHPILOT_MAX_LIMIT", "7")
    assert get_settings().max_limit == 100  # 仍是缓存值
    clear_settings_cache()
    assert get_settings().max_limit == 7
    get_settings.cache_clear()
    assert get_settings() is not first
