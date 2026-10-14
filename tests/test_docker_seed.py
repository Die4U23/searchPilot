"""空表才导入，已有数据的表不再重写。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "docker_seed.py"
    name = "docker_seed"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_steps_skips_tables_that_already_have_rows() -> None:
    script = _script()
    assert script.seed_steps({"items": 0, "catalog": 0, "experiments": 0}) == (
        "items",
        "catalog",
        "experiments",
    )
    assert script.seed_steps({"items": 65238, "catalog": 100, "experiments": 6}) == ()
    assert script.seed_steps({"items": 0, "catalog": 100, "experiments": 6}) == ("items",)


def test_seed_connection_stays_out_of_subprocess_command(monkeypatch) -> None:
    script = _script()
    database_url = "postgresql://fixture:private-password@localhost/test"
    monkeypatch.setenv("SEARCHPILOT_DATABASE_URL", "postgresql://wrong-database")
    calls = []

    def fail(command, *, check, env):
        calls.append((command, check, env))
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(script.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        script._run(["scripts/seed_items.py", "items.parquet"], database_url)
    assert database_url not in str(error.value)
    assert "private-password" not in str(error.value)
    command, check, environment = calls[0]
    assert command == [sys.executable, "scripts/seed_items.py", "items.parquet"]
    assert check is True
    assert environment["SEARCHPILOT_DATABASE_URL"] == database_url
    assert environment["SEARCHPILOT_DB_STATEMENT_TIMEOUT_MS"] == "120000"


@pytest.mark.parametrize("present", [False, True])
def test_table_presence_checks_use_bounded_query(monkeypatch, present) -> None:
    script = _script()
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (present,)
    monkeypatch.setattr("psycopg.connect", lambda url: connection)
    assert script.table_has_rows("postgresql://fixture", "items") is present
    cursor.execute.assert_called_once_with("SELECT EXISTS (SELECT 1 FROM items)")
