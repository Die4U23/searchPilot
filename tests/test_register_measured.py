from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path("scripts/register_measured.py")
    spec = importlib.util.spec_from_file_location("register_measured", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_register_measured_writes_json_without_database(tmp_path: Path) -> None:
    module = _load_script()
    out = tmp_path / "registry.json"

    assert module.main(["--out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    ids = {item["experiment_id"] for item in payload}
    assert {"sp-bm25", "sp-vector", "sp-hybrid", "sp-ltr", "sp-ctr", "evorec-itemcf"} <= ids


def test_register_measured_requires_database_url_only_when_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    monkeypatch.setenv("SEARCHPILOT_DATABASE_URL", "postgresql://unused")
    out = tmp_path / "registry.json"

    assert module.main(["--out", str(out)]) == 0
    assert out.is_file()
