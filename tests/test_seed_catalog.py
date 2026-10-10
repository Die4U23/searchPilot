from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load():
    path = Path("scripts/seed_catalog.py")
    spec = importlib.util.spec_from_file_location("seed_catalog", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_queries_normalizes_and_rejects_unknown_type(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "queries.csv"
    path.write_text(
        "query_id,query_text,query_type,split\nq1,  Hello   World  ,exact_entity,train\n",
        encoding="utf-8",
    )
    rows = module.read_queries(path)
    assert rows == [("q1", "  Hello   World  ", "hello world", "exact_entity", "train")]

    path.write_text(
        "query_id,query_text,query_type,split\nq1,Hello,blog,train\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="query_type"):
        module.read_queries(path)


def test_read_labels_keeps_annotator_and_rejects_bad_grade(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "labels.csv"
    path.write_text(
        "query_id,item_id,grade,annotator\nq1,N1,2,grok-a\n",
        encoding="utf-8",
    )
    assert module.read_labels(path, {"q1"}) == [("q1", "N1", 2, "grok-a")]

    path.write_text(
        "query_id,item_id,grade,annotator\nq1,N1,4,grok-a\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside 0..3"):
        module.read_labels(path, {"q1"})


def test_collect_model_versions_skips_missing_files(tmp_path: Path) -> None:
    module = _load()
    artifact = tmp_path / "artifacts"
    meta = artifact / "search" / "bm25"
    meta.mkdir(parents=True)
    (meta / "meta.json").write_text(
        json.dumps({"model_version": "bm25-aaaaaaaa", "data_version": "d1"}),
        encoding="utf-8",
    )
    rows = module.collect_model_versions(artifact, tmp_path / "data")
    assert [row[0] for row in rows] == ["bm25-aaaaaaaa"]
    assert rows[0][1] == "bm25"
    assert len(rows[0][4]) == 64
