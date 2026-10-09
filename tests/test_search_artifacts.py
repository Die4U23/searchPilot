"""索引产物的落盘、版本号与损坏检测。"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchpilot.ports import Document
from searchpilot.search.artifacts import (
    INDEX_FILENAME,
    META_FILENAME,
    load_index,
    make_meta,
    save_index,
)
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.inverted_index import InvertedIndex

DOCS = [
    Document("N2", "Celtics trade rumors", "Boston Celtics eye a trade", "sports", "nba"),
    Document("N1", "Lakers beat Celtics", "Los Angeles Lakers win in Boston", "sports", "nba"),
    Document("N3", "Rain in Boston", "Heavy rain expected overnight", "weather", "local"),
]

_VERSION_RE = re.compile(r"bm25-[0-9a-f]{8}\Z")


def _index() -> InvertedIndex:
    return InvertedIndex.build_from_documents(DOCS)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    index = _index()
    meta = make_meta(
        index,
        data_version="dv-test",
        input_sha256={"items.parquet": "abc123"},
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    directory = tmp_path / "search" / "bm25"
    save_index(index, meta, directory)
    loaded, loaded_meta = load_index(directory)

    assert loaded.item_ids == index.item_ids
    assert loaded.categories == index.categories
    assert set(loaded.fields) == set(index.fields)
    for name in index.fields:
        assert loaded.fields[name].postings == index.fields[name].postings
        assert loaded.fields[name].doc_lengths == index.fields[name].doc_lengths
    assert loaded_meta == meta


def test_model_version_stable_for_same_input() -> None:
    first_index = _index()
    second_index = _index()
    first = make_meta(
        first_index,
        data_version="dv-test",
        input_sha256={"items.parquet": "abc123"},
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = make_meta(
        second_index,
        data_version="dv-test",
        input_sha256={"items.parquet": "abc123"},
        built_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert first.built_at != second.built_at
    assert first.model_version == second.model_version
    assert _VERSION_RE.fullmatch(first.model_version)


def test_meta_json_has_contract_fields(tmp_path: Path) -> None:
    index = _index()
    meta = make_meta(
        index,
        data_version="dv-test",
        input_sha256={"items.parquet": "deadbeef"},
        built_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    save_index(index, meta, tmp_path)
    payload = json.loads((tmp_path / META_FILENAME).read_text(encoding="utf-8"))
    required = {
        "model_version",
        "k1",
        "b",
        "data_version",
        "doc_count",
        "vocab_size",
        "built_at",
        "input_sha256",
    }
    assert required <= set(payload)
    assert payload["model_version"] == meta.model_version
    assert payload["k1"] == pytest.approx(1.2)
    assert payload["b"] == pytest.approx(0.75)
    assert payload["data_version"] == "dv-test"
    assert payload["doc_count"] == index.doc_count == 3
    assert payload["vocab_size"] == index.vocab_size
    assert payload["built_at"]
    assert payload["input_sha256"] == {"items.parquet": "deadbeef"}


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(SearchNotReadyError):
        load_index(tmp_path / "does-not-exist")


def test_missing_index_json_raises(tmp_path: Path) -> None:
    index = _index()
    meta = make_meta(
        index,
        data_version="dv-test",
        input_sha256={"items.parquet": "abc123"},
    )
    save_index(index, meta, tmp_path)
    (tmp_path / INDEX_FILENAME).unlink()
    with pytest.raises(SearchNotReadyError, match="index.json"):
        load_index(tmp_path)


def test_tampered_doc_count_raises(tmp_path: Path) -> None:
    index = _index()
    meta = make_meta(
        index,
        data_version="dv-test",
        input_sha256={"items.parquet": "abc123"},
    )
    save_index(index, meta, tmp_path)
    meta_path = tmp_path / META_FILENAME
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["doc_count"] = index.doc_count + 1
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SearchNotReadyError, match="documents"):
        load_index(tmp_path)
