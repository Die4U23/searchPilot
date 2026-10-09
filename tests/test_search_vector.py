"""向量索引与 vector / hybrid 模式。不下载嵌入模型。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from searchpilot.ports import Document
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.service import InMemorySearchService
from searchpilot.search.vector_index import (
    VectorIndex,
    l2_normalize_rows,
    load_vector_index,
    make_vector_meta,
    save_vector_index,
)

DOCS = [
    Document("BOTH", "alpha sports", "alpha sports recap", "sports", "nba"),
    Document("BM25", "alpha zzzzz", "alpha keyword only", "sports", "nba"),
    Document("VEC", "unrelated title", "no shared keyword", "news", "world"),
    Document("OTHER", "alpha weather", "alpha forecast", "weather", "local"),
]


def _index() -> VectorIndex:
    # BOTH 与 VEC 靠近查询向量 [1, 0]；BM25 正交；OTHER 略偏。
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    meta = make_vector_meta(
        matrix=l2_normalize_rows(matrix),
        model_name="test-encoder",
        data_version="test",
        input_sha256={"items.parquet": "abc"},
        query_prefix="",
    )
    categories = tuple(doc.category for doc in DOCS)
    item_ids = tuple(doc.item_id for doc in DOCS)
    normalized = l2_normalize_rows(matrix)
    return VectorIndex(item_ids, categories, normalized, meta)


def _encode(text: str) -> np.ndarray:
    del text
    return np.array([1.0, 0.0], dtype=np.float32)


def _service() -> InMemorySearchService:
    service = InMemorySearchService(DOCS)
    service.attach_vector_index(_index(), _encode)
    return service


def test_vector_ranks_by_cosine_and_breaks_ties_by_item_id() -> None:
    service = _service()
    result = service.search("alpha", 2, "vector")
    assert result.mode == "vector"
    assert result.model_version.startswith("vector-")
    assert [hit.channel for hit in result.hits] == ["vector", "vector"]
    assert result.hits[0].item_id == "BOTH"
    assert result.hits[0].rank == 1
    assert result.hits[1].item_id == "VEC"


def test_vector_category_filter() -> None:
    service = _service()
    result = service.search_with_filters("alpha", 10, "vector", filters={"category": "news"})
    assert [hit.item_id for hit in result.hits] == ["VEC"]


def test_hybrid_puts_dual_match_first() -> None:
    service = _service()
    result = service.search("alpha", 4, "hybrid")
    assert result.mode == "hybrid"
    assert result.model_version.startswith("hybrid-")
    assert {hit.channel for hit in result.hits} == {"rrf"}
    assert result.hits[0].item_id == "BOTH"
    assert {"BM25", "VEC"} <= {hit.item_id for hit in result.hits}


def test_missing_vector_index_still_raises() -> None:
    service = InMemorySearchService(DOCS)
    with pytest.raises(SearchNotReadyError):
        service.search("alpha", 5, "vector")
    with pytest.raises(SearchNotReadyError):
        service.search("alpha", 5, "hybrid")
    with pytest.raises(SearchNotReadyError):
        service.search("alpha", 5, "ltr")


def test_vector_roundtrip(tmp_path: Path) -> None:
    index = _index()
    save_vector_index(tmp_path, index)
    loaded = load_vector_index(tmp_path)
    assert loaded.model_version == index.model_version
    assert loaded.search(np.array([1.0, 0.0]), 1)[0][0] == "BOTH"


def test_vector_tie_break_uses_item_id_at_cutoff() -> None:
    matrix = np.ones((3, 2), dtype=np.float32)
    meta = make_vector_meta(
        matrix=l2_normalize_rows(matrix),
        model_name="test-encoder",
        data_version="ties",
        input_sha256={},
        query_prefix="",
    )
    index = VectorIndex(("C", "A", "B"), ("news", "news", "news"), l2_normalize_rows(matrix), meta)
    ranked = index.search(np.array([1.0, 0.0], dtype=np.float32), 2)
    assert [item_id for item_id, _score in ranked] == ["A", "B"]


def test_vector_dim_mismatch_raises() -> None:
    index = _index()
    with pytest.raises(ValueError, match="query dim"):
        index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 2)


def test_vector_hash_mismatch_raises(tmp_path: Path) -> None:
    index = _index()
    save_vector_index(tmp_path, index)
    meta_path = tmp_path / "meta.json"
    text = meta_path.read_text(encoding="utf-8").replace(index.meta.embeddings_sha256, "0" * 64)
    meta_path.write_text(text, encoding="utf-8")
    with pytest.raises(SearchNotReadyError):
        load_vector_index(tmp_path)
