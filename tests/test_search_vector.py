"""向量索引与 vector / hybrid 模式。不下载嵌入模型。"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import numpy as np
import pytest

from searchpilot.ports import Document
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.service import InMemorySearchService
from searchpilot.search.vector_index import (
    SentenceTransformerQueryEncoder,
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


def test_concurrent_cold_queries_construct_one_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    start = Barrier(8, timeout=5)
    counts_lock = Lock()
    constructions: list[str] = []
    inputs: list[str] = []

    class FakeModel:
        def __init__(self, name: str) -> None:
            with counts_lock:
                constructions.append(name)
            # Let the other workers reach the cold encoder while construction is in flight.
            time.sleep(0.05)

        def encode(self, text: str, **kwargs: object) -> list[float]:
            with counts_lock:
                inputs.append(text)
            return [1.0, 0.0]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel)
    )
    encoder = SentenceTransformerQueryEncoder("fixture", "prefix: ")

    def query(i: int) -> np.ndarray:
        start.wait()
        return encoder(str(i))

    with ThreadPoolExecutor(max_workers=8) as pool:
        vectors = list(pool.map(query, range(8)))
    assert constructions == ["fixture"]
    assert sorted(inputs) == [f"prefix: {i}" for i in range(8)]
    for vector in vectors:
        np.testing.assert_array_equal(vector, np.array([1.0, 0.0], dtype=np.float32))


def test_failed_encoder_initialization_can_recover(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    class FakeModel:
        def encode(self, text: str, **kwargs: object) -> list[float]:
            return [1.0, 0.0]

    def construct(name: str) -> FakeModel:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("fixture load failure")
        return FakeModel()

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=construct)
    )
    encoder = SentenceTransformerQueryEncoder("fixture", "")
    with pytest.raises(RuntimeError, match="fixture load failure"):
        encoder("first")
    np.testing.assert_array_equal(encoder("second"), [1.0, 0.0])
    encoder("third")
    assert attempts == 2


def _reference_vector_search(
    index: VectorIndex, query: np.ndarray, limit: int, category: str | None
) -> list[tuple[str, float]]:
    """Independent full-sort oracle: no heap or vectorized candidate-selection path."""
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0 or not index.item_ids:
        return []
    scores = index.matrix @ (vector / norm).astype(np.float32)
    candidates = [
        (item_id, float(scores[row]))
        for row, item_id in enumerate(index.item_ids)
        if np.isfinite(scores[row])
        and (category is None or index.categories[row].casefold() == category.casefold())
    ]
    return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:limit]


def _selection_index(matrix: np.ndarray, categories: tuple[str, ...]) -> VectorIndex:
    # IDs run opposite to row order so stable row order cannot impersonate item_id tie-breaking.
    ids = tuple(f"N{row:04}" for row in reversed(range(len(categories))))
    meta = make_vector_meta(
        matrix=matrix, model_name="fixture", data_version="selection", input_sha256={}
    )
    return VectorIndex(ids, categories, matrix, meta)


@pytest.mark.parametrize("limit", [1, 2, 10])
@pytest.mark.parametrize("category", [None, "NEWS", "sports", "missing"])
def test_candidate_selection_preserves_finite_scores_and_filters(limit, category) -> None:
    matrix = np.array(
        [[0.5, 1], [np.nan, 1], [np.inf, 1], [-np.inf, 1], [-0.5, 1], [0.5, 1]],
        dtype=np.float32,
    )
    index = _selection_index(matrix, ("news", "news", "news", "news", "sports", "News"))
    query = np.array([1, 0], dtype=np.float32)
    assert index.search(query, limit, category=category) == _reference_vector_search(
        index, query, limit, category
    )


@pytest.mark.parametrize("seed", [0, 42, 20261014])
def test_candidate_selection_matches_full_sort_on_seeded_inputs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for size in (0, 1, 257):
        matrix = l2_normalize_rows(rng.integers(-3, 4, size=(size, 8)).astype(np.float32))
        # Duplicate complete rows create ties; varying categories exercise post-score filtering.
        if size > 1:
            matrix[1:9] = matrix[0]
        categories = tuple("news" if row % 3 else "sports" for row in range(size))
        index = _selection_index(matrix, categories)
        queries = [rng.normal(size=8).astype(np.float32), np.zeros(8, dtype=np.float32)]
        for query in queries:
            for limit in (1, 10, 300):
                for category in (None, "NEWS", "missing"):
                    assert index.search(
                        query, limit, category=category
                    ) == _reference_vector_search(index, query, limit, category)


def test_candidate_selection_all_nonfinite_returns_empty() -> None:
    matrix = np.array([[np.nan, 1], [np.inf, 1], [-np.inf, 1]], dtype=np.float32)
    index = _selection_index(matrix, ("news",) * 3)
    assert index.search(np.array([1, 0], dtype=np.float32), 100) == []


def test_candidate_selection_many_cutoff_ties_uses_item_id() -> None:
    index = _selection_index(np.ones((257, 2), dtype=np.float32), ("news",) * 257)
    result = index.search(np.array([1, 0], dtype=np.float32), 100)
    assert result == [(f"N{row:04}", 1.0) for row in range(100)]
