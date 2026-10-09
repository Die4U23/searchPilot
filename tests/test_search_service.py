"""InMemorySearchService / build_search_service 的契约行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from searchpilot.ports import Document, SearchPort
from searchpilot.search.artifacts import bm25_artifact_dir, save_index
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.service import InMemorySearchService, build_search_service

DOCS = [
    Document("B", "Bravo bulletin", "A bravo story about sports", "sports", "nba"),
    Document("A", "Alpha bravo", "Alpha meets bravo in sports", "sports", "nba"),
    Document("E", "Alpha orbit", "Alpha satellite orbit science notes", "science", "space"),
    Document("C", "Alpha forecast", "Alpha weather forecast today", "weather", "local"),
    Document("S", "Alpha science", "Another alpha science briefing", "science", "space"),
    Document("D", "Banana bread", "Easy banana bread recipe", "food", "baking"),
]


def _service() -> InMemorySearchService:
    return InMemorySearchService(DOCS)


def test_in_memory_service_satisfies_search_port() -> None:
    service = _service()
    port: SearchPort = service
    result = port.search("alpha", 10, "bm25")
    assert result.mode == "bm25"
    assert result.model_version == port.model_version
    assert port.model_version.startswith("bm25-")


def test_bm25_mode_rank_channel_and_normalized_query() -> None:
    service = _service()
    result = service.search("\u3000The   ALPHA  ", 10, "bm25")
    assert result.normalized_query == "the alpha"
    assert result.mode == "bm25"
    assert result.hits
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))
    assert {hit.channel for hit in result.hits} == {"bm25"}


def test_unimplemented_modes_raise() -> None:
    service = _service()
    with pytest.raises(SearchNotReadyError):
        service.search("alpha", 5, "vector")
    with pytest.raises(SearchNotReadyError):
        service.search("alpha", 5, "hybrid")
    with pytest.raises(SearchNotReadyError):
        service.search("alpha", 5, "ltr")


def test_empty_and_stopword_queries_return_no_hits() -> None:
    service = _service()
    empty = service.search("", 5, "bm25")
    assert empty.hits == ()
    assert empty.normalized_query == ""
    stopwords = service.search("the and of", 5, "bm25")
    assert stopwords.hits == ()
    assert stopwords.normalized_query == "the and of"


def test_limit_truncates_preserving_order() -> None:
    service = _service()
    full = service.search("alpha", 10, "bm25")
    assert len(full.hits) >= 3
    trimmed = service.search("alpha", 2, "bm25")
    assert [hit.item_id for hit in trimmed.hits] == [hit.item_id for hit in full.hits[:2]]
    assert [hit.score for hit in trimmed.hits] == [hit.score for hit in full.hits[:2]]
    assert [hit.rank for hit in trimmed.hits] == [1, 2]


def test_category_filter_reranks_contiguously() -> None:
    service = _service()
    filtered = service.search_with_filters("alpha", 10, "bm25", filters={"category": "science"})
    assert {hit.item_id for hit in filtered.hits} == {"E", "S"}
    assert [hit.rank for hit in filtered.hits] == [1, 2]
    unfiltered = service.search("alpha", 10, "bm25")
    assert "C" in {hit.item_id for hit in unfiltered.hits}
    assert "C" not in {hit.item_id for hit in filtered.hits}
    assert len(unfiltered.hits) > len(filtered.hits)


def test_build_search_service_matches_in_memory(tmp_path: Path) -> None:
    memory = _service()
    save_index(memory.index, memory.meta, bm25_artifact_dir(tmp_path))
    loaded = build_search_service(tmp_path)
    assert loaded.model_version == memory.model_version
    for query, limit in (("alpha", 10), ("alpha", 2), ("", 5), ("the and of", 5), ("bravo", 3)):
        assert loaded.search(query, limit, "bm25") == memory.search(query, limit, "bm25")


def test_build_search_service_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(SearchNotReadyError):
        build_search_service(tmp_path)
