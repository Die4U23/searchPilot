"""LTR 特征、pairwise 训练与 mode=ltr。不下载嵌入模型。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from searchpilot.ports import Document
from searchpilot.search.artifacts import bm25_artifact_dir, save_index
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.ltr import (
    FEATURE_NAMES,
    LtrModel,
    QuerySignals,
    assemble_ltr_model,
    feature_matrix,
    fit_standardizer,
    freshness_days,
    labels_for_queries,
    load_ltr,
    mode_category,
    pairwise_indices,
    save_ltr,
    train_click_counts,
    train_pairwise,
)
from searchpilot.search.service import (
    BM25SearchService,
    InMemorySearchService,
    build_search_service,
)
from searchpilot.search.vector_index import (
    VectorIndex,
    l2_normalize_rows,
    make_vector_meta,
    save_vector_index,
)

DOCS = [
    Document("BOTH", "alpha sports", "alpha sports recap", "sports", "nba"),
    Document("BM25", "alpha zzzzz", "alpha keyword only", "sports", "nba"),
    Document("VEC", "unrelated title", "no shared keyword", "news", "world"),
    Document("OTHER", "alpha weather", "alpha forecast", "weather", "local"),
]


def _vector_index() -> VectorIndex:
    matrix = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.2, 0.8]],
        dtype=np.float32,
    )
    normalized = l2_normalize_rows(matrix)
    meta = make_vector_meta(
        matrix=normalized,
        model_name="test-encoder",
        data_version="test",
        input_sha256={"items.parquet": "abc"},
        query_prefix="",
    )
    return VectorIndex(
        tuple(doc.item_id for doc in DOCS),
        tuple(doc.category for doc in DOCS),
        normalized,
        meta,
    )


def _encode(text: str) -> np.ndarray:
    del text
    return np.array([1.0, 0.0], dtype=np.float32)


def _service() -> InMemorySearchService:
    service = InMemorySearchService(DOCS)
    service.attach_vector_index(_vector_index(), _encode)
    return service


def _model(
    weights: list[float] | None = None,
    *,
    clicks: dict[str, int] | None = None,
    title_chars: dict[str, int] | None = None,
    first_seen_at: dict[str, datetime] | None = None,
    reference_time: datetime | None = None,
) -> LtrModel:
    values = [0.0] * len(FEATURE_NAMES) if weights is None else weights
    return LtrModel(
        weights=tuple(values),
        bias=0.0,
        mean=(0.0,) * len(FEATURE_NAMES),
        std=(1.0,) * len(FEATURE_NAMES),
        reference_time=reference_time,
        clicks=clicks or {},
        title_chars=title_chars or {},
        first_seen_at=first_seen_at or {},
        model_version="ltr-handmade",
    )


def test_feature_names_are_fixed() -> None:
    assert FEATURE_NAMES == (
        "bm25_score",
        "vector_score",
        "bm25_rank",
        "vector_rank",
        "log1p_train_clicks",
        "freshness_days",
        "category_match",
        "title_chars",
    )


def test_feature_row_ranks_freshness_clicks_and_category() -> None:
    reference = datetime(2019, 11, 11, tzinfo=UTC)
    signals = QuerySignals(
        bm25_score={"A": 3.5},
        bm25_rank={"A": 2},
        vector_score={"A": 0.25},
        vector_rank={},
        category_mode="sports",
    )
    rows = feature_matrix(
        ["A", "B"],
        signals,
        categories={"A": "Sports", "B": "news"},
        clicks={"A": 3},
        title_chars={"A": 12, "B": 4},
        first_seen_at={"A": datetime(2019, 11, 9, tzinfo=UTC)},
        reference_time=reference,
    )
    assert rows[0, 0] == pytest.approx(3.5)
    assert rows[0, 1] == pytest.approx(0.25)
    assert rows[0, 2] == pytest.approx(2)
    assert rows[0, 3] == pytest.approx(101)
    assert rows[0, 4] == pytest.approx(np.log1p(3))
    assert rows[0, 5] == pytest.approx(2.0)
    assert rows[0, 6] == pytest.approx(1.0)
    assert rows[0, 7] == pytest.approx(12)
    assert rows[1].tolist() == pytest.approx([0.0, 0.0, 101.0, 101.0, 0.0, 0.0, 0.0, 4.0])


def test_category_mode_tie_breaks_by_earliest() -> None:
    assert mode_category(["sports", "news", "sports", "news"]) == "sports"
    assert mode_category(["news", "sports", "sports"]) == "sports"
    assert mode_category([]) is None


def test_freshness_days_uses_reference_time() -> None:
    reference = datetime(2019, 11, 11, tzinfo=UTC)
    assert freshness_days(None, reference) == 0.0
    assert freshness_days(datetime(2019, 11, 9, tzinfo=UTC), None) == 0.0
    assert freshness_days(datetime(2019, 11, 9, tzinfo=UTC), reference) == pytest.approx(2.0)
    assert freshness_days(datetime(2019, 11, 12, tzinfo=UTC), reference) == 0.0


def test_train_clicks_ignore_dev_and_unclicked() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "C", "A"],
            "clicked": [1, 0, 1, 1, 1],
            "split": ["train", "train", "train", "dev", "dev"],
        }
    )
    assert train_click_counts(frame) == {"A": 1, "B": 1}


def test_labels_for_queries_drops_other_splits() -> None:
    labels = {"q_train": {"N1": 3}, "q_test": {"N2": 1}}
    assert labels_for_queries(labels, {"q_train"}) == {"q_train": {"N1": 3}}


def test_pairs_stay_inside_query_and_skip_equal_grades() -> None:
    higher, lower = pairwise_indices([3, 0, 0, 3], ["a", "a", "b", "b"])
    assert set(zip(higher.tolist(), lower.tolist(), strict=True)) == {(0, 1), (3, 2)}
    higher, lower = pairwise_indices([1, 1], ["q", "q"])
    assert higher.size == 0
    assert lower.size == 0
    higher, lower = pairwise_indices([0, 2, 2], ["q", "q", "q"])
    assert set(zip(higher.tolist(), lower.tolist(), strict=True)) == {(1, 0), (2, 0)}


def test_constant_feature_std_is_one() -> None:
    raw = np.ones((4, len(FEATURE_NAMES)), dtype=np.float64)
    normalized, _mean, std = fit_standardizer(raw)
    assert np.allclose(std, 1.0)
    assert np.allclose(normalized, 0.0)


def test_pairwise_training_scores_higher_grade_above() -> None:
    raw = np.array(
        [
            [0.0, 0.0, 101.0, 101.0, 0.0, 0.0, 0.0, 5.0],
            [8.0, 0.2, 1.0, 3.0, 1.0, 0.0, 1.0, 12.0],
        ],
        dtype=np.float64,
    )
    normalized, mean, std = fit_standardizer(raw)
    higher, lower = pairwise_indices([0, 3], ["q", "q"])
    weights, bias, loss_first, loss_last = train_pairwise(normalized, higher, lower)
    model = assemble_ltr_model(
        weights,
        bias,
        mean,
        std,
        reference_time=None,
        clicks={},
        title_chars={},
        first_seen_at={},
    )
    scores = model.score_rows(raw)
    assert scores[1] > scores[0]
    assert loss_last <= loss_first
    assert model.model_version.startswith("ltr-")
    assert len(model.model_version) == 12


def test_save_load_roundtrip_and_rejects_tampered_weights(tmp_path: Path) -> None:
    raw = np.array(
        [
            [1.0, 0.2, 4.0, 8.0, 0.0, 1.0, 1.0, 10.0],
            [0.5, 0.1, 9.0, 2.0, 1.5, 3.0, 0.0, 20.0],
        ],
        dtype=np.float64,
    )
    _normalized, mean, std = fit_standardizer(raw)
    reference = datetime(2019, 11, 11, tzinfo=UTC)
    model = assemble_ltr_model(
        np.linspace(-0.2, 0.3, len(FEATURE_NAMES)),
        0.1,
        mean,
        std,
        reference_time=reference,
        clicks={"A": 2, "B": 0},
        title_chars={"A": 5, "B": 1},
        first_seen_at={"A": datetime(2019, 11, 9, tzinfo=UTC)},
    )
    assert model.clicks == {"A": 2}
    save_ltr(tmp_path, model, stats={"pair_count": 1})
    loaded = load_ltr(tmp_path)
    assert loaded.model_version == model.model_version
    assert np.allclose(loaded.score_rows(raw), model.score_rows(raw))
    assert loaded.reference_time == reference

    weights_path = tmp_path / "weights.json"
    document = json.loads(weights_path.read_text(encoding="utf-8"))
    document["weights"][0] = float(document["weights"][0]) + 1.0
    weights_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SearchNotReadyError):
        load_ltr(tmp_path)


def test_load_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(SearchNotReadyError):
        load_ltr(tmp_path)


def test_ltr_vector_weight_reranks_ahead_of_keywords() -> None:
    service = _service()
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("vector_score")] = 1.0
    service.attach_ltr(_model(weights))
    result = service.search("alpha", 4, "ltr")
    assert result.mode == "ltr"
    assert result.model_version == "ltr-handmade"
    assert [hit.channel for hit in result.hits] == ["ltr"] * 4
    assert [hit.item_id for hit in result.hits] == ["BOTH", "VEC", "OTHER", "BM25"]
    assert [hit.rank for hit in result.hits] == [1, 2, 3, 4]


def test_ltr_clicks_and_freshness_are_read_from_the_artifact() -> None:
    service = _service()
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("log1p_train_clicks")] = 1.0
    service.attach_ltr(_model(weights, clicks={"BOTH": 10, "VEC": 4}))
    clicked = service.search("alpha", 4, "ltr")
    assert [hit.item_id for hit in clicked.hits] == ["BOTH", "VEC", "BM25", "OTHER"]

    fresh_weights = [0.0] * len(FEATURE_NAMES)
    fresh_weights[FEATURE_NAMES.index("freshness_days")] = -1.0
    reference = datetime(2019, 11, 11, tzinfo=UTC)
    service.attach_ltr(
        _model(
            fresh_weights,
            reference_time=reference,
            first_seen_at={"BOTH": datetime(2019, 11, 1, tzinfo=UTC)},
        )
    )
    aged = service.search("alpha", 4, "ltr")
    assert aged.hits[-1].item_id == "BOTH"


def test_ltr_category_match_prefers_bm25_mode() -> None:
    service = _service()
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("category_match")] = 1.0
    service.attach_ltr(_model(weights))
    result = service.search("alpha", 4, "ltr")
    assert [hit.item_id for hit in result.hits] == ["BM25", "BOTH", "OTHER", "VEC"]


def test_ltr_category_filter_and_limit_and_empty_query() -> None:
    service = _service()
    service.attach_ltr(_model())
    assert [hit.item_id for hit in service.search("alpha", 10, "ltr").hits] == [
        "BM25",
        "BOTH",
        "OTHER",
        "VEC",
    ]
    trimmed = service.search("alpha", 2, "ltr")
    assert [hit.item_id for hit in trimmed.hits] == ["BM25", "BOTH"]
    assert [hit.rank for hit in trimmed.hits] == [1, 2]
    filtered = service.search_with_filters("alpha", 10, "ltr", filters={"category": "news"})
    assert [hit.item_id for hit in filtered.hits] == ["VEC"]
    empty = service.search("", 5, "ltr")
    assert empty.hits == ()
    assert empty.mode == "ltr"
    assert empty.normalized_query == ""


def test_ltr_ready_flags_and_missing_artifact_raises() -> None:
    bare = InMemorySearchService(DOCS)
    assert bare.vector_ready is False
    assert bare.ltr_ready is False
    bare.attach_ltr(_model())
    assert bare.ltr_ready is True
    with pytest.raises(SearchNotReadyError):
        bare.search("alpha", 3, "ltr")

    service = _service()
    assert service.vector_ready is True
    assert service.ltr_ready is False
    with pytest.raises(SearchNotReadyError):
        service.search("alpha", 3, "ltr")
    service.attach_ltr(_model())
    assert service.ltr_ready is True
    assert service.search("alpha", 1, "ltr").hits


def test_build_search_service_loads_ltr(tmp_path: Path) -> None:
    memory = InMemorySearchService(DOCS)
    save_index(memory.index, memory.meta, bm25_artifact_dir(tmp_path))
    save_vector_index(tmp_path / "search" / "vector", _vector_index())
    loaded_vector = build_search_service(tmp_path)
    assert isinstance(loaded_vector, BM25SearchService)
    assert loaded_vector.vector_ready is True
    assert loaded_vector.ltr_ready is False

    model = assemble_ltr_model(
        [0.0] * len(FEATURE_NAMES),
        0.0,
        [0.0] * len(FEATURE_NAMES),
        [1.0] * len(FEATURE_NAMES),
        reference_time=None,
        clicks={},
        title_chars={"BOTH": 12},
        first_seen_at={},
    )
    save_ltr(tmp_path / "search" / "ltr", model)
    loaded = build_search_service(tmp_path)
    assert isinstance(loaded, BM25SearchService)
    assert loaded.vector_ready is True
    assert loaded.ltr_ready is True
    assert loaded._ltr is not None
    assert loaded._ltr.model_version == model.model_version
