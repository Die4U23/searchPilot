from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from searchpilot.recommend.service import (
    InMemoryRecommendService,
    RecommendNotReadyError,
    build_recommend_service,
)


def test_cold_start_uses_popular() -> None:
    svc = InMemoryRecommendService(
        {"u1": ("A", "B")},
        model_version="dv1",
    )
    result = svc.recommend("unknown", limit=5, exclude=frozenset())
    assert result.channel == "popular"
    assert result.cold_start is True
    assert result.model_version == "popular-dv1"


def test_itemcf_path_when_history_and_neighbors() -> None:
    histories = {
        "u1": ("A", "B"),
        "u2": ("A", "C"),
        "u3": ("B", "D"),
    }
    svc = InMemoryRecommendService(histories, model_version="dv1")
    result = svc.recommend("u1", limit=5, exclude=frozenset())
    assert result.channel == "itemcf"
    assert result.cold_start is False
    assert result.model_version == "itemcf-dv1"
    assert len(result.hits) >= 1


def test_recommend_no_duplicate_and_excludes() -> None:
    histories = {"u1": ("A", "B"), "u2": ("A", "C")}
    svc = InMemoryRecommendService(histories, model_version="dv1")
    result = svc.recommend(
        "u1",
        limit=10,
        exclude=frozenset({"C"}),
    )
    ids = [h.item_id for h in result.hits]
    assert len(ids) == len(set(ids))
    assert "A" not in ids
    assert "B" not in ids
    assert "C" not in ids
    assert all(h.rank == i for i, h in enumerate(result.hits, start=1))


def _write_snapshot(base: Path, data_version: str = "abc123") -> Path:
    root = base / "data" / "processed" / data_version
    root.mkdir(parents=True)
    impressions = pd.DataFrame(
        {
            "impression_id": ["i1", "i2", "i3"],
            "user_id": ["u1", "u1", "u2"],
            "shown_at": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"], utc=True),
            "item_id": ["A", "B", "A"],
            "clicked": [1, 1, 1],
            "split": ["train", "train", "train"],
            "source": ["mind"] * 3,
        }
    )
    impressions.to_parquet(root / "impressions.parquet")
    table = pa.table(
        {
            "user_id": ["u1", "u2"],
            "history": [["A", "B"], ["A"]],
        }
    )
    pq.write_table(table, root / "user_history.parquet")
    return base / "data"


def test_build_recommend_service_from_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _write_snapshot(tmp_path)
    monkeypatch.delenv("SEARCHPILOT_DATA_VERSION", raising=False)
    svc = build_recommend_service(data_dir)
    result = svc.recommend("u2", limit=3, exclude=frozenset())
    assert result.hits
    assert result.model_version.endswith("abc123")


def test_build_recommend_missing_processed(tmp_path: Path) -> None:
    with pytest.raises(RecommendNotReadyError):
        build_recommend_service(tmp_path / "data")


def test_build_recommend_multiple_versions_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_snapshot(tmp_path, "v1")

    root = tmp_path / "data" / "processed" / "v2"
    root.mkdir()
    pd.DataFrame(
        {
            "impression_id": ["x"],
            "user_id": ["u"],
            "shown_at": [datetime(2020, 1, 1, tzinfo=UTC)],
            "item_id": ["A"],
            "clicked": [1],
            "split": ["train"],
            "source": ["mind"],
        }
    ).to_parquet(root / "impressions.parquet")
    pq.write_table(
        pa.table({"user_id": ["u"], "history": [["A"]]}),
        root / "user_history.parquet",
    )
    monkeypatch.delenv("SEARCHPILOT_DATA_VERSION", raising=False)
    with pytest.raises(RecommendNotReadyError, match="multiple"):
        build_recommend_service(tmp_path / "data")
