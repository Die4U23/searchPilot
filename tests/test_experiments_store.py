from __future__ import annotations

import pytest

from searchpilot.experiments.store import (
    ExperimentRecord,
    InMemoryExperimentStore,
    MetricPoint,
    compare_experiments,
)


def _record(source: str, protocol: str, value: float = 0.5) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=f"exp-{source}-{protocol}",
        kind="search",
        config={"seed": 20261009},
        data_version="d3a904f41240",
        protocol_version=protocol,
        source=source,
        source_ref={"repo": "searchpilot", "commit": "abc", "sha256": "deadbeef"},
        metrics=(MetricPoint(name="ndcg@10", split="test", value=value, source=source),),
    )


def test_unknown_source_rejected() -> None:
    store = InMemoryExperimentStore()
    with pytest.raises(ValueError, match="unknown source"):
        store.put(_record("blog", "v1"))


def test_cross_source_is_not_comparable() -> None:
    result = compare_experiments(_record("searchpilot", "v1"), _record("evorec", "v1"), ["ndcg@10"])
    assert result["comparable"] is False
    assert result["reason"] == "source or protocol_version differs"


def test_same_source_reports_values() -> None:
    result = compare_experiments(
        _record("searchpilot", "v1", 0.2),
        _record("searchpilot", "v1", 0.4),
        ["ndcg@10", "missing"],
    )
    assert result["comparable"] is True
    assert result["left"]["ndcg@10"] == 0.2
    assert result["right"]["missing"] is None


def test_registry_roundtrip(tmp_path) -> None:
    store = InMemoryExperimentStore()
    store.put(_record("searchpilot", "v1"))
    path = tmp_path / "registry.json"
    store.save(path)
    loaded = InMemoryExperimentStore.load(path)
    got = loaded.get("exp-searchpilot-v1")
    assert got is not None
    assert got.metrics[0].value == 0.5
    assert got.source_ref is not None
    assert got.source_ref["sha256"] == "deadbeef"
