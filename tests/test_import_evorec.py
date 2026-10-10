from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from searchpilot.experiments.import_evorec import import_evorec_file, import_evorec_result
from searchpilot.experiments.store import (
    ExperimentRecord,
    InMemoryExperimentStore,
    MetricPoint,
    compare_experiments,
)

_PAYLOAD = {
    "experiment_id": "evorec-itemcf-1",
    "kind": "recommend",
    "data_version": "evorec-d1",
    "protocol_version": "evorec-v1",
    "metrics": [
        {"name": "ndcg@10", "split": "test", "value": 0.42, "segment": None},
        {"name": "recall@20", "split": "test", "value": 0.31, "segment": "all"},
    ],
}


def _write_results(path: Path) -> str:
    path.write_text(json.dumps(_PAYLOAD, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import(
    path: Path,
    store: InMemoryExperimentStore,
    expect_sha256: str | None,
) -> ExperimentRecord:
    return import_evorec_result(
        path,
        store,
        repo="Die4U23/EvoRec",
        commit="abc123",
        run_id="run-1",
        expect_sha256=expect_sha256,
    )


def test_matching_sha256_imports(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    digest = _write_results(path)
    store = InMemoryExperimentStore()

    record = _import(path, store, digest)

    assert record.source == "evorec"
    assert record.source_ref == {
        "repo": "Die4U23/EvoRec",
        "commit": "abc123",
        "path": path.as_posix(),
        "sha256": digest,
        "run_id": "run-1",
    }
    loaded = store.get("evorec-itemcf-1")
    assert loaded is record
    assert [metric.name for metric in loaded.metrics] == ["ndcg@10", "recall@20"]
    assert [metric.value for metric in loaded.metrics] == [0.42, 0.31]
    assert all(metric.source == "evorec" for metric in loaded.metrics)


def test_mismatched_sha256_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    digest = _write_results(path)
    store = InMemoryExperimentStore()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _import(path, store, "0" * 64)
    assert store.get("evorec-itemcf-1") is None

    path.write_text(path.read_text(encoding="utf-8").replace("0.42", "0.99"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _import(path, store, digest)
    assert store.get("evorec-itemcf-1") is None


def test_imported_evorec_is_not_comparable_to_searchpilot(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    digest = _write_results(path)
    store = InMemoryExperimentStore()
    imported = _import(path, store, digest)
    local = ExperimentRecord(
        experiment_id="sp-search-1",
        kind="search",
        config={},
        data_version="d3a904f41240",
        protocol_version="evorec-v1",
        source="searchpilot",
        metrics=(MetricPoint(name="ndcg@10", split="test", value=0.42, source="searchpilot"),),
    )

    result = compare_experiments(local, imported, ["ndcg@10"])
    assert result["comparable"] is False
    assert result["reason"] == "source or protocol_version differs"


def test_published_results_keep_cohort_metric_names(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "protocol_id": "2624e6561e71af4b",
                "protocol": {"catalog_sha256": "catalogsha"},
                "configuration": {"stage": "R05-ranker"},
                "test_results": [
                    {
                        "name": "RRF",
                        "metrics": {
                            "cohorts": {
                                "cold_user": {
                                    "ndcg@10": 0.2,
                                    "recall@20": 0.1,
                                    "ranking_latency_ms_p50": None,
                                }
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = InMemoryExperimentStore()
    records = import_evorec_file(
        path,
        store,
        repo="Die4U23/EvoRec",
        commit="5ce1d96b80ddaf3c205a5ef8a6a5e39df6ab0ff5",
        run_id="r05-ranker",
        source_path="docs/experiments/r05-ranker/results.json",
    )
    assert [record.experiment_id for record in records] == ["r05-ranker-rrf"]
    loaded = store.get("r05-ranker-rrf")
    assert loaded is not None
    assert loaded.protocol_version == "2624e6561e71af4b"
    assert loaded.source_ref is not None
    assert loaded.source_ref["path"] == "docs/experiments/r05-ranker/results.json"
    assert {(metric.name, metric.segment, metric.value) for metric in loaded.metrics} == {
        ("ndcg@10", "cold_user", 0.2),
        ("recall@20", "cold_user", 0.1),
    }
