from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from searchpilot.bootstrap import build_default_app, create_app
from searchpilot.config import get_settings
from searchpilot.experiments.store import ExperimentRecord, InMemoryExperimentStore, MetricPoint


def _record() -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id="exp-searchpilot-v1",
        kind="search",
        config={"seed": 20261009},
        data_version="d3a904f41240",
        protocol_version="v1",
        source="searchpilot",
        code_commit=None,
        source_ref={"repo": "searchpilot", "commit": "abc", "sha256": "deadbeef"},
        metrics=(
            MetricPoint(
                name="ndcg@10",
                split="test",
                segment=None,
                value=0.5,
                source="searchpilot",
            ),
        ),
    )


def test_get_experiment_returns_source_and_source_ref() -> None:
    store = InMemoryExperimentStore()
    store.put(_record())
    client = TestClient(create_app(experiments=store))
    response = client.get("/experiments/exp-searchpilot-v1")
    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"] == "exp-searchpilot-v1"
    assert body["source"] == "searchpilot"
    assert body["source_ref"] == {"repo": "searchpilot", "commit": "abc", "sha256": "deadbeef"}
    assert body["metrics"] == [
        {
            "name": "ndcg@10",
            "split": "test",
            "segment": None,
            "value": 0.5,
            "source": "searchpilot",
        }
    ]


def test_unknown_experiment_404_without_leak(fakes: SimpleNamespace) -> None:
    store = InMemoryExperimentStore()
    client = TestClient(create_app(experiments=store))
    experiment_id = "missing_SELECT_secret_table_C_secret_registry.json"
    response = client.get(f"/experiments/{experiment_id}")
    assert response.status_code == 404
    error = fakes.assert_error_envelope(response.json(), "ITEM_NOT_FOUND", retryable=False)
    assert error["request_id"] == response.headers["X-Request-Id"]
    assert "secret_table" not in response.text
    assert "registry.json" not in response.text
    assert "SELECT" not in response.text
    assert "postgresql" not in response.text.lower()


def test_default_app_without_database_uses_empty_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SEARCHPILOT_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("SEARCHPILOT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    app = build_default_app()
    assert app.state.experiments is not None
    client = TestClient(app)
    response = client.get("/experiments/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ITEM_NOT_FOUND"


def test_default_app_loads_registry_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = InMemoryExperimentStore()
    store.put(_record())
    store.save(tmp_path / "experiments" / "registry.json")
    monkeypatch.setenv("SEARCHPILOT_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("SEARCHPILOT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    client = TestClient(build_default_app())
    response = client.get("/experiments/exp-searchpilot-v1")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "searchpilot"
    assert body["source_ref"]["sha256"] == "deadbeef"
