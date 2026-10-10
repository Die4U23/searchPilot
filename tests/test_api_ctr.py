from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from searchpilot.bootstrap import create_app
from searchpilot.ctr.runtime import FEATURE_NAMES, CtrNotReadyError, load_ctr

MakeClient = Callable[..., TestClient]


def _write_ctr_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "model_version": "ctr-test0001",
                "calibrator_version": "temp-test0001",
                "feature_names": list(FEATURE_NAMES),
                "mean": [0.0, 0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    (directory / "lr.json").write_text(
        json.dumps({"weight": [0.1, -0.2, 0.3, -0.1], "bias": -0.5}),
        encoding="utf-8",
    )
    (directory / "temperature.json").write_text(json.dumps({"T": 1.5}), encoding="utf-8")


def test_ctr_score_not_loaded_503(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client().post(
        "/ctr/score", json={"user_id": "U1", "candidates": [{"item_id": "N1"}]}
    )
    assert response.status_code == 503
    body = response.json()
    assert "results" not in body
    fakes.assert_error_envelope(body, "NOT_READY", retryable=True)


def test_ctr_score_ok_after_loading_tmp_artifacts(tmp_path: Path) -> None:
    _write_ctr_artifacts(tmp_path / "ctr")
    runtime = load_ctr(tmp_path)
    client = TestClient(create_app(ctr=runtime))
    response = client.post(
        "/ctr/score",
        json={
            "user_id": "U1",
            "candidates": [{"item_id": "N1", "bid": 1.25}, {"item_id": "N2"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-Id"]
    assert body["model_version"] == "ctr-test0001"
    assert body["calibrator_version"] == "temp-test0001"
    assert len(body["results"]) == 2
    first, second = body["results"]
    assert first["item_id"] == "N1"
    assert 0.0 <= first["pctr"] <= 1.0
    assert 0.0 <= first["pctr_calibrated"] <= 1.0
    assert first["ecpm"] == pytest.approx(first["pctr_calibrated"] * 1.25)
    assert second["item_id"] == "N2"
    assert 0.0 <= second["pctr"] <= 1.0
    assert second["ecpm"] is None


def test_ctr_score_more_than_100_candidates_422(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    payload = {
        "user_id": "U1",
        "candidates": [{"item_id": f"N{i}"} for i in range(101)],
    }
    response = make_client().post("/ctr/score", json=payload)
    assert response.status_code == 422
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT", retryable=False)


def test_ctr_rejects_mismatched_feature_names(tmp_path: Path) -> None:
    _write_ctr_artifacts(tmp_path / "ctr")
    meta_path = tmp_path / "ctr" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["feature_names"] = ["other_feature", "still_wrong", "nope", "no"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(CtrNotReadyError, match="feature_names"):
        load_ctr(tmp_path)
