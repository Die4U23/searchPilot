from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from searchpilot.ports import RecommendationHit

MakeClient = Callable[..., TestClient]


def test_recommend_ok(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    service = fakes.FakeRecommend()
    response = make_client(recommend=service).post(
        "/recommend", json={"user_id": "U1", "limit": 5, "context": {"page": "home"}}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-Id"]
    assert body["model_version"] == "popular-test"
    assert body["channel"] == "popular"
    assert body["cold_start"] is False
    assert body["results"] == [
        {"item_id": "N9", "score": 0.9, "rank": 1, "channel": "popular"},
        {"item_id": "N8", "score": 0.8, "rank": 2, "channel": "popular"},
    ]
    assert service.calls == [("U1", 5, frozenset())]


def test_recommend_cold_start_passthrough(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(recommend=fakes.FakeRecommend(cold_start=True)).post(
        "/recommend", json={"user_id": "new-user"}
    )
    assert response.status_code == 200
    assert response.json()["cold_start"] is True


def test_recommend_results_are_deduplicated_and_reranked(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    hits = [
        RecommendationHit("N1", 0.9, 1, "popular"),
        RecommendationHit("N1", 0.8, 2, "popular"),
        RecommendationHit("N2", 0.7, 3, "itemcf"),
    ]
    response = make_client(recommend=fakes.FakeRecommend(hits=hits)).post(
        "/recommend", json={"user_id": "U1"}
    )
    results = response.json()["results"]
    assert [(r["item_id"], r["rank"]) for r in results] == [("N1", 1), ("N2", 2)]
    assert results[1]["channel"] == "itemcf"


def test_recommend_empty_is_200(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(recommend=fakes.FakeRecommend(hits=[])).post(
        "/recommend", json={"user_id": "U1"}
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_recommend_service_missing_503(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client().post("/recommend", json={"user_id": "U1"})
    assert response.status_code == 503
    fakes.assert_error_envelope(response.json(), "NOT_READY", retryable=True)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user_id": ""},
        {"user_id": "u" * 65},
        {"user_id": "U1", "limit": 0},
        {"user_id": "U1", "limit": 101},
        {"user_id": "U1", "context": "not-an-object"},
    ],
)
def test_recommend_invalid_input_422(
    make_client: MakeClient, fakes: SimpleNamespace, payload: dict[str, object]
) -> None:
    response = make_client(recommend=fakes.FakeRecommend()).post("/recommend", json=payload)
    assert response.status_code == 422
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT", retryable=False)


def test_recommend_user_id_boundaries(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    client = make_client(recommend=fakes.FakeRecommend())
    assert client.post("/recommend", json={"user_id": "u"}).status_code == 200
    assert client.post("/recommend", json={"user_id": "u" * 64}).status_code == 200
