from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from searchpilot.ports import Document

MakeClient = Callable[..., TestClient]


def test_feedback_created_201(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    store = fakes.FakeFeedbackStore()
    response = make_client(feedback=store).post("/feedback", json=fakes.feedback_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["replayed"] is False
    assert body["event_id"]
    assert body["received_at"]
    assert body["request_id"] == response.headers["X-Request-Id"]
    assert store.count == 1


def test_feedback_minimal_payload(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    payload = fakes.feedback_payload()
    del payload["user_id"], payload["position"]
    response = make_client(feedback=fakes.FakeFeedbackStore()).post("/feedback", json=payload)
    assert response.status_code == 201


def test_feedback_replay_returns_original(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    store = fakes.FakeFeedbackStore()
    client = make_client(feedback=store)
    payload = fakes.feedback_payload()
    first = client.post("/feedback", json=payload)
    # 同一时刻换一种时区写法：规范化后内容相同，仍是重放
    replay_payload = {**payload, "event_at": "2026-01-01T19:04:05+00:00"}
    second = client.post("/feedback", json=replay_payload)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["replayed"] is True
    assert second.json()["event_id"] == first.json()["event_id"]
    assert second.json()["received_at"] == first.json()["received_at"]
    assert store.count == 1


def test_feedback_conflict_409(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    client = make_client(feedback=fakes.FakeFeedbackStore())
    payload = fakes.feedback_payload()
    assert client.post("/feedback", json=payload).status_code == 201
    response = client.post("/feedback", json={**payload, "kind": "like"})
    assert response.status_code == 409
    fakes.assert_error_envelope(response.json(), "IDEMPOTENCY_CONFLICT", retryable=False)


def test_feedback_naive_event_at_422(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(feedback=fakes.FakeFeedbackStore()).post(
        "/feedback", json=fakes.feedback_payload(event_at="2026-01-02T03:04:05")
    )
    assert response.status_code == 422
    error = fakes.assert_error_envelope(response.json(), "INVALID_INPUT")
    assert "event_at" in error["message"]


@pytest.mark.parametrize("item_id", ["", "bad id", "a/b", "x" * 129, "新闻", "N1\n"])
def test_feedback_invalid_item_id_422(
    make_client: MakeClient, fakes: SimpleNamespace, item_id: str
) -> None:
    response = make_client(feedback=fakes.FakeFeedbackStore()).post(
        "/feedback", json=fakes.feedback_payload(item_id=item_id)
    )
    assert response.status_code == 422
    error = fakes.assert_error_envelope(response.json(), "INVALID_INPUT")
    assert "item_id" in error["message"]


@pytest.mark.parametrize("item_id", ["N1", "a.b:c_d-e", "x" * 128])
def test_feedback_valid_item_id(
    make_client: MakeClient, fakes: SimpleNamespace, item_id: str
) -> None:
    response = make_client(feedback=fakes.FakeFeedbackStore()).post(
        "/feedback", json=fakes.feedback_payload(item_id=item_id)
    )
    assert response.status_code == 201


@pytest.mark.parametrize(
    "overrides",
    [
        {"idempotency_key": "not-a-uuid"},
        {"kind": "share"},
        {"user_id": ""},
        {"user_id": "u" * 65},
        {"position": -1},
        {"event_at": "yesterday"},
        {"request_id": ""},
    ],
)
def test_feedback_other_invalid_fields_422(
    make_client: MakeClient, fakes: SimpleNamespace, overrides: dict[str, object]
) -> None:
    response = make_client(feedback=fakes.FakeFeedbackStore()).post(
        "/feedback", json=fakes.feedback_payload(**overrides)
    )
    assert response.status_code == 422
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT")


def test_feedback_missing_fields_422(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(feedback=fakes.FakeFeedbackStore()).post("/feedback", json={})
    assert response.status_code == 422
    error = fakes.assert_error_envelope(response.json(), "INVALID_INPUT")
    assert "idempotency_key" in error["message"]


def test_feedback_store_missing_503(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client().post("/feedback", json=fakes.feedback_payload())
    assert response.status_code == 503
    fakes.assert_error_envelope(response.json(), "NOT_READY", retryable=True)


def test_feedback_unknown_item_404_when_item_store_present(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    items = fakes.FakeItems({"N1": Document("N1", "t", "", "news", "x")})
    client = make_client(feedback=fakes.FakeFeedbackStore(), items=items)
    assert client.post("/feedback", json=fakes.feedback_payload(item_id="N1")).status_code == 201
    response = client.post("/feedback", json=fakes.feedback_payload(item_id="N404"))
    assert response.status_code == 404
    fakes.assert_error_envelope(response.json(), "ITEM_NOT_FOUND", retryable=False)
