from __future__ import annotations

import re
from collections.abc import Callable
from types import SimpleNamespace

from fastapi.testclient import TestClient

MakeClient = Callable[..., TestClient]
REQ_RE = re.compile(r"^req_[0-9a-f]{32}$")


def _ready_client(make_client: MakeClient, fakes: SimpleNamespace, **kwargs: object) -> TestClient:
    return make_client(
        search=fakes.FakeSearch(),
        recommend=fakes.FakeRecommend(),
        feedback=fakes.FakeFeedbackStore(),
        **kwargs,
    )


def test_live_ok_without_any_dependency(make_client: MakeClient) -> None:
    response = make_client().get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_200_with_all_dependencies(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    client = _ready_client(make_client, fakes, probes=[fakes.FakeProbe({"database": True})])
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {
        "search": True,
        "recommend": True,
        "feedback": True,
        "database": True,
    }


def test_ready_503_when_probe_false(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    client = _ready_client(
        make_client, fakes, probes=[fakes.FakeProbe({"database": False, "search_index": True})]
    )
    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["database"] is False
    assert body["checks"]["search_index"] is True
    fakes.assert_error_envelope(body, "NOT_READY", retryable=True)


def test_ready_503_when_any_of_several_probes_false(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    client = _ready_client(
        make_client,
        fakes,
        probes=[fakes.FakeProbe({"database": True}), fakes.FakeProbe({"vector_index": False})],
    )
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["vector_index"] is False


def test_ready_503_when_ports_missing(make_client: MakeClient) -> None:
    response = make_client().get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"] == {"search": False, "recommend": False, "feedback": False}


def test_ready_probe_exception_is_not_ready_and_does_not_leak(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    client = _ready_client(make_client, fakes, probes=[fakes.RaisingProbe()])
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["RaisingProbe"] is False
    assert "secret" not in response.text


def test_request_id_generated_when_absent(make_client: MakeClient) -> None:
    response = make_client().get("/health/live")
    assert REQ_RE.match(response.headers["X-Request-Id"])


def test_request_id_reused_when_valid(make_client: MakeClient) -> None:
    rid = "req_" + "0123456789abcdef" * 2
    response = make_client().get("/health/live", headers={"X-Request-Id": rid})
    assert response.headers["X-Request-Id"] == rid


def test_request_id_replaced_when_invalid(make_client: MakeClient) -> None:
    client = make_client()
    for bad in ("abc", "req_XYZ", "req_" + "a" * 31, "req_" + "A" * 32):
        response = client.get("/health/live", headers={"X-Request-Id": bad})
        assert response.headers["X-Request-Id"] != bad
        assert REQ_RE.match(response.headers["X-Request-Id"])


def test_unknown_route_uses_error_envelope(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client().get("/nope")
    assert response.status_code == 404
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT", retryable=False)
    assert response.json()["error"]["request_id"] == response.headers["X-Request-Id"]
