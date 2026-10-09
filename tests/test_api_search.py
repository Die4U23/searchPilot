from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from searchpilot.config import get_settings
from searchpilot.ports import Document, SearchHit

MakeClient = Callable[..., TestClient]


def test_search_ok(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    service = fakes.FakeSearch()
    response = make_client(search=service).post(
        "/search", json={"query": "  Hello World  ", "limit": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-Id"]
    assert body["normalized_query"] == "hello world"
    assert body["model_version"] == "bm25-test0001"
    assert body["mode"] == "bm25"
    assert body["results"] == [
        {"item_id": "N1", "score": 3.0, "rank": 1, "channel": "bm25"},
        {"item_id": "N2", "score": 2.0, "rank": 2, "channel": "bm25"},
    ]
    # query 去首尾空白后才传给服务；默认 mode 为 bm25
    assert service.calls == [("Hello World", 2, "bm25")]


def test_search_defaults(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    service = fakes.FakeSearch()
    response = make_client(search=service).post("/search", json={"query": "x"})
    assert response.status_code == 200
    assert service.calls == [("x", 10, "bm25")]


def test_search_mode_passthrough(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    service = fakes.FakeSearch(hits=[SearchHit("N1", 0.5, 1, "rrf")])
    response = make_client(search=service).post("/search", json={"query": "x", "mode": "hybrid"})
    assert response.status_code == 200
    assert response.json()["mode"] == "hybrid"
    assert response.json()["results"][0]["channel"] == "rrf"


def test_search_empty_results_is_200(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(search=fakes.FakeSearch(hits=[])).post("/search", json={"query": "zzz"})
    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_search_blank_query_422(
    make_client: MakeClient, fakes: SimpleNamespace, query: str
) -> None:
    response = make_client(search=fakes.FakeSearch()).post("/search", json={"query": query})
    assert response.status_code == 422
    error = fakes.assert_error_envelope(response.json(), "INVALID_INPUT", retryable=False)
    assert "query" in error["message"]


def test_search_query_at_limit_ok_and_over_limit_422(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    client = make_client(search=fakes.FakeSearch())
    assert client.post("/search", json={"query": "a" * 256}).status_code == 200
    response = client.post("/search", json={"query": "a" * 257})
    assert response.status_code == 422
    error = fakes.assert_error_envelope(response.json(), "INVALID_INPUT")
    assert "256" in error["message"]


def test_search_extremely_long_query_422(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(search=fakes.FakeSearch()).post("/search", json={"query": "a" * 100_000})
    assert response.status_code == 422
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT")


def test_search_query_limit_follows_settings(
    make_client: MakeClient, fakes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEARCHPILOT_MAX_QUERY_CHARS", "10")
    get_settings.cache_clear()
    client = make_client(search=fakes.FakeSearch())
    assert client.post("/search", json={"query": "a" * 10}).status_code == 200
    assert client.post("/search", json={"query": "a" * 11}).status_code == 422


@pytest.mark.parametrize("limit", [0, -1, 101, 1001])
def test_search_limit_out_of_range_422(
    make_client: MakeClient, fakes: SimpleNamespace, limit: int
) -> None:
    response = make_client(search=fakes.FakeSearch()).post(
        "/search", json={"query": "x", "limit": limit}
    )
    assert response.status_code == 422
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT")


def test_search_limit_follows_settings(
    make_client: MakeClient, fakes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEARCHPILOT_MAX_LIMIT", "5")
    get_settings.cache_clear()
    client = make_client(search=fakes.FakeSearch())
    assert client.post("/search", json={"query": "x", "limit": 5}).status_code == 200
    assert client.post("/search", json={"query": "x", "limit": 6}).status_code == 422


def test_search_unknown_mode_422(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(search=fakes.FakeSearch()).post(
        "/search", json={"query": "x", "mode": "magic"}
    )
    assert response.status_code == 422
    error = fakes.assert_error_envelope(response.json(), "INVALID_INPUT")
    assert "body.mode" in error["message"]


def test_search_missing_body_and_bad_json_422(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    client = make_client(search=fakes.FakeSearch())
    assert client.post("/search").status_code == 422
    response = client.post(
        "/search", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT")


def test_search_service_missing_503(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client().post("/search", json={"query": "x"})
    assert response.status_code == 503
    fakes.assert_error_envelope(response.json(), "NOT_READY", retryable=True)


def test_search_service_not_ready_exception_503(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    class SearchNotReadyError(Exception):
        pass

    service = fakes.FakeSearch(error=SearchNotReadyError("index missing at C:\\secret\\path"))
    response = make_client(search=service).post("/search", json={"query": "x", "mode": "vector"})
    assert response.status_code == 503
    fakes.assert_error_envelope(response.json(), "NOT_READY", retryable=True)
    assert "secret" not in response.text


def test_search_unexpected_exception_500_without_leak(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    service = fakes.FakeSearch(error=RuntimeError("SELECT * FROM secret_table"))
    response = make_client(search=service).post("/search", json={"query": "x"})
    assert response.status_code == 500
    fakes.assert_error_envelope(response.json(), "INTERNAL", retryable=False)
    assert "secret_table" not in response.text
    assert response.json()["error"]["request_id"] == response.headers["X-Request-Id"]


def test_search_category_filter(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    docs = {
        "N1": Document("N1", "t", "", "sports", "x"),
        "N2": Document("N2", "t", "", "news", "x"),
        "N3": Document("N3", "t", "", "Sports", "y"),
    }
    client = make_client(search=fakes.FakeSearch(), items=fakes.FakeItems(docs))
    response = client.post(
        "/search", json={"query": "x", "limit": 5, "filters": {"category": "sports"}}
    )
    assert response.status_code == 200
    assert [(h["item_id"], h["rank"]) for h in response.json()["results"]] == [
        ("N1", 1),
        ("N3", 2),
    ]


def test_search_category_filter_uses_index_native_filter(make_client: MakeClient) -> None:
    """搜索服务自带 search_with_filters 时，不需要 ItemStore 也能按类目过滤。"""
    from searchpilot.search.service import InMemorySearchService

    docs = [
        Document("N1", "patriots sign quarterback", "", "sports", "football"),
        Document("N2", "patriots fans protest tax", "", "news", "politics"),
        Document("N3", "Patriots quarterback injured", "", "Sports", "football"),
    ]
    client = make_client(search=InMemorySearchService(docs))
    response = client.post(
        "/search", json={"query": "patriots", "limit": 5, "filters": {"category": "sports"}}
    )
    assert response.status_code == 200
    body = response.json()
    assert sorted(h["item_id"] for h in body["results"]) == ["N1", "N3"]
    assert [h["rank"] for h in body["results"]] == [1, 2]


def test_search_category_filter_without_item_store_503(
    make_client: MakeClient, fakes: SimpleNamespace
) -> None:
    response = make_client(search=fakes.FakeSearch()).post(
        "/search", json={"query": "x", "filters": {"category": "sports"}}
    )
    assert response.status_code == 503
    fakes.assert_error_envelope(response.json(), "NOT_READY")


def test_search_rejects_unknown_fields(make_client: MakeClient, fakes: SimpleNamespace) -> None:
    response = make_client(search=fakes.FakeSearch()).post(
        "/search", json={"query": "x", "bogus": 1}
    )
    assert response.status_code == 422
    fakes.assert_error_envelope(response.json(), "INVALID_INPUT")
