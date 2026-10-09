"""共享 fixture 与假对象：只依赖 ports.py 的 Protocol，不访问网络 / 数据库 / 模型。"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from searchpilot.bootstrap import create_app
from searchpilot.config import get_settings
from searchpilot.ports import (
    Document,
    FeedbackEvent,
    FeedbackStore,
    FeedbackWriteResult,
    IdempotencyConflictError,
    ItemStore,
    ReadinessProbe,
    RecommendationHit,
    RecommendationResult,
    RecommendPort,
    SearchHit,
    SearchMode,
    SearchPort,
    SearchResult,
)


class FakeSearch:
    def __init__(
        self,
        hits: Sequence[SearchHit] | None = None,
        model_version: str = "bm25-test0001",
        error: Exception | None = None,
    ) -> None:
        self.hits = (
            tuple(hits)
            if hits is not None
            else (
                SearchHit("N1", 3.0, 1, "bm25"),
                SearchHit("N2", 2.0, 2, "bm25"),
                SearchHit("N3", 1.0, 3, "bm25"),
            )
        )
        self._model_version = model_version
        self.error = error
        self.calls: list[tuple[str, int, str]] = []

    @property
    def model_version(self) -> str:
        return self._model_version

    def search(self, query: str, limit: int, mode: SearchMode) -> SearchResult:
        self.calls.append((query, limit, mode))
        if self.error is not None:
            raise self.error
        return SearchResult(
            normalized_query=query.lower(),
            hits=self.hits[:limit],
            model_version=self._model_version,
            mode=mode,
        )


class FakeRecommend:
    def __init__(
        self,
        hits: Sequence[RecommendationHit] | None = None,
        cold_start: bool = False,
        model_version: str = "popular-test",
    ) -> None:
        self.hits = (
            tuple(hits)
            if hits is not None
            else (
                RecommendationHit("N9", 0.9, 1, "popular"),
                RecommendationHit("N8", 0.8, 2, "popular"),
            )
        )
        self.cold_start = cold_start
        self._model_version = model_version
        self.calls: list[tuple[str, int, frozenset[str]]] = []

    @property
    def model_version(self) -> str:
        return self._model_version

    def recommend(self, user_id: str, limit: int, exclude: frozenset[str]) -> RecommendationResult:
        self.calls.append((user_id, limit, exclude))
        return RecommendationResult(
            hits=self.hits[:limit],
            model_version=self._model_version,
            channel="popular",
            cold_start=self.cold_start,
        )


class FakeFeedbackStore:
    """内存幂等语义：同键同内容重放，同键不同内容冲突。"""

    def __init__(self) -> None:
        self._events: dict[str, tuple[FeedbackEvent, FeedbackWriteResult]] = {}

    @staticmethod
    def _canonical(e: FeedbackEvent) -> tuple[object, ...]:
        return (e.request_id, e.item_id, e.kind, e.event_at.astimezone(UTC), e.user_id, e.position)

    def write(self, event: FeedbackEvent) -> FeedbackWriteResult:
        known = self._events.get(event.idempotency_key)
        if known is not None:
            if self._canonical(known[0]) != self._canonical(event):
                raise IdempotencyConflictError(event.idempotency_key)
            return FeedbackWriteResult(known[1].event_id, True, known[1].received_at)
        result = FeedbackWriteResult(f"evt_{len(self._events) + 1}", False, datetime.now(UTC))
        self._events[event.idempotency_key] = (event, result)
        return result

    @property
    def count(self) -> int:
        return len(self._events)


class FakeItems:
    def __init__(self, docs: Mapping[str, Document]) -> None:
        self._docs = dict(docs)

    def get(self, item_id: str) -> Document | None:
        return self._docs.get(item_id)

    def exists(self, item_id: str) -> bool:
        return item_id in self._docs


class FakeProbe:
    def __init__(self, checks: Mapping[str, bool]) -> None:
        self._checks = dict(checks)

    def check(self) -> Mapping[str, bool]:
        return self._checks


class RaisingProbe:
    def check(self) -> Mapping[str, bool]:
        raise RuntimeError("boom: postgresql://user:secret@db/x")


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """隔离环境变量与 Settings 缓存，保证每个测试看到默认配置。"""
    for key in [k for k in os.environ if k.startswith("SEARCHPILOT_")]:
        monkeypatch.delenv(key)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def make_client() -> ClientFactory:
    def factory(
        *,
        search: SearchPort | None = None,
        recommend: RecommendPort | None = None,
        feedback: FeedbackStore | None = None,
        items: ItemStore | None = None,
        probes: Sequence[ReadinessProbe] = (),
    ) -> TestClient:
        app = create_app(
            search=search, recommend=recommend, feedback=feedback, items=items, probes=probes
        )
        return TestClient(app)

    return factory


def feedback_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "idempotency_key": str(uuid.uuid4()),
        "request_id": "req_" + "a" * 32,
        "item_id": "N1",
        "kind": "click",
        "event_at": "2026-01-02T03:04:05+08:00",
        "user_id": "U1",
        "position": 1,
    }
    payload.update(overrides)
    return payload


def assert_error_envelope(
    response_json: Mapping[str, object], code: str, *, retryable: bool | None = None
) -> Mapping[str, object]:
    assert set(response_json) >= {"error"}
    error = response_json["error"]
    assert isinstance(error, dict)
    assert set(error) == {"code", "message", "retryable", "request_id"}
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["retryable"], bool)
    assert isinstance(error["request_id"], str) and error["request_id"].startswith("req_")
    if retryable is not None:
        assert error["retryable"] is retryable
    return error


@pytest.fixture
def fakes() -> SimpleNamespace:
    """假对象与断言工具的统一入口，避免测试文件之间 import conftest。"""
    return SimpleNamespace(
        FakeSearch=FakeSearch,
        FakeRecommend=FakeRecommend,
        FakeFeedbackStore=FakeFeedbackStore,
        FakeItems=FakeItems,
        FakeProbe=FakeProbe,
        RaisingProbe=RaisingProbe,
        feedback_payload=feedback_payload,
        assert_error_envelope=assert_error_envelope,
    )
