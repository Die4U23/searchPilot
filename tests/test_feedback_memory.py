from __future__ import annotations

import threading
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from searchpilot.feedback.memory import InMemoryFeedbackStore, content_hash
from searchpilot.ports import FeedbackEvent, IdempotencyConflictError


def _event(**kwargs: object) -> FeedbackEvent:
    defaults = {
        "idempotency_key": "key-1",
        "request_id": "req_abc",
        "item_id": "item1",
        "kind": "click",
        "event_at": datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        "user_id": "u1",
        "position": 1,
    }
    defaults.update(kwargs)
    return FeedbackEvent(**defaults)  # type: ignore[arg-type]


def test_first_write_not_replayed() -> None:
    store = InMemoryFeedbackStore()
    result = store.write(_event())
    assert result.replayed is False
    assert result.event_id
    assert store.count() == 1


def test_replay_same_key_same_content() -> None:
    store = InMemoryFeedbackStore()
    first = store.write(_event())
    second = store.write(_event())
    assert second.replayed is True
    assert second.event_id == first.event_id
    assert store.count() == 1


def test_conflict_same_key_different_content() -> None:
    store = InMemoryFeedbackStore()
    store.write(_event(item_id="a"))
    with pytest.raises(IdempotencyConflictError):
        store.write(_event(item_id="b"))


def test_replay_keeps_first_model_version() -> None:
    store = InMemoryFeedbackStore()
    first = store.write(_event(model_version="bm25-aaaa1111"))
    second = store.write(_event(model_version="bm25-bbbb2222"))
    assert second.replayed is True
    assert second.event_id == first.event_id
    assert store.count() == 1
    stored = store.get_event("key-1")
    assert stored is not None
    assert stored.model_version == "bm25-aaaa1111"


def test_content_hash_timezone_equivalent() -> None:
    utc = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    eastern = datetime(2024, 6, 1, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    assert content_hash(_event(event_at=utc)) == content_hash(_event(event_at=eastern))


def test_concurrent_same_key_single_write() -> None:
    store = InMemoryFeedbackStore()
    barrier = threading.Barrier(20)
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        res = store.write(_event(idempotency_key="shared"))
        with lock:
            results.append(res.event_id)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.count() == 1
    assert len(set(results)) == 1
