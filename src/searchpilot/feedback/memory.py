"""幂等反馈存储的内存参考实现。"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime

from searchpilot.ports import (
    FeedbackEvent,
    FeedbackStore,
    FeedbackWriteResult,
    IdempotencyConflictError,
    feedback_content_hash,
)


def content_hash(event: FeedbackEvent) -> str:
    """规范化内容摘要；与 PostgreSQL 实现共用 ``ports.feedback_content_hash``。"""
    return feedback_content_hash(event)


class InMemoryFeedbackStore(FeedbackStore):
    """线程安全的内存 ``FeedbackStore``。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[str, tuple[str, FeedbackWriteResult, FeedbackEvent]] = {}

    def write(self, event: FeedbackEvent) -> FeedbackWriteResult:
        digest = content_hash(event)
        with self._lock:
            existing = self._by_key.get(event.idempotency_key)
            if existing is not None:
                stored_digest, result, _stored = existing
                if stored_digest != digest:
                    raise IdempotencyConflictError(
                        f"idempotency_key {event.idempotency_key!r} "
                        "already used with different content"
                    )
                return FeedbackWriteResult(
                    event_id=result.event_id,
                    replayed=True,
                    received_at=result.received_at,
                )

            received_at = datetime.now(tz=UTC)
            result = FeedbackWriteResult(
                event_id=str(uuid.uuid4()),
                replayed=False,
                received_at=received_at,
            )
            self._by_key[event.idempotency_key] = (digest, result, event)
            return result

    def get_event(self, idempotency_key: str) -> FeedbackEvent | None:
        with self._lock:
            existing = self._by_key.get(idempotency_key)
            return None if existing is None else existing[2]

    def count(self) -> int:
        with self._lock:
            return len(self._by_key)

    def all_events(self) -> tuple[FeedbackWriteResult, ...]:
        with self._lock:
            return tuple(pair[1] for pair in self._by_key.values())
