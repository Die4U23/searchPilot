"""PostgreSQL 反馈仓储及跨实现共享的载荷规范化。"""

from __future__ import annotations

from psycopg_pool import ConnectionPool

from searchpilot.ports import (
    FeedbackEvent,
    FeedbackWriteResult,
    IdempotencyConflictError,
    feedback_content_hash,
)


def content_hash(event: FeedbackEvent) -> str:
    """规范化内容摘要；与内存实现共用 ``ports.feedback_content_hash``。"""
    return feedback_content_hash(event)


class PostgresFeedbackStore:
    """依赖唯一幂等键实现并发安全的反馈写入。"""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def write(self, event: FeedbackEvent) -> FeedbackWriteResult:
        digest = content_hash(event)
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO feedback_events (
                    idempotency_key, request_id, item_id, kind, event_at,
                    user_id, position, content_hash, model_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING event_id, received_at
                """,
                (
                    event.idempotency_key,
                    event.request_id,
                    event.item_id,
                    event.kind,
                    event.event_at,
                    event.user_id,
                    event.position,
                    digest,
                    event.model_version,
                ),
            ).fetchone()
            if row is not None:
                return FeedbackWriteResult(event_id=str(row[0]), replayed=False, received_at=row[1])

            existing = connection.execute(
                """
                SELECT event_id, received_at, content_hash
                FROM feedback_events
                WHERE idempotency_key = %s
                """,
                (event.idempotency_key,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("idempotency conflict row was not visible")
            if existing[2] != digest:
                raise IdempotencyConflictError(
                    "idempotency key was already used for different feedback content"
                )
            return FeedbackWriteResult(
                event_id=str(existing[0]),
                replayed=True,
                received_at=existing[1],
            )
