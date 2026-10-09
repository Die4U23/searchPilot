from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from searchpilot.db.feedback_store import content_hash
from searchpilot.feedback.memory import content_hash as memory_content_hash
from searchpilot.ports import FeedbackEvent, feedback_content_hash


@pytest.mark.parametrize(
    "event",
    [
        # 零微秒：isoformat() 会省略微秒，timespec="microseconds" 不会
        FeedbackEvent("k", "req_1", "N1", "click", datetime(2026, 2, 3, tzinfo=UTC), "U1", 0),
        # 非 ASCII：ensure_ascii 开关会改变字节
        FeedbackEvent(
            "k", "req_1", "N1", "like", datetime(2026, 2, 3, 4, 5, 6, 7, tzinfo=UTC), "用户", 1
        ),
        # 非 UTC 时区 + None 字段
        FeedbackEvent(
            "k",
            "req_1",
            "N1",
            "hide",
            datetime(2026, 2, 3, 12, tzinfo=timezone(timedelta(hours=8))),
            None,
            None,
        ),
    ],
)
def test_memory_and_postgres_hash_are_byte_identical(event: FeedbackEvent) -> None:
    """build-plan §2：内存与 PostgreSQL 实现必须使用同一套规范化。"""
    expected = feedback_content_hash(event)
    assert content_hash(event) == expected
    assert memory_content_hash(event) == expected


def test_content_hash_rejects_naive_datetime() -> None:
    event = FeedbackEvent("k", "req_1", "N1", "click", datetime(2026, 2, 3), None, None)
    with pytest.raises(ValueError, match="timezone-aware"):
        memory_content_hash(event)
    with pytest.raises(ValueError, match="timezone-aware"):
        content_hash(event)


def test_content_hash_normalizes_equivalent_timezones() -> None:
    utc_event = FeedbackEvent(
        idempotency_key="65bc8c09-55f7-4bf7-ac6a-49823dc66368",
        request_id="req_1",
        item_id="N12345",
        kind="click",
        event_at=datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=UTC),
        user_id="U1",
        position=2,
    )
    offset_event = FeedbackEvent(
        idempotency_key="different-key-is-not-part-of-content",
        request_id="req_1",
        item_id="N12345",
        kind="click",
        event_at=datetime(
            2026,
            1,
            2,
            11,
            4,
            5,
            123456,
            tzinfo=timezone(timedelta(hours=8)),
        ),
        user_id="U1",
        position=2,
    )

    assert content_hash(utc_event) == content_hash(offset_event)


def test_content_hash_has_stable_field_order_and_excludes_idempotency_key() -> None:
    event = FeedbackEvent(
        idempotency_key="first-key",
        request_id="req_order",
        item_id="N12345",
        kind="impression",
        event_at=datetime(2026, 2, 3, tzinfo=UTC),
        user_id=None,
        position=None,
    )
    same_content = FeedbackEvent(
        idempotency_key="second-key",
        request_id=event.request_id,
        item_id=event.item_id,
        kind=event.kind,
        event_at=event.event_at,
        user_id=event.user_id,
        position=event.position,
    )

    assert content_hash(event) == content_hash(same_content)
    assert content_hash(event) == (
        "f52bb2e3ce751437155cbbfe6e92e229f7dd3ddc5a0d4597fa9134a8cfe92877"
    )
