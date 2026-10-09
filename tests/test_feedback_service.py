from __future__ import annotations

from datetime import UTC, datetime

import pytest

from searchpilot.feedback.memory import InMemoryFeedbackStore
from searchpilot.feedback.service import FeedbackService
from searchpilot.ports import FeedbackEvent


def test_feedback_service_delegates() -> None:
    store = InMemoryFeedbackStore()
    svc = FeedbackService(store)
    event = FeedbackEvent(
        idempotency_key="k1",
        request_id="req_1",
        item_id="i1",
        kind="impression",
        event_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    result = svc.write(event)
    assert result.replayed is False


def test_feedback_service_rejects_naive_datetime() -> None:
    svc = FeedbackService(InMemoryFeedbackStore())
    event = FeedbackEvent(
        idempotency_key="k1",
        request_id="req_1",
        item_id="i1",
        kind="click",
        event_at=datetime(2024, 1, 1),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        svc.write(event)
