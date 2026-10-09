"""反馈写入服务薄封装。"""

from __future__ import annotations

from searchpilot.ports import FeedbackEvent, FeedbackStore, FeedbackWriteResult


class FeedbackService:
    """校验 ``event_at`` 时区后委托 ``FeedbackStore.write``。"""

    def __init__(self, store: FeedbackStore) -> None:
        self._store = store

    def write(self, event: FeedbackEvent) -> FeedbackWriteResult:
        if event.event_at.tzinfo is None or event.event_at.utcoffset() is None:
            raise ValueError("event_at must be timezone-aware")
        return self._store.write(event)
