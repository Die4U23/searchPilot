"""POST /feedback（幂等写入）。"""

from __future__ import annotations

from fastapi import APIRouter

from searchpilot.api.deps import FeedbackDep, ItemsDep
from searchpilot.contracts import ErrorResponse, FeedbackRequest, FeedbackResponse
from searchpilot.errors import ITEM_NOT_FOUND, NOT_READY, ApiError
from searchpilot.observability import get_request_id, new_request_id
from searchpilot.ports import FeedbackEvent

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    status_code=201,
    response_model=FeedbackResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def feedback(body: FeedbackRequest, store: FeedbackDep, items: ItemsDep) -> FeedbackResponse:
    if store is None:
        raise ApiError(NOT_READY, "feedback store is not ready", 503, retryable=True)
    # 注入了 ItemStore 才校验 item 是否存在；IdempotencyConflictError 交给全局 handler 映射为 409。
    if items is not None and not items.exists(body.item_id):
        raise ApiError(ITEM_NOT_FOUND, "item does not exist", 404)

    written = store.write(
        FeedbackEvent(
            idempotency_key=str(body.idempotency_key),
            request_id=body.request_id,
            item_id=body.item_id,
            kind=body.kind,
            event_at=body.event_at,
            user_id=body.user_id,
            position=body.position,
            model_version=body.model_version,
        )
    )
    return FeedbackResponse(
        request_id=get_request_id() or new_request_id(),
        event_id=written.event_id,
        replayed=written.replayed,
        received_at=written.received_at,
    )
