"""POST /recommend（仅回退通道）。"""

from __future__ import annotations

from fastapi import APIRouter

from searchpilot.api.deps import RecommendDep, SettingsDep
from searchpilot.contracts import (
    ErrorResponse,
    RecommendHitOut,
    RecommendRequest,
    RecommendResponse,
)
from searchpilot.errors import INVALID_INPUT, NOT_READY, ApiError
from searchpilot.observability import get_request_id, new_request_id

router = APIRouter(tags=["recommend"])


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def recommend(
    body: RecommendRequest,
    settings: SettingsDep,
    service: RecommendDep,
) -> RecommendResponse:
    if body.limit > settings.max_limit:
        raise ApiError(INVALID_INPUT, f"limit must be between 1 and {settings.max_limit}", 422)
    if service is None:
        raise ApiError(NOT_READY, "recommend service is not ready", 503, retryable=True)

    result = service.recommend(body.user_id, body.limit, frozenset())

    # 契约要求结果不重复：服务层若漏过，这里兜底去重并重排名次。
    seen: set[str] = set()
    results: list[RecommendHitOut] = []
    for hit in result.hits:
        if hit.item_id in seen:
            continue
        seen.add(hit.item_id)
        results.append(
            RecommendHitOut(
                item_id=hit.item_id, score=hit.score, rank=len(results) + 1, channel=hit.channel
            )
        )
        if len(results) == body.limit:
            break

    return RecommendResponse(
        request_id=get_request_id() or new_request_id(),
        model_version=result.model_version,
        channel=result.channel,
        cold_start=result.cold_start,
        results=results,
    )
