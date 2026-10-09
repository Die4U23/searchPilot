"""POST /search。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from fastapi import APIRouter

from searchpilot.api.deps import ItemsDep, SearchDep, SettingsDep
from searchpilot.contracts import ErrorResponse, SearchHitOut, SearchRequest, SearchResponse
from searchpilot.errors import INVALID_INPUT, NOT_READY, ApiError
from searchpilot.observability import get_request_id, new_request_id
from searchpilot.ports import ItemStore, SearchHit, SearchMode, SearchResult

router = APIRouter(tags=["search"])


@runtime_checkable
class _FilterableSearch(Protocol):
    """搜索服务可选实现的索引内过滤扩展（BM25SearchService 提供）；不改 SearchPort 协议。"""

    def search_with_filters(
        self,
        query: str,
        limit: int,
        mode: SearchMode,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> SearchResult: ...


def _filter_by_category(
    hits: tuple[SearchHit, ...], category: str, items: ItemStore, limit: int
) -> list[SearchHit]:
    wanted = category.casefold()
    kept: list[SearchHit] = []
    for hit in hits:
        doc = items.get(hit.item_id)
        if doc is not None and doc.category.casefold() == wanted:
            kept.append(hit)
            if len(kept) == limit:
                break
    return kept


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def search(
    body: SearchRequest,
    settings: SettingsDep,
    service: SearchDep,
    items: ItemsDep,
) -> SearchResponse:
    if len(body.query) > settings.max_query_chars:
        raise ApiError(INVALID_INPUT, f"query exceeds {settings.max_query_chars} characters", 422)
    if body.limit > settings.max_limit:
        raise ApiError(INVALID_INPUT, f"limit must be between 1 and {settings.max_limit}", 422)
    if service is None:
        raise ApiError(NOT_READY, "search service is not ready", 503, retryable=True)

    category = body.filters.category
    if category is None:
        result = service.search(body.query, body.limit, body.mode)
        hits = list(result.hits[: body.limit])
    elif isinstance(service, _FilterableSearch):
        # 优先用索引内过滤：在打分阶段就排除其他类目，limit 内结果完整。
        result = service.search_with_filters(
            body.query, body.limit, body.mode, filters={"category": category}
        )
        hits = list(result.hits[: body.limit])
    else:
        # SearchPort 不支持过滤：多取一批候选，再按 ItemStore 里的 category 过滤。
        if items is None:
            raise ApiError(NOT_READY, "category filter is not available", 503, retryable=True)
        result = service.search(body.query, settings.max_limit, body.mode)
        hits = _filter_by_category(result.hits, category, items, body.limit)

    return SearchResponse(
        request_id=get_request_id() or new_request_id(),
        normalized_query=result.normalized_query,
        model_version=result.model_version,
        mode=result.mode,
        results=[
            SearchHitOut(item_id=h.item_id, score=h.score, rank=rank, channel=h.channel)
            for rank, h in enumerate(hits, start=1)
        ],
    )
