"""HTTP 请求/响应模型（Pydantic v2），字段与限制见 docs/dev/build-plan.md 第 4 节。

模型是静态的，而 query 长度、limit 上限来自 Settings：这里只设置安全的硬上限，
实际阈值（默认 256 / 100）由路由读取 Settings 后二次校验并抛 ApiError(INVALID_INPUT)。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from searchpilot.ports import FeedbackKind, RecommendChannel, SearchChannel, SearchMode

ITEM_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,128}$"
USER_ID_MAX_CHARS = 64
# 仅用于挡住极端输入；业务上限在路由中按配置校验。
HARD_MAX_QUERY_CHARS = 4096
HARD_MAX_LIMIT = 1000


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Response(BaseModel):
    request_id: str


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------


class SearchFilters(_Request):
    category: str | None = Field(default=None, min_length=1, max_length=128)


class SearchRequest(_Request):
    query: str = Field(min_length=1, max_length=HARD_MAX_QUERY_CHARS)
    limit: int = Field(default=10, ge=1, le=HARD_MAX_LIMIT)
    mode: SearchMode = "bm25"
    filters: SearchFilters = Field(default_factory=SearchFilters)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class SearchHitOut(BaseModel):
    item_id: str
    score: float
    rank: int
    channel: SearchChannel


class SearchResponse(_Response):
    normalized_query: str
    model_version: str
    mode: SearchMode
    results: list[SearchHitOut]


# ---------------------------------------------------------------------------
# 回退推荐
# ---------------------------------------------------------------------------


class RecommendRequest(_Request):
    user_id: str = Field(min_length=1, max_length=USER_ID_MAX_CHARS)
    limit: int = Field(default=10, ge=1, le=HARD_MAX_LIMIT)
    context: dict[str, Any] | None = None


class RecommendHitOut(BaseModel):
    item_id: str
    score: float
    rank: int
    channel: RecommendChannel


class RecommendResponse(_Response):
    model_version: str
    channel: RecommendChannel
    cold_start: bool
    results: list[RecommendHitOut]


# ---------------------------------------------------------------------------
# 反馈
# ---------------------------------------------------------------------------


class FeedbackRequest(_Request):
    idempotency_key: UUID
    request_id: str = Field(pattern=ITEM_ID_PATTERN)
    item_id: str = Field(pattern=ITEM_ID_PATTERN)
    kind: FeedbackKind
    event_at: AwareDatetime
    user_id: str | None = Field(default=None, min_length=1, max_length=USER_ID_MAX_CHARS)
    position: int | None = Field(default=None, ge=0)
    model_version: str | None = Field(default=None, min_length=1, max_length=128)


class FeedbackResponse(_Response):
    event_id: str
    replayed: bool
    received_at: datetime


# ---------------------------------------------------------------------------
# 健康检查与错误信封
# ---------------------------------------------------------------------------


class LiveResponse(BaseModel):
    status: str = "ok"


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, bool]
    error: ErrorBody | None = None


# ---------------------------------------------------------------------------
# CTR 打分
# ---------------------------------------------------------------------------


class CtrCandidate(_Request):
    item_id: str = Field(pattern=ITEM_ID_PATTERN)
    bid: float | None = Field(default=None, gt=0)


class CtrScoreRequest(_Request):
    user_id: str = Field(min_length=1, max_length=USER_ID_MAX_CHARS)
    candidates: list[CtrCandidate] = Field(min_length=1, max_length=100)
    calibrated: bool = True


class CtrScoreHitOut(BaseModel):
    item_id: str
    pctr: float
    pctr_calibrated: float
    ecpm: float | None


class CtrScoreResponse(_Response):
    model_version: str
    calibrator_version: str
    results: list[CtrScoreHitOut]
