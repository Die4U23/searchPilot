"""模块间契约：值对象与 Protocol。

这是并行开发的锚点，各模块只依赖这里的类型，不互相 import 实现。
修改本文件需要技术主管确认（见 docs/dev/build-plan.md）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

# ---------------------------------------------------------------------------
# 枚举字面量
# ---------------------------------------------------------------------------

SearchMode = Literal["bm25", "vector", "hybrid", "ltr"]
SearchChannel = Literal["bm25", "vector", "rrf", "ltr"]
RecommendChannel = Literal["popular", "itemcf"]
FeedbackKind = Literal["impression", "click", "like", "hide"]
QueryType = Literal[
    "exact_entity",
    "synonym",
    "multi_condition",
    "misspelling_or_abbrev",
    "no_answer",
    "long_tail_popular_distractor",
]

# ---------------------------------------------------------------------------
# 内容与搜索
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    """搜索语料中的一条文档（对应 items.parquet 的一行）。"""

    item_id: str
    title: str
    abstract: str
    category: str
    subcategory: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: str
    score: float
    rank: int  # 从 1 开始
    channel: SearchChannel


@dataclass(frozen=True, slots=True)
class SearchResult:
    normalized_query: str
    hits: tuple[SearchHit, ...]
    model_version: str
    mode: SearchMode


class SearchPort(Protocol):
    """在线搜索入口。实现方：searchpilot.search.service。"""

    def search(self, query: str, limit: int, mode: SearchMode) -> SearchResult: ...

    @property
    def model_version(self) -> str: ...

    @property
    def vector_ready(self) -> bool: ...

    @property
    def ltr_ready(self) -> bool: ...


class ItemStore(Protocol):
    """按 ID 读取文档。实现方：searchpilot.db（PostgreSQL）或内存实现。"""

    def get(self, item_id: str) -> Document | None: ...

    def exists(self, item_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# 回退推荐
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecommendationHit:
    item_id: str
    score: float
    rank: int  # 从 1 开始
    channel: RecommendChannel


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    hits: tuple[RecommendationHit, ...]
    model_version: str
    channel: RecommendChannel
    cold_start: bool


class RecommendPort(Protocol):
    """无 query 或零结果时的回退推荐。实现方：searchpilot.recommend.service。"""

    def recommend(
        self, user_id: str, limit: int, exclude: frozenset[str]
    ) -> RecommendationResult: ...

    @property
    def model_version(self) -> str: ...


class HistoryStore(Protocol):
    """用户点击历史（按时间升序的 item_id 序列）。"""

    def get_history(self, user_id: str) -> tuple[str, ...]: ...


# ---------------------------------------------------------------------------
# 反馈事件
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    idempotency_key: str  # 客户端提供的 UUID 字符串
    request_id: str  # 产生该曝光/点击的服务请求 ID
    item_id: str
    kind: FeedbackKind
    event_at: datetime  # 必须带时区
    user_id: str | None = None
    position: int | None = None


@dataclass(frozen=True, slots=True)
class FeedbackWriteResult:
    event_id: str
    replayed: bool  # True 表示命中幂等键、返回的是原结果
    received_at: datetime


class IdempotencyConflictError(Exception):
    """同一幂等键携带了不同的事件内容。"""


def feedback_content_hash(event: FeedbackEvent) -> str:
    """幂等比较用的规范化内容摘要；**所有** FeedbackStore 实现必须用这一个函数。

    规范化规则（任何实现不得自行变体）：
    1. ``event_at`` 必须带时区，否则抛 ``ValueError``；
    2. 六元组顺序固定：``request_id, item_id, kind, event_at, user_id, position``；
    3. ``event_at`` 转 UTC 后 ``isoformat(timespec="microseconds")``（始终六位微秒 + ``+00:00``）；
    4. 编码为 JSON 数组，``ensure_ascii=False``、``separators=(",", ":")``、``allow_nan=False``；
    5. 对 UTF-8 字节做 SHA-256，返回十六进制。
    """
    if event.event_at.tzinfo is None or event.event_at.utcoffset() is None:
        raise ValueError("event_at must be timezone-aware")
    normalized = [
        event.request_id,
        event.item_id,
        event.kind,
        event.event_at.astimezone(UTC).isoformat(timespec="microseconds"),
        event.user_id,
        event.position,
    ]
    payload = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FeedbackStore(Protocol):
    """幂等写入反馈事件。实现方：searchpilot.feedback.memory 与 searchpilot.db.feedback_store。"""

    def write(self, event: FeedbackEvent) -> FeedbackWriteResult: ...


# ---------------------------------------------------------------------------
# 就绪探针
# ---------------------------------------------------------------------------


class ReadinessProbe(Protocol):
    """返回各依赖的就绪状态，例如 {"database": True, "search_index": False}。"""

    def check(self) -> Mapping[str, bool]: ...
