"""搜索离线评估：读查询集与标注，调用任意 ``SearchPort``，输出总体 / 分段指标与失败样本。

输入格式（build-plan 3.2）：

- ``queries.csv``：``query_id, query_text, query_type, split``
- ``labels.csv``：``query_id, item_id, grade, annotator``
  （同一 ``(query_id, item_id)`` 多条标注时取最大 grade）

指标：``ndcg@10``、``mrr@10``、``recall@50``、``precision@10``、``zero_result_rate``
（k 值可配）。nDCG / MRR / Recall 只在**有相关文档**的查询上求均值，
无相关文档的查询被跳过并计入 ``skipped_no_relevant``（理由见 ``eval.metrics``）；
零结果率在全部查询上统计，另外单独给出「有答案却零结果」的数量，它们一定进失败样本。
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

from searchpilot.eval.metrics import (
    has_relevant,
    mean_defined,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from searchpilot.ports import QueryType, SearchMode, SearchPort

VALID_QUERY_TYPES: frozenset[str] = frozenset(get_args(QueryType))
VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query_id: str
    query_text: str
    query_type: str
    split: str


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """评估参数；全部写进报告，保证可复现。"""

    mode: SearchMode = "bm25"
    ndcg_k: int = 10
    mrr_k: int = 10
    recall_k: int = 50
    precision_k: int = 10
    failure_count: int = 10
    failure_top_n: int = 10  # FR-4：失败样例附带实际 Top-10

    @property
    def search_limit(self) -> int:
        return max(self.ndcg_k, self.mrr_k, self.recall_k, self.precision_k)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return (
            f"ndcg@{self.ndcg_k}",
            f"mrr@{self.mrr_k}",
            f"recall@{self.recall_k}",
            f"precision@{self.precision_k}",
            "zero_result_rate",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ndcg_k": self.ndcg_k,
            "mrr_k": self.mrr_k,
            "recall_k": self.recall_k,
            "precision_k": self.precision_k,
            "search_limit": self.search_limit,
            "failure_count": self.failure_count,
            "failure_top_n": self.failure_top_n,
        }


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """单条查询的评估结果。"""

    query: EvalQuery
    normalized_query: str
    returned: tuple[tuple[str, float], ...]
    expected: tuple[tuple[str, int], ...]
    has_relevant: bool
    zero_result: bool
    metrics: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    query_count: int
    skipped_no_relevant: int
    answerable_zero_result_count: int
    metrics: Mapping[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "skipped_no_relevant": self.skipped_no_relevant,
            "answerable_zero_result_count": self.answerable_zero_result_count,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class FailureSample:
    query_id: str
    query_text: str
    normalized_query: str
    query_type: str
    ndcg: float | None
    zero_result: bool
    top_results: tuple[tuple[str, float], ...]
    expected: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "normalized_query": self.normalized_query,
            "query_type": self.query_type,
            "ndcg": self.ndcg,
            "zero_result": self.zero_result,
            "top_results": [{"item_id": i, "score": s} for i, s in self.top_results],
            "expected": [{"item_id": i, "grade": g} for i, g in self.expected],
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    model_version: str
    split: str | None
    config: EvalConfig
    evaluated_at: str
    overall: SegmentMetrics
    segments: Mapping[str, SegmentMetrics]
    failures: tuple[FailureSample, ...]
    per_query: tuple[QueryEvaluation, ...] = field(repr=False)

    @property
    def query_count(self) -> int:
        return self.overall.query_count

    @property
    def skipped_no_relevant(self) -> int:
        return self.overall.skipped_no_relevant


# ------------------------------------------------------------------ loading


def load_queries(path: Path, *, split: str | None = None) -> list[EvalQuery]:
    """读取 ``queries.csv``；``split`` 不为 ``None`` 时只保留该切分。按 ``query_id`` 升序返回。"""
    required = {"query_id", "query_text", "query_type", "split"}
    queries: list[EvalQuery] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            query_id = row["query_id"].strip()
            if not query_id:
                continue
            if query_id in seen:
                raise ValueError(f"duplicate query_id in {path}: {query_id!r}")
            seen.add(query_id)
            query_type = row["query_type"].strip()
            row_split = row["split"].strip()
            if query_type not in VALID_QUERY_TYPES:
                raise ValueError(f"query {query_id!r} has unknown query_type {query_type!r}")
            if row_split not in VALID_SPLITS:
                raise ValueError(f"query {query_id!r} has unknown split {row_split!r}")
            if split is not None and row_split != split:
                continue
            queries.append(EvalQuery(query_id, row["query_text"], query_type, row_split))
    return sorted(queries, key=lambda q: q.query_id)


def load_labels(path: Path) -> dict[str, dict[str, int]]:
    """读取 ``labels.csv`` 为 ``{query_id: {item_id: grade}}``；重复标注取最大 grade。"""
    required = {"query_id", "item_id", "grade"}
    labels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            query_id = row["query_id"].strip()
            item_id = row["item_id"].strip()
            if not query_id or not item_id:
                continue
            try:
                grade = int(row["grade"])
            except ValueError as exc:
                raise ValueError(f"non-integer grade for ({query_id}, {item_id})") from exc
            if not 0 <= grade <= 3:
                raise ValueError(f"grade out of range 0-3 for ({query_id}, {item_id}): {grade}")
            per_query = labels.setdefault(query_id, {})
            per_query[item_id] = max(per_query.get(item_id, 0), grade)
    return labels


# --------------------------------------------------------------- evaluation


def evaluate_query(
    port: SearchPort, query: EvalQuery, grades: Mapping[str, int], config: EvalConfig
) -> QueryEvaluation:
    result = port.search(query.query_text, config.search_limit, config.mode)
    ranked = [hit.item_id for hit in result.hits]
    returned = tuple((hit.item_id, hit.score) for hit in result.hits)
    expected = tuple(
        sorted(((i, g) for i, g in grades.items() if g > 0), key=lambda ig: (-ig[1], ig[0]))
    )
    metrics: dict[str, float | None] = {
        f"ndcg@{config.ndcg_k}": ndcg_at_k(ranked, grades, config.ndcg_k),
        f"mrr@{config.mrr_k}": reciprocal_rank(ranked, grades, config.mrr_k),
        f"recall@{config.recall_k}": recall_at_k(ranked, grades, config.recall_k),
        f"precision@{config.precision_k}": precision_at_k(ranked, grades, config.precision_k),
        "zero_result_rate": 1.0 if not ranked else 0.0,
    }
    return QueryEvaluation(
        query=query,
        normalized_query=result.normalized_query,
        returned=returned,
        expected=expected,
        has_relevant=has_relevant(grades),
        zero_result=not ranked,
        metrics=metrics,
    )


def aggregate(evaluations: Iterable[QueryEvaluation], config: EvalConfig) -> SegmentMetrics:
    items = list(evaluations)
    metrics: dict[str, float | None] = {}
    for name in config.metric_names:
        metrics[name] = mean_defined([e.metrics[name] for e in items])
    return SegmentMetrics(
        query_count=len(items),
        skipped_no_relevant=sum(1 for e in items if not e.has_relevant),
        answerable_zero_result_count=sum(1 for e in items if e.has_relevant and e.zero_result),
        metrics=metrics,
    )


def select_failures(
    evaluations: Sequence[QueryEvaluation], config: EvalConfig
) -> tuple[FailureSample, ...]:
    """nDCG@k 最低的 N 条**有相关文档**的查询（并列按 query_id 升序）；有答案却零结果的排最前。"""
    ndcg_name = f"ndcg@{config.ndcg_k}"
    candidates = [e for e in evaluations if e.has_relevant]

    def sort_key(e: QueryEvaluation) -> tuple[int, float, str]:
        value = e.metrics[ndcg_name]
        return (0 if e.zero_result else 1, value if value is not None else 0.0, e.query.query_id)

    candidates.sort(key=sort_key)
    return tuple(
        FailureSample(
            query_id=e.query.query_id,
            query_text=e.query.query_text,
            normalized_query=e.normalized_query,
            query_type=e.query.query_type,
            ndcg=e.metrics[ndcg_name],
            zero_result=e.zero_result,
            top_results=e.returned[: config.failure_top_n],
            expected=e.expected,
        )
        for e in candidates[: config.failure_count]
    )


def evaluate_search(
    port: SearchPort,
    queries: Sequence[EvalQuery],
    labels: Mapping[str, Mapping[str, int]],
    *,
    split: str | None = None,
    config: EvalConfig | None = None,
    now: datetime | None = None,
) -> EvalReport:
    """逐查询评估并聚合。``queries`` 中不属于 ``split`` 的会被过滤；无标注的查询视为无相关文档。"""
    cfg = config or EvalConfig()
    selected = [q for q in queries if split is None or q.split == split]
    selected.sort(key=lambda q: q.query_id)
    evaluations = tuple(evaluate_query(port, q, labels.get(q.query_id, {}), cfg) for q in selected)

    by_type: dict[str, list[QueryEvaluation]] = {}
    for e in evaluations:
        by_type.setdefault(e.query.query_type, []).append(e)
    segments = {name: aggregate(items, cfg) for name, items in sorted(by_type.items())}

    stamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    return EvalReport(
        model_version=port.model_version,
        split=split,
        config=cfg,
        evaluated_at=stamp,
        overall=aggregate(evaluations, cfg),
        segments=segments,
        failures=select_failures(evaluations, cfg),
        per_query=evaluations,
    )


def evaluate_from_files(
    port: SearchPort,
    queries_path: Path,
    labels_path: Path,
    *,
    split: str | None = None,
    config: EvalConfig | None = None,
) -> EvalReport:
    """文件版入口：读 CSV 后调用 :func:`evaluate_search`。"""
    queries = load_queries(queries_path, split=split)
    labels = load_labels(labels_path)
    return evaluate_search(port, queries, labels, split=split, config=config)
