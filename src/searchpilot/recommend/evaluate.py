"""回退推荐离线评估（dev 切分留一 / 时序）。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

from searchpilot.recommend.itemcf import ItemCFModel
from searchpilot.recommend.popular import PopularModel

SegmentName = Literal["cold_user", "history_present"]
ModelName = Literal["popular", "itemcf"]


@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    recall_at_20: float
    ndcg_at_10: float
    catalog_coverage_at_10: float
    user_count: int


@dataclass(frozen=True, slots=True)
class EvalReport:
    source: str
    model_version: str
    model: ModelName
    metrics: SegmentMetrics
    segments: dict[SegmentName, SegmentMetrics]
    evaluated_at: datetime
    user_count: int
    catalog_size: int

    def to_json_dict(self) -> dict[str, Any]:
        def seg_dict(m: SegmentMetrics) -> dict[str, Any]:
            return {
                "recall_at_20": m.recall_at_20,
                "ndcg_at_10": m.ndcg_at_10,
                "catalog_coverage_at_10": m.catalog_coverage_at_10,
                "user_count": m.user_count,
            }

        return {
            "source": self.source,
            "model_version": self.model_version,
            "model": self.model,
            "metrics": seg_dict(self.metrics),
            "segments": {name: seg_dict(m) for name, m in self.segments.items()},
            "evaluated_at": self.evaluated_at.isoformat(),
            "user_count": self.user_count,
            "catalog_size": self.catalog_size,
        }


def _binary_ndcg_at_k(relevant: frozenset[str], ranked: Sequence[str], k: int) -> float:
    dcg = 0.0
    for i, item_id in enumerate(ranked[:k]):
        if item_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _recall_at_k(relevant: frozenset[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item_id in ranked[:k] if item_id in relevant)
    return hits / float(len(relevant))


def evaluate_fallback(
    impressions: pd.DataFrame,
    histories: Mapping[str, Sequence[str]],
    *,
    data_version: str,
    model: ModelName,
    k_recall: int = 20,
    k_ndcg: int = 10,
    k_coverage: int = 10,
    recommend_limit: int = 20,
) -> EvalReport:
    """对 dev 中有点击的用户评估；训练侧仅用 train 历史 / 计数。"""
    train_impressions = impressions.loc[impressions["split"] == "train"]
    dev_clicks = impressions.loc[(impressions["split"] == "dev") & (impressions["clicked"] == 1)]

    catalog = frozenset(str(x) for x in impressions["item_id"].unique())

    popular = PopularModel()
    popular.fit(train_impressions)

    train_histories = dict(histories)
    itemcf = ItemCFModel()
    itemcf.fit(train_histories)

    dev_truth: dict[str, frozenset[str]] = {}
    for user_id, group in dev_clicks.groupby("user_id", sort=False):
        items = frozenset(str(x) for x in group["item_id"].tolist())
        if items:
            dev_truth[str(user_id)] = items

    segment_users: dict[SegmentName, list[str]] = {
        "cold_user": [],
        "history_present": [],
    }
    for user_id in dev_truth:
        hist = train_histories.get(user_id, ())
        if hist:
            segment_users["history_present"].append(user_id)
        else:
            segment_users["cold_user"].append(user_id)

    def recommend_for_user(user_id: str) -> list[str]:
        history = tuple(str(x) for x in train_histories.get(user_id, ()))
        blocked = frozenset(history)
        if model == "itemcf" and history:
            pairs = itemcf.predict(history, blocked, recommend_limit)
            if pairs:
                return [item_id for item_id, _ in pairs]
        pairs = popular.predict(blocked, recommend_limit)
        return [item_id for item_id, _ in pairs]

    def eval_users(user_ids: list[str]) -> SegmentMetrics:
        if not user_ids:
            return SegmentMetrics(0.0, 0.0, 0.0, 0)

        recalls: list[float] = []
        ndcgs: list[float] = []
        covered: set[str] = set()

        for user_id in user_ids:
            ranked = recommend_for_user(user_id)
            rel = dev_truth[user_id]
            recalls.append(_recall_at_k(rel, ranked, k_recall))
            ndcgs.append(_binary_ndcg_at_k(rel, ranked, k_ndcg))
            covered.update(ranked[:k_coverage])

        coverage = len(covered) / float(len(catalog)) if catalog else 0.0
        return SegmentMetrics(
            recall_at_20=sum(recalls) / len(recalls),
            ndcg_at_10=sum(ndcgs) / len(ndcgs),
            catalog_coverage_at_10=coverage,
            user_count=len(user_ids),
        )

    all_users = list(dev_truth.keys())
    overall = eval_users(all_users)
    segments: dict[SegmentName, SegmentMetrics] = {
        "cold_user": eval_users(segment_users["cold_user"]),
        "history_present": eval_users(segment_users["history_present"]),
    }

    model_version = f"{model}-{data_version}"
    return EvalReport(
        source="searchpilot",
        model_version=model_version,
        model=model,
        metrics=overall,
        segments=segments,
        evaluated_at=datetime.now(tz=UTC),
        user_count=len(all_users),
        catalog_size=len(catalog),
    )


def format_eval_markdown(reports: Sequence[EvalReport]) -> str:
    """简短 markdown 表（仅 searchpilot 来源指标）。"""
    lines = [
        "| model | segment | recall@20 | ndcg@10 | coverage@10 | users |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        for segment_name, metrics in report.segments.items():
            lines.append(
                f"| {report.model} | {segment_name} | "
                f"{metrics.recall_at_20:.4f} | {metrics.ndcg_at_10:.4f} | "
                f"{metrics.catalog_coverage_at_10:.4f} | {metrics.user_count} |"
            )
        m = report.metrics
        lines.append(
            f"| {report.model} | **all** | "
            f"{m.recall_at_20:.4f} | {m.ndcg_at_10:.4f} | "
            f"{m.catalog_coverage_at_10:.4f} | {m.user_count} |"
        )
    return "\n".join(lines) + "\n"
