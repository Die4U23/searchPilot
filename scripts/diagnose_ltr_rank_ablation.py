"""不重新训练，只改已保存 LTR 的 bm25_rank 打分方式。

两种对照都用同一批 RRF 候选：

- 名次权重置 0
- 缺席 top 100 的名次 101 改成训练均值，使这项 z-score 为 0
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from searchpilot.eval.metrics import judged_ranking, mean_defined, ndcg_at_k
from searchpilot.eval.search_eval import (
    EvalConfig,
    EvalQuery,
    load_labels,
    load_queries,
    merge_label_maps,
)
from searchpilot.search.ltr import (
    FEATURE_NAMES,
    RETRIEVAL_DEPTH,
    LtrModel,
    feature_matrix,
    load_ltr,
    ltr_artifact_dir,
    rank_by_score,
)
from searchpilot.search.service import BM25SearchService, build_search_service

SPLITS = ("test", "val")
CONFIG = EvalConfig()
RANK_INDEX = FEATURE_NAMES.index("bm25_rank")
MISSING_RANK = float(RETRIEVAL_DEPTH + 1)
VARIANTS = ("original", "weight_zero", "missing_at_mean")


def zero_weight(weights: tuple[float, ...], index: int) -> tuple[float, ...]:
    """把一个权重改成 0，不改入参。"""
    updated = list(weights)
    updated[index] = 0.0
    return tuple(updated)


def fill_missing_rank(raw: np.ndarray, index: int, missing: float, fill: float) -> np.ndarray:
    """缺席名次改成 fill。真实名次不动，也不改入参。"""
    updated = np.array(raw, dtype=np.float64, copy=True)
    column = updated[:, index]
    updated[:, index] = np.where(np.isclose(column, missing), fill, column)
    return updated


def _with_weights(model: LtrModel, weights: tuple[float, ...]) -> LtrModel:
    return replace(model, weights=weights)


def _categories(service: BM25SearchService) -> dict[str, str]:
    index = service.index
    return {
        item_id: category
        for item_id, category in zip(index.item_ids, index.categories, strict=True)
    }


def _order(item_ids: tuple[str, ...] | list[str], scores: np.ndarray) -> list[str]:
    if len(item_ids) == 0:
        return []
    return [item_id for item_id, _score in rank_by_score(item_ids, scores, CONFIG.search_limit)]


def _mean(
    queries: list[EvalQuery], orders: dict[str, list[str]], labels: dict, *, judged: bool
) -> float | None:
    values: list[float | None] = []
    for query in queries:
        grades = labels.get(query.query_id, {})
        order = orders[query.query_id]
        if judged:
            order = judged_ranking(order, grades)
        values.append(ndcg_at_k(order, grades, CONFIG.ndcg_k))
    return mean_defined(values)


def _losses(
    queries: list[EvalQuery],
    left: dict[str, list[str]],
    right: dict[str, list[str]],
    labels: dict,
) -> int:
    count = 0
    for query in queries:
        grades = labels.get(query.query_id, {})
        ltr = ndcg_at_k(judged_ranking(left[query.query_id], grades), grades, CONFIG.ndcg_k)
        bm25 = ndcg_at_k(judged_ranking(right[query.query_id], grades), grades, CONFIG.ndcg_k)
        if ltr is not None and bm25 is not None and ltr < bm25:
            count += 1
    return count


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render(
    service: BM25SearchService,
    model: LtrModel,
    queries_path: Path,
    labels_path: Path,
    tool_path: Path,
) -> str:
    queries = [query for query in load_queries(queries_path) if query.split in SPLITS]
    labels = merge_label_maps(load_labels(labels_path), load_labels(tool_path))
    categories = _categories(service)
    zeroed = _with_weights(model, zero_weight(model.weights, RANK_INDEX))
    by_split = {split: [query for query in queries if query.split == split] for split in SPLITS}
    bm25: dict[str, dict[str, list[str]]] = {split: {} for split in SPLITS}
    orders: dict[str, dict[str, dict[str, list[str]]]] = {
        name: {split: {} for split in SPLITS} for name in VARIANTS
    }
    for split in SPLITS:
        for query in by_split[split]:
            bm25_hits = service.search(query.query_text, CONFIG.search_limit, "bm25")
            bm25[split][query.query_id] = [hit.item_id for hit in bm25_hits.hits]
            prep = service.prepare_ltr(query.query_text)
            raw = feature_matrix(
                prep.candidates,
                prep.signals,
                categories=categories,
                clicks=model.clicks,
                title_chars=model.title_chars,
                first_seen_at=model.first_seen_at,
                reference_time=model.reference_time,
            )
            filled = fill_missing_rank(raw, RANK_INDEX, MISSING_RANK, model.mean[RANK_INDEX])
            orders["original"][split][query.query_id] = _order(
                prep.candidates, model.score_rows(raw)
            )
            orders["weight_zero"][split][query.query_id] = _order(
                prep.candidates, zeroed.score_rows(raw)
            )
            orders["missing_at_mean"][split][query.query_id] = _order(
                prep.candidates, model.score_rows(filled)
            )
    lines = [
        "# bm25_rank 打分对照",
        "",
        f"2026-10-11。模型 `{model.model_version}`，没有重新训练，也没有改候选。",
        "标注是 `labels.csv` 与 `labels_tool.csv` 的并集。",
        f"`bm25_rank` 训练均值 {model.mean[RANK_INDEX]:.4f}，"
        f"权重 {model.weights[RANK_INDEX]:.4f}。",
        "名次权重置 0 只把这一项乘子清掉。缺席名次改均值只替换 101，真实名次不动。",
        "",
        "| split | 口径 | 原权重 | 名次权重置 0 | 缺席名次改均值 | BM25 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        selected = by_split[split]
        for judged, label in ((False, "未标注当 0"), (True, "只评已标注")):
            cells = [
                _fmt(_mean(selected, orders[name][split], labels, judged=judged))
                for name in VARIANTS
            ]
            bm25_value = _fmt(_mean(selected, bm25[split], labels, judged=judged))
            lines.append(f"| {split} | {label} | {' | '.join(cells)} | {bm25_value} |")
    lines.extend(["", "## 只评已标注时仍低于 BM25 的查询数", ""])
    for split in SPLITS:
        selected = by_split[split]
        bits = []
        for name in VARIANTS:
            lost = _losses(selected, orders[name][split], bm25[split], labels)
            bits.append(f"{name} {lost}/{len(selected)}")
        lines.append(f"- `{split}`：{'，'.join(bits)}")
    lines.extend(
        [
            "",
            "查询数含没有相关文档的查询。低于 BM25 只统计两边 nDCG 都有定义的查询。",
            "缺席名次改均值只动 101。top 100 里偏大的名次仍被正权重往前推。",
            "",
        ]
    )
    for split in SPLITS:
        selected = by_split[split]
        raw = _mean(selected, orders["weight_zero"][split], labels, judged=False)
        raw_bm25 = _mean(selected, bm25[split], labels, judged=False)
        judged = _mean(selected, orders["weight_zero"][split], labels, judged=True)
        judged_bm25 = _mean(selected, bm25[split], labels, judged=True)
        lines.append(
            f"`{split}` 名次权重置 0，未标注当 0 是 {_fmt(raw)}，BM25 是 {_fmt(raw_bm25)}。"
        )
        lines.append(
            f"`{split}` 名次权重置 0，只评已标注是 {_fmt(judged)}，BM25 是 {_fmt(judged_bm25)}。"
        )
    lines.extend(
        [
            "",
            "其他权重是在 `bm25_rank` 还在时学的。置 0 不是新的线上模型。",
            "向量和 RRF 的并集数字见 `docs/data/pool-label-eval.md`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--tool-labels", type=Path, default=Path("data/queries/labels_tool.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/ltr-rank-ablation.md"))
    args = parser.parse_args()
    service = build_search_service(args.artifact_dir)
    if not isinstance(service, BM25SearchService) or not service.vector_ready:
        raise TypeError("vector search service is required")
    model = load_ltr(ltr_artifact_dir(args.artifact_dir))
    text = render(service, model, args.queries, args.labels, args.tool_labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
