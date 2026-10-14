"""列出 val / test 上只评已标注时 LTR 低于 BM25 的查询。

不重新训练。名次缺席 top 100 记为 101。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from searchpilot.eval.metrics import judged_ranking, ndcg_at_k
from searchpilot.eval.search_eval import (
    EvalConfig,
    EvalQuery,
    load_labels,
    load_queries,
    merge_label_maps,
)
from searchpilot.ports import SearchMode
from searchpilot.search.ltr import RETRIEVAL_DEPTH
from searchpilot.search.service import BM25SearchService, build_search_service

MODES: tuple[SearchMode, ...] = ("bm25", "vector", "hybrid", "ltr")
DETAIL_SPLITS = ("test", "val")
CONFIG = EvalConfig()
MISSING_RANK = RETRIEVAL_DEPTH + 1
SHOW = 6


def zeros_before(order: list[str], grades: dict[str, int], minimum: int) -> int | None:
    """第一篇达到 minimum 的文档前面有几篇明确 0 分。没有则返回 None。"""
    zeros = 0
    for item_id in order:
        if item_id not in grades:
            continue
        grade = grades[item_id]
        if grade >= minimum:
            return zeros
        if grade == 0:
            zeros += 1
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _rank(ranks: Mapping[str, int], item_id: str) -> int:
    return ranks.get(item_id, MISSING_RANK)


def _cell(order: list[str], grades: dict[str, int], index: int) -> str:
    judged = [item_id for item_id in order if item_id in grades]
    if index >= len(judged):
        return ""
    item_id = judged[index]
    return f"{item_id} g{grades[item_id]}"


def render(
    service: BM25SearchService, queries_path: Path, labels_path: Path, tool_path: Path
) -> str:
    queries = [query for query in load_queries(queries_path) if query.split in DETAIL_SPLITS]
    merged = merge_label_maps(load_labels(labels_path), load_labels(tool_path))
    by_split = {
        split: [query for query in queries if query.split == split] for split in DETAIL_SPLITS
    }
    ranked: dict[str, dict[str, list[str]]] = {split: {} for split in DETAIL_SPLITS}
    for split in DETAIL_SPLITS:
        for mode in MODES:
            ranked[split][mode] = {}
            for query in by_split[split]:
                result = service.search(query.query_text, CONFIG.search_limit, mode)
                ranked[split][mode][query.query_id] = [hit.item_id for hit in result.hits]

    lines = [
        "# LTR 低于 BM25 的查询",
        "",
        "2026-10-11。只评已标注：未标注文档移出名单，明确 0 分留在原位置。",
        "比较的是并集标注上的 nDCG@10。只列出 LTR 低于 BM25 的 val / test 查询。",
        "名次 101 表示不在该路 top 100。",
        "",
    ]
    losers: list[EvalQuery] = []
    for query in queries:
        grades = merged.get(query.query_id, {})
        ltr = ndcg_at_k(
            judged_ranking(ranked[query.split]["ltr"][query.query_id], grades),
            grades,
            CONFIG.ndcg_k,
        )
        bm25 = ndcg_at_k(
            judged_ranking(ranked[query.split]["bm25"][query.query_id], grades),
            grades,
            CONFIG.ndcg_k,
        )
        if ltr is not None and bm25 is not None and ltr < bm25:
            losers.append(query)
    lines.append(f"共 {len(losers)} 条。")
    lines.append("")
    lines.append("| split | query | type | bm25 | vector | hybrid | ltr |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for query in losers:
        grades = merged.get(query.query_id, {})
        cells = []
        for mode in MODES:
            value = ndcg_at_k(
                judged_ranking(ranked[query.split][mode][query.query_id], grades),
                grades,
                CONFIG.ndcg_k,
            )
            cells.append(_fmt(value))
        lines.append(
            f"| {query.split} | {query.query_id} | {query.query_type} | {' | '.join(cells)} |"
        )
    lines.append("")
    short = 0
    boosted = 0
    sections: list[str] = []
    for query in losers:
        section, judged_count, first_rank = _query_section(
            service, query, ranked[query.split], merged.get(query.query_id, {})
        )
        sections.extend(section)
        short += int(judged_count < SHOW)
        boosted += int(first_rank is not None and first_rank >= 50)
    lines.append(
        f"{len(losers)} 条里，{short} 条的 LTR 前 {CONFIG.search_limit} "
        f"里已标注文档少于 {SHOW} 篇。"
    )
    lines.append(f"{boosted} 条的已标注第一名，BM25 名次大于或等于 50。")
    lines.append("bm25_rank 权重为正，名次数字越大越靠前。")
    lines.append("因此 BM25 名次很差的文档会被抬到 grade 3 前面。")
    lines.append("把这项权重置 0 的对照见 `docs/data/ltr-rank-ablation.md`。")
    lines.append("")
    lines.extend(sections)
    return "\n".join(lines)


def _query_section(
    service: BM25SearchService,
    query: EvalQuery,
    ranked: dict[str, dict[str, list[str]]],
    grades: dict[str, int],
) -> list[str]:
    prep = service.prepare_ltr(query.query_text, sorted(grades))
    lines = [
        f"## {query.query_id} {query.split} {query.query_type}",
        "",
        query.query_text,
        "",
        "| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |",
        "|---|---:|---:|",
    ]
    for mode in ("bm25", "vector", "ltr"):
        judged = judged_ranking(ranked[mode][query.query_id], grades)
        first = zeros_before(judged, grades, 1)
        strong = zeros_before(judged, grades, 2)
        lines.append(f"| {mode} | {_fmt_count(first)} | {_fmt_count(strong)} |")
    lines.extend(
        [
            "",
            "| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    ltr_order = ranked["ltr"][query.query_id]
    for index in range(SHOW):
        ltr_item = _judged_item(ltr_order, grades, index)
        rank_bm25 = ""
        rank_vector = ""
        if ltr_item is not None:
            rank_bm25 = str(_rank(prep.signals.bm25_rank, ltr_item))
            rank_vector = str(_rank(prep.signals.vector_rank, ltr_item))
        lines.append(
            f"| {index + 1} | {_cell(ranked['bm25'][query.query_id], grades, index)} | "
            f"{_cell(ranked['vector'][query.query_id], grades, index)} | "
            f"{_cell(ltr_order, grades, index)} | {rank_bm25} | {rank_vector} |"
        )
    judged_count = sum(item_id in grades for item_id in ltr_order)
    first_item = _judged_item(ltr_order, grades, 0)
    first_rank = None if first_item is None else _rank(prep.signals.bm25_rank, first_item)
    lines.append("")
    return lines, judged_count, first_rank


def _judged_item(order: list[str], grades: dict[str, int], index: int) -> str | None:
    judged = [item_id for item_id in order if item_id in grades]
    if index >= len(judged):
        return None
    return judged[index]


def _fmt_count(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--tool-labels", type=Path, default=Path("data/queries/labels_tool.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/ltr-query-gaps.md"))
    args = parser.parse_args()
    service = build_search_service(args.artifact_dir)
    if not isinstance(service, BM25SearchService):
        raise TypeError("search service does not expose prepare_ltr")
    text = render(service, args.queries, args.labels, args.tool_labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
