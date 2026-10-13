"""在 labels.csv 与补标并集上对照四路检索。

并集是 ``labels.csv`` 加上 ``labels_tool.csv``。两份文件的
``(query_id, item_id)`` 不能重叠。不改 ``labels.csv``。
train 是 LTR 的训练切分，表内数字不能当验收。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from searchpilot.eval.metrics import (
    judged_ranking,
    mean_defined,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from searchpilot.eval.search_eval import (
    EvalConfig,
    EvalQuery,
    load_labels,
    load_queries,
    merge_label_maps,
)
from searchpilot.ports import SearchMode, SearchPort
from searchpilot.search.service import SearchNotReadyError, build_search_service

MODES: tuple[SearchMode, ...] = ("bm25", "vector", "hybrid", "ltr")
SPLITS = ("test", "val", "train")
DETAIL_SPLITS = ("test", "val")
QUERY_TYPES = (
    "exact_entity",
    "synonym",
    "multi_condition",
    "misspelling_or_abbrev",
    "long_tail_popular_distractor",
    "no_answer",
)
CONFIG = EvalConfig()


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _positive(labels: dict[str, dict[str, int]], query_ids: set[str]) -> int:
    return sum(
        1
        for query_id, grades in labels.items()
        if query_id in query_ids
        for grade in grades.values()
        if grade > 0
    )


def _overlap(
    left: dict[str, dict[str, int]], right: dict[str, dict[str, int]]
) -> list[tuple[str, str]]:
    shared: list[tuple[str, str]] = []
    for query_id, grades in left.items():
        other = right.get(query_id, {})
        shared.extend((query_id, item_id) for item_id in grades if item_id in other)
    return shared


def _search_once(
    service: SearchPort, queries: list[EvalQuery], mode: SearchMode
) -> dict[str, list[str]]:
    ranked: dict[str, list[str]] = {}
    for query in queries:
        result = service.search(query.query_text, CONFIG.search_limit, mode)
        ranked[query.query_id] = [hit.item_id for hit in result.hits]
    return ranked


def _score(
    queries: list[EvalQuery],
    ranked: dict[str, list[str]],
    labels: dict[str, dict[str, int]],
) -> tuple[float | None, float | None, float | None, int]:
    ndcg: list[float | None] = []
    mrr: list[float | None] = []
    recall: list[float | None] = []
    for query in queries:
        grades = labels.get(query.query_id, {})
        order = ranked[query.query_id]
        ndcg.append(ndcg_at_k(order, grades, CONFIG.ndcg_k))
        mrr.append(reciprocal_rank(order, grades, CONFIG.mrr_k))
        recall.append(recall_at_k(order, grades, CONFIG.recall_k))
    skipped = sum(value is None for value in ndcg)
    return mean_defined(ndcg), mean_defined(mrr), mean_defined(recall), skipped


def _score_judged(
    queries: list[EvalQuery],
    ranked: dict[str, list[str]],
    labels: dict[str, dict[str, int]],
) -> tuple[float | None, float | None]:
    """未标注文档移出名单。明确 0 分仍留在原位置。"""
    ndcg: list[float | None] = []
    mrr: list[float | None] = []
    for query in queries:
        grades = labels.get(query.query_id, {})
        order = judged_ranking(ranked[query.query_id], grades)
        ndcg.append(ndcg_at_k(order, grades, CONFIG.ndcg_k))
        mrr.append(reciprocal_rank(order, grades, CONFIG.mrr_k))
    return mean_defined(ndcg), mean_defined(mrr)


def _per_query_ndcg(
    queries: list[EvalQuery],
    ranked: dict[str, list[str]],
    labels: dict[str, dict[str, int]],
    *,
    judged: bool,
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for query in queries:
        grades = labels.get(query.query_id, {})
        order = ranked[query.query_id]
        if judged:
            order = judged_ranking(order, grades)
        scores[query.query_id] = ndcg_at_k(order, grades, CONFIG.ndcg_k)
    return scores


def _mean_for_type(
    queries: list[EvalQuery],
    scores: dict[str, float | None],
    query_type: str,
) -> tuple[float | None, int]:
    chosen = [query.query_id for query in queries if query.query_type == query_type]
    values = [scores[query_id] for query_id in chosen]
    counted = sum(value is not None for value in values)
    return mean_defined(values), counted


def _type_rows(
    queries: list[EvalQuery],
    scores_by_mode: dict[str, dict[str, float | None]],
    split: str,
) -> tuple[list[str], list[str]]:
    selected = [query for query in queries if query.split == split]
    rows: list[str] = []
    behind_vector: list[str] = []
    behind_bm25: list[str] = []
    for query_type in QUERY_TYPES:
        cells: list[str] = []
        means: dict[str, float | None] = {}
        counted = 0
        for mode in MODES:
            value, counted = _mean_for_type(selected, scores_by_mode[mode], query_type)
            means[mode] = value
            cells.append(_fmt(value))
        rows.append(f"| {split} | {query_type} | {counted} | {' | '.join(cells)} |")
        ltr = means["ltr"]
        vector = means["vector"]
        bm25 = means["bm25"]
        if ltr is not None and vector is not None and ltr < vector:
            behind_vector.append(query_type)
        if ltr is not None and bm25 is not None and ltr < bm25:
            behind_bm25.append(query_type)
    return rows, behind_vector, behind_bm25


def render(service: SearchPort, queries_path: Path, base_path: Path, tool_path: Path) -> str:
    queries = load_queries(queries_path)
    base = load_labels(base_path)
    tool = load_labels(tool_path)
    shared = _overlap(base, tool)
    if shared:
        sample = ", ".join(f"{query_id}/{item_id}" for query_id, item_id in shared[:5])
        raise ValueError(f"{len(shared)} overlapping label pairs, e.g. {sample}")
    merged = merge_label_maps(base, tool)
    by_split = {split: [query for query in queries if query.split == split] for split in SPLITS}
    lines = [
        "# 补标并集上的检索对照",
        "",
        "2026-10-11。每一行检索只跑一次，再用两套标注打分。",
        f"基础标注是 `{base_path.as_posix()}`，补标是 `{tool_path.as_posix()}`。",
        "两份文件没有相同的 `(query_id, item_id)`。并集不写回 `labels.csv`。",
        "未出现在这两份文件里的文档仍按 0 分。nDCG / MRR / Recall 跳过没有相关文档的查询。",
        "train 是 LTR 的训练切分，不能当验收。val 和 test 没有进入 LTR 损失。",
        "",
    ]
    for split in SPLITS:
        ids = {query.query_id for query in by_split[split]}
        lines.append(
            f"`{split}` 有关文档：基础 {_positive(base, ids)}，并集 {_positive(merged, ids)}。"
        )
    lines.extend(
        [
            "",
            "## 总体",
            "",
            "| split | mode | nDCG@10 基础 | nDCG@10 并集 | MRR@10 基础 | MRR@10 并集 | "
            "Recall@50 基础 | Recall@50 并集 | 跳过 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    notes: list[str] = []
    ndcg_base: dict[str, list[tuple[str, float | None]]] = {split: [] for split in SPLITS}
    ndcg_merged: dict[str, list[tuple[str, float | None]]] = {split: [] for split in SPLITS}
    ndcg_judged: dict[str, list[tuple[str, float | None]]] = {split: [] for split in SPLITS}
    judged_rows: list[str] = []
    judged_scores: dict[str, dict[str, dict[str, float | None]]] = {
        split: {} for split in DETAIL_SPLITS
    }
    for split in SPLITS:
        selected = by_split[split]
        for mode in MODES:
            try:
                ranked = _search_once(service, selected, mode)
            except SearchNotReadyError as exc:
                lines.append(f"| {split} | {mode} | n/a | n/a | n/a | n/a | n/a | n/a | |")
                judged_rows.append(f"| {split} | {mode} | n/a | n/a | n/a | n/a |")
                notes.append(f"`{split}` / `{mode}` 未就绪：{exc}")
                continue
            base_metrics = _score(selected, ranked, base)
            merged_metrics = _score(selected, ranked, merged)
            judged_ndcg, judged_mrr = _score_judged(selected, ranked, merged)
            ndcg_base[split].append((mode, base_metrics[0]))
            ndcg_merged[split].append((mode, merged_metrics[0]))
            ndcg_judged[split].append((mode, judged_ndcg))
            if split in judged_scores:
                judged_scores[split][mode] = _per_query_ndcg(selected, ranked, merged, judged=True)
            judged_rows.append(
                f"| {split} | {mode} | {_fmt(merged_metrics[0])} | {_fmt(judged_ndcg)} | "
                f"{_fmt(merged_metrics[1])} | {_fmt(judged_mrr)} |"
            )
            if base_metrics[3] != merged_metrics[3]:
                skipped = f"{base_metrics[3]}/{merged_metrics[3]}"
            else:
                skipped = str(base_metrics[3])
            lines.append(
                f"| {split} | {mode} | {_fmt(base_metrics[0])} | {_fmt(merged_metrics[0])} | "
                f"{_fmt(base_metrics[1])} | {_fmt(merged_metrics[1])} | "
                f"{_fmt(base_metrics[2])} | {_fmt(merged_metrics[2])} | {skipped} |"
            )
    lines.extend(
        [
            "",
            "跳过列是没有相关文档的查询数。基础和并集不同时写成 `基础/并集`。",
            f"检索深度 {CONFIG.search_limit}。",
            "",
            "## 排序",
            "",
        ]
    )
    for split in SPLITS:
        lines.append(f"- `{split}` 基础 nDCG@10：{_order(ndcg_base[split])}")
        lines.append(f"- `{split}` 并集 nDCG@10：{_order(ndcg_merged[split])}")
    held = _names(ndcg_merged["test"]) == _names(ndcg_merged["val"])
    if held and _names(ndcg_merged["test"]):
        lines.append("")
        lines.append(
            "test 和 val 都没有进入 LTR 损失。两边的并集 nDCG@10 名次相同："
            f"{' > '.join(_names(ndcg_merged['test']))}。"
        )
    base_winner = _names(ndcg_base["val"])[:1]
    merged_winner = _names(ndcg_merged["val"])[:1]
    if base_winner and merged_winner and base_winner != merged_winner:
        lines.append(
            f"val 在基础标注上 nDCG@10 最高的是 {base_winner[0]}，并集后变成 {merged_winner[0]}。"
        )
    lines.append("train 名次只作诊断，不能用来声称 LTR 或某一路检索已通过验收。")
    lines.extend(
        [
            "",
            "## 并集上只评已标注文档",
            "",
            "未标注文档从名单里拿掉，相对顺序不变，不另打分。明确 0 分留在原位置。",
            "并集没有覆盖 LTR 自己挖出来的文档，所以这一列只说明已标注文档之间的顺序。",
            "",
            "| split | mode | nDCG@10 未标注当 0 | nDCG@10 只评已标注 | "
            "MRR@10 未标注当 0 | MRR@10 只评已标注 |",
            "|---|---|---:|---:|---:|---:|",
            *judged_rows,
            "",
        ]
    )
    for split in ("test", "val"):
        lines.append(f"- `{split}` 只评已标注 nDCG@10：{_order(ndcg_judged[split])}")
    lines.extend(
        [
            "",
            "## 只评已标注，按查询类型",
            "",
            "计入列是有相关文档的查询数。`no_answer` 不进入 nDCG。",
            "val 里 exact_entity 和 synonym 各只有 1 条计入查询。",
            "",
            "| split | query_type | 计入 | bm25 | vector | hybrid | ltr |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    reported: list[tuple[str, list[str], list[str]]] = []
    for split in DETAIL_SPLITS:
        if set(judged_scores[split]) != set(MODES):
            notes.append(f"`{split}` 有模式未就绪，查询类型表跳过。")
            continue
        rows, behind_vector, behind_bm25 = _type_rows(queries, judged_scores[split], split)
        lines.extend(rows)
        reported.append((split, behind_vector, behind_bm25))
    lines.append("")
    for split, behind_vector, behind_bm25 in reported:
        if behind_vector:
            lines.append(f"{split} 上 LTR 低于向量的类型：{'、'.join(behind_vector)}。")
        else:
            lines.append(f"{split} 上没有 LTR 低于向量的类型。")
        if behind_bm25:
            lines.append(f"{split} 上 LTR 低于 BM25 的类型：{'、'.join(behind_bm25)}。")
        else:
            lines.append(f"{split} 上没有 LTR 低于 BM25 的类型。")
    zero_order = _names(ndcg_merged["test"])
    judged_order = _names(ndcg_judged["test"])
    if zero_order and judged_order and zero_order != judged_order:
        lines.append(
            "test 上，未标注当 0 的名次是 "
            f"{' > '.join(zero_order)}，只评已标注变成 {' > '.join(judged_order)}。"
        )
    elif zero_order and judged_order:
        lines.append(f"test 上两种口径的名次相同：{' > '.join(judged_order)}。")
    val_zero = _names(ndcg_merged["val"])
    val_judged = _names(ndcg_judged["val"])
    if val_zero and val_judged and val_zero == val_judged:
        lines.append(f"val 上两种口径的名次相同：{' > '.join(val_judged)}。")
    elif val_zero and val_judged:
        lines.append(
            "val 上，未标注当 0 的名次是 "
            f"{' > '.join(val_zero)}，只评已标注变成 {' > '.join(val_judged)}。"
        )
    lines.append("只评已标注不能当成某一路检索已经超过另一路的验收。")
    lines.append("现有 LTR 权重的覆盖见 `docs/data/ltr-pool-features.md`。")
    lines.append("LTR 低于 BM25 的逐条对照见 `docs/data/ltr-query-gaps.md`。")
    lines.append("")
    lines.extend(notes)
    if notes:
        lines.append("")
    return "\n".join(lines)


def _names(pairs: list[tuple[str, float | None]]) -> tuple[str, ...]:
    usable = [(mode, value) for mode, value in pairs if value is not None]
    usable.sort(key=lambda item: (-item[1], item[0]))
    return tuple(mode for mode, _value in usable)


def _order(pairs: list[tuple[str, float | None]]) -> str:
    usable = [(mode, value) for mode, value in pairs if value is not None]
    usable.sort(key=lambda item: (-item[1], item[0]))
    if not usable:
        return "n/a"
    return " > ".join(f"{mode} {_fmt(value)}" for mode, value in usable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--tool-labels", type=Path, default=Path("data/queries/labels_tool.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/pool-label-eval.md"))
    args = parser.parse_args()
    service = build_search_service(args.artifact_dir)
    text = render(service, args.queries, args.labels, args.tool_labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
