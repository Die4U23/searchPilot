"""同一份 test 排序上，比较「未标注当 0」和「只保留已标注文档」的 nDCG / MRR。

第二种做法不给未标注文档打分，只把它们从名单里拿掉。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from searchpilot.eval.metrics import judged_ranking, mean_defined, ndcg_at_k, reciprocal_rank
from searchpilot.eval.search_eval import EvalConfig, load_labels, load_queries
from searchpilot.search.service import SearchNotReadyError, build_search_service

MODES = ("bm25", "vector", "hybrid", "ltr")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _mean(values: list[float | None]) -> float | None:
    return mean_defined(values)


def evaluate(artifact_dir: Path, queries_path: Path, labels_path: Path, split: str) -> str:
    service = build_search_service(artifact_dir)
    labels = load_labels(labels_path)
    queries = [query for query in load_queries(queries_path) if query.split == split]
    config = EvalConfig()
    lines = [
        "# 未标注当 0 与只评已标注文档",
        "",
        f"切分 `{split}`，查询 {len(queries)} 条。检索深度 {config.search_limit}。",
        "未标注当 0 是现有评估。只评已标注文档时，未标注的结果被移出名单，相对顺序不变，不新打分。",
        "nDCG / MRR 仍跳过没有相关文档的查询。",
        "",
        "| mode | nDCG@10 未标注当 0 | nDCG@10 只评已标注 | "
        "MRR@10 未标注当 0 | MRR@10 只评已标注 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        config = EvalConfig(mode=mode)  # type: ignore[arg-type]
        standard_ndcg: list[float | None] = []
        judged_ndcg: list[float | None] = []
        standard_mrr: list[float | None] = []
        judged_mrr: list[float | None] = []
        try:
            for query in queries:
                grades = labels.get(query.query_id, {})
                result = service.search(query.query_text, config.search_limit, mode)  # type: ignore[arg-type]
                ranked = [hit.item_id for hit in result.hits]
                condensed = judged_ranking(ranked, grades)
                standard_ndcg.append(ndcg_at_k(ranked, grades, config.ndcg_k))
                judged_ndcg.append(ndcg_at_k(condensed, grades, config.ndcg_k))
                standard_mrr.append(reciprocal_rank(ranked, grades, config.mrr_k))
                judged_mrr.append(reciprocal_rank(condensed, grades, config.mrr_k))
        except SearchNotReadyError as exc:
            lines.append(f"| {mode} | n/a | n/a | n/a | n/a |")
            lines.append("")
            lines.append(f"`{mode}` 未就绪：{exc}")
            continue
        lines.append(
            f"| {mode} | {_fmt(_mean(standard_ndcg))} | {_fmt(_mean(judged_ndcg))} | "
            f"{_fmt(_mean(standard_mrr))} | {_fmt(_mean(judged_mrr))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("docs/data/judged-only.md"))
    args = parser.parse_args()
    text = evaluate(args.artifact_dir, args.queries, args.labels, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
