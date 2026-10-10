"""比较现有标注和 BM25 / 向量 / RRF 前排的重合。不生成新分数。

标注候选当初主要来自 BM25 前 10。这里只统计缺口，并写出尚未标注的候选。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from searchpilot.eval.search_eval import load_labels, load_queries
from searchpilot.search.service import build_search_service

MODES = ("bm25", "vector", "hybrid")


def retrieve(service: object, query: str, limit: int) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for mode in MODES:
        result = service.search(query, limit, mode)  # type: ignore[attr-defined]
        found[mode] = [hit.item_id for hit in result.hits]
    return found


def coverage(labeled: set[str], retrieved: set[str]) -> tuple[int, int]:
    if not labeled:
        return 0, 0
    return len(labeled & retrieved), len(labeled)


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, tuple[int, int]]]:
    """按 split 汇总标注命中数。键是 ``all`` 和各个 split。"""
    groups: dict[str, list[dict[str, object]]] = {"all": rows}
    for row in rows:
        groups.setdefault(str(row["split"]), []).append(row)
    summary: dict[str, dict[str, tuple[int, int]]] = {}
    for name, group in groups.items():
        stats: dict[str, tuple[int, int]] = {}
        for kind in ("labeled", "positive"):
            for mode in MODES:
                for depth in (10, 50):
                    hit = 0
                    total = 0
                    key = f"{mode}@{depth}"
                    field = f"{kind}_{key}"
                    for row in group:
                        pair = row[field]
                        assert isinstance(pair, tuple)
                        hit += int(pair[0])
                        total += int(pair[1])
                    stats[f"{kind}:{key}"] = (hit, total)
        unlabeled = sum(int(row["unlabeled_k10"]) for row in group)
        stats["unlabeled_k10"] = (unlabeled, unlabeled)
        summary[name] = stats
    return summary


def _rate(pair: tuple[int, int]) -> str:
    hit, total = pair
    if total == 0:
        return "n/a"
    return f"{hit}/{total} = {hit / total:.4f}"


def render(summary: dict[str, dict[str, tuple[int, int]]]) -> str:
    lines = [
        "# 标注池和检索前排的重合",
        "",
        "现有 `labels.csv` 的候选主要是 BM25 前 10，种子新闻不在其中才补上。",
        "下表是这些已标注文档落在 BM25、向量、RRF 前 10 和前 50 里的比例。",
        "没有给新文档打分。`grade > 0` 记为有关。",
        "",
    ]
    for name in ("all", "train", "val", "test"):
        stats = summary.get(name)
        if not stats:
            continue
        lines.extend(
            [
                f"## {name}",
                "",
                "| 范围 | 已标注命中 | 有关文档命中 |",
                "|---|---:|---:|",
            ]
        )
        for mode in MODES:
            for depth in (10, 50):
                key = f"{mode}@{depth}"
                labeled = _rate(stats[f"labeled:{key}"])
                positive = _rate(stats[f"positive:{key}"])
                lines.append(f"| {key} | {labeled} | {positive} |")
        unlabeled, _total = stats["unlabeled_k10"]
        lines.extend(
            [
                "",
                f"三路前 10 的并集里，尚未出现在 `labels.csv` 的文档有 {unlabeled} 条。",
                "",
            ]
        )
    lines.append("尚未标注的候选在 `data/queries/candidate_pool_k10.csv`，没有 grade 列。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/label-pool.md"))
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/queries/candidate_pool_k10.csv")
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.limit < 10:
        parser.error("--limit must be at least 10")

    service = build_search_service(args.artifact_dir)
    labels = load_labels(args.labels)
    queries = load_queries(args.queries)
    rows: list[dict[str, object]] = []
    pending: list[tuple[str, str, str, str, str]] = []
    for query in queries:
        found = retrieve(service, query.query_text, args.limit)
        sets = {mode: set(ids) for mode, ids in found.items()}
        top10 = {mode: set(ids[:10]) for mode, ids in found.items()}
        graded = labels.get(query.query_id, {})
        labeled = set(graded)
        positive = {item_id for item_id, grade in graded.items() if grade > 0}
        row: dict[str, object] = {"split": query.split}
        for kind, pool in (("labeled", labeled), ("positive", positive)):
            for mode in MODES:
                row[f"{kind}_{mode}@10"] = coverage(pool, top10[mode])
                row[f"{kind}_{mode}@50"] = coverage(pool, sets[mode])
        union = top10["bm25"] | top10["vector"] | top10["hybrid"]
        missing = sorted(union - labeled)
        row["unlabeled_k10"] = len(missing)
        rows.append(row)
        for item_id in missing:
            pending.append(
                (
                    query.query_id,
                    item_id,
                    "1" if item_id in top10["bm25"] else "0",
                    "1" if item_id in top10["vector"] else "0",
                    "1" if item_id in top10["hybrid"] else "0",
                )
            )

    text = render(summarize(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_id", "item_id", "in_bm25", "in_vector", "in_hybrid"))
        writer.writerows(pending)
    print(text)
    print(f"wrote {args.out}")
    print(f"wrote {args.candidates} rows={len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
