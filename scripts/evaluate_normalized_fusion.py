"""分数归一化融合对照。默认融合仍是 RRF，这里只报告对照指标。"""

from __future__ import annotations

import argparse
from pathlib import Path

from searchpilot.eval.metrics import mean_defined, ndcg_at_k, recall_at_k, reciprocal_rank
from searchpilot.eval.search_eval import load_labels, load_queries
from searchpilot.search.fusion import normalized_score_fusion
from searchpilot.search.service import SearchNotReadyError, build_search_service


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate min-max score fusion on the test split")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/normalized-fusion.md"))
    args = parser.parse_args()
    try:
        service = build_search_service(args.artifact_dir)
    except SearchNotReadyError as exc:
        args.out.write_text(f"# 分数归一化融合\n\nUNRUN：{exc}\n", encoding="utf-8")
        print(f"UNRUN: {exc}")
        return 2
    if not getattr(service, "vector_ready", False):
        args.out.write_text("# 分数归一化融合\n\nUNRUN：向量索引未加载。\n", encoding="utf-8")
        print("UNRUN: vector index missing")
        return 2

    queries = [query for query in load_queries(args.queries) if query.split == "test"]
    labels = load_labels(args.labels)
    ndcgs: list[float | None] = []
    mrrs: list[float | None] = []
    recalls: list[float | None] = []
    for query in queries:
        bm25 = service.search(query.query_text, 50, "bm25")
        vector = service.search(query.query_text, 50, "vector")
        fused = normalized_score_fusion(
            [
                [(hit.item_id, hit.score) for hit in bm25.hits],
                [(hit.item_id, hit.score) for hit in vector.hits],
            ]
        )
        ranked = [item_id for item_id, _score in fused][:50]
        grades = labels.get(query.query_id, {})
        ndcgs.append(ndcg_at_k(ranked, grades, 10))
        mrrs.append(reciprocal_rank(ranked, grades, 10))
        recalls.append(recall_at_k(ranked, grades, 50))

    def fmt(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    body = "\n".join(
        [
            "# 分数归一化融合对照",
            "",
            "每一路分数先缩放到 0–1 再平均。这不是默认融合；默认仍是 RRF。",
            "test 查询与 `search_mode_compare_test.md` 相同。",
            "",
            f"- 查询数：{len(queries)}",
            f"- nDCG@10：{fmt(mean_defined(ndcgs))}",
            f"- MRR@10：{fmt(mean_defined(mrrs))}",
            f"- Recall@50：{fmt(mean_defined(recalls))}",
            "",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
