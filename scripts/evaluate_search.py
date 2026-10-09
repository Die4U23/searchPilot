"""用查询集与标注离线评估 BM25 搜索产物。

用法::

    python scripts/evaluate_search.py --artifact-dir ./artifacts \
        --queries data/queries/queries.csv --labels data/queries/labels.csv \
        --split test --out artifacts/search/eval

输出 ``<out>/search_eval_<split>_<model_version>.json`` 与同名 ``.md``，并在控制台打印汇总表。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from searchpilot.eval.report import summary_lines, write_json, write_markdown
from searchpilot.eval.search_eval import EvalConfig, evaluate_from_files
from searchpilot.search.service import SearchNotReadyError, load_bm25_service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the BM25 search service offline")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_ARTIFACT_DIR", "./artifacts")),
    )
    data_dir = Path(os.environ.get("SEARCHPILOT_DATA_DIR", "./data"))
    parser.add_argument("--queries", type=Path, default=data_dir / "queries" / "queries.csv")
    parser.add_argument("--labels", type=Path, default=data_dir / "queries" / "labels.csv")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="which query split to evaluate ('all' disables filtering)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/search/eval"))
    parser.add_argument("--failures", type=int, default=10, help="number of failure samples")
    args = parser.parse_args(argv)

    for path in (args.queries, args.labels):
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
    try:
        service = load_bm25_service(args.artifact_dir)
    except SearchNotReadyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    split = None if args.split == "all" else args.split
    config = EvalConfig(failure_count=args.failures)
    try:
        report = evaluate_from_files(service, args.queries, args.labels, split=split, config=config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stem = f"search_eval_{args.split}_{report.model_version}"
    json_path = args.out / f"{stem}.json"
    md_path = args.out / f"{stem}.md"
    write_json(report, json_path)
    write_markdown(report, md_path)

    for line in summary_lines(report):
        print(line)
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
