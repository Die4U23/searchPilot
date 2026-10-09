"""对 dev 切分评估 Popular / ItemCF 回退推荐。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from searchpilot.recommend.evaluate import evaluate_fallback, format_eval_markdown
from searchpilot.recommend.history import load_user_history
from searchpilot.recommend.service import RecommendNotReadyError, resolve_data_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fallback recommenders")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_DATA_DIR", "./data")),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/recommend/eval_fallback.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/recommend/eval_fallback.md"),
    )
    args = parser.parse_args(argv)

    processed_dir = args.data_dir / "processed"
    try:
        data_version = resolve_data_version(processed_dir)
    except RecommendNotReadyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    version_dir = processed_dir / data_version
    impressions = pd.read_parquet(version_dir / "impressions.parquet")
    histories = load_user_history(version_dir / "user_history.parquet")

    reports = [
        evaluate_fallback(impressions, histories, data_version=data_version, model="popular"),
        evaluate_fallback(impressions, histories, data_version=data_version, model="itemcf"),
    ]

    payload = {
        "source": "searchpilot",
        "data_version": data_version,
        "reports": [r.to_json_dict() for r in reports],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(format_eval_markdown(reports), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
