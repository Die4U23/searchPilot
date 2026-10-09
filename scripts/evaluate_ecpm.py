"""用测试集 CTR 分数做合成出价 eCPM 排序对照。

用法::

    python scripts/evaluate_ecpm.py \\
        --scores artifacts/ctr/test_scores.parquet \\
        --out docs/data/ecpm-report.md

分数文件不存在时打印缺失并退出码 2，不编造分数。
出价 ``source=synthetic``，结论不是真实广告收益。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from searchpilot.ctr.ecpm import compare_ecpm, format_ecpm_markdown

DEFAULT_SCORES = Path("artifacts/ctr/test_scores.parquet")
DEFAULT_OUT = Path("docs/data/ecpm-report.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare raw vs calibrated eCPM ranking with synthetic bids"
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.scores.is_file():
        print(f"缺失: {args.scores}", file=sys.stderr)
        return 2

    try:
        frame = pd.read_parquet(args.scores)
        result = compare_ecpm(frame)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_ecpm_markdown(result), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"source={result.source} impressions={result.impression_count}")
    print(f"agreement_rate={result.agreement_rate:.4f}")
    print(f"raw proxy_revenue={result.raw.proxy_revenue:.6f} click_hits={result.raw.click_hits}")
    print(
        "calibrated proxy_revenue="
        f"{result.calibrated.proxy_revenue:.6f} click_hits={result.calibrated.click_hits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
