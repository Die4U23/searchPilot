"""从 MIND TSV 构建版本化 Parquet 快照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchpilot.data.manifest import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mind"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_dataset(args.raw_dir, args.out_dir)
    summary = {
        "data_version": manifest.data_version,
        "rows": {entry.name: entry.rows for entry in manifest.outputs},
        "time_bounds": manifest.to_dict()["time_bounds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
