"""下载并解压 MIND Small。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchpilot.data.download import download_mind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mind"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = download_mind(args.raw_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
