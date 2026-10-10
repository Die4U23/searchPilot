"""把已测实验写入 artifacts/experiments/registry.json。没有数据库时 API 读这份文件。"""

from __future__ import annotations

import argparse
from pathlib import Path

from searchpilot.agent.demo import demo_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the measured experiment registry")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/experiments/registry.json"),
    )
    args = parser.parse_args()
    store = demo_store()
    store.save(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
