"""下载固定 commit 的 EvoRec results.json。不改文件字节。"""

from __future__ import annotations

import urllib.request
from pathlib import Path

COMMIT = "5ce1d96b80ddaf3c205a5ef8a6a5e39df6ab0ff5"
REPO = "Die4U23/EvoRec"
FILES = (
    "docs/experiments/r03-content/results.json",
    "docs/experiments/r04-gating/results.json",
    "docs/experiments/r05-cold-replication/results.json",
    "docs/experiments/r05-ranker/results.json",
    "docs/experiments/r06-multi-interest/results.json",
)


def main() -> int:
    root = Path("third_party/evorec") / COMMIT
    for relative in FILES:
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{relative}"
        urllib.request.urlretrieve(url, dest)
        print(f"{dest.stat().st_size} {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
