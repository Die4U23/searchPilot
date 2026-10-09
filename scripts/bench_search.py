"""对本机已加载的 BM25 服务做一次固定规模计时。不开启 --reload。"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

from searchpilot.search.service import SearchNotReadyError, build_search_service


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Time in-process BM25 search")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("docs/data/bench.md"))
    args = parser.parse_args()
    try:
        service = build_search_service(args.artifact_dir)
    except SearchNotReadyError as exc:
        args.out.write_text(f"# 压测\n\nUNRUN：{exc}\n", encoding="utf-8")
        print(f"UNRUN: {exc}")
        return 2
    # 预热一次，不计入延迟。
    service.search("election update", 10, "bm25")
    samples: list[float] = []
    started = time.perf_counter()
    for index in range(args.requests):
        query = f"election update {index % 7}"
        tick = time.perf_counter()
        service.search(query, 10, "bm25")
        samples.append((time.perf_counter() - tick) * 1000)
    elapsed = time.perf_counter() - started
    throughput = args.requests / elapsed if elapsed else 0.0
    body = "\n".join(
        [
            "# BM25 进程内压测",
            "",
            "单进程顺序请求，没有 `--reload`。向量和 CTR 的分阶段计时未单独测量，记为 UNRUN。",
            "",
            f"- 机器：`{platform.platform()}`",
            f"- Python：`{platform.python_version()}`",
            f"- 请求数：{args.requests}",
            "- 预热：1 次，不计入",
            f"- p50：{percentile(samples, 0.50):.3f} ms",
            f"- p95：{percentile(samples, 0.95):.3f} ms",
            f"- p99：{percentile(samples, 0.99):.3f} ms",
            f"- 吞吐：{throughput:.2f} req/s",
            "",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
