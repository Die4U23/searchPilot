"""分阶段计时：BM25、向量、RRF、LTR、CTR。冷启动单独记第一次，其后为预热后延迟。"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

from searchpilot.ctr.runtime import CtrNotReadyError, load_ctr
from searchpilot.search.service import SearchNotReadyError, build_search_service

QUERIES = [f"election update {index}" for index in range(30)]


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def time_calls(calls: list) -> list[float]:
    samples: list[float] = []
    for call in calls:
        tick = time.perf_counter()
        call()
        samples.append((time.perf_counter() - tick) * 1000)
    return samples


def summarize(name: str, samples: list[float]) -> str:
    if not samples:
        return f"- {name}：UNRUN"
    if len(samples) == 1:
        return f"- {name} 冷启动：{samples[0]:.3f} ms（1 次）"
    return (
        f"- {name} 预热后 {len(samples)} 次：p50 {percentile(samples, 0.50):.3f} ms，"
        f"p95 {percentile(samples, 0.95):.3f} ms，p99 {percentile(samples, 0.99):.3f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Time BM25, vector, hybrid, LTR, and CTR")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/bench-stages.md"))
    args = parser.parse_args()
    lines = [
        "# 分阶段计时",
        "",
        "单进程顺序调用，没有 `--reload`。冷启动是该阶段的第一次调用，不计入后面的分位数。",
        "",
        f"- 机器：`{platform.platform()}`",
        f"- Python：`{platform.python_version()}`",
        "",
    ]
    try:
        service = build_search_service(args.artifact_dir)
    except SearchNotReadyError as exc:
        lines.append(f"UNRUN：{exc}")
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 2

    def bm25() -> None:
        service.search("election update", 10, "bm25")

    lines.append(summarize("BM25", time_calls([bm25])))
    lines.append(summarize("BM25", time_calls([bm25] * 30)))

    if getattr(service, "vector_ready", False):

        def vector() -> None:
            service.search("election update", 10, "vector")

        def hybrid() -> None:
            service.search("election update", 10, "hybrid")

        lines.append(summarize("向量", time_calls([vector])))
        lines.append(summarize("向量", time_calls([vector] * 30)))
        lines.append(summarize("RRF", time_calls([hybrid])))
        lines.append(summarize("RRF", time_calls([hybrid] * 30)))
    else:
        lines.append("- 向量：UNRUN")
        lines.append("- RRF：UNRUN")

    if getattr(service, "ltr_ready", False) and getattr(service, "vector_ready", False):

        def ltr() -> None:
            service.search("election update", 10, "ltr")

        lines.append(summarize("LTR", time_calls([ltr])))
        lines.append(summarize("LTR", time_calls([ltr] * 30)))
    else:
        lines.append("- LTR：UNRUN")

    try:
        ctr = load_ctr(args.artifact_dir)
        candidates = [(f"N{index}", 0.5) for index in range(10)]

        def score() -> None:
            ctr.score("U1", candidates, calibrated=True)

        lines.append(summarize("CTR", time_calls([score])))
        lines.append(summarize("CTR", time_calls([score] * 30)))
    except CtrNotReadyError as exc:
        lines.append(f"- CTR：UNRUN（{exc.__class__.__name__}）")

    lines.append("")
    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
