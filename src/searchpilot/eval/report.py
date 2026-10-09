"""把 :class:`EvalReport` 写成 JSON 与 Markdown。

JSON 顶层字段：``model_version``、``split``、``mode``、``config``、``metrics``、``segments``、
``failures``、``evaluated_at``、``query_count``、``skipped_no_relevant``、
``answerable_zero_result_count``。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from searchpilot.eval.search_eval import EvalReport


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "source": "searchpilot",
        "model_version": report.model_version,
        "split": report.split,
        "mode": report.config.mode,
        "config": report.config.to_dict(),
        "evaluated_at": report.evaluated_at,
        "query_count": report.query_count,
        "skipped_no_relevant": report.skipped_no_relevant,
        "answerable_zero_result_count": report.overall.answerable_zero_result_count,
        "metrics": dict(report.overall.metrics),
        "segments": {name: seg.to_dict() for name, seg in report.segments.items()},
        "failures": [f.to_dict() for f in report.failures],
    }


def write_json(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _metrics_table(
    rows: Sequence[tuple[str, int, int, Mapping[str, float | None]]], metric_names: Sequence[str]
) -> list[str]:
    header = ["segment", "queries", "skipped (no relevant)", *metric_names]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, count, skipped, metrics in rows:
        cells = [name, str(count), str(skipped), *(_fmt(metrics.get(m)) for m in metric_names)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_markdown(report: EvalReport) -> str:
    names = report.config.metric_names
    lines: list[str] = [
        "# Search evaluation",
        "",
        f"- model_version: `{report.model_version}`",
        f"- mode: `{report.config.mode}`",
        f"- split: `{report.split if report.split is not None else 'all'}`",
        f"- evaluated_at: {report.evaluated_at}",
        f"- queries: {report.query_count}"
        f" (skipped for nDCG/MRR/Recall because no relevant item: {report.skipped_no_relevant})",
        f"- answerable queries with zero results: {report.overall.answerable_zero_result_count}",
        f"- search limit: {report.config.search_limit}",
        "",
        "## Overall",
        "",
        *_metrics_table(
            [
                (
                    "all",
                    report.overall.query_count,
                    report.overall.skipped_no_relevant,
                    report.overall.metrics,
                )
            ],
            names,
        ),
        "",
        "## By query_type",
        "",
        *_metrics_table(
            [
                (name, seg.query_count, seg.skipped_no_relevant, seg.metrics)
                for name, seg in report.segments.items()
            ],
            names,
        ),
        "",
        f"## Failures (lowest ndcg@{report.config.ndcg_k}, answerable queries only)",
        "",
    ]
    if not report.failures:
        lines.append("_none_")
    for i, f in enumerate(report.failures, start=1):
        lines.append(
            f"### {i}. `{f.query_id}` [{f.query_type}] ndcg={_fmt(f.ndcg)}"
            + (" ZERO RESULTS" if f.zero_result else "")
        )
        lines.append("")
        lines.append(f"- query: {f.query_text}")
        lines.append(f"- normalized: {f.normalized_query}")
        top = ", ".join(f"{item} ({score:.3f})" for item, score in f.top_results) or "(none)"
        lines.append(f"- top-{report.config.failure_top_n}: {top}")
        expected = ", ".join(f"{item} (grade {g})" for item, g in f.expected) or "(none)"
        lines.append(f"- expected: {expected}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")


def summary_lines(report: EvalReport) -> list[str]:
    """给命令行打印的纯 ASCII 摘要（不含查询文本，避免控制台编码问题）。"""
    names = report.config.metric_names
    rows = [
        (
            "all",
            report.overall.query_count,
            report.overall.skipped_no_relevant,
            report.overall.metrics,
        )
    ] + [
        (name, seg.query_count, seg.skipped_no_relevant, seg.metrics)
        for name, seg in report.segments.items()
    ]
    return [
        f"model_version={report.model_version} split={report.split} queries={report.query_count}"
        f" skipped_no_relevant={report.skipped_no_relevant}",
        *_metrics_table(rows, names),
    ]
