"""对比 bm25、vector、hybrid、ltr 的离线检索指标。

用法::

    python scripts/compare_search_modes.py --artifact-dir ./artifacts \
        --queries data/queries/queries.csv --labels data/queries/labels.csv \
        --split test --out artifacts/search/eval

依次调用 ``evaluate_from_files``，不另行实现指标。输出
``<out>/search_mode_compare_<split>.md``。某个 mode 抛出 ``SearchNotReadyError``
时在总表记一行失败原因并继续其余 mode，进程退出码为 2。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from searchpilot.eval.search_eval import (
    EvalConfig,
    EvalReport,
    SegmentMetrics,
    evaluate_from_files,
)
from searchpilot.ports import SearchMode
from searchpilot.search.service import SearchNotReadyError, build_search_service

MODES: tuple[SearchMode, ...] = ("bm25", "vector", "hybrid", "ltr")


@dataclass(frozen=True, slots=True)
class _ModeOutcome:
    mode: SearchMode
    report: EvalReport | None
    error: str | None


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _metric_header(config: EvalConfig) -> list[str]:
    return [
        f"nDCG@{config.ndcg_k}",
        f"MRR@{config.mrr_k}",
        f"Recall@{config.recall_k}",
    ]


def _metric_cells(metrics: Mapping[str, float | None], config: EvalConfig) -> list[str]:
    keys = (
        f"ndcg@{config.ndcg_k}",
        f"mrr@{config.mrr_k}",
        f"recall@{config.recall_k}",
    )
    return [_fmt(metrics[key]) for key in keys]


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    return lines


def _overall_header(config: EvalConfig) -> list[str]:
    return [
        "mode",
        "model_version",
        "queries",
        "skipped_no_relevant",
        *_metric_header(config),
        "error",
    ]


def _overall_row(outcome: _ModeOutcome) -> list[str]:
    if outcome.report is None:
        return [outcome.mode, "", "", "", "", "", "", outcome.error or ""]
    report = outcome.report
    return [
        outcome.mode,
        report.model_version,
        str(report.query_count),
        str(report.skipped_no_relevant),
        *_metric_cells(report.overall.metrics, report.config),
        "",
    ]


def _segment_header(config: EvalConfig) -> list[str]:
    return [
        "mode",
        "model_version",
        "queries",
        "skipped_no_relevant",
        *_metric_header(config),
    ]


def _segment_row(mode: str, report: EvalReport, segment: SegmentMetrics) -> list[str]:
    return [
        mode,
        report.model_version,
        str(segment.query_count),
        str(segment.skipped_no_relevant),
        *_metric_cells(segment.metrics, report.config),
    ]


def _query_types(outcomes: Sequence[_ModeOutcome]) -> list[str]:
    names: set[str] = set()
    for outcome in outcomes:
        if outcome.report is not None:
            names.update(outcome.report.segments)
    return sorted(names)


def render_comparison(outcomes: Sequence[_ModeOutcome], *, split_label: str) -> str:
    """总表一行一个 mode；失败 mode 的原因写在 error 列。再按 query_type 分组。"""
    config = next(
        (item.report.config for item in outcomes if item.report is not None), EvalConfig()
    )
    lines = [
        "# Search mode comparison",
        "",
        f"- split: `{split_label}`",
        "",
        "## Overall",
        "",
        *_table(_overall_header(config), [_overall_row(item) for item in outcomes]),
        "",
        "## By query_type",
        "",
    ]
    query_types = _query_types(outcomes)
    if not query_types:
        lines.append("_none_")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    header = _segment_header(config)
    for query_type in query_types:
        lines.append(f"### {query_type}")
        lines.append("")
        rows: list[list[str]] = []
        for outcome in outcomes:
            report = outcome.report
            if report is None or query_type not in report.segments:
                continue
            rows.append(_segment_row(outcome.mode, report, report.segments[query_type]))
        lines.extend(_table(header, rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare bm25, vector, hybrid, and ltr metrics")
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/search/eval"),
        help="directory for search_mode_compare_<split>.md",
    )
    args = parser.parse_args(argv)

    for path in (args.queries, args.labels):
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
    try:
        service = build_search_service(args.artifact_dir)
    except SearchNotReadyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    split = None if args.split == "all" else args.split
    outcomes: list[_ModeOutcome] = []
    for mode in MODES:
        try:
            report = evaluate_from_files(
                service,
                args.queries,
                args.labels,
                split=split,
                config=EvalConfig(mode=mode),
            )
        except SearchNotReadyError as exc:
            message = str(exc)
            outcomes.append(_ModeOutcome(mode, None, message))
            print(f"error: {mode}: {message}", file=sys.stderr)
            continue
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        outcomes.append(_ModeOutcome(mode, report, None))
        print(
            f"{mode}: model_version={report.model_version} "
            f"queries={report.query_count} "
            f"skipped_no_relevant={report.skipped_no_relevant}"
        )

    out_path = args.out / f"search_mode_compare_{args.split}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_comparison(outcomes, split_label=args.split),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    return 2 if any(item.error for item in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
