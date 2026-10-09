"""用小语料和临时查询集跑通搜索离线评估。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from searchpilot.eval.report import render_markdown, write_json, write_markdown
from searchpilot.eval.search_eval import evaluate_from_files
from searchpilot.ports import Document
from searchpilot.search.service import InMemorySearchService

DOCS = [
    Document(
        "N1",
        "Xylophone quartet",
        "The xylophone quartet played downtown",
        "arts",
        "music",
    ),
    Document(
        "N2",
        "Banana bread recipe",
        "Easy banana bread for weeknights",
        "food",
        "baking",
    ),
    Document(
        "N3",
        "Popular celebrity wedding",
        "A famous couple married in Paris",
        "entertainment",
        "celebs",
    ),
]

_JSON_FIELDS = {
    "model_version",
    "split",
    "mode",
    "config",
    "metrics",
    "segments",
    "failures",
    "evaluated_at",
    "query_count",
    "skipped_no_relevant",
    "answerable_zero_result_count",
}


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _write_eval_files(directory: Path) -> tuple[Path, Path]:
    queries = directory / "queries.csv"
    labels = directory / "labels.csv"
    _write_csv(
        queries,
        [
            ["query_id", "query_text", "query_type", "split"],
            ["q_exact", "Xylophone quartet", "exact_entity", "test"],
            ["q_syn", "Banana bread", "synonym", "test"],
            ["q_multi", "celebrity wedding", "multi_condition", "test"],
            ["q_none", "underwater basket", "no_answer", "test"],
            ["q_train", "Xylophone quartet", "exact_entity", "train"],
        ],
    )
    _write_csv(
        labels,
        [
            ["query_id", "item_id", "grade", "annotator"],
            ["q_exact", "N1", "3", "ann"],
            ["q_syn", "N2", "3", "ann"],
            ["q_multi", "N1", "3", "ann"],
            ["q_none", "N3", "0", "ann"],
            ["q_train", "N1", "3", "ann"],
        ],
    )
    return queries, labels


def test_split_segments_skipped_and_failures(tmp_path: Path) -> None:
    queries, labels = _write_eval_files(tmp_path)
    service = InMemorySearchService(DOCS)
    report = evaluate_from_files(service, queries, labels, split="test")

    assert report.split == "test"
    assert report.query_count == 4
    assert [item.query.query_id for item in report.per_query] == [
        "q_exact",
        "q_multi",
        "q_none",
        "q_syn",
    ]
    assert all(item.query.split == "test" for item in report.per_query)
    assert set(report.segments) == {
        "exact_entity",
        "multi_condition",
        "no_answer",
        "synonym",
    }
    assert report.skipped_no_relevant == 1
    assert report.segments["no_answer"].skipped_no_relevant == 1
    assert report.segments["no_answer"].query_count == 1

    # q_multi 的相关文档不在结果里（ndcg 0），其余两条命中唯一相关文档（ndcg 1）。
    assert [sample.query_id for sample in report.failures] == ["q_multi", "q_exact", "q_syn"]
    ndcgs = [sample.ndcg for sample in report.failures]
    assert all(value is not None for value in ndcgs)
    assert ndcgs == sorted(ndcgs)
    assert ndcgs[0] == pytest.approx(0.0)
    assert ndcgs[-1] == pytest.approx(1.0)

    unfiltered = evaluate_from_files(service, queries, labels, split=None)
    assert unfiltered.query_count == 5
    assert any(item.query.query_id == "q_train" for item in unfiltered.per_query)
    assert unfiltered.skipped_no_relevant == 1


def test_report_json_fields_and_markdown(tmp_path: Path) -> None:
    queries, labels = _write_eval_files(tmp_path)
    report = evaluate_from_files(InMemorySearchService(DOCS), queries, labels, split="test")
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    write_json(report, json_path)
    write_markdown(report, md_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) >= _JSON_FIELDS
    assert payload["split"] == "test"
    assert payload["mode"] == "bm25"
    assert payload["query_count"] == 4
    assert payload["skipped_no_relevant"] == 1
    for name in ("ndcg@10", "mrr@10", "recall@50", "zero_result_rate"):
        assert name in payload["metrics"]
    assert set(payload["segments"]) == set(report.segments)
    assert payload["failures"]
    failure = payload["failures"][0]
    assert {"query_text", "expected", "top_results"} <= set(failure)

    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.strip()
    assert markdown == render_markdown(report)
    assert "Search evaluation" in markdown
