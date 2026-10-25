"""Offline evidence accounting; no application, model, database or network execution."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from scripts import summarize_soak
from scripts.summarize_soak import main, summarize


def bench(warmup: int = 0, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = [
        row
        for row in events or []
        if row.get("event") == "result" and row.get("phase") == "measured"
    ]
    successes = sum(row["outcome"] == "success" for row in records)
    start = datetime(2026, 10, 23, tzinfo=UTC)
    return {
        "status": "COMPLETED",
        "config": {"duration_seconds": 3600, "warmup": warmup},
        "requested_modes": ["bm25"],
        "runs": [
            {
                "mode": "mixed",
                "dispatch_window_seconds": 3600,
                "started_at": start.isoformat(),
                "finished_at": (start + timedelta(seconds=3601)).isoformat(),
                "last_dispatch_seconds": max(
                    (row["dispatched_ms"] / 1000 for row in records), default=0
                ),
                "summary": {
                    "total": len(records),
                    "successes": successes,
                    "failures": len(records) - successes,
                    "elapsed_seconds": 3601,
                },
            }
        ],
    }


def pair(
    request_id: str,
    dispatch: float = 0,
    *,
    phase: str = "measured",
    outcome: str = "success",
    elapsed: float = 100,
) -> list[dict[str, Any]]:
    base = {"request_id": request_id, "phase": phase, "query": "query", "mode": "bm25", "index": 0}
    if phase == "measured":
        base["dispatched_ms"] = dispatch
    return [
        {**base, "event": "pending", "outcome": "pending"},
        {
            **base,
            "event": "result",
            "outcome": outcome,
            "elapsed_ms": elapsed,
            "status_code": 200 if outcome == "success" else 500,
            "result_count": 1,
            "model_version": "model",
            "result_sha256": "a" * 64,
            "ranking_sha256": "b" * 64,
        },
    ]


def test_unknown_stays_in_denominator_without_invented_latency() -> None:
    events = pair("ok") + pair("lost", 10)[:1]
    report = summarize(bench(), events)
    measured = report["phases"]["measured"]
    assert report["status"] == "FAILED"
    assert (measured["issued"], measured["success"], measured["unknown"]) == (2, 1, 1)
    assert measured["success_rate"] == 0.5
    assert measured["latency_all_completed"]["count"] == 1
    assert measured["unknown_request_ids"] == ["lost"]


def test_duplicates_orphans_and_mismatches_are_not_successes() -> None:
    duplicate_pending = pair("dp")
    duplicate_result = pair("dr")
    mismatch = pair("mismatch")
    mismatch[1]["query"] = "other"
    events = (
        duplicate_pending
        + duplicate_pending[:1]
        + duplicate_result
        + duplicate_result[1:]
        + mismatch
        + pair("orphan")[1:]
    )
    report = summarize(bench(), events)
    assert report["ledger"]["pending_events"] == 4
    assert report["ledger"]["unique_issued"] == 3
    assert report["ledger"]["paired"] == 0
    assert len(report["ledger"]["duplicates"]) == 2
    assert report["ledger"]["orphan_result_request_ids"] == ["orphan"]
    assert report["phases"]["measured"]["unknown"] == 3


def test_warmup_failure_is_separate_and_fails_report() -> None:
    report = summarize(bench(1), pair("w", phase="warmup", outcome="timeout") + pair("m"))
    assert report["status"] == "FAILED"
    assert report["phases"]["warmup"]["issued"] == 1
    assert report["phases"]["warmup"]["failed"] == 1
    assert report["phases"]["measured"]["issued"] == 1
    assert sum(window["issued"] for window in report["windows"]) == 1


def test_dispatch_window_boundary_and_cross_window_completion() -> None:
    events = pair("before", 299999, elapsed=9000) + pair("boundary", 300000) + pair("last", 3599999)
    report = summarize(bench(events=events), events)
    assert len(report["windows"]) == 12
    assert [report["windows"][i]["issued"] for i in (0, 1, 11)] == [1, 1, 1]
    assert report["windows"][0]["latency_success"]["p99_ms"] == 9000
    excluded = summarize(bench(), events + pair("outside", 3600000))
    assert excluded["status"] == "FAILED"
    assert excluded["phases"]["measured"]["issued"] == 4
    assert excluded["ledger"]["window_unassigned_request_ids"] == ["outside"]


def test_all_and_success_percentiles_use_nearest_rank() -> None:
    events = []
    for i in range(1, 21):
        events.extend(pair(str(i), elapsed=i, outcome="timeout" if i == 20 else "success"))
    mode = summarize(bench(), events)["phases"]["measured"]["by_mode"]["bm25"]
    assert mode["latency_all_completed"]["p95_ms"] == 19
    assert mode["latency_all_completed"]["p99_ms"] == 20
    assert mode["latency_success"]["p99_ms"] == 19


def test_fingerprint_stability_uses_query_and_mode_and_includes_warmup() -> None:
    events = pair("warm", phase="warmup") + pair("measured")
    events[3]["result_sha256"] = "c" * 64
    report = summarize(bench(1), events)
    assert report["status"] == "FAILED"
    assert report["fingerprints"]["group_count"] == 1
    assert report["fingerprints"]["groups"][0]["stable"] is False
    assert report["fingerprints"]["groups"][0]["count"] == 2


def test_cli_saves_failure_with_partial_malformed_input(tmp_path: Path) -> None:
    benchmark, ledger, out = (
        tmp_path / name for name in ("bench.json", "ledger.jsonl", "report.json")
    )
    benchmark.write_text(json.dumps(bench()), encoding="utf-8")
    ledger.write_text(json.dumps(pair("lost")[0]) + "\n{broken\n", encoding="utf-8")
    assert main(["--bench", str(benchmark), "--ledger", str(ledger), "--out", str(out)]) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["phases"]["measured"]["unknown"] == 1
    assert any(issue["kind"] == "invalid_jsonl" for issue in report["issues"])


def test_cli_missing_inputs_still_writes_failure(tmp_path: Path) -> None:
    missing, out = tmp_path / "missing", tmp_path / "report.json"
    assert main(["--bench", str(missing), "--ledger", str(missing), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "FAILED"


def test_empty_partial_window_and_sample_inventory() -> None:
    payload = bench()
    payload["config"]["duration_seconds"] = 350
    report = summarize(
        payload,
        pair("m", 349999),
        duration_seconds=350,
        samples=[{"phase": "recovery", "error": "timeout"}],
    )
    assert len(report["windows"]) == 2
    assert report["windows"][0]["latency_success"]["p95_ms"] is None
    assert report["windows"][1]["end_seconds"] == 350
    assert report["resources"]["errors"]["sample:timeout"] == 1


def test_complete_run_crosscheck_and_declared_duration_are_required() -> None:
    events = [event for i in range(12) for event in pair(str(i), i * 300000)]
    payload = bench(events=events)
    assert summarize(payload, events)["status"] == "PASSED"
    payload["runs"][0]["summary"]["total"] += 1
    report = summarize(payload, events)
    assert report["status"] == "FAILED"
    assert any(issue["kind"] == "run_ledger_count_mismatch" for issue in report["issues"])
    payload = bench(events=events)
    payload["runs"][0]["summary"]["elapsed_seconds"] = 60
    payload["runs"][0]["finished_at"] = "2026-10-23T00:01:00+00:00"
    payload["runs"][0]["last_dispatch_seconds"] = 3600
    kinds = {issue["kind"] for issue in summarize(payload, events)["issues"]}
    assert {
        "insufficient_elapsed_duration",
        "insufficient_wall_duration",
        "invalid_run_last_dispatch",
    } <= kinds


def test_missing_run_modes_or_empty_windows_cannot_pass() -> None:
    payload = bench()
    payload.pop("runs")
    payload["requested_modes"] = ["bm25", "vector"]
    kinds = {issue["kind"] for issue in summarize(payload, pair("only"))["issues"]}
    assert {
        "expected_single_duration_run",
        "missing_measured_mode",
        "empty_dispatch_window",
    } <= kinds


def test_missing_fingerprint_cannot_be_labeled_stable() -> None:
    events = pair("first") + pair("second")
    del events[3]["result_sha256"]
    report = summarize(bench(events=events), events)
    group = report["fingerprints"]["groups"][0]
    assert group["stable"] is False
    assert group["invalid_fingerprint_count"] == 1


def test_result_before_pending_is_unknown() -> None:
    events = list(reversed(pair("reversed")))
    report = summarize(bench(events=events), events)
    assert report["phases"]["measured"]["unknown"] == 1


def query_workload(tmp_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    queries = tmp_path / "queries.csv"
    raw = b"query_text\nfirst query\nsecond query\n"
    queries.write_bytes(raw)
    events = []
    modes = ["bm25", "vector"]
    for index in range(12):
        request = pair(str(index), index * 300000)
        for event in request:
            event.update(
                index=index,
                mode=modes[index % 2],
                query=["first query", "second query"][(index // 2) % 2],
            )
        events.extend(request)
    payload = bench(events=events)
    payload["requested_modes"] = modes
    payload["config"]["concurrency"] = 2
    payload["queries"] = {"count": 2, "sha256": hashlib.sha256(raw).hexdigest()}
    return queries, payload, events


def test_expected_query_coverage_hash_repeats_and_input_order(tmp_path: Path) -> None:
    queries, payload, events = query_workload(tmp_path)
    report = summarize(payload, events, queries=queries, min_repeats=3)
    assert report["status"] == "PASSED"
    assert report["coverage"]["verified"] is True
    assert report["coverage"]["expected_group_count"] == 4
    assert {row["successful"] for row in report["coverage"]["groups"]} == {3}
    assert report["inflight"]["peak"] == 1
    assert report["inflight"]["configuration_checked"] is True
    assert "not server execution" in report["inflight"]["scope"]
    assert "resources are descriptive" in report["status_scope"]


@pytest.mark.parametrize(
    "change", ["missing", "unexpected", "insufficient", "hash", "count", "order"]
)
def test_expected_workload_rejects_incomplete_or_mismatched_evidence(
    tmp_path: Path, change: str
) -> None:
    queries, payload, events = query_workload(tmp_path)
    expected = {
        "missing": "missing_query_mode",
        "unexpected": "unexpected_query_mode",
        "insufficient": "insufficient_query_mode_repeats",
        "hash": "queries_sha256_mismatch",
        "count": "queries_count_mismatch",
        "order": "workload_order_mismatch",
    }[change]
    if change == "missing":
        events = [
            row
            for row in events
            if not (row["query"] == "second query" and row["mode"] == "vector")
        ]
    elif change == "unexpected":
        for row in events[:2]:
            row["query"] = "unrecorded query"
    elif change == "hash":
        # Same parsed workload, different raw input bytes must not match the run provenance.
        queries.write_bytes(queries.read_bytes().replace(b"\n", b"\r\n"))
    elif change == "count":
        payload["queries"]["count"] = 3
    elif change == "order":
        # The set and repeat counts are identical; only the ordered workload is wrong.
        for row in events[:2]:
            row["query"] = "second query"
        for row in events[4:6]:
            row["query"] = "first query"
    report = summarize(
        payload, events, queries=queries, min_repeats=4 if change == "insufficient" else 3
    )
    assert report["status"] == "FAILED"
    assert report["coverage"]["verified"] is False
    assert expected in {issue["kind"] for issue in report["issues"]}


def test_warmup_and_unknown_results_cannot_supply_expected_repeats(tmp_path: Path) -> None:
    queries, payload, events = query_workload(tmp_path)
    events[-1]["query"] = "mismatch"
    warm = pair("warmup", phase="warmup")
    for row in warm:
        row.update(query="second query", mode="vector")
    report = summarize(payload, warm + events, queries=queries, min_repeats=3)
    group = next(
        row
        for row in report["coverage"]["groups"]
        if row["query"] == "second query" and row["mode"] == "vector"
    )
    assert group["issued"] == 3 and group["successful"] == 2
    assert "insufficient_query_mode_repeats" in {issue["kind"] for issue in report["issues"]}


def test_single_success_per_group_is_insufficient_and_empty_success_still_fails(
    tmp_path: Path,
) -> None:
    queries, payload, events = query_workload(tmp_path)
    report = summarize(payload, events[:8], queries=queries, min_repeats=2)
    assert len(report["coverage"]["insufficient_repeat_groups"]) == 4
    assert {group["successful"] for group in report["coverage"]["groups"]} == {1}
    events[1]["result_count"] = 0
    report = summarize(payload, events, queries=queries)
    assert report["status"] == "FAILED"
    assert "empty_success" in {issue["kind"] for issue in report["issues"]}


@pytest.mark.parametrize("index", [1, -1, True, None])
def test_expected_workload_requires_contiguous_integer_indices(tmp_path: Path, index: Any) -> None:
    queries, payload, events = query_workload(tmp_path)
    for row in events[:2]:
        row["index"] = index
    report = summarize(payload, events, queries=queries)
    assert "workload_order_mismatch" in {issue["kind"] for issue in report["issues"]}


def test_inflight_uses_original_event_order_and_rejects_overlimit_or_residuals() -> None:
    requests = [pair(str(i), i * 300000) for i in range(12)]
    sequential = [event for request in requests for event in request]
    payload = bench(events=sequential)
    payload["config"]["concurrency"] = 2
    concurrent = [requests[0][0], requests[1][0], requests[0][1], requests[1][1]] + sequential[4:]
    report = summarize(payload, concurrent)
    assert report["status"] == "PASSED"
    assert report["inflight"]["peak"] == 2
    overlimit = [request[0] for request in requests] + [request[1] for request in requests]
    report = summarize(payload, overlimit)
    assert report["status"] == "FAILED"
    assert report["inflight"]["peak"] == 12
    assert report["inflight"]["pending_events_above_limit"] == 10
    assert "concurrency_exceeded" in {issue["kind"] for issue in report["issues"]}
    residual = summarize(payload, sequential[:-1])
    assert residual["inflight"]["pending_events_at_end"] == 1
    assert residual["inflight"]["pending_request_ids_at_end"] == ["11"]
    assert "inflight_requests_remaining" in {issue["kind"] for issue in residual["issues"]}


def test_optional_queries_preserve_legacy_results_but_repeats_require_them(tmp_path: Path) -> None:
    events = [event for i in range(12) for event in pair(str(i), i * 300000)]
    report = summarize(bench(events=events), events)
    assert report["status"] == "PASSED"
    assert report["coverage"]["provided"] is False
    assert report["inflight"]["configuration_checked"] is False
    with pytest.raises(ValueError, match="requires queries"):
        summarize(bench(events=events), events, min_repeats=2)
    queries, payload, events = query_workload(tmp_path)
    del payload["config"]["concurrency"]
    assert "invalid_configured_concurrency" in {
        issue["kind"] for issue in summarize(payload, events, queries=queries)["issues"]
    }


def test_cli_queries_cover_shared_input_validation_and_output_alias(tmp_path: Path) -> None:
    queries, payload, events = query_workload(tmp_path)
    benchmark, ledger, out = (
        tmp_path / name for name in ("bench.json", "ledger.jsonl", "report.json")
    )
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    ledger.write_text("\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8")
    argv = [
        "--bench",
        str(benchmark),
        "--ledger",
        str(ledger),
        "--queries",
        str(queries),
        "--min-repeats",
        "3",
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    assert json.loads(out.read_text())["coverage"]["verified"] is True
    with pytest.raises(SystemExit):
        main([*argv[:-1], str(queries)])
    queries.write_text("query_text\nsame\nsame\n", encoding="utf-8")
    failed_out = tmp_path / "failed.json"
    assert main([*argv[:-1], str(failed_out)]) == 1
    assert "queries_read_error" in {
        issue["kind"] for issue in json.loads(failed_out.read_text())["issues"]
    }


def resource_sample(tmp_path: Path) -> dict[str, Any]:
    """Use the sampler's real envelope from synthetic proc/cgroup files, never live PIDs."""
    from scripts.soak_memory import sample_linux_process

    process = tmp_path / "proc" / "1"
    process.mkdir(parents=True)
    (process / "stat").write_text("1 (python) S " + "0 " * 18 + "123 0\n")
    (process / "status").write_text("VmRSS: 100 kB\nThreads: 4\n")
    (process / "smaps_rollup").write_text(
        "Rss: 100 kB\nPss: 90 kB\nAnonymous: 80 kB\nPrivate_Dirty: 70 kB\n"
    )
    (process / "cmdline").write_bytes(b"python\0")
    (process / "fd").mkdir()
    (process / "fd" / "0").touch()
    (process / "cgroup").write_text("0::/\n")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    for name, value in {
        "current": "200000",
        "peak": "300000",
        "stat": "anon 100000\nfile 60000\nkernel 40000\n",
        "events": "oom 1\noom_kill 0\n",
    }.items():
        (cgroup / f"memory.{name}").write_text(value)
    return {
        "at": "2026-10-23T00:00:00+00:00",
        "phase": "load",
        "server": sample_linux_process(proc_root=process.parent, cgroup_root=cgroup),
        "client": {
            "available": True,
            "pid": 42,
            "sources": {
                "memory": {"available": True, "data": {"rss_bytes": 1000, "private_bytes": 2000}}
            },
            "errors": {},
        },
        "live": {"available": True, "status": 200, "seconds": 0.1},
        "ready": {"available": True, "status": 200, "seconds": 0.1},
    }


def test_resource_real_envelope_values_windows_and_counter_delta(tmp_path: Path) -> None:
    first = resource_sample(tmp_path)
    second = deepcopy(first)
    second["at"] = "2026-10-23T00:05:00+00:00"
    second["server"]["sources"]["smaps_rollup"]["data"]["Rss_bytes"] = 150000
    second["server"]["sources"]["cgroup_memory_events"]["data"]["oom"] = 3
    second["client"]["sources"]["memory"]["data"]["private_bytes"] = 2500
    before = deepcopy(first)
    before["at"] = "2026-10-22T23:59:59+00:00"  # load phase can include pre-measurement warmup
    report = summarize(bench(), [], samples=[second, before, first])
    resources = report["resources"]
    overall = resources["overall"]
    assert resources["coverage"]["aligned_to_measured_windows"] == 2
    assert resources["coverage"]["outside_measured_window"] == 1
    assert resources["windows"][0]["sample_rows"] == 1
    assert resources["windows"][1]["sample_rows"] == 1
    assert overall["coverage"]["max_interval_seconds"] == 300
    metrics = overall["metrics"]
    assert metrics["pid1_rss_bytes"]["first"] == 102400
    assert metrics["pid1_rss_bytes"]["last"] == 150000
    assert metrics["pid1_pss_bytes"]["min"] == 92160
    assert metrics["pid1_anonymous_bytes"]["min"] == 81920
    assert metrics["pid1_private_dirty_bytes"]["min"] == 71680
    assert metrics["cgroup_current_bytes"]["max"] == 200000
    assert metrics["cgroup_kernel_bytes"]["max"] == 40000
    assert metrics["client_private_bytes"]["last"] == 2500
    assert metrics["pid1_fd_count"]["min"] == 1
    assert metrics["pid1_thread_count"]["max"] == 4
    assert overall["memory_events"]["oom"]["delta"] == 2
    assert overall["memory_events"]["oom_group_kill"]["delta"] is None
    assert overall["identity"]["start_time_ticks_unique"] == [123]


def test_resource_missing_restart_and_health_do_not_change_ledger_status(tmp_path: Path) -> None:
    first = resource_sample(tmp_path)
    second = deepcopy(first)
    second["at"] = "2026-10-23T00:00:10+00:00"
    second["server"]["sources"]["stat"]["data"]["start_time_ticks"] = 456
    second["server"]["sources"]["stat_after"]["data"]["start_time_ticks"] = 456
    second["server"]["sources"]["smaps_rollup"] = {"available": False}
    second["server"]["errors"] = {"smaps_rollup": {"error_type": "PermissionError"}}
    second["server"]["sources"]["cgroup_memory_events"]["data"]["oom"] = 0
    second["ready"] = {"available": True, "status": 503}
    second["live"] = {"available": False, "error_type": "TimeoutError"}
    missing = {"at": "2026-10-23T00:00:20+00:00", "phase": "recovery", "server": None}
    events = [event for i in range(12) for event in pair(str(i), i * 300000)]
    report = summarize(bench(events=events), events, samples=[first, second, missing])
    assert report["status"] == "PASSED"
    resources = report["resources"]
    overall = resources["overall"]
    assert overall["metrics"]["pid1_rss_bytes"]["available"] == 1
    assert overall["metrics"]["pid1_rss_bytes"]["missing"] == 2
    assert overall["identity"]["start_time_ticks_unique"] == [123, 456]
    assert overall["identity"]["unknown"] == 1
    assert overall["memory_events"]["oom"]["counter_decreases"] == 1
    assert overall["memory_events"]["oom"]["delta"] is None
    assert overall["health"]["live"]["transport_errors"] == 1
    assert overall["health"]["ready"]["http_errors"] == 1
    assert overall["health"]["ready"]["status_200"] == 1
    assert overall["health"]["ready"]["missing"] == 1
    assert resources["errors"]["server.smaps_rollup:PermissionError"] == 1
    assert resources["by_phase"]["recovery"]["metrics"]["client_rss_bytes"]["min"] is None


def test_resource_unstable_identity_and_untimed_values_are_explicit(tmp_path: Path) -> None:
    sample = resource_sample(tmp_path)
    sample["server"]["identity_consistent"] = False
    sample["at"] = "not-a-timestamp"
    resources = summarize(bench(), [], samples=[sample])["resources"]
    assert resources["coverage"]["unaligned"] == 1
    assert resources["overall"]["identity"]["inconsistent"] == 1
    metric = resources["overall"]["metrics"]["pid1_rss_bytes"]
    assert metric["first"] is None and metric["min"] == 102400
    assert metric["available_without_timestamp"] == 1


def test_cli_sample_read_failure_is_resource_only(tmp_path: Path) -> None:
    events = [event for i in range(12) for event in pair(str(i), i * 300000)]
    benchmark, ledger, out = (
        tmp_path / name for name in ("bench.json", "ledger.jsonl", "out.json")
    )
    benchmark.write_text(json.dumps(bench(events=events)), encoding="utf-8")
    ledger.write_text("\n".join(json.dumps(row) for row in events), encoding="utf-8")
    assert (
        main(
            [
                "--bench",
                str(benchmark),
                "--ledger",
                str(ledger),
                "--out",
                str(out),
                "--samples",
                str(tmp_path / "missing.jsonl"),
            ]
        )
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "PASSED"
    assert report["resources"]["input_errors"][0]["kind"] == "input_read_error"


@pytest.mark.parametrize("source", ["bench", "ledger", "samples"])
@pytest.mark.parametrize("alias", ["same", "normalized", "hardlink"])
def test_cli_refuses_output_aliases_without_changing_evidence(
    tmp_path: Path,
    source: str,
    alias: str,
) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in ("bench", "ledger", "samples")}
    original = b"original evidence bytes\r\n"
    for path in paths.values():
        path.write_bytes(original)
    output = paths[source]
    if alias == "normalized":
        (tmp_path / "subdir").mkdir()
        output = tmp_path / "subdir" / ".." / output.name
    elif alias == "hardlink":
        output = tmp_path / "alias.json"
        output.hardlink_to(paths[source])
    arguments = [part for name, path in paths.items() for part in (f"--{name}", str(path))]
    with pytest.raises(SystemExit) as exc:
        main([*arguments, "--out", str(output)])
    assert exc.value.code == 2
    assert all(path.read_bytes() == original for path in paths.values())
    assert output.read_bytes() == original


def test_cli_preserves_existing_report(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"
    output.write_bytes(b"prior failed analysis\n")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--bench",
                str(tmp_path / "bench.json"),
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--out",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert output.read_bytes() == b"prior failed analysis\n"


def test_cli_exclusive_create_preserves_file_added_during_analysis(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "analysis.json"
    original_summarize = summarize_soak.summarize

    def concurrent_file(*args, **kwargs):
        result = original_summarize(*args, **kwargs)
        output.write_bytes(b"another writer's evidence\n")
        return result

    monkeypatch.setattr(summarize_soak, "summarize", concurrent_file)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--bench",
                str(tmp_path / "bench.json"),
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--out",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert output.read_bytes() == b"another writer's evidence\n"
