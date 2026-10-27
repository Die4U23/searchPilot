"""Offline soak ledger audit. O(N) memory; never imports or starts the application.

Dispatch windows are half-open and use pending.dispatched_ms, not completion time.
Unknown/ambiguous requests remain in issued denominators but have no invented latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.bench_concurrent import MAX_DURATION_SECONDS, MODES, latency_summary, read_queries


def _number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _timestamp(value: Any) -> float | None:
    try:
        stamp = datetime.fromisoformat(value)
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _source_data(row: dict[str, Any], owner: str, source: str) -> dict[str, Any]:
    envelope = row.get(owner)
    if not isinstance(envelope, dict):
        return {}
    sources = envelope.get("sources")
    item = sources.get(source) if isinstance(sources, dict) else None
    if not isinstance(item, dict) or item.get("available") is not True:
        return {}
    data = item.get("data")
    return data if isinstance(data, dict) else {}


RESOURCE_METRICS = {
    "pid1_rss_bytes": ("server", "smaps_rollup", ("Rss_bytes",)),
    "pid1_pss_bytes": ("server", "smaps_rollup", ("Pss_bytes",)),
    "pid1_anonymous_bytes": ("server", "smaps_rollup", ("Anonymous_bytes",)),
    "pid1_private_dirty_bytes": ("server", "smaps_rollup", ("Private_Dirty_bytes",)),
    "pid1_status_rss_bytes": ("server", "status", ("VmRSS_bytes",)),
    "pid1_fd_count": ("server", "fd", ("count",)),
    "pid1_thread_count": ("server", "status", ("Threads",)),
    "cgroup_current_bytes": ("server", "cgroup_memory_current", ("bytes",)),
    "cgroup_anon_bytes": ("server", "cgroup_memory_stat", ("bytes", "anon")),
    "cgroup_file_bytes": ("server", "cgroup_memory_stat", ("bytes", "file")),
    "cgroup_kernel_bytes": ("server", "cgroup_memory_stat", ("bytes", "kernel")),
    "client_rss_bytes": ("client", "memory", ("rss_bytes",)),
    "client_private_bytes": ("client", "memory", ("private_bytes",)),
}


def _metric_value(row: dict[str, Any], name: str) -> float | int | None:
    owner, source, keys = RESOURCE_METRICS[name]
    server = row.get("server") if isinstance(row.get("server"), dict) else {}
    if name.startswith("pid1_") and server.get("pid") != 1:
        return None
    value: Any = _source_data(row, owner, source)
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value if _number(value) and value >= 0 else None


def _series(values: list[tuple[float | None, float | int | None]]) -> dict[str, Any]:
    valid = [(at, value) for at, value in values if value is not None]
    timed = sorted(((at, value) for at, value in valid if at is not None), key=lambda pair: pair[0])
    return {
        "available": len(valid),
        "missing": len(values) - len(valid),
        "available_without_timestamp": len(valid) - len(timed),
        "first": timed[0][1] if timed else None,
        "last": timed[-1][1] if timed else None,
        "min": min((value for _, value in valid), default=None),
        "max": max((value for _, value in valid), default=None),
    }


def _health_counts(rows: list[dict[str, Any]], endpoint: str) -> dict[str, Any]:
    counts = Counter(available=0, missing=0, status_200=0, http_errors=0, transport_errors=0)
    statuses: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for row in rows:
        probe = row.get(endpoint)
        if not isinstance(probe, dict):
            counts["missing"] += 1
            continue
        status = probe.get("status")
        # Historical records predate available; a recorded HTTP status remains evidence.
        if probe.get("available") is not False and isinstance(status, int):
            counts["available"] += 1
            statuses[str(status)] += 1
            counts["status_200" if status == 200 else "http_errors"] += 1
        elif probe.get("available") is False or probe.get("error_type"):
            counts["transport_errors"] += 1
            errors[str(probe.get("error_type", "unavailable"))] += 1
        else:
            counts["missing"] += 1
    return {**counts, "status_codes": dict(statuses), "error_types": dict(errors)}


def _resource_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [_timestamp(row.get("at")) for row in rows]
    ordered_times = sorted(at for at in times if at is not None)
    identities: Counter[str] = Counter(consistent=0, inconsistent=0, unknown=0)
    starts: set[int] = set()
    pids: set[int] = set()
    client_pids: set[int] = set()
    starts_missing = 0
    memberships: set[str] = set()
    memberships_missing = 0
    errors: Counter[str] = Counter()
    for row in rows:
        server = row.get("server") if isinstance(row.get("server"), dict) else {}
        flag = server.get("identity_consistent")
        identities[
            "consistent" if flag is True else "inconsistent" if flag is False else "unknown"
        ] += 1
        start = _source_data(row, "server", "stat").get("start_time_ticks")
        if isinstance(start, int) and not isinstance(start, bool):
            starts.add(start)
        else:
            starts_missing += 1
        after = _source_data(row, "server", "stat_after").get("start_time_ticks")
        if isinstance(after, int) and not isinstance(after, bool):
            starts.add(after)
        membership = _source_data(row, "server", "cgroup_membership").get("path")
        if isinstance(membership, str):
            memberships.add(membership)
        else:
            memberships_missing += 1
        for owner, destination in (("server", pids), ("client", client_pids)):
            envelope = row.get(owner) if isinstance(row.get(owner), dict) else {}
            pid = envelope.get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool):
                destination.add(pid)
            if envelope.get("error_type"):
                errors[f"{owner}:{envelope['error_type']}"] += 1
            source_errors = envelope.get("errors")
            if isinstance(source_errors, dict):
                for source, error in source_errors.items():
                    kind = (
                        error.get("error_type", "unknown") if isinstance(error, dict) else "unknown"
                    )
                    errors[f"{owner}.{source}:{kind}"] += 1
        if row.get("error"):
            errors[f"sample:{row['error']}"] += 1
        stats = row.get("stats")
        if isinstance(stats, dict) and stats.get("error_type"):
            errors[f"stats:{stats['error_type']}"] += 1
    stable_identity = (
        len(starts) == 1
        and not starts_missing
        and pids == {1}
        and not identities["inconsistent"]
        and not identities["unknown"]
    )
    counters = {}
    for name in ("oom", "oom_kill", "oom_group_kill"):
        observations = []
        for row, at in zip(rows, times, strict=True):
            value = _source_data(row, "server", "cgroup_memory_events").get(name)
            observations.append((at, value if _number(value) and value >= 0 else None))
        ordered = sorted(
            ((at, value) for at, value in observations if at is not None and value is not None),
            key=lambda pair: pair[0],
        )
        resets = sum(right[1] < left[1] for left, right in zip(ordered, ordered[1:], strict=False))
        known = (
            len(ordered) >= 2
            and not resets
            and stable_identity
            and len(memberships) == 1
            and not memberships_missing
        )
        counters[name] = {
            **_series(observations),
            "counter_decreases": resets,
            "delta": ordered[-1][1] - ordered[0][1] if known else None,
            "delta_available": known,
        }
    return {
        "sample_rows": len(rows),
        "coverage": {
            "timestamp_available": len(ordered_times),
            "timestamp_missing": len(rows) - len(ordered_times),
            "max_interval_seconds": max(
                (
                    right - left
                    for left, right in zip(ordered_times, ordered_times[1:], strict=False)
                ),
                default=None,
            ),
        },
        "metrics": {
            name: _series(
                [(at, _metric_value(row, name)) for row, at in zip(rows, times, strict=True)]
            )
            for name in RESOURCE_METRICS
        },
        "identity": {
            **identities,
            "start_time_ticks_unique": sorted(starts),
            "start_time_ticks_missing": starts_missing,
            "server_pids": sorted(pids),
            "client_pids": sorted(client_pids),
            "single_consistent_pid1_observed": stable_identity,
            "cgroup_membership_paths": sorted(memberships),
            "cgroup_membership_missing": memberships_missing,
        },
        "memory_events": counters,
        "health": {endpoint: _health_counts(rows, endpoint) for endpoint in ("live", "ready")},
        "errors": dict(sorted(errors.items())),
    }


def summarize_resources(
    bench: dict[str, Any],
    samples: list[dict[str, Any]] | None,
    *,
    duration_seconds: float,
    window_seconds: float,
) -> dict[str, Any]:
    """Descriptive samples only; deliberately independent of ledger acceptance status."""
    rows = samples or []
    runs = bench.get("runs")
    run = runs[0] if isinstance(runs, list) and len(runs) == 1 and isinstance(runs[0], dict) else {}
    anchor = _timestamp(run.get("started_at"))
    buckets: list[list[dict[str, Any]]] = [
        [] for _ in range(math.ceil(duration_seconds / window_seconds))
    ]
    outside = 0
    unaligned = 0
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phases[str(row.get("phase", "unknown"))].append(row)
        at = _timestamp(row.get("at"))
        if anchor is None or at is None:
            unaligned += 1
        elif 0 <= at - anchor < duration_seconds:
            buckets[int((at - anchor) // window_seconds)].append(row)
        else:
            outside += 1
    overall = _resource_group(rows)
    return {
        "provided": samples is not None,
        "interpretation": (
            "Descriptive discrete samples; no leak threshold or resource pass/fail. "
            "Top-level status covers ledger/benchmark acceptance only. Missing is not zero. "
            "Windows approximately align sample.at to run.started_at; sources and probes are "
            "not atomic. Process RSS/PSS, cgroup memory and client memory have different scopes. "
            "First/last use timestamped observations; extrema include untimed values. "
            "Mixed process identities invalidate counter deltas; extrema may span processes. "
            "OOM delta requires two counter observations, consistent PID1 identity, "
            "unchanged membership path and no decrease; "
            "missing/unsampled intervals and cgroup recreation cannot prove absence of OOM."
        ),
        "anchor_started_at": run.get("started_at"),
        "coverage": {
            "sample_rows": len(rows),
            "aligned_to_measured_windows": sum(map(len, buckets)),
            "outside_measured_window": outside,
            "unaligned": unaligned,
            **overall["coverage"],
        },
        "errors": overall["errors"],
        "input_errors": [],
        "overall": overall,
        "by_phase": {phase: _resource_group(group) for phase, group in sorted(phases.items())},
        "windows": [
            {
                "index": index,
                "start_seconds": index * window_seconds,
                "end_seconds": min((index + 1) * window_seconds, duration_seconds),
                **_resource_group(group),
            }
            for index, group in enumerate(buckets)
        ],
    }


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in rows if row["result"] is not None]
    successful = [row for row in paired if row["result"].get("outcome") == "success"]
    outcomes = Counter(
        row["result"].get("outcome", "INVALID") if row["result"] else "UNKNOWN" for row in rows
    )
    valid_latency = [
        row["result"]["elapsed_ms"]
        for row in paired
        if _number(row["result"].get("elapsed_ms")) and row["result"]["elapsed_ms"] >= 0
    ]
    return {
        "issued": len(rows),
        "paired": len(paired),
        "success": len(successful),
        "failed": len(paired) - len(successful),
        "unknown": len(rows) - len(paired),
        "success_rate": len(successful) / len(rows) if rows else None,
        "outcomes": dict(sorted(outcomes.items())),
        "unknown_request_ids": [row["request_id"] for row in rows if row["result"] is None],
        "failed_request_ids": [
            row["request_id"] for row in paired if row["result"].get("outcome") != "success"
        ],
        "empty_successes": sum(row["result"].get("result_count") == 0 for row in successful),
        "latency_all_completed": latency_summary(valid_latency),
        "latency_success": latency_summary(
            [
                row["result"]["elapsed_ms"]
                for row in successful
                if _number(row["result"].get("elapsed_ms")) and row["result"]["elapsed_ms"] >= 0
            ]
        ),
        "completed_without_valid_latency": len(paired) - len(valid_latency),
    }


def _group_counts(rows: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    return {
        **_counts(rows),
        "by_mode": {
            mode: _counts([row for row in rows if row["pending"].get("mode") == mode])
            for mode in modes
        },
    }


def _query_coverage(
    bench: dict[str, Any],
    rows: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    modes: list[str],
    path: Path | None,
    min_repeats: int,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check the supplied input and the measured issue order, excluding warmup repeats."""
    report: dict[str, Any] = {"provided": path is not None, "min_repeats": min_repeats}
    if path is None:
        return {**report, "verified": False, "reason": "expected queries not supplied"}
    report["path"] = str(path)
    try:
        queries, raw = read_queries(path)
    except (OSError, ValueError, UnicodeError) as exc:
        issues.append({"kind": "queries_read_error", "error": type(exc).__name__})
        return {**report, "verified": False}
    first_issue = len(issues)
    declared = bench.get("queries") if isinstance(bench.get("queries"), dict) else {}
    digest = hashlib.sha256(raw).hexdigest()
    report.update(sha256=digest, query_count=len(queries))
    if digest != declared.get("sha256"):
        issues.append(
            {
                "kind": "queries_sha256_mismatch",
                "expected": digest,
                "benchmark": declared.get("sha256"),
            }
        )
    if type(declared.get("count")) is not int or declared["count"] != len(queries):
        issues.append(
            {
                "kind": "queries_count_mismatch",
                "expected": len(queries),
                "benchmark": declared.get("count"),
            }
        )
    issued: Counter[tuple[str, str]] = Counter()
    successful: Counter[tuple[str, str]] = Counter()
    for row in rows:
        event, result = row["pending"], row["result"]
        query, mode = event.get("query"), event.get("mode")
        if (
            event.get("phase") != "measured"
            or not isinstance(query, str)
            or not isinstance(mode, str)
        ):
            continue
        issued[(query, mode)] += 1
        if result is not None and result.get("outcome") == "success":
            successful[(query, mode)] += 1
    expected = {(query, mode) for query in queries for mode in modes}
    missing = sorted(expected - issued.keys())
    unexpected = sorted(issued.keys() - expected)
    groups = [
        {
            "query": query,
            "mode": mode,
            "issued": issued[(query, mode)],
            "successful": successful[(query, mode)],
        }
        for query in queries
        for mode in modes
    ]
    for query, mode in missing:
        issues.append({"kind": "missing_query_mode", "query": query, "mode": mode})
    for query, mode in unexpected:
        issues.append({"kind": "unexpected_query_mode", "query": query, "mode": mode})
    insufficient = [group for group in groups if group["successful"] < min_repeats]
    for group in insufficient:
        issues.append({"kind": "insufficient_query_mode_repeats", **group, "expected": min_repeats})
    order_errors = 0
    for index, event in enumerate(pending):
        query, mode = queries[(index // len(modes)) % len(queries)], modes[index % len(modes)]
        if (
            type(event.get("index")) is not int
            or event["index"] != index
            or event.get("query") != query
            or event.get("mode") != mode
        ):
            order_errors += 1
            issues.append(
                {
                    "kind": "workload_order_mismatch",
                    "request_id": event["request_id"],
                    "expected_index": index,
                    "observed_index": event.get("index"),
                    "expected_query": query,
                    "observed_query": event.get("query"),
                    "expected_mode": mode,
                    "observed_mode": event.get("mode"),
                }
            )
    return {
        **report,
        "verified": len(issues) == first_issue,
        "scope": (
            "measured requests only; repeats require paired successful results; "
            "input order is significant"
        ),
        "expected_group_count": len(expected),
        "observed_group_count": len(issued),
        "missing_groups": [{"query": q, "mode": m} for q, m in missing],
        "unexpected_groups": [{"query": q, "mode": m} for q, m in unexpected],
        "insufficient_repeat_groups": insufficient,
        "groups": groups,
        "workload_order_mismatches": order_errors,
    }


def summarize(
    bench: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    duration_seconds: float = 3600,
    window_seconds: float = 300,
    samples: list[dict[str, Any]] | None = None,
    queries: Path | None = None,
    min_repeats: int = 1,
) -> dict[str, Any]:
    """Reconstruct unique issued IDs; duplicates are ambiguous, never silently deduplicated."""
    if not _number(duration_seconds) or not 0 < duration_seconds <= MAX_DURATION_SECONDS:
        raise ValueError(f"duration_seconds must be finite and in (0, {MAX_DURATION_SECONDS}]")
    if not _number(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if type(min_repeats) is not int or min_repeats < 1:
        raise ValueError("min_repeats must be a positive integer")
    if min_repeats > 1 and queries is None:
        raise ValueError("min_repeats > 1 requires queries")
    issues: list[dict[str, Any]] = []

    def issue(kind: str, **details: Any) -> None:
        issues.append({"kind": kind, **details})

    grouped: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = defaultdict(
        lambda: {"pending": [], "result": []}
    )
    event_counts: Counter[str] = Counter()
    config = bench.get("config") if isinstance(bench.get("config"), dict) else {}
    concurrency = config.get("concurrency")
    concurrency_valid = type(concurrency) is int and concurrency > 0
    if not concurrency_valid and ("concurrency" in config or queries is not None):
        issue("invalid_configured_concurrency", observed=concurrency)
    active: Counter[str] = Counter()
    in_flight = peak_in_flight = exceeded_events = 0
    measured_pending: list[dict[str, Any]] = []
    for line, event in enumerate(events, 1):
        if not isinstance(event, dict):
            issue("invalid_event", line=line)
            continue
        event_type, request_id = event.get("event"), event.get("request_id")
        if (
            event_type not in ("pending", "result")
            or not isinstance(request_id, str)
            or not request_id
        ):
            issue("invalid_event", line=line)
            continue
        event_counts[event_type] += 1
        grouped[request_id][event_type].append((line, event))
        if event_type == "pending":
            active[request_id] += 1
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            if concurrency_valid and in_flight > concurrency:
                exceeded_events += 1
            if event.get("phase") == "measured":
                measured_pending.append(event)
        elif active[request_id]:
            active[request_id] -= 1
            in_flight -= 1
            if not active[request_id]:
                del active[request_id]
        else:
            issue("result_without_inflight_request", line=line, request_id=request_id)
    if exceeded_events:
        issue("concurrency_exceeded", configured=concurrency, observed_peak=peak_in_flight)
    if active:
        issue("inflight_requests_remaining", pending_events=in_flight, request_ids=sorted(active))

    rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    orphans: list[str] = []
    modes = list(MODES)
    match_fields = ("phase", "mode", "query", "index", "dispatched_ms")
    for request_id, group in grouped.items():
        pending, results = group["pending"], group["result"]
        for kind, entries in group.items():
            if len(entries) > 1:
                duplicate = {"request_id": request_id, "event": kind, "count": len(entries)}
                duplicates.append(duplicate)
                issue("duplicate_event", **duplicate)
        if not pending:
            orphans.append(request_id)
            issue("orphan_result", request_id=request_id)
            continue
        first_line, first = pending[0]
        result = None
        if len(pending) == 1 and len(results) == 1:
            result_line, candidate = results[0]
            mismatch = [key for key in match_fields if first.get(key) != candidate.get(key)]
            if mismatch or result_line <= first_line:
                issue("pair_mismatch", request_id=request_id, fields=mismatch)
            else:
                result = candidate
        if result is None:
            issue("unknown_request", request_id=request_id)
        mode = first.get("mode")
        if mode not in MODES:
            issue("invalid_mode", request_id=request_id)
        if first.get("phase") not in ("warmup", "measured"):
            issue("invalid_phase", request_id=request_id)
        if not isinstance(first.get("query"), str) or not first["query"].strip():
            issue("invalid_query", request_id=request_id)
        if result is not None:
            if not isinstance(result.get("outcome"), str):
                issue("invalid_outcome", request_id=request_id)
                result = {**result, "outcome": "INVALID"}
            if not _number(result.get("elapsed_ms")) or result["elapsed_ms"] < 0:
                issue("invalid_latency", request_id=request_id)
            if result.get("outcome") == "success":
                count = result.get("result_count")
                if (
                    result.get("status_code") != 200
                    or not isinstance(result.get("model_version"), str)
                    or not result.get("model_version")
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 0
                ):
                    issue("invalid_success", request_id=request_id)
                if result.get("result_count") == 0:
                    issue("empty_success", request_id=request_id)
            else:
                issue("request_failed", request_id=request_id, outcome=result.get("outcome"))
        rows.append({"request_id": request_id, "pending": first, "result": result})

    phases = {
        phase: _group_counts([row for row in rows if row["pending"].get("phase") == phase], modes)
        for phase in ("warmup", "measured")
    }
    windows: list[dict[str, Any]] = []
    buckets: list[list[dict[str, Any]]] = [
        [] for _ in range(math.ceil(duration_seconds / window_seconds))
    ]
    unassigned: list[str] = []
    for row in rows:
        if row["pending"].get("phase") != "measured":
            continue
        dispatch = row["pending"].get("dispatched_ms")
        if not _number(dispatch) or not 0 <= dispatch < duration_seconds * 1000:
            unassigned.append(row["request_id"])
            issue("invalid_dispatch_time", request_id=row["request_id"])
            continue
        buckets[int(dispatch // (window_seconds * 1000))].append(row)
    for index, bucket in enumerate(buckets):
        if not bucket:
            issue("empty_dispatch_window", index=index)
        windows.append(
            {
                "index": index,
                "start_seconds": index * window_seconds,
                "end_seconds": min((index + 1) * window_seconds, duration_seconds),
                **_group_counts(bucket, modes),
            }
        )

    fingerprints: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        result, pending = row["result"], row["pending"]
        if result is None or result.get("outcome") != "success":
            continue
        if not isinstance(pending.get("query"), str) or pending.get("mode") not in MODES:
            continue
        key = (pending["query"], pending["mode"])
        entry = fingerprints.setdefault(
            key,
            {
                "count": 0,
                "invalid_fingerprint_count": 0,
                "result_sha256": set(),
                "ranking_sha256": set(),
                "model_versions": set(),
            },
        )
        entry["count"] += 1
        for field in ("result_sha256", "ranking_sha256"):
            digest = result.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                issue("missing_or_invalid_fingerprint", request_id=row["request_id"], field=field)
                entry["invalid_fingerprint_count"] += 1
            else:
                entry[field].add(digest)
        version = result.get("model_version")
        if isinstance(version, str):
            entry["model_versions"].add(version)
    fingerprint_rows = []
    for (query, mode), entry in sorted(fingerprints.items()):
        stable = all(len(entry[field]) == 1 for field in ("result_sha256", "ranking_sha256"))
        stable = (
            stable and len(entry["model_versions"]) == 1 and not entry["invalid_fingerprint_count"]
        )
        if not stable:
            issue("unstable_fingerprint", query=query, mode=mode)
        fingerprint_rows.append(
            {
                "query": query,
                "mode": mode,
                "count": entry["count"],
                "invalid_fingerprint_count": entry["invalid_fingerprint_count"],
                "stable": stable,
                **{
                    field: sorted(entry[field])
                    for field in ("result_sha256", "ranking_sha256", "model_versions")
                },
            }
        )
    if bench.get("status") != "COMPLETED":
        issue("benchmark_not_completed", status=bench.get("status"))
    if config.get("duration_seconds") != duration_seconds:
        issue(
            "duration_mismatch", benchmark=config.get("duration_seconds"), expected=duration_seconds
        )
    if not phases["measured"]["issued"]:
        issue("no_measured_requests")
    requested_modes = bench.get("requested_modes", list(MODES))
    if (
        not isinstance(requested_modes, list)
        or not requested_modes
        or any(mode not in MODES for mode in requested_modes)
        or len(set(requested_modes)) != len(requested_modes)
    ):
        issue("invalid_requested_modes")
        requested_modes = list(MODES)
    coverage = _query_coverage(
        bench, rows, measured_pending, requested_modes, queries, min_repeats, issues
    )
    for mode in dict.fromkeys(requested_modes):
        if not phases["measured"]["by_mode"][mode]["issued"]:
            issue("missing_measured_mode", mode=mode)
    warmup = config.get("warmup")
    if isinstance(requested_modes, list) and isinstance(warmup, int) and warmup >= 0:
        for mode in dict.fromkeys(str(mode) for mode in requested_modes):
            observed = phases["warmup"]["by_mode"].get(mode, {}).get("issued", 0)
            if observed != warmup:
                issue("warmup_count_mismatch", mode=mode, expected=warmup, observed=observed)
    runs = bench.get("runs")
    crosscheck: dict[str, Any] = {}
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        issue("expected_single_duration_run")
    else:
        run = runs[0]
        if run.get("mode") != "mixed":
            issue("expected_mixed_duration_run")
        if run.get("dispatch_window_seconds") != duration_seconds:
            issue("run_duration_mismatch", observed=run.get("dispatch_window_seconds"))
        run_summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        measured = phases["measured"]
        for field, expected in (
            ("total", measured["paired"]),
            ("successes", measured["success"]),
            ("failures", measured["failed"]),
        ):
            if run_summary.get(field) != expected:
                issue(
                    "run_ledger_count_mismatch",
                    field=field,
                    expected=expected,
                    observed=run_summary.get(field),
                )
        elapsed = run_summary.get("elapsed_seconds")
        if not _number(elapsed) or elapsed < duration_seconds:
            issue("insufficient_elapsed_duration", observed=elapsed)
        wall_seconds = None
        try:
            start = datetime.fromisoformat(run["started_at"])
            end = datetime.fromisoformat(run["finished_at"])
            if start.tzinfo is None or end.tzinfo is None:
                raise ValueError("timestamps require timezone")
            wall_seconds = (end - start).total_seconds()
            if wall_seconds < duration_seconds:
                issue("insufficient_wall_duration", observed=wall_seconds)
        except (KeyError, TypeError, ValueError):
            issue("invalid_run_timestamps")
        dispatched = [
            row["pending"]["dispatched_ms"] / 1000
            for row in rows
            if row["pending"].get("phase") == "measured"
            and _number(row["pending"].get("dispatched_ms"))
        ]
        observed_last = max(dispatched) if dispatched else None
        last = run.get("last_dispatch_seconds")
        if not _number(last) or not 0 <= last < duration_seconds:
            issue("invalid_run_last_dispatch", observed=last)
        elif observed_last is None or not math.isclose(last, observed_last, abs_tol=1e-6):
            issue("last_dispatch_mismatch", observed=last, ledger=observed_last)
        crosscheck = {
            "elapsed_seconds": elapsed,
            "wall_seconds": wall_seconds,
            "declared_last_dispatch_seconds": last,
            "ledger_last_dispatch_seconds": observed_last,
        }
    return {
        "schema_version": 1,
        "status": "FAILED" if issues else "PASSED",
        "status_scope": "ledger and benchmark accounting only; resources are descriptive evidence",
        "method": {
            "memory": "O(N) offline audit; not service memory or load-generator resource sampling",
            "denominator": (
                "unique IDs with pending; duplicate event counts reported separately; "
                "first pending supplies phase/mode when duplicate IDs are ambiguous"
            ),
            "pairing": "exactly one pending and one later matching result; otherwise UNKNOWN",
            "latency": "nearest rank among completed observations; UNKNOWN has no imputed latency",
            "windows": "half-open dispatch intervals; completions remain in their dispatch window",
            "fingerprints": "successful warmup and measured responses grouped by (query, mode)",
        },
        "duration_seconds": duration_seconds,
        "window_seconds": window_seconds,
        "benchmark_crosscheck": crosscheck,
        "coverage": coverage,
        "inflight": {
            "scope": (
                "client pending/result event order, including warmup; "
                "not server execution concurrency"
            ),
            "configured_concurrency": concurrency,
            "configuration_checked": concurrency_valid,
            "peak": peak_in_flight,
            "pending_events_at_end": in_flight,
            "pending_request_ids_at_end": sorted(active),
            "pending_events_above_limit": exceeded_events,
        },
        "issues": issues,
        "ledger": {
            "event_rows": len(events),
            "pending_events": event_counts["pending"],
            "result_events": event_counts["result"],
            "unique_issued": len(rows),
            "paired": sum(row["result"] is not None for row in rows),
            "duplicates": duplicates,
            "orphan_result_request_ids": orphans,
            "window_unassigned_request_ids": unassigned,
        },
        "phases": phases,
        "windows": windows,
        "fingerprints": {"groups": fingerprint_rows, "group_count": len(fingerprint_rows)},
        "resources": summarize_resources(
            bench, samples, duration_seconds=duration_seconds, window_seconds=window_seconds
        ),
    }


def _read_jsonl(path: Path, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                    rows.append(row)
                except (ValueError, TypeError):
                    errors.append({"kind": "invalid_jsonl", "path": str(path), "line": number})
    except (OSError, UnicodeError) as exc:
        errors.append({"kind": "input_read_error", "path": str(path), "error": type(exc).__name__})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--min-repeats", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=3600,
        help=f"expected dispatch window in (0, {MAX_DURATION_SECONDS}] seconds; default: 3600",
    )
    parser.add_argument("--window-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    inputs = [
        path for path in (args.bench, args.ledger, args.samples, args.queries) if path is not None
    ]
    try:
        for path in inputs:
            if args.out.resolve() == path.resolve() or (
                args.out.exists() and path.exists() and args.out.samefile(path)
            ):
                parser.error("output aliases an input; choose a new output path")
        if args.out.exists():
            parser.error("output already exists; choose a new output path")
    except OSError as exc:
        parser.error(f"cannot validate output path: {exc}")
    errors: list[dict[str, Any]] = []
    bench: dict[str, Any] = {}
    try:
        bench = json.loads(args.bench.read_text(encoding="utf-8"))
        if not isinstance(bench, dict):
            raise ValueError("expected benchmark object")
    except (OSError, ValueError, UnicodeError) as exc:
        errors.append({"kind": "benchmark_read_error", "error": type(exc).__name__})
        bench = {}
    events = _read_jsonl(args.ledger, errors)
    sample_errors: list[dict[str, Any]] = []
    samples = _read_jsonl(args.samples, sample_errors) if args.samples else None
    try:
        report = summarize(
            bench,
            events,
            duration_seconds=args.duration_seconds,
            window_seconds=args.window_seconds,
            samples=samples,
            queries=args.queries,
            min_repeats=args.min_repeats,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "FAILED",
            "issues": [
                {"kind": "analysis_error", "error": type(exc).__name__, "message": str(exc)}
            ],
        }
    report["issues"].extend(errors)
    report.setdefault("resources", {"input_errors": []})["input_errors"].extend(sample_errors)
    if errors:
        report["status"] = "FAILED"
    report["inputs"] = {
        "bench": str(args.bench),
        "ledger": str(args.ledger),
        "samples": str(args.samples) if args.samples else None,
        "queries": str(args.queries) if args.queries else None,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also rejects aliases/files introduced after the initial check.
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except OSError as exc:
        parser.error(f"cannot create output exclusively; choose a new output path: {exc}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
