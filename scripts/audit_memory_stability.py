"""Offline resource stability audit; does not replace request/business acceptance.

All dispatch windows are half-open. RSS and anonymous memory have fixed gates;
PSS, VmData and cgroup memory are separate descriptive series, never added together.
This tool reads evidence only and never starts an application, container or load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
METRICS = {
    "rss": ("smaps_rollup", "Rss_bytes"),
    "anonymous": ("smaps_rollup", "Anonymous_bytes"),
    "pss": ("smaps_rollup", "Pss_bytes"),
    "vmdata": ("status", "VmData_bytes"),
    "cgroup_current": ("cgroup_memory_current", "bytes"),
}
GATED = ("rss", "anonymous")
LIMITS = {
    "stable_growth_mib": 2.0,
    "stable_ols_mib_per_minute": 0.03,
    "head_tail_ten_minute_median_growth_mib": 2.0,
    "five_minute_median_range_mib": 2.0,
    "rolling_twenty_growth_mib": 2.0,
    "rolling_twenty_ols_mib_per_minute": 0.1,
    "sample_interval_seconds": 10,
    "maximum_gap_seconds": 15,
    "boundary_tolerance_seconds": 10,
    "benchmark_clock_tolerance_seconds": 1,
    "floating_point_tolerance": 1e-9,
}


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _at_most(value: float | None, maximum: float) -> bool:
    return value is not None and _number(value) and value <= maximum + 1e-9


def _timestamp(value: Any) -> float | None:
    try:
        stamp = datetime.fromisoformat(value)
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _source(row: dict[str, Any], name: str) -> dict[str, Any]:
    server = _mapping(row.get("server"))
    source = _mapping(_mapping(server.get("sources")).get(name))
    return _mapping(source.get("data")) if source.get("available") is True else {}


def window(rows: list[dict[str, Any]], left: float, right: float) -> dict[str, Any]:
    """Describe a fixed half-open window without sorting or dropping missing metric values."""
    selected = [row for row in rows if left <= row["seconds"] < right]
    xs = [row["seconds"] for row in selected]
    minimum = max(2, math.floor((right - left) / 10) - 2)
    gaps = [b - a for a, b in zip(xs, xs[1:], strict=False)]
    coverage = (
        len(xs) >= minimum
        and bool(gaps)
        and all(0 < gap <= 15 for gap in gaps)
        and xs[0] <= left + 10
        and xs[-1] >= right - 10
    )
    metrics: dict[str, Any] = {}
    for name in METRICS:
        valid = [
            (row["seconds"], row["values"][name])
            for row in selected
            if row["values"][name] is not None
        ]
        complete = len(valid) == len(selected) and len(valid) >= 2
        result: dict[str, Any] = {
            "available": complete,
            "valid_count": len(valid),
            "missing_count": len(selected) - len(valid),
        }
        if complete:
            ys = [value for _, value in valid]
            mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
            denominator = sum((x - mean_x) ** 2 for x in xs)
            slope = (
                (
                    sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
                    / denominator
                    * 60
                )
                if denominator
                else None
            )
            if not _number(slope):
                metrics[name] = {**result, "available": False, "reason": "invalid derived slope"}
                continue
            result.update(
                first_mib=ys[0],
                last_mib=ys[-1],
                growth_mib=ys[-1] - ys[0],
                minimum_mib=min(ys),
                maximum_mib=max(ys),
                median_mib=statistics.median(ys),
                ols_mib_per_minute=slope,
            )
        metrics[name] = result
    return {
        "start_seconds": left,
        "end_seconds": right,
        "count": len(selected),
        "minimum_count": minimum,
        "coverage_passed": coverage,
        "first_seconds": xs[0] if xs else None,
        "last_seconds": xs[-1] if xs else None,
        "span_seconds": xs[-1] - xs[0] if xs else None,
        "maximum_gap_seconds": max(gaps) if gaps else None,
        "metrics": metrics,
    }


def _benchmark(
    bench: Any, duration: float, errors: list[str]
) -> tuple[float | None, dict[str, Any]]:
    bench = _mapping(bench)
    runs = bench.get("runs")
    if bench.get("status") != "COMPLETED" or not isinstance(runs, list) or len(runs) != 1:
        errors.append("benchmark must be COMPLETED with exactly one run")
        return None, {}
    run = _mapping(runs[0])
    summary = _mapping(run.get("summary"))
    start, end = _timestamp(run.get("started_at")), _timestamp(run.get("finished_at"))
    elapsed = summary.get("elapsed_seconds")
    declared = (
        _mapping(bench.get("config")).get("duration_seconds"),
        run.get("dispatch_window_seconds"),
    )
    if run.get("mode") != "mixed" or any(not _number(v) or v != duration for v in declared):
        errors.append("benchmark dispatch duration/mode does not match requested audit")
    if (
        start is None
        or end is None
        or not _number(elapsed)
        or elapsed < duration
        or end <= start
        or abs(end - start - elapsed) > 1
    ):
        errors.append("benchmark wall-clock and elapsed duration are missing or inconsistent")
    requests = run.get("requests")
    if not isinstance(requests, list) or not requests:
        errors.append("benchmark has no request timing evidence")
        return start, {}
    if type(summary.get("total")) is not int or summary["total"] != len(requests):
        errors.append("benchmark request count does not match timing records")
    dispatches, completions = [], []
    for index, record in enumerate(requests):
        record = _mapping(record)
        dispatch, latency = record.get("dispatched_ms"), record.get("elapsed_ms")
        if (
            not _number(dispatch)
            or not _number(latency)
            or not 0 <= dispatch < duration * 1000
            or latency < 0
        ):
            errors.append(f"invalid request timing at record {index}")
            continue
        dispatches.append(dispatch / 1000)
        completions.append((dispatch + latency) / 1000)
    if len(dispatches) != len(requests):
        return start, {"valid_request_timings": len(dispatches)}
    last = run.get("last_dispatch_seconds")
    if (
        any(b < a for a, b in zip(dispatches, dispatches[1:], strict=False))
        or dispatches[0] > 10
        or dispatches[-1] < duration - 10
        or not _number(last)
        or abs(last - dispatches[-1]) > 0.001
    ):
        errors.append("request dispatch ordering or boundary coverage is inconsistent")
    if not _number(elapsed) or abs(max(completions) - elapsed) > 1:
        errors.append("last request completion does not corroborate elapsed duration")
    return start, {
        "valid_request_timings": len(dispatches),
        "first_dispatch_seconds": dispatches[0],
        "last_dispatch_seconds": dispatches[-1],
        "last_completion_seconds": max(completions),
        "wall_clock_seconds": end - start if start is not None and end is not None else None,
        "elapsed_seconds": elapsed if _number(elapsed) else None,
    }


def audit(
    bench: Any,
    samples: list[Any],
    *,
    duration_seconds: float = 7200,
    stable_after_seconds: float = 3600,
) -> dict[str, Any]:
    """Apply predeclared gates; malformed/short evidence cannot become an N/A pass."""
    errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAILED",
        "scope": "resource_stability_only",
        "duration_seconds": duration_seconds if _number(duration_seconds) else None,
        "stable_after_seconds": stable_after_seconds if _number(stable_after_seconds) else None,
        "limits": dict(LIMITS),
        "errors": errors,
        "limitations": [
            "Finite sampled evidence does not prove absence of leaks.",
            "Request ledger, query coverage, health and business checks are separate.",
            "PSS, VmData and cgroup memory are descriptive, not RSS causes.",
        ],
    }
    if (
        not _number(duration_seconds)
        or not _number(stable_after_seconds)
        or not 0 <= stable_after_seconds < duration_seconds <= 14400
        or duration_seconds - stable_after_seconds < 3600
    ):
        errors.append("duration must be <=14400 and stable period must be at least 3600 seconds")
        return report
    start, timing = _benchmark(bench, duration_seconds, errors)
    report["benchmark_timing"] = timing
    if start is None:
        return report
    load, recovery = [], []
    previous = None
    starts: set[int] = set()
    server_times: list[float] = []
    for line, raw in enumerate(samples, 1):
        row = _mapping(raw)
        at = _timestamp(row.get("at"))
        if at is None:
            errors.append(f"sample {line}: missing timezone-aware timestamp")
            continue
        if previous is not None and at <= previous:
            errors.append(f"sample {line}: timestamps must strictly increase")
        previous = at
        seconds = at - start
        formal = 0 <= seconds < duration_seconds
        if formal and row.get("phase") != "load":
            errors.append(f"sample {line}: non-load phase inside dispatch window")
        if not (formal or row.get("phase") == "recovery"):
            continue
        values: dict[str, float | None] = {}
        for name, (source, key) in METRICS.items():
            data = _source(row, source)
            value = data.get(key)
            good = _number(value) and value >= 0
            values[name] = value / MIB if good else None
            if formal and ((name in GATED and not good) or (value is not None and not good)):
                errors.append(f"sample {line}: unavailable or invalid {name}")
        if formal:
            server = _mapping(row.get("server"))
            before, after = _source(row, "stat"), _source(row, "stat_after")
            ticks = before.get("start_time_ticks")
            if (
                server.get("available") is not True
                or server.get("identity_consistent") is not True
                or any(
                    type(item.get("pid")) is not int or item.get("pid") != 1
                    for item in (server, before, after)
                )
                or type(ticks) is not int
                or ticks < 0
                or type(after.get("start_time_ticks")) is not int
                or after.get("start_time_ticks") != ticks
            ):
                errors.append(f"sample {line}: missing or inconsistent PID identity")
            elif type(ticks) is int:
                starts.add(ticks)
            server_at = _timestamp(server.get("at"))
            if server_at is None or not 0 <= server_at - at <= 15:
                errors.append(f"sample {line}: server timestamp missing or outside sample interval")
            else:
                server_times.append(server_at)
        normalized = {"seconds": seconds, "values": values, "line": line}
        (load if formal else recovery).append(normalized)
    if len(starts) != 1 or any(
        b <= a for a, b in zip(server_times, server_times[1:], strict=False)
    ):
        errors.append("formal samples do not establish one process with increasing server times")
    report["pid_start_time_ticks"] = sorted(starts)
    full = window(load, 0, duration_seconds)
    stable = window(load, stable_after_seconds, duration_seconds)
    five = [
        window(load, left, min(left + 300, duration_seconds))
        for left in range(0, math.ceil(duration_seconds), 300)
    ]
    stable_five = [
        window(load, left, min(left + 300, duration_seconds))
        for left in (
            stable_after_seconds + 300 * index
            for index in range(math.ceil((duration_seconds - stable_after_seconds) / 300))
        )
    ]
    rolling = [
        window(load, stable_after_seconds + 300 * index, stable_after_seconds + 300 * index + 1200)
        for index in range(math.floor((duration_seconds - stable_after_seconds - 1200) / 300) + 1)
    ]
    head = window(load, stable_after_seconds, stable_after_seconds + 600)
    tail = window(load, duration_seconds - 600, duration_seconds)
    last_twenty = window(load, duration_seconds - 1200, duration_seconds)
    report["windows"] = {
        "full_dispatch": full,
        "stable": stable,
        "five_minutes": five,
        "stable_five_minutes": stable_five,
        "rolling_twenty_minutes": rolling,
        "stable_first_ten": head,
        "last_ten": tail,
        "last_twenty": last_twenty,
        "recovery": window(recovery, recovery[0]["seconds"], recovery[-1]["seconds"] + 10)
        if recovery
        else {"available": False, "reason": "no recovery samples"},
    }
    required = [full, stable, head, tail, last_twenty, *five, *stable_five, *rolling]
    if any(not item["coverage_passed"] for item in required):
        errors.append("one or more fixed windows lack boundary/count/gap coverage")
    gates: dict[str, Any] = {}
    for name in GATED:
        if any(not item["metrics"][name]["available"] for item in required):
            gates[name] = {"passed": False, "reason": "incomplete metric evidence"}
            continue
        whole = stable["metrics"][name]
        medians = [item["metrics"][name]["median_mib"] for item in stable_five]
        median_growth = tail["metrics"][name]["median_mib"] - head["metrics"][name]["median_mib"]
        median_range = max(medians) - min(medians)
        checks = {
            "stable_growth": _at_most(whole["growth_mib"], LIMITS["stable_growth_mib"]),
            "stable_ols": _at_most(
                whole["ols_mib_per_minute"], LIMITS["stable_ols_mib_per_minute"]
            ),
            "head_tail_ten_median": _at_most(
                median_growth, LIMITS["head_tail_ten_minute_median_growth_mib"]
            ),
            "five_minute_median_range": _at_most(
                median_range, LIMITS["five_minute_median_range_mib"]
            ),
            "every_rolling_twenty": all(
                _at_most(item["metrics"][name]["growth_mib"], LIMITS["rolling_twenty_growth_mib"])
                and _at_most(
                    item["metrics"][name]["ols_mib_per_minute"],
                    LIMITS["rolling_twenty_ols_mib_per_minute"],
                )
                for item in rolling
            ),
        }
        gates[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "head_tail_ten_median_growth_mib": median_growth,
            "five_minute_median_range_mib": median_range,
        }
    report["gates"] = gates
    if not errors and all(item["passed"] for item in gates.values()):
        report["status"] = "PASSED"
    return report


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--duration-seconds", type=float, default=7200)
    parser.add_argument("--stable-after-seconds", type=float, default=3600)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        bench_raw, samples_raw = args.bench.read_bytes(), args.samples.read_bytes()
        bench = json.loads(bench_raw, parse_constant=_reject_constant)
        samples = [
            json.loads(line, parse_constant=_reject_constant)
            for line in samples_raw.decode("utf-8").splitlines()
        ]
        report = audit(
            bench,
            samples,
            duration_seconds=args.duration_seconds,
            stable_after_seconds=args.stable_after_seconds,
        )
        report["inputs"] = {
            "bench_sha256": hashlib.sha256(bench_raw).hexdigest(),
            "samples_sha256": hashlib.sha256(samples_raw).hexdigest(),
        }
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        report = {
            "status": "FAILED",
            "scope": "resource_stability_only",
            "errors": [f"invalid input: {type(exc).__name__}: {exc}"],
        }
    with args.out.open("x", encoding="utf-8") as sink:
        json.dump(report, sink, indent=2, allow_nan=False)
        sink.write("\n")
    print(json.dumps({"status": report["status"], "errors": report.get("errors", [])}))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
