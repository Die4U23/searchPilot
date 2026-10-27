"""Adversarial resource evidence, without Docker, models, network or application startup."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from scripts.audit_memory_stability import MIB, audit, main

START = datetime(2026, 10, 26, tzinfo=UTC)


def timestamp(seconds: float) -> str:
    return (START + timedelta(seconds=seconds)).isoformat()


def evidence(duration: int = 7200):
    requests = [
        {"dispatched_ms": seconds * 1000, "elapsed_ms": 1000} for seconds in range(duration)
    ]
    bench = {
        "status": "COMPLETED",
        "config": {"duration_seconds": duration},
        "runs": [
            {
                "mode": "mixed",
                "dispatch_window_seconds": duration,
                "started_at": timestamp(0),
                "finished_at": timestamp(duration),
                "last_dispatch_seconds": duration - 1,
                "summary": {"elapsed_seconds": duration, "total": len(requests)},
                "requests": requests,
            }
        ],
    }
    samples = []
    for seconds in range(0, duration, 10):
        identity = {"pid": 1, "start_time_ticks": 1234}
        data = {
            "stat": dict(identity),
            "stat_after": dict(identity),
            "smaps_rollup": {
                "Rss_bytes": 1000 * MIB,
                "Pss_bytes": 990 * MIB,
                "Anonymous_bytes": 900 * MIB,
            },
            "status": {"VmData_bytes": 3000 * MIB},
            "cgroup_memory_current": {"bytes": 1100 * MIB},
        }
        samples.append(
            {
                "at": timestamp(seconds),
                "phase": "load",
                "server": {
                    "pid": 1,
                    "at": timestamp(seconds + 0.1),
                    "available": True,
                    "identity_consistent": True,
                    "sources": {
                        key: {"available": True, "data": value} for key, value in data.items()
                    },
                },
            }
        )
    return bench, samples


def increase(row, amount: float, name="Rss_bytes") -> None:
    row["server"]["sources"]["smaps_rollup"]["data"][name] += amount * MIB


def test_two_hour_fixed_windows_and_half_open_boundary() -> None:
    bench, samples = evidence()
    outside = json.loads(json.dumps(samples[-1]))
    outside["at"] = timestamp(7200)
    outside["server"]["at"] = timestamp(7200.1)
    increase(outside, 100)
    result = audit(bench, samples + [outside])
    assert result["status"] == "PASSED"
    assert result["windows"]["full_dispatch"]["count"] == 720
    assert len(result["windows"]["five_minutes"]) == 24
    assert [w["start_seconds"] for w in result["windows"]["rolling_twenty_minutes"]] == list(
        range(3600, 6001, 300)
    )
    assert result["windows"]["stable"]["count"] == 360


def test_positive_trend_cannot_hide_behind_flat_last_twenty() -> None:
    bench, samples = evidence()
    for index, row in enumerate(samples):
        increase(row, max(0, min(index * 10 - 3600, 2400)) / 60 * 0.08)
    result = audit(bench, samples)
    assert result["windows"]["last_twenty"]["metrics"]["rss"]["growth_mib"] == 0
    assert result["status"] == "FAILED"
    assert result["gates"]["rss"]["checks"]["stable_ols"] is False


def test_step_inside_window_is_not_masked_by_stable_endpoints() -> None:
    bench, samples = evidence()
    for index in range(420, 480):
        increase(samples[index], 5)
    result = audit(bench, samples)
    assert result["windows"]["stable"]["metrics"]["rss"]["growth_mib"] == 0
    assert result["gates"]["rss"]["checks"]["head_tail_ten_median"] is True
    assert result["gates"]["rss"]["checks"]["five_minute_median_range"] is False
    assert result["gates"]["rss"]["checks"]["every_rolling_twenty"] is False
    assert result["status"] == "FAILED"


def test_anonymous_is_gated_even_when_rss_is_constant() -> None:
    bench, samples = evidence()
    for index, row in enumerate(samples):
        increase(row, max(0, index * 10 - 3600) / 60 * 0.1, "Anonymous_bytes")
    result = audit(bench, samples)
    assert result["gates"]["rss"]["passed"] is True
    assert result["gates"]["anonymous"]["passed"] is False
    assert result["status"] == "FAILED"


@pytest.mark.parametrize("slope,expected", [(0.03, "PASSED"), (0.03001, "FAILED")])
def test_exact_slope_limit_uses_only_roundoff_tolerance(slope: float, expected: str) -> None:
    bench, samples = evidence()
    for index, row in enumerate(samples):
        increase(row, max(0, index * 10 - 3600) / 60 * slope)
    assert audit(bench, samples)["status"] == expected


def test_four_hour_configuration_keeps_every_stable_window() -> None:
    bench, samples = evidence(14400)
    result = audit(bench, samples, duration_seconds=14400)
    assert result["status"] == "PASSED"
    assert len(result["windows"]["rolling_twenty_minutes"]) == 33


def test_boolean_pid_is_not_a_process_identity() -> None:
    bench, samples = evidence()
    samples[420]["server"]["sources"]["stat_after"]["data"]["pid"] = True
    assert audit(bench, samples)["status"] == "FAILED"


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "nan",
        "negative",
        "new_pid",
        "identity_flag",
        "source_flag",
        "duplicate_time",
        "reverse_time",
        "server_time",
    ],
)
def test_invalid_samples_cannot_pass(fault: str) -> None:
    bench, samples = evidence()
    row = samples[420]
    source = row["server"]["sources"]
    if fault == "missing":
        source["smaps_rollup"]["data"].pop("Rss_bytes")
    elif fault in ("nan", "negative"):
        source["smaps_rollup"]["data"]["Anonymous_bytes"] = float("nan") if fault == "nan" else -1
    elif fault == "new_pid":
        for name in ("stat", "stat_after"):
            source[name]["data"]["start_time_ticks"] = 5678
    elif fault == "identity_flag":
        row["server"].pop("identity_consistent")
    elif fault == "source_flag":
        source["smaps_rollup"].pop("available")
    elif fault in ("duplicate_time", "reverse_time"):
        row["at"] = samples[419 if fault == "duplicate_time" else 418]["at"]
    else:
        row["server"]["at"] = timestamp(100)
    result = audit(bench, samples)
    assert result["status"] == "FAILED"
    assert result["errors"]


@pytest.mark.parametrize("fault", ["gap", "start_boundary", "end_boundary"])
def test_sampling_coverage(fault: str) -> None:
    bench, samples = evidence()
    if fault == "gap":
        del samples[420]
    elif fault == "start_boundary":
        samples = samples[2:]
    else:
        samples = samples[:-1]
    assert audit(bench, samples)["status"] == "FAILED"


def test_missing_descriptive_series_are_explicit_not_zero() -> None:
    bench, samples = evidence()
    for row in samples:
        row["server"]["sources"]["smaps_rollup"]["data"].pop("Pss_bytes")
        row["server"]["sources"].pop("cgroup_memory_current")
    result = audit(bench, samples)
    assert result["status"] == "PASSED"
    for name in ("pss", "cgroup_current"):
        metric = result["windows"]["stable"]["metrics"][name]
        assert metric == {"available": False, "valid_count": 0, "missing_count": 360}


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "declared_duration",
        "clock",
        "last_dispatch",
        "request_latency",
        "empty_requests",
        "incomplete",
    ],
)
def test_benchmark_duration_requires_corrobating_evidence(fault: str) -> None:
    bench, samples = evidence()
    run = bench["runs"][0]
    if fault == "short":
        result = audit(bench, samples, duration_seconds=3600, stable_after_seconds=1200)
        assert result["status"] == "FAILED"
        assert "at least 3600" in result["errors"][0]
        return
    if fault == "declared_duration":
        bench["config"]["duration_seconds"] = 3600
    elif fault == "clock":
        run["finished_at"] = timestamp(3600)
    elif fault == "last_dispatch":
        run["requests"] = run["requests"][:3600]
        run["summary"]["total"] = 3600
        run["last_dispatch_seconds"] = 3599
    elif fault == "request_latency":
        run["requests"][0]["elapsed_ms"] = float("inf")
    elif fault == "empty_requests":
        run["requests"] = []
    else:
        bench["status"] = "RUNNING"
    result = audit(bench, samples)
    assert result["status"] == "FAILED"
    assert result["errors"]


def write_inputs(tmp_path: Path, bench, samples) -> list[str]:
    bench_path, samples_path = tmp_path / "bench.json", tmp_path / "samples.jsonl"
    bench_path.write_text(json.dumps(bench), encoding="utf-8")
    samples_path.write_text("\n".join(json.dumps(row) for row in samples), encoding="utf-8")
    return [
        "--bench",
        str(bench_path),
        "--samples",
        str(samples_path),
        "--out",
        str(tmp_path / "audit.json"),
    ]


def test_cli_writes_report_hashes_and_refuses_overwrite(tmp_path: Path) -> None:
    args = write_inputs(tmp_path, *evidence())
    assert main(args) == 0
    report = json.loads((tmp_path / "audit.json").read_text())
    assert len(report["inputs"]["bench_sha256"]) == 64
    with pytest.raises(FileExistsError):
        main(args)


@pytest.mark.parametrize("bad", ["NaN", "{", "[]"])
def test_cli_malformed_input_fails_explicitly(tmp_path: Path, bad: str) -> None:
    args = write_inputs(tmp_path, *evidence())
    (tmp_path / "samples.jsonl").write_text(bad, encoding="utf-8")
    assert main(args) == 1
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["status"] == "FAILED"
    assert report["errors"]


def test_cli_nonfinite_window_argument_writes_failure(tmp_path: Path) -> None:
    args = write_inputs(tmp_path, *evidence()) + ["--duration-seconds", "nan"]
    assert main(args) == 1
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["status"] == "FAILED"
