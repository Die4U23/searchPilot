"""Memory evidence parsing and independent failure handling, without Docker."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from scripts import soak_memory


def proc_stat(start: int = 12345) -> str:
    return "1 (python worker ) main) S " + "0 " * 18 + str(start) + " 0 0\n"


def fixture_roots(tmp_path: Path, membership: str = "/") -> tuple[Path, Path]:
    proc = tmp_path / "proc"
    process = proc / "1"
    process.mkdir(parents=True)
    (process / "stat").write_text(proc_stat())
    (process / "status").write_text("Name:\tpython\nVmRSS:\t1024 kB\nThreads:\t4\n")
    (process / "smaps_rollup").write_text(
        "55bc59100000-7fff8b92b000 ---p 00000000 00:00 0 [rollup]\nRss: 1024 kB\nPss: 900 kB\n"
    )
    (process / "cmdline").write_bytes(b"python\0-m\0uvicorn\0")
    (process / "fd").mkdir()
    (process / "fd" / "0").touch()
    (process / "fd" / "1").touch()
    (process / "cgroup").write_text("0::" + membership + "\n")
    root = tmp_path / "cgroup"
    group = root / membership.lstrip("/")
    group.mkdir(parents=True)
    for name, content in {
        "current": "2097152\n",
        "peak": "3145728\n",
        "stat": "anon 1048576\nfile 524288\npgfault 100\nnew_kernel_metric 7\n",
        "events": "low 0\nhigh 1\noom 0\noom_kill 0\n",
    }.items():
        (group / ("memory." + name)).write_text(content)
    return proc, root


def test_status_byte_units_and_missing_fields() -> None:
    parsed = soak_memory.parse_proc_fields("Name: python\nVmRSS: 5 kB\nThreads: 3\n")
    assert parsed == {"Name": "python", "VmRSS_bytes": 5120, "Threads": 3}
    assert "RssAnon_bytes" not in parsed


@pytest.mark.parametrize("name", ["python: worker", "123", "8 kB"])
def test_status_process_name_is_always_text(name: str) -> None:
    assert soak_memory.parse_proc_fields(f"Name:\t{name}\n") == {"Name": name}


def test_address_header_alone_is_not_a_metric() -> None:
    with pytest.raises(ValueError, match="no proc fields"):
        soak_memory.parse_proc_fields("55bc59100000-7fff8b92b000 ---p 00000000 00:00 0 [rollup]")


def test_proc_stat_with_spaces_and_parentheses() -> None:
    assert soak_memory.parse_proc_stat(proc_stat()) == {
        "pid": 1,
        "comm": "python worker ) main",
        "state": "S",
        "start_time_ticks": 12345,
    }


@pytest.mark.parametrize("membership", ["/", "/nested/service"])
def test_process_and_exact_cgroup_sources(tmp_path: Path, membership: str) -> None:
    proc, cgroup = fixture_roots(tmp_path, membership)
    report = soak_memory.sample_linux_process(proc_root=proc, cgroup_root=cgroup)
    assert report["available"] and report["identity_consistent"]
    assert report["errors"] == {}
    sources = report["sources"]
    assert all(source["available"] for source in sources.values())
    assert sources["status"]["data"]["VmRSS_bytes"] == 1048576
    assert sources["smaps_rollup"]["data"] == {"Rss_bytes": 1048576, "Pss_bytes": 921600}
    assert sources["cmdline"]["data"]["argv"] == ["python", "-m", "uvicorn"]
    assert sources["fd"]["data"]["count"] == 2
    assert sources["cgroup_memory_current"]["data"] == {"bytes": 2097152}
    assert sources["cgroup_memory_stat"]["data"]["bytes"] == {"anon": 1048576, "file": 524288}
    assert sources["cgroup_memory_stat"]["data"]["raw"]["new_kernel_metric"] == 7
    assert sources["cgroup_memory_events"]["data"]["oom_kill"] == 0


def test_missing_and_malformed_sources_do_not_hide_other_evidence(tmp_path: Path) -> None:
    proc, cgroup = fixture_roots(tmp_path)
    (proc / "1" / "smaps_rollup").unlink()
    (cgroup / "memory.peak").unlink()
    (cgroup / "memory.current").write_text("invalid")
    report = soak_memory.sample_linux_process(proc_root=proc, cgroup_root=cgroup)
    assert report["available"]
    assert set(report["errors"]) == {"smaps_rollup", "cgroup_memory_peak", "cgroup_memory_current"}
    assert report["sources"]["smaps_rollup"] == {"available": False}
    assert report["sources"]["status"]["available"]
    assert report["sources"]["cgroup_memory_events"]["available"]


def test_permission_error_is_explicit_and_independent(tmp_path: Path, monkeypatch) -> None:
    proc, cgroup = fixture_roots(tmp_path)
    original = Path.read_text

    def denied(path: Path, *args, **kwargs):
        if path.name == "smaps_rollup":
            raise PermissionError("fixture permission denial")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    report = soak_memory.sample_linux_process(proc_root=proc, cgroup_root=cgroup)
    assert report["errors"]["smaps_rollup"]["error_type"] == "PermissionError"
    assert report["sources"]["cgroup_memory_current"]["available"]


def test_unknown_membership_never_samples_root_group(tmp_path: Path) -> None:
    proc, cgroup = fixture_roots(tmp_path)
    (proc / "1" / "cgroup").write_text("0::/missing\n")
    report = soak_memory.sample_linux_process(proc_root=proc, cgroup_root=cgroup)
    assert not report["sources"]["cgroup_memory_current"]["available"]
    assert "data" not in report["sources"]["cgroup_memory_current"]


def test_process_replacement_is_detected(tmp_path: Path, monkeypatch) -> None:
    proc, cgroup = fixture_roots(tmp_path)
    original = soak_memory.parse_proc_stat
    starts = iter([123, 456])
    monkeypatch.setattr(
        soak_memory, "parse_proc_stat", lambda text: original(proc_stat(next(starts)))
    )
    report = soak_memory.sample_linux_process(proc_root=proc, cgroup_root=cgroup)
    assert report["identity_consistent"] is False


def test_vanished_process_has_no_fabricated_zeroes(tmp_path: Path) -> None:
    report = soak_memory.sample_linux_process(proc_root=tmp_path, cgroup_root=tmp_path)
    assert not report["available"]
    assert report["identity_consistent"] is None
    assert all(source == {"available": False} for source in report["sources"].values())


@pytest.mark.parametrize("text", ["0::/../../other", "0::relative", "2:memory:/group"])
def test_invalid_or_v1_membership_is_not_guessed(text: str) -> None:
    with pytest.raises(ValueError):
        soak_memory.parse_cgroup_membership(text)


def test_windows_sampling_unsupported_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(soak_memory.sys, "platform", "linux")
    report = soak_memory.sample_windows_process(1)
    assert report["available"] is False
    assert report["sources"]["memory"] == {"available": False}
    assert "unsupported" in report["errors"]["memory"]["message"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API boundary")
def test_windows_reads_own_process_and_reports_nonexistent_pid() -> None:
    report = soak_memory.sample_windows_process(os.getpid())
    assert report["errors"] == {}
    data = report["sources"]["memory"]["data"]
    assert data["rss_bytes"] > 0 and data["private_bytes"] > 0
    assert data["peak_rss_bytes"] >= data["rss_bytes"]
    unavailable = soak_memory.sample_windows_process(0xFFFFFFFE)
    assert not unavailable["available"]
    assert unavailable["errors"]["memory"]["error_type"] in {"OSError", "PermissionError"}


def test_cli_emits_exactly_one_json_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr(soak_memory, "sample_linux_process", lambda pid: {"pid": pid})
    assert soak_memory.main(["--pid", "7"]) == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert json.loads(output) == {"pid": 7}


def test_cli_rejects_invalid_pid() -> None:
    with pytest.raises(SystemExit) as error:
        soak_memory.main(["--pid", "0"])
    assert error.value.code == 2
