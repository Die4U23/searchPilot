"""One-shot memory evidence; standard library only, suitable for docker exec.

Missing sources are reported independently and never represented as zero bytes.
Process RSS and cgroup memory describe different scopes and are not interchangeable.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

CGROUP_BYTE_KEYS = frozenset(
    [
        "anon",
        "file",
        "kernel",
        "kernel_stack",
        "pagetables",
        "sec_pagetables",
        "percpu",
        "sock",
        "vmalloc",
        "shmem",
        "zswap",
        "zswapped",
        "file_mapped",
        "file_dirty",
        "file_writeback",
        "swapcached",
        "anon_thp",
        "file_thp",
        "shmem_thp",
        "inactive_anon",
        "active_anon",
        "inactive_file",
        "active_file",
        "unevictable",
        "slab_reclaimable",
        "slab_unreclaimable",
        "slab",
    ]
)


def parse_proc_fields(text: str) -> dict[str, Any]:
    """Parse status/smaps values, explicitly converting Linux kB (1024 bytes)."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            continue  # The smaps address header contains a device colon, not a field.
        tokens = value.split()
        if key == "Name":
            result[key] = value.strip()
        elif len(tokens) == 2 and tokens[1] == "kB":
            result[key + "_bytes"] = int(tokens[0]) * 1024
        elif len(tokens) == 1 and tokens[0].isdecimal():
            result[key] = int(tokens[0])
        else:
            result[key] = value.strip()
    if not result:
        raise ValueError("no proc fields found")
    return result


def parse_proc_stat(text: str) -> dict[str, Any]:
    """Field 22 is the process start time, even when comm contains spaces or ')'."""
    opening, closing = text.find("("), text.rfind(")")
    if opening < 1 or closing <= opening:
        raise ValueError("invalid proc stat command field")
    fields = text[closing + 1 :].split()
    if len(fields) < 20:
        raise ValueError("proc stat lacks start time")
    return {
        "pid": int(text[:opening].strip()),
        "comm": text[opening + 1 : closing],
        "state": fields[0],
        "start_time_ticks": int(fields[19]),
    }


def parse_key_values(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        if key in result:
            raise ValueError(f"duplicate metric: {key}")
        result[key] = int(value)
    if not result:
        raise ValueError("no metrics found")
    return result


def parse_cgroup_stat(text: str) -> dict[str, Any]:
    values = parse_key_values(text)
    return {
        "bytes": {key: value for key, value in values.items() if key in CGROUP_BYTE_KEYS},
        # Preserve unknown/new kernel fields without guessing their units.
        "raw": values,
    }


def parse_cgroup_membership(text: str) -> dict[str, str]:
    for line in text.splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            parsed = PurePosixPath(path)
            if not parsed.is_absolute() or ".." in parsed.parts:
                raise ValueError("cgroup v2 path is not a safe absolute path")
            return {"path": str(parsed)}
    raise ValueError("cgroup v2 membership not found")


def _new_sample(pid: int, platform: str) -> dict[str, Any]:
    if pid < 1:
        raise ValueError("pid must be positive")
    return {
        "schema_version": 1,
        "pid": pid,
        "platform": platform,
        "at": datetime.now(UTC).isoformat(),
        "available": False,
        "sources": {},
        "errors": {},
    }


def _source(report: dict[str, Any], name: str, read: Callable[[], Any]) -> None:
    try:
        report["sources"][name] = {"available": True, "data": read()}
        report["available"] = True
    except (OSError, ValueError) as exc:
        report["sources"][name] = {"available": False}
        report["errors"][name] = {"error_type": type(exc).__name__, "message": str(exc)}


def sample_linux_process(
    pid: int = 1,
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    """Read one process and its cgroup v2 under the supplied mounted cgroup root.

    cgroup_root must expose the same cgroup namespace as proc_root. Docker's
    default private cgroup namespace exposes the container membership as '/'.
    A non-root membership is resolved beneath cgroup_root, never silently replaced
    with the root's counters. Sources are sequential, not an atomic snapshot.
    """
    report = _new_sample(pid, "linux")
    process = proc_root / str(pid)
    _source(report, "stat", lambda: parse_proc_stat((process / "stat").read_text()))

    def read_proc_fields(filename: str) -> dict[str, Any]:
        return parse_proc_fields((process / filename).read_text())

    for filename in ("status", "smaps_rollup"):
        _source(report, filename, partial(read_proc_fields, filename))
    _source(
        report,
        "cmdline",
        lambda: {
            "argv": [
                value.decode("utf-8", errors="replace")
                for value in (process / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        },
    )
    _source(report, "fd", lambda: {"count": sum(1 for _ in (process / "fd").iterdir())})
    _source(
        report,
        "cgroup_membership",
        lambda: parse_cgroup_membership((process / "cgroup").read_text()),
    )

    def read_cgroup(filename: str) -> str:
        membership = report["sources"]["cgroup_membership"]
        if not membership["available"]:
            raise ValueError("cgroup v2 membership unavailable; target counters unknown")
        directory = cgroup_root / membership["data"]["path"].lstrip("/")
        return (directory / filename).read_text()

    def read_cgroup_bytes(suffix: str) -> dict[str, int]:
        return {"bytes": int(read_cgroup("memory." + suffix).strip())}

    for suffix in ("current", "peak"):
        _source(report, "cgroup_memory_" + suffix, partial(read_cgroup_bytes, suffix))
    _source(report, "cgroup_memory_stat", lambda: parse_cgroup_stat(read_cgroup("memory.stat")))
    _source(report, "cgroup_memory_events", lambda: parse_key_values(read_cgroup("memory.events")))
    _source(report, "stat_after", lambda: parse_proc_stat((process / "stat").read_text()))
    before, after = report["sources"]["stat"], report["sources"]["stat_after"]
    report["identity_consistent"] = (
        before["data"]["start_time_ticks"] == after["data"]["start_time_ticks"]
        if before["available"] and after["available"]
        else None
    )
    return report


def _read_windows_memory(pid: int) -> dict[str, int]:
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "rss_bytes": counters.WorkingSetSize,
            "private_bytes": counters.PrivateUsage,
            "peak_rss_bytes": counters.PeakWorkingSetSize,
        }
    finally:
        kernel.CloseHandle(handle)


def sample_windows_process(pid: int) -> dict[str, Any]:
    """Read Windows working set and private committed bytes for exactly this PID."""
    report = _new_sample(pid, sys.platform)

    def read() -> dict[str, int]:
        if sys.platform != "win32":
            raise OSError("Windows process memory sampling is unsupported on this platform")
        return _read_windows_memory(pid)

    _source(report, "memory", read)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=1)
    args = parser.parse_args(argv)
    if args.pid < 1:
        parser.error("pid must be positive")
    print(json.dumps(sample_linux_process(args.pid), ensure_ascii=True), flush=True)
    return 0  # Partial/unavailable evidence remains valid JSON for the caller to inspect.


if __name__ == "__main__":
    raise SystemExit(main())
