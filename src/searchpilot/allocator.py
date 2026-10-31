"""Opt-in glibc free-page residency maintenance; no GC or application imports.

Default disabled creates no thread/native binding. Diagnostic noop deliberately
uses the same single worker/schedule but never calls malloc_trim. This prototype
does not establish an allocation call stack or prove absence of a live leak.
"""

from __future__ import annotations

import ctypes
import logging
import math
import platform
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass

type Scalar = str | int | float | bool | None
type Tick = dict[str, Scalar]
type Trim = Callable[[], int]
type TickCallback = Callable[[Tick], None]


class UnsupportedAllocatorError(RuntimeError):
    """Explicitly enabled maintenance requires Linux/glibc with malloc_trim."""


UnsupportedAllocator = UnsupportedAllocatorError


def _native_trim() -> Trim:
    if sys.platform != "linux" or platform.libc_ver()[0] != "glibc":
        raise UnsupportedAllocator("allocator residency maintenance requires Linux/glibc")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.malloc_trim
    except (OSError, AttributeError) as exc:
        raise UnsupportedAllocator("glibc malloc_trim is unavailable") from exc
    function.argtypes = [ctypes.c_size_t]
    function.restype = ctypes.c_int

    def trim() -> int:
        # CDLL, not PyDLL: native call releases the GIL. Return0/1 is NOT bytes.
        return int(function(0))

    return trim


def next_deadline(scheduled: float, finished: float, interval: float) -> tuple[float, int]:
    """Advance fixed phase; overdue periods are skipped, never queued/caught up."""
    upcoming = scheduled + interval
    missed = 0
    if upcoming <= finished:
        missed = math.floor((finished - upcoming) / interval) + 1
        upcoming += missed * interval
    return upcoming, missed


@dataclass(slots=True)
class _Telemetry:
    ticks: int = 0
    completed_ticks: int = 0
    actual_calls: int = 0
    noop_ticks: int = 0
    return_zero: int = 0
    return_one: int = 0
    errors: int = 0
    callback_errors: int = 0
    overlap: int = 0
    missed: int = 0
    close_timeouts: int = 0
    close_timed_out: bool = False
    started_mono: float | None = None
    first_due_mono: float | None = None
    next_due_mono: float | None = None
    stop_requested_mono: float | None = None
    closed_mono: float | None = None
    last_scheduled_mono: float | None = None
    last_started_mono: float | None = None
    last_ended_mono: float | None = None
    last_completed_ended_mono: float | None = None
    last_duration_seconds: float | None = None
    total_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    last_lateness_seconds: float | None = None
    max_lateness_seconds: float = 0.0
    max_early_seconds: float = 0.0
    last_start_gap_seconds: float | None = None
    max_start_gap_seconds: float = 0.0
    last_callback_duration_seconds: float | None = None
    total_callback_duration_seconds: float = 0.0
    max_tick_total_seconds: float = 0.0
    last_error_type: str | None = None
    last_callback_error_type: str | None = None


class ResidencyReclaimer:
    def __init__(
        self,
        interval_seconds: float = 7.0,
        *,
        enabled: bool = False,
        diagnostic_noop: bool = False,
        on_tick: TickCallback | None = None,
        _trim: Trim | None = None,
    ) -> None:
        if type(enabled) is not bool or type(diagnostic_noop) is not bool:
            raise ValueError("enabled/diagnostic_noop must be booleans")
        if enabled and diagnostic_noop:
            raise ValueError("real maintenance and diagnostic noop are mutually exclusive")
        if (
            type(interval_seconds) not in (int, float)
            or not math.isfinite(interval_seconds)
            or not 0 < interval_seconds <= threading.TIMEOUT_MAX
        ):
            raise ValueError("interval_seconds must be finite and positive")
        if on_tick is not None and not callable(on_tick):
            raise ValueError("on_tick must be callable or None")
        if _trim is not None and not callable(_trim):
            raise ValueError("injected trim must be callable")
        self._interval_seconds = float(interval_seconds)
        self._enabled, self._diagnostic_noop = enabled, diagnostic_noop
        self._mode = "trim" if enabled else "noop" if diagnostic_noop else "disabled"
        # Both active diagnostic arms bind the same backend, noop never invokes.
        self._trim = (
            (_trim if _trim is not None else _native_trim()) if self.mode != "disabled" else None
        )
        self._on_tick = on_tick
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = self._closing = self._closed = False
        self._active = 0
        self._stats = _Telemetry()
        self._warned = False

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def diagnostic_noop(self) -> bool:
        return self._diagnostic_noop

    @property
    def mode(self) -> str:
        return self._mode

    def _warn_once(self, error_type: str) -> None:
        with self._lock:
            if self._warned:
                return
            self._warned = True
        with suppress(Exception):
            logging.getLogger(__name__).warning(
                "allocator residency maintenance stopped: %s", error_type
            )

    def start(self) -> bool:
        with self._lock:
            if self._started or self._closed or self._closing or self.mode == "disabled":
                return False
            began = time.monotonic()
            self._stats.started_mono = began
            self._stats.first_due_mono = self._stats.next_due_mono = began + self.interval_seconds
            self._thread = threading.Thread(
                target=self._run, name="allocator-residency-reclaimer", daemon=True
            )
            self._started = True
            try:
                self._thread.start()
            except BaseException as exc:
                self._stats.errors += 1
                self._stats.last_error_type = type(exc).__name__[:128]
                self._closed = True
                self._stats.closed_mono = time.monotonic()
                self._thread = None
                raise
            return True

    def _run(self) -> None:
        assert self._stats.first_due_mono is not None
        scheduled = self._stats.first_due_mono
        try:
            while not self._stop.is_set():
                if self._stop.wait(max(0.0, scheduled - time.monotonic())):
                    break
                began = time.monotonic()
                if began < scheduled:  # Event timers may wake one clock tick early.
                    continue
                with self._lock:
                    if self._closing or self._stop.is_set():
                        break
                    if self._active:
                        self._stats.overlap += 1
                        self._stats.errors += 1
                        break
                    self._active = 1
                    s = self._stats
                    s.ticks += 1
                    seq, skipped = s.ticks, s.missed
                    if s.last_started_mono is not None:
                        s.last_start_gap_seconds = began - s.last_started_mono
                        s.max_start_gap_seconds = max(
                            s.max_start_gap_seconds, s.last_start_gap_seconds
                        )
                    s.last_scheduled_mono, s.last_started_mono = scheduled, began
                    s.last_lateness_seconds = began - scheduled
                    s.max_lateness_seconds = max(s.max_lateness_seconds, began - scheduled)
                    s.max_early_seconds = max(s.max_early_seconds, scheduled - began)
                    if self.enabled:
                        s.actual_calls += 1
                    else:
                        s.noop_ticks += 1
                result, error_type = None, None
                try:
                    if self.enabled:
                        assert self._trim is not None
                        result = self._trim()
                        if type(result) is not int or result not in (0, 1):
                            result = None
                            raise ValueError("malloc_trim returned a value outside0/1")
                except BaseException as exc:
                    error_type = type(exc).__name__[:128]
                ended = time.monotonic()
                with self._lock:
                    s = self._stats
                    duration = ended - began
                    s.last_ended_mono, s.last_duration_seconds = ended, duration
                    s.total_duration_seconds += duration
                    s.max_duration_seconds = max(s.max_duration_seconds, duration)
                    if error_type:
                        s.errors += 1
                        s.last_error_type = error_type
                    elif self.enabled:
                        if result == 0:
                            s.return_zero += 1
                        else:
                            s.return_one += 1
                tick: Tick = dict(
                    seq=seq,
                    mode=self.mode,
                    did_call=self.enabled,
                    scheduled_mono=scheduled,
                    started_mono=began,
                    ended_mono=ended,
                    result=result,
                    error_type=error_type,
                    skipped_before=skipped,
                )
                callback_start = time.monotonic()
                callback_error_type = None
                try:
                    if self._on_tick is not None:
                        self._on_tick(tick)
                except BaseException as exc:
                    callback_error_type = type(exc).__name__[:128]
                    with self._lock:
                        self._stats.callback_errors += 1
                        self._stats.errors += 1
                        self._stats.last_callback_error_type = callback_error_type
                finally:
                    del tick
                finished = time.monotonic()
                failure = error_type or callback_error_type
                with self._lock:
                    s = self._stats
                    callback_seconds = finished - callback_start
                    s.last_callback_duration_seconds = callback_seconds
                    s.total_callback_duration_seconds += callback_seconds
                    s.max_tick_total_seconds = max(s.max_tick_total_seconds, finished - began)
                    s.completed_ticks += 1
                    s.last_completed_ended_mono = ended
                    self._active = 0
                    if failure:
                        self._closing = True
                        if s.stop_requested_mono is None:
                            s.stop_requested_mono = finished
                        self._stop.set()
                    if not self._stop.is_set():
                        scheduled, missed = next_deadline(
                            scheduled, finished, self.interval_seconds
                        )
                        s.missed += missed
                        s.next_due_mono = scheduled
                if failure:
                    self._warn_once(failure)
                if self._stop.is_set():
                    break
        except BaseException as exc:
            failure = type(exc).__name__[:128]
            with self._lock:
                self._stats.errors += 1
                self._stats.last_error_type = failure
                self._closing = True
                if self._stats.stop_requested_mono is None:
                    self._stats.stop_requested_mono = time.monotonic()
                self._stop.set()
            self._warn_once(failure)
        finally:
            with self._lock:
                self._active = 0
                self._closed = True
                self._stats.next_due_mono = None
                self._stats.closed_mono = time.monotonic()

    def status(self) -> dict[str, Scalar]:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {
                **asdict(self._stats),
                "mode": self.mode,
                "enabled": self.enabled,
                "diagnostic_noop": self.diagnostic_noop,
                "interval_seconds": self.interval_seconds,
                "started": self._started,
                "running": alive,
                "alive": alive,
                "closing": self._closing,
                "closed": self._closed,
                "active": self._active,
                "pending": self._active,
                "native_id": self._thread.native_id if self._thread is not None else None,
            }

    def close(self, timeout_seconds: float = 5.0) -> dict[str, Scalar]:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 <= timeout_seconds <= threading.TIMEOUT_MAX
        ):
            raise ValueError("timeout_seconds must be finite and nonnegative")
        with self._lock:
            if not self._closing:
                self._stats.stop_requested_mono = time.monotonic()
            self._closing = True
            self._stop.set()
            thread = self._thread
            if thread is None:
                self._closed = True
                if self._stats.closed_mono is None:
                    self._stats.closed_mono = time.monotonic()
                self._stats.next_due_mono = None
        if thread is not None and thread is not threading.current_thread():
            # A C call is not force-cancelled. A timeout is explicit invalid state.
            thread.join(timeout_seconds)
            if thread.is_alive():
                with self._lock:
                    self._stats.errors += 1
                    self._stats.close_timeouts += 1
                    self._stats.close_timed_out = True
        return self.status()
