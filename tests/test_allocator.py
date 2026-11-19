"""Offline injected-function tests; no application, native trim, model or network."""

import gc
import json
import threading
import time
import weakref

import pytest

from searchpilot import allocator as module
from searchpilot.allocator import ResidencyReclaimer, UnsupportedAllocator, next_deadline


def until(predicate, timeout=2):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("bounded test wait expired")


def test_disabled_production_has_no_native_binding_or_thread(monkeypatch):
    monkeypatch.setattr(module, "_native_trim", lambda: pytest.fail("disabled native binding"))
    policy = ResidencyReclaimer()
    assert policy.start() is False and policy.start() is False
    assert policy.status()["started"] is False
    status = policy.close()
    assert status["closed"] and not status["alive"] and status["actual_calls"] == 0
    assert policy.close() == status and policy.start() is False


@pytest.mark.parametrize(
    "value", [0, -1, True, float("nan"), float("inf"), threading.TIMEOUT_MAX * 2]
)
def test_bad_interval_rejected(value):
    with pytest.raises(ValueError):
        ResidencyReclaimer(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"diagnostic_noop": "yes"},
        {"enabled": True, "diagnostic_noop": True},
        {"on_tick": 1},
    ],
)
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ResidencyReclaimer(**kwargs)


@pytest.mark.parametrize("platform_name,libc", [("win32", "glibc"), ("linux", "musl")])
def test_unsupported_enabled_fails_explicitly(monkeypatch, platform_name, libc):
    monkeypatch.setattr(module.sys, "platform", platform_name)
    monkeypatch.setattr(module.platform, "libc_ver", lambda: (libc, "test"))
    with pytest.raises(UnsupportedAllocator):
        ResidencyReclaimer(enabled=True)


def test_native_adapter_is_cdll_size_t_int_without_executing_real_trim(monkeypatch):
    class Function:
        argtypes = restype = None

        def __call__(self, pad):
            assert pad == 0
            return 0

    function = Function()

    class Library:
        malloc_trim = function

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.platform, "libc_ver", lambda: ("glibc", "2.41"))
    monkeypatch.setattr(module.ctypes, "CDLL", lambda name, use_errno: Library())
    trim = module._native_trim()
    assert function.argtypes == [module.ctypes.c_size_t] and function.restype == module.ctypes.c_int
    assert trim() == 0


def test_no_call_before_first_due_and_close_wakes_wait():
    called = threading.Event()
    policy = ResidencyReclaimer(0.2, enabled=True, _trim=lambda: called.set() or 0)
    assert policy.start()
    time.sleep(0.02)
    assert not called.is_set()
    started = time.perf_counter()
    status = policy.close()
    assert time.perf_counter() - started < 0.5
    assert status["closed"] and not status["alive"] and status["ticks"] == 0


@pytest.mark.parametrize("result", [0, 1])
def test_valid_native_returns_and_bounded_scalar_telemetry(result):
    policy = ResidencyReclaimer(0.03, enabled=True, _trim=lambda: result)
    before = set(policy.status())
    policy.start()
    until(lambda: policy.status()["completed_ticks"] >= 2)
    status = policy.close()
    assert status["actual_calls"] >= 2 and status["errors"] == status["overlap"] == 0
    assert status["return_zero" if result == 0 else "return_one"] == status["actual_calls"]
    assert set(status) == before and all(
        type(v) in (type(None), str, bool, int, float) for v in status.values()
    )
    json.dumps(status)
    count = status["actual_calls"]
    time.sleep(0.08)
    assert policy.status()["actual_calls"] == count


def test_diagnostic_noop_uses_same_worker_but_never_calls_native():
    ticks = []
    policy = ResidencyReclaimer(
        0.03,
        diagnostic_noop=True,
        _trim=lambda: pytest.fail("noop called trim"),
        on_tick=ticks.append,
    )
    policy.start()
    until(lambda: policy.status()["completed_ticks"] >= 2)
    status = policy.close()
    assert (
        status["ticks"] >= 2
        and status["actual_calls"] == 0
        and status["noop_ticks"] == status["ticks"]
    )
    assert all(t["did_call"] is False and t["result"] is None for t in ticks)


def test_concurrent_start_creates_only_one_worker():
    policy = ResidencyReclaimer(0.03, enabled=True, _trim=lambda: 0)
    starts = []
    callers = [threading.Thread(target=lambda: starts.append(policy.start())) for _ in range(12)]
    for thread in callers:
        thread.start()
    for thread in callers:
        thread.join()
    assert starts.count(True) == 1
    until(lambda: policy.status()["completed_ticks"] >= 1)
    status = policy.close()
    assert status["overlap"] == 0 and status["closed"] and not status["alive"]


def test_start_failure_is_terminal_and_repeated_close_never_joins_unstarted(monkeypatch):
    class BrokenThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("injected start failure")

    monkeypatch.setattr(module.threading, "Thread", BrokenThread)
    policy = ResidencyReclaimer(0.03, enabled=True, _trim=lambda: 0)
    with pytest.raises(RuntimeError):
        policy.start()
    status = policy.close()
    assert status["closed"] and not status["alive"] and status["errors"] == 1
    assert policy.close() == status and policy.start() is False


@pytest.mark.parametrize("kind", ["native", "callback", "return"])
def test_first_error_stops_without_retry_or_traceback_retention(kind, caplog):
    def trim():
        if kind == "native":
            raise OSError("private payload must not be logged")
        return 2 if kind == "return" else 0

    def callback(_tick):
        if kind == "callback":
            raise LookupError("private callback payload")

    policy = ResidencyReclaimer(0.03, enabled=True, _trim=trim, on_tick=callback)
    policy.start()
    until(lambda: policy.status()["closed"])
    status = policy.close()
    assert status["actual_calls"] == 1 and status["errors"] == 1
    assert status["callback_errors"] == (1 if kind == "callback" else 0)
    assert len(caplog.records) == 1 and caplog.records[0].exc_info is None
    assert "private" not in caplog.text
    time.sleep(0.08)
    assert policy.status()["actual_calls"] == 1


def test_close_timeout_is_explicit_and_later_join_cannot_erase_failure():
    entered, release = threading.Event(), threading.Event()

    def trim():
        entered.set()
        assert release.wait(2)
        return 0

    policy = ResidencyReclaimer(0.03, enabled=True, _trim=trim)
    policy.start()
    assert entered.wait(1)
    # Native operation holds no telemetry lock: status and bounded close proceed.
    assert policy.status()["active"] == 1
    status = policy.close(0.01)
    assert status["close_timed_out"] and status["alive"] and not status["closed"]
    release.set()
    status = policy.close()
    assert status["closed"] and not status["alive"] and status["pending"] == 0
    assert status["errors"] >= 1 and status["close_timed_out"]
    assert status["actual_calls"] == 1


def test_fixed_deadlines_skip_overrun_without_backlog():
    assert next_deadline(10, 10.1, 7) == (17, 0)
    assert next_deadline(10, 31, 7) == (38, 3)
    assert next_deadline(10, 17, 7) == (24, 1)


def test_callback_can_cancel_own_worker_without_self_join():
    signalled = threading.Event()
    policy = None

    def callback(_tick):
        policy.close()
        signalled.set()

    policy = ResidencyReclaimer(0.03, enabled=True, _trim=lambda: 0, on_tick=callback)
    policy.start()
    assert signalled.wait(1)
    status = policy.close()
    assert status["closed"] and status["actual_calls"] == 1 and status["errors"] == 0


def test_exception_payload_is_not_retained():
    class Payload:
        pass

    payload = Payload()
    reference = weakref.ref(payload)
    holder = [payload]
    del payload

    def trim():
        raise RuntimeError(holder.pop())

    policy = ResidencyReclaimer(0.03, enabled=True, _trim=trim)
    policy.start()
    until(lambda: policy.status()["closed"])
    policy.close()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("name", ["interval_seconds", "enabled", "diagnostic_noop", "mode"])
def test_configuration_is_read_only(name):
    policy = ResidencyReclaimer()
    with pytest.raises(AttributeError):
        setattr(policy, name, 0)


def test_completed_end_does_not_advance_while_next_callback_is_blocked():
    blocked, release = threading.Event(), threading.Event()

    def callback(tick):
        if tick["seq"] == 2:
            blocked.set()
            assert release.wait(2)

    policy = ResidencyReclaimer(0.03, enabled=True, _trim=lambda: 0, on_tick=callback)
    policy.start()
    try:
        until(lambda: policy.status()["completed_ticks"] == 1)
        first = policy.status()["last_completed_ended_mono"]
        assert blocked.wait(1)
        pending = policy.status()
        assert pending["pending"] == 1 and pending["completed_ticks"] == 1
        assert pending["last_completed_ended_mono"] == first
        assert pending["last_ended_mono"] > first
    finally:
        release.set()
        final = policy.close()
    assert final["completed_ticks"] == 2 and final["errors"] == 0
    assert final["last_completed_ended_mono"] == final["last_ended_mono"]
