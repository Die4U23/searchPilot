"""Lifespan/config fault tests with fake policy only; no native calls, model or network."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from searchpilot import bootstrap
from searchpilot.config import Settings, get_settings
from searchpilot.observability import JsonFormatter


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import os

    for key in [k for k in os.environ if k.startswith("SEARCHPILOT_")]:
        monkeypatch.delenv(key)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_policy(monkeypatch: pytest.MonkeyPatch) -> type:
    class FakePolicy:
        instances: list[FakePolicy] = []
        constructor_error: BaseException | None = None
        start_error: BaseException | None = None
        close_error: BaseException | None = None
        start_result = True
        timeout = False

        def __init__(self, *, interval_seconds: float, enabled: bool) -> None:
            if self.constructor_error:
                raise self.constructor_error
            self.interval = interval_seconds
            self.enabled = enabled
            self.events: list[object] = []
            self.closed = False
            self.instances.append(self)

        def start(self) -> bool:
            self.events.append("start")
            if self.start_error:
                raise self.start_error
            return self.start_result

        def status(self) -> dict[str, object]:
            return {
                "mode": "trim",
                "interval_seconds": self.interval,
                "closed": self.closed,
                "alive": self.timeout or not self.closed,
                "errors": int(self.timeout),
            }

        def close(self, timeout_seconds: float) -> dict[str, object]:
            self.events.append(("close", timeout_seconds))
            if self.close_error:
                raise self.close_error
            self.closed = not self.timeout
            return self.status()

    monkeypatch.setattr(bootstrap, "ResidencyReclaimer", FakePolicy)
    return FakePolicy


def enabled_app(interval: float = 7):
    app = bootstrap.create_app()
    app.state.settings = Settings(allocator_trim_interval_seconds=interval)
    return app


def test_default_disabled_has_no_policy_constructor_or_logs(fake_policy, caplog) -> None:
    app = bootstrap.create_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            pass

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    assert fake_policy.instances == [] and closed == [True]
    assert not any(r.getMessage().startswith("allocator_reclaimer_") for r in caplog.records)


def test_reads_app_state_at_lifespan_and_emits_two_json_policy_records(fake_policy, caplog) -> None:
    app = enabled_app(9)
    assert fake_policy.instances == []  # Factory assembly never starts a thread.
    app.state.settings = Settings(allocator_trim_interval_seconds=7)
    closed: list[object] = []
    app.state.closers.append(lambda: closed.append(fake_policy.instances[0].closed))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            policy = fake_policy.instances[0]
            assert policy.interval == 7 and policy.enabled and policy.events == ["start"]
        assert policy.events == ["start", ("close", 5.0)]

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    records = [r for r in caplog.records if r.getMessage().startswith("allocator_reclaimer_")]
    formatted = [json.loads(JsonFormatter().format(r)) for r in records]
    assert [r["msg"] for r in formatted] == [
        "allocator_reclaimer_started",
        "allocator_reclaimer_stopped",
    ]
    assert [r["policy"]["closed"] for r in formatted] == [False, True]
    assert all(r["policy"]["interval_seconds"] == 7 for r in formatted)
    assert closed == [True]
    assert all(
        not getattr(r, "path", "").startswith(("/policy", "/runtime", "/trim", "/gc"))
        for r in app.routes
    )


@pytest.mark.parametrize(
    "failure", [ValueError("body failed"), asyncio.CancelledError("body cancelled")]
)
def test_body_failure_or_cancellation_still_stops_and_runs_closers(fake_policy, failure) -> None:
    app = enabled_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            raise failure

    with pytest.raises(type(failure)) as captured:
        asyncio.run(run())
    assert captured.value is failure
    assert fake_policy.instances[0].events == ["start", ("close", 5.0)] and closed == [True]


def test_real_task_cancellation_cleans_up_before_parent_observes_cancel(fake_policy) -> None:
    app = enabled_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))

    async def run() -> None:
        entered = asyncio.Event()

        async def serve() -> None:
            async with app.router.lifespan_context(app):
                entered.set()
                await asyncio.Event().wait()

        child = asyncio.create_task(serve())
        await entered.wait()
        child.cancel()
        with pytest.raises(asyncio.CancelledError):
            await child
        assert fake_policy.instances[0].closed and closed == [True]

    asyncio.run(run())


@pytest.mark.parametrize("where", ["constructor", "start", "false_start"])
def test_startup_faults_fail_before_yield_and_run_existing_closers(
    fake_policy, where, caplog
) -> None:
    error = RuntimeError("startup failed")
    if where == "constructor":
        fake_policy.constructor_error = error
    elif where == "start":
        fake_policy.start_error = error
    else:
        fake_policy.start_result = False
    app = enabled_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("Startup failure must not enter application body")

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        asyncio.run(run())
    assert closed == [True]
    assert not any(r.getMessage() == "allocator_reclaimer_started" for r in caplog.records)
    if where != "constructor":
        assert fake_policy.instances[0].events[-1] == ("close", 5.0)


@pytest.mark.parametrize(
    "original", [RuntimeError("start failed"), asyncio.CancelledError("cancelled")]
)
def test_secondary_close_failure_preserves_original_exception(
    fake_policy, original, caplog
) -> None:
    fake_policy.start_error = original
    fake_policy.close_error = ValueError("secondary close failure")
    app = enabled_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("startup fault yielded")

    with pytest.raises(type(original)) as captured:
        asyncio.run(run())
    assert captured.value is original and closed == [True]
    assert any("shutdown failed" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "close_error", [RuntimeError("close failed"), asyncio.CancelledError("close cancelled")]
)
def test_close_failure_propagates_when_body_succeeded_and_closers_still_run(
    fake_policy, close_error
) -> None:
    fake_policy.close_error = close_error
    app = enabled_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(type(close_error)) as captured:
        asyncio.run(run())
    assert captured.value is close_error and closed == [True]


def test_join_timeout_fails_shutdown_and_preserves_honest_status(fake_policy, caplog) -> None:
    fake_policy.timeout = True
    app = enabled_app()
    closed: list[bool] = []
    app.state.closers.append(lambda: closed.append(True))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            pass

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="shut down cleanly"):
        asyncio.run(run())
    record = next(r for r in caplog.records if r.getMessage() == "allocator_reclaimer_stopped")
    assert record.policy["alive"] and not record.policy["closed"] and record.policy["errors"] == 1
    assert closed == [True]


def test_one_original_closer_failure_does_not_skip_later_closers(fake_policy) -> None:
    app = enabled_app()
    called: list[int] = []

    def bad_close() -> None:
        called.append(1)
        raise RuntimeError("old hook failed")

    app.state.closers.extend([bad_close, lambda: called.append(2)])

    async def run() -> None:
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(run())
    assert called == [1, 2]


@pytest.mark.parametrize(
    "value", [-1, float("nan"), float("inf"), float("-inf"), threading.TIMEOUT_MAX * 2]
)
def test_interval_invalid_rejected_in_configuration(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(allocator_trim_interval_seconds=value)


@pytest.mark.parametrize("value", [0.0, 0.01, 7.0, threading.TIMEOUT_MAX])
def test_interval_valid_bounds(value: float) -> None:
    assert Settings(allocator_trim_interval_seconds=value).allocator_trim_interval_seconds == value


def test_interval_default_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().allocator_trim_interval_seconds == 0
    monkeypatch.setenv("SEARCHPILOT_ALLOCATOR_TRIM_INTERVAL_SECONDS", "7")
    assert Settings().allocator_trim_interval_seconds == 7
