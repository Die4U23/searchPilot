from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from scripts.validate_container_soak import (
    Harness,
    bind_windows_worker,
    clean_environment,
    main,
    probe_health,
    stop_owned_process,
    wait_owned_process,
    wait_with_sampler,
)


def test_counts_sets_server_side_query_deadlines(tmp_path, monkeypatch):
    from scripts import validate_container_soak

    def connect(url, **kwargs):
        assert kwargs["connect_timeout"] == 5
        assert kwargs["options"] == "-c statement_timeout=10000 -c lock_timeout=3000"
        raise TimeoutError("bounded fixture")

    monkeypatch.setattr(validate_container_soak.psycopg, "connect", connect)
    with pytest.raises(TimeoutError, match="bounded fixture"):
        Harness(arguments(tmp_path)).counts()


@pytest.mark.parametrize(("status", "exit_code"), [("COMPLETED", 0), ("FAILED", 1)])
def test_main_exit_code_reflects_final_report_status(tmp_path, monkeypatch, status, exit_code):
    async def run(harness):
        harness.report["status"] = status

    monkeypatch.setattr(Harness, "run", run)
    assert main(["--out", str(tmp_path)]) == exit_code


@pytest.mark.parametrize("duration", [None, 7200, 14400])
def test_cli_duration_default_and_long_window_forwarded(tmp_path, monkeypatch, duration):
    expected = 3600 if duration is None else duration

    async def run(harness):
        # Construction validates the same duration through bench_concurrent.Config.
        assert harness.args.duration_seconds == expected
        command = harness.benchmark_command()
        assert float(command[command.index("--duration-seconds") + 1]) == expected
        harness.report["status"] = "COMPLETED"

    monkeypatch.setattr(Harness, "run", run)
    args = ["--out", str(tmp_path)]
    if duration is not None:
        args += ["--duration-seconds", str(duration)]
    assert main(args) == 0


@pytest.mark.parametrize("duration", ["0", "-1", "14400.01", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_duration_before_docker(tmp_path, monkeypatch, duration):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not call Docker"))
    with pytest.raises(SystemExit) as error:
        main(["--out", str(tmp_path / "new"), f"--duration-seconds={duration}"])
    assert error.value.code == 2
    assert not (tmp_path / "new").exists()


def launch_child():
    child = subprocess.Popen(
        [sys.executable, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    pid = int(child.stdout.readline())
    child.stdout.close()
    bind_windows_worker(child, pid)
    return child


def arguments(out):
    return argparse.Namespace(
        out=out,
        project="sp-test-owned",
        image="validation:test",
        api_port=18082,
        db_port=15433,
        duration_seconds=1,
        concurrency=4,
        queries=Path("data/queries/queries.csv"),
        baseline_seconds=0,
        recovery_seconds=0,
        data_version="fixture",
    )


def test_environment_never_inherits_private_service_or_compose_settings(monkeypatch):
    for key in (
        "SEARCHPILOT_API_TOKEN",
        "OPENAI_API_KEY",
        "ARCBENCH_API_KEY",
        "POSTGRES_PASSWORD",
        "COMPOSE_FILE",
        "COMPOSE_ENV_FILES",
        "MALLOC_ARENA_MAX",
        "MALLOC_TRIM_THRESHOLD_",
        "GLIBC_TUNABLES",
        "LD_PRELOAD",
        "PYTHONMALLOC",
        "PYTHONTRACEMALLOC",
    ):
        monkeypatch.setenv(key, "private-fixture-value")
    monkeypatch.setenv("PATH", "preserved-path")
    actual = clean_environment()
    assert actual["POSTGRES_PASSWORD"] == "searchpilot"
    assert actual["PATH"] == "preserved-path"
    assert "private-fixture-value" not in actual.values()


@pytest.mark.asyncio
async def test_existing_output_is_never_overwritten_or_used(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    original = b"original failure evidence"
    (output / "summary.json").write_bytes(original)
    harness = Harness(arguments(output))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not call Docker"))
    with pytest.raises(FileExistsError):
        await harness.run()
    assert (output / "summary.json").read_bytes() == original
    assert list(output.iterdir()) == [output / "summary.json"]


@pytest.mark.parametrize("labelled", [True, False])
def test_rejects_existing_stopped_or_unlabelled_resources(tmp_path, monkeypatch, labelled):
    harness = Harness(arguments(tmp_path / "new"))
    calls = []

    def command(args):
        calls.append(args)
        assert args[1:3] == ["container", "ls"]
        assert "-a" in args
        if "--filter" in args:
            return "stopped-container-id" if labelled else ""
        return harness.api

    monkeypatch.setattr(harness, "command", command)
    with pytest.raises(ValueError, match="already"):
        harness.prepare()
    assert len(calls) == (1 if labelled else 2)


@pytest.mark.asyncio
async def test_timeout_reaps_owned_child_without_stopping_other_child():
    owned, other = launch_child(), launch_child()
    try:
        with pytest.raises(TimeoutError):
            await wait_owned_process(owned, 0.05)
        assert owned.poll() is not None
        assert other.poll() is None
    finally:
        stop_owned_process(owned)
        stop_owned_process(other)


@pytest.mark.asyncio
async def test_cancellation_reaps_owned_child():
    owned = launch_child()
    task = asyncio.create_task(wait_owned_process(owned, 30))
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert owned.poll() is not None


@pytest.mark.asyncio
async def test_sampler_failure_stops_owned_load_immediately():
    owned = launch_child()

    async def fail():
        await asyncio.sleep(0.03)
        raise OSError("evidence disk failed")

    sampler = asyncio.create_task(fail())
    with pytest.raises(OSError, match="evidence disk failed"):
        await wait_with_sampler(owned, 30, sampler)
    assert owned.poll() is not None


@pytest.mark.asyncio
async def test_sampler_error_does_not_hide_original_failure_or_prevent_summary(
    tmp_path, monkeypatch
):
    harness = Harness(arguments(tmp_path))

    def prepare():
        harness.owns_output = True

    async def sample(client):
        raise ValueError("sampler failure")

    def command(*args):
        raise RuntimeError("startup failure")

    monkeypatch.setattr(harness, "prepare", prepare)
    monkeypatch.setattr(harness, "samples", sample)
    monkeypatch.setattr(harness, "command", command)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="startup failure"):
        await harness.run()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["error_type"] == "RuntimeError"
    assert summary["sampler_error"] == "ValueError"
    assert summary["status"] == "FAILED"


def test_unresponsive_child_is_killed_and_reaped():
    class StubbornChild:
        def __init__(self):
            self.calls = []

        def poll(self):
            return None

        def terminate(self):
            self.calls.append("terminate")

        def kill(self):
            self.calls.append("kill")

        def wait(self, timeout):
            self.calls.append("wait")
            if "kill" not in self.calls:
                raise subprocess.TimeoutExpired("owned-child", timeout)
            return -9

    child = StubbornChild()
    stop_owned_process(child, grace=0.01)
    assert child.calls == ["terminate", "wait", "kill", "wait"]


@pytest.mark.asyncio
async def test_failed_live_probe_does_not_hide_ready_or_memory(tmp_path, monkeypatch):
    harness = Harness(arguments(tmp_path))

    async def command(args, **kwargs):
        harness.stop.set()
        if "stats" in args:
            return {"available": False, "error_type": "TimeoutExpired"}
        return {"available": True, "data": {"available": True, "pid": 1, "sources": {}}}

    monkeypatch.setattr(harness, "sample_command", command)

    def respond(request):
        if request.url.path == "/health/live":
            raise httpx.ReadError("fixture failure")
        return httpx.Response(200, json={"status": "ready"})

    async with httpx.AsyncClient(
        base_url="http://fixture", transport=httpx.MockTransport(respond)
    ) as client:
        await harness.samples(client)
    row = json.loads((tmp_path / "samples.jsonl").read_text())
    assert row["server"]["pid"] == 1
    assert row["live"]["error_type"] == "ReadError"
    assert row["ready"]["status"] == 200
    assert row["stats"]["error_type"] == "TimeoutExpired"


@pytest.mark.asyncio
async def test_health_body_is_bounded():
    async with httpx.AsyncClient(
        base_url="http://fixture",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 65537)),
    ) as client:
        row = await probe_health(client, "live")
    assert row["available"] is False
    assert row["error_type"] == "ValueError"


@pytest.mark.parametrize("invalid", ["concurrency", "duplicate", "missing", "long", "empty"])
def test_invalid_workload_rejected_before_docker(tmp_path, monkeypatch, invalid):
    queries = tmp_path / "input.csv"
    text = {"duplicate": "one\none", "long": "x" * 257, "empty": "   "}.get(invalid, "one")
    if invalid != "missing":
        queries.write_text("query_text\n" + text + "\n", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not call Docker"))
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--out",
                str(out),
                "--queries",
                str(queries),
                "--concurrency",
                "257" if invalid == "concurrency" else "8",
            ]
        )
    assert error.value.code == 2
    assert not out.exists()


def test_cli_freezes_query_bytes_and_forwards_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    raw = b"\xef\xbb\xbfquery_text\r\none\r\ntwo\r\n"
    source.write_bytes(raw)
    out = tmp_path / "run"

    async def run(harness):
        source.write_text("query_text\nchanged\n", encoding="utf-8")

        def command(args):
            if "inspect" in args:
                return '[{"Id":"fixture-image"}]'
            if "info" in args:
                return "{}"
            return ""

        monkeypatch.setattr(harness, "command", command)
        harness.prepare()
        command = harness.benchmark_command()
        snapshot = Path(command[command.index("--queries") + 1])
        assert snapshot == out / "queries.csv"
        assert snapshot.read_bytes() == raw
        assert harness.queries == ["one", "two"]
        assert harness.report["queries"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert command[command.index("--concurrency") + 1] == "8"
        assert command[command.index("--limit") + 1] == "10"
        assert command[command.index("--timeout") + 1] == "10"
        assert harness.report["config"]["queries"] == str(source.resolve())
        harness.report["status"] = "COMPLETED"

    monkeypatch.setattr(Harness, "run", run)
    assert main(["--out", str(out), "--queries", str(source), "--concurrency", "8"]) == 0


def test_state_exposes_only_native_environment_allowlist(tmp_path, monkeypatch):
    harness = Harness(arguments(tmp_path))
    row = {"Id": "fixture", "Name": "api", "Image": "image", "RestartCount": 0, "State": {}}
    row["Config"] = {
        "Cmd": ["python", "scripts/docker_entrypoint.py"],
        "Entrypoint": None,
        "Env": [
            "MALLOC_ARENA_MAX=2",
            "SEARCHPILOT_DISABLE_THP=true",
            "MALLOC_TRIM_THRESHOLD_=4096",
            "LD_PRELOAD",
            "PRIVATE_TOKEN=hidden",
            "DATABASE_URL=hidden",
        ],
    }
    monkeypatch.setattr(harness, "command", lambda args: json.dumps([row]))
    actual = harness.state()[0]
    assert actual["command"] == row["Config"]["Cmd"]
    assert actual["entrypoint"] is None
    assert actual["native_env"]["MALLOC_ARENA_MAX"] == "2"
    assert actual["native_env"]["SEARCHPILOT_DISABLE_THP"] == "true"
    assert actual["native_env"]["MALLOC_TRIM_THRESHOLD_"] == "4096"
    assert actual["native_env"]["LD_PRELOAD"] is None
    assert "hidden" not in json.dumps(actual)
