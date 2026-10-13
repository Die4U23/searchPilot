"""Loopback TCP boundary tests; no external network, database, model or service required."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from scripts import bench_concurrent
from scripts.bench_concurrent import Config, latency_summary, run_mode

Handler = Callable[[dict[str, str], dict[str, Any], asyncio.StreamWriter], Awaitable[None]]


@asynccontextmanager
async def serve(handler: Handler) -> AsyncIterator[str]:
    tasks: set[asyncio.Task[Any]] = set()

    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            headers = {}
            for line in head.decode().split("\r\n")[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.lower()] = value.strip()
            body = await reader.readexactly(int(headers["content-length"]))
            await handler(headers, json.loads(body), writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), 0.5)
            tasks.discard(task)

    server = await asyncio.start_server(connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.close()
        pending = list(tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await server.wait_closed()


def payload(headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": headers["x-request-id"],
        "normalized_query": body["query"],
        "model_version": "bm25-fixture",
        "mode": body["mode"],
        "results": [{"item_id": "N1", "score": 1.0, "rank": 1, "channel": "bm25"}],
    }


async def respond(
    writer: asyncio.StreamWriter,
    headers: dict[str, str],
    body: bytes,
    status: int = 200,
    extra_length: int = 0,
) -> None:
    writer.write(
        (
            f"HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body) + extra_length}\r\nConnection: close\r\n"
            f"X-Request-Id: {headers['x-request-id']}\r\n\r\n"
        ).encode()
        + body
    )
    await writer.drain()


async def test_real_concurrency_denominator_and_warmup() -> None:
    active = peak = calls = 0
    ids = set()

    async def handler(headers, body, writer):
        nonlocal active, peak, calls
        active += 1
        peak = max(peak, active)
        calls += 1
        ids.add(headers["x-request-id"])
        try:
            await asyncio.sleep(0.02)
            await respond(writer, headers, json.dumps(payload(headers, body)).encode())
        finally:
            active -= 1

    async with serve(handler) as url:
        report = await run_mode(Config(url, 11, 3, 2, 2), "bm25", ["one", "two"])
    summary = report["summary"]
    assert 2 <= peak <= 3
    assert calls == len(ids) == 13
    assert summary["total"] == summary["successes"] == 11
    assert summary["failures"] == 0
    assert summary["all_latency"]["count"] == summary["success_latency"]["count"] == 11
    assert len(report["warmup"]) == 2
    assert [row["index"] for row in report["requests"]] == list(range(11))
    assert [row["query"] for row in report["requests"]] == ["one", "two"] * 5 + ["one"]


@pytest.mark.parametrize("failure", ["http_error", "malformed_response", "incomplete_response"])
async def test_failure_keeps_every_request(failure: str) -> None:
    calls = 0

    async def handler(headers, body, writer):
        nonlocal calls
        calls += 1
        if body["query"] == "success":
            await respond(writer, headers, json.dumps(payload(headers, body)).encode())
        elif failure == "http_error":
            await respond(writer, headers, b'{"error":"not ready"}', status=503)
        elif failure == "malformed_response":
            await respond(writer, headers, b"not JSON")
        else:
            await respond(writer, headers, b'{"truncated":', extra_length=100)

    async with serve(handler) as url:
        report = await run_mode(Config(url, 6, 2, 2, 0), "bm25", ["success", "failure"])
    summary = report["summary"]
    assert calls == summary["total"] == 6
    assert summary["successes"] == summary["failures"] == 3
    assert summary["success_rate"] == 0.5
    assert summary["outcomes"] == {failure: 3, "success": 3}
    assert summary["all_latency"]["count"] == 6
    assert summary["success_latency"]["count"] == 3


@pytest.mark.parametrize("drip_body", [False, True])
async def test_total_timeout_includes_headers_and_slow_body(drip_body: bool) -> None:
    async def handler(headers, body, writer):
        if not drip_body:
            await asyncio.sleep(2)
        else:
            writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            while True:
                writer.write(b"1\r\n \r\n")
                await writer.drain()
                await asyncio.sleep(0.005)

    async with serve(handler) as url:
        report = await run_mode(Config(url, 3, 2, 0.08, 0), "bm25", ["timeout"])
    summary = report["summary"]
    assert summary["outcomes"] == {"timeout": 3}, report["requests"]
    assert summary["timeout_rate"] == 1.0
    assert summary["success_latency"]["p99_ms"] is None
    assert summary["all_latency"]["count"] == 3
    assert summary["elapsed_seconds"] < 1.5
    assert all(row["elapsed_ms"] >= 60 for row in report["requests"])
    assert all(row["status_code"] == (200 if drip_body else None) for row in report["requests"])


@pytest.mark.parametrize("defect", ["schema", "id", "mode", "duplicate", "nan", "rank"])
async def test_http_200_does_not_imply_success(defect: str) -> None:
    async def handler(headers, body, writer):
        value = payload(headers, body)
        if defect == "schema":
            del value["results"]
        elif defect == "id":
            value["request_id"] = "req_" + "0" * 32
        elif defect == "mode":
            value["mode"] = "vector"
        elif defect == "duplicate":
            value["results"] *= 2
        elif defect == "nan":
            value["results"][0]["score"] = float("nan")
        else:
            value["results"][0]["rank"] = 0
        await respond(writer, headers, json.dumps(value).encode())

    async with serve(handler) as url:
        report = await run_mode(Config(url, 1, 1, 2, 0), "bm25", ["invalid"])
    assert report["summary"]["outcomes"] == {"invalid_response": 1}
    assert report["requests"][0]["status_code"] == 200


async def test_empty_results_are_contract_success() -> None:
    async def handler(headers, body, writer):
        value = payload(headers, body)
        value["results"] = []
        await respond(writer, headers, json.dumps(value).encode())

    async with serve(handler) as url:
        report = await run_mode(Config(url, 1, 1, 2, 0), "bm25", ["empty"])
    assert report["summary"]["successes"] == report["summary"]["empty_successes"] == 1


async def test_connection_closed_before_headers_is_counted() -> None:
    async def handler(headers, body, writer):
        return

    async with serve(handler) as url:
        report = await run_mode(Config(url, 2, 2, 1, 0), "bm25", ["q"])
    assert report["summary"]["outcomes"] == {"transport_error": 2}
    assert report["summary"]["all_latency"]["count"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"requests": 0},
        {"concurrency": 0},
        {"concurrency": 257},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"warmup": -1},
        {"limit": 101},
        {"base_url": "https://user:secret@example.test"},
    ],
)
def test_invalid_limits_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Config(**{"base_url": "http://127.0.0.1:8000", **kwargs})


def test_percentile_nearest_rank_and_empty() -> None:
    result = latency_summary(list(range(1, 101)))
    assert result == {"count": 100, "p50_ms": 50, "p95_ms": 95, "p99_ms": 99, "max_ms": 100}
    assert latency_summary([])["max_ms"] is None


def cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    queries = tmp_path / "queries.csv"
    queries.write_text("query_text\none\ntwo\n", encoding="utf-8")
    return queries, tmp_path / "result.json"


def test_cli_refuses_existing_output_before_requests(tmp_path: Path, monkeypatch) -> None:
    queries, out = cli_paths(tmp_path)
    out.write_text("original failure evidence", encoding="utf-8")
    monkeypatch.setattr(bench_concurrent, "code_metadata", lambda root: {})

    async def unexpected(*args):
        pytest.fail("existing output must be rejected before sending requests")

    monkeypatch.setattr(bench_concurrent, "run_mode", unexpected)
    with pytest.raises(SystemExit) as error:
        bench_concurrent.main(["--queries", str(queries), "--out", str(out)])
    assert error.value.code == 2
    assert out.read_text(encoding="utf-8") == "original failure evidence"


async def test_cli_real_http_marks_complete(tmp_path: Path, monkeypatch) -> None:
    queries, out = cli_paths(tmp_path)
    monkeypatch.setattr(bench_concurrent, "code_metadata", lambda root: {})

    async def handler(headers, body, writer):
        in_progress = json.loads(out.read_text(encoding="utf-8"))
        assert in_progress["status"] == "RUNNING"
        assert in_progress["active_mode"] == "bm25"
        events = [
            json.loads(line) for line in out.with_suffix(".requests.jsonl").read_text().splitlines()
        ]
        assert any(
            event["event"] == "pending" and event["request_id"] == headers["x-request-id"]
            for event in events
        )
        await respond(writer, headers, json.dumps(payload(headers, body)).encode())

    async with serve(handler) as url:
        result = await asyncio.to_thread(
            bench_concurrent.main,
            [
                "--base-url",
                url,
                "--queries",
                str(queries),
                "--out",
                str(out),
                "--requests",
                "3",
                "--concurrency",
                "2",
                "--warmup",
                "0",
            ],
        )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert result == 0
    assert report["status"] == "COMPLETED"
    assert report["requested_modes"] == ["bm25"]
    assert report["active_mode"] is None
    assert report["runs"][0]["summary"]["successes"] == 3
    events = [
        json.loads(line) for line in out.with_suffix(".requests.jsonl").read_text().splitlines()
    ]
    assert len(events) == 6
    assert sum(event["event"] == "result" for event in events) == 3


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_cli_partial_modes_are_not_complete(tmp_path: Path, monkeypatch, error) -> None:
    queries, out = cli_paths(tmp_path)
    monkeypatch.setattr(bench_concurrent, "code_metadata", lambda root: {})

    async def partial(config, mode, queries, **kwargs):
        if mode == "vector":
            raise error()
        return {"mode": mode, "summary": {"failures": 0}, "warmup": []}

    monkeypatch.setattr(bench_concurrent, "run_mode", partial)
    with pytest.raises(error):
        bench_concurrent.main(
            ["--queries", str(queries), "--out", str(out), "--modes", "bm25", "vector"]
        )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == ("INTERRUPTED" if error is KeyboardInterrupt else "FAILED")
    assert report["requested_modes"] == ["bm25", "vector"]
    assert len(report["runs"]) == 1
    assert report["active_mode"] == "vector"
    assert "UNKNOWN" in report["incomplete_mode_ledger"]


def test_cli_returns_failure_for_warmup_timeout(tmp_path: Path, monkeypatch) -> None:
    queries, out = cli_paths(tmp_path)
    monkeypatch.setattr(bench_concurrent, "code_metadata", lambda root: {})

    async def failed_warmup(*args, **kwargs):
        return {"summary": {"failures": 0}, "warmup": [{"outcome": "timeout"}]}

    monkeypatch.setattr(bench_concurrent, "run_mode", failed_warmup)
    assert bench_concurrent.main(["--queries", str(queries), "--out", str(out)]) == 1


async def test_cancel_retains_completed_and_unknown_pending_requests(tmp_path: Path) -> None:
    waiting = asyncio.Event()

    async def handler(headers, body, writer):
        if body["query"] == "complete":
            await respond(writer, headers, json.dumps(payload(headers, body)).encode())
        else:
            waiting.set()
            await asyncio.Event().wait()

    ledger_path = tmp_path / "ledger.jsonl"
    with ledger_path.open("x", encoding="utf-8") as file:

        def record(event):
            file.write(json.dumps(event) + "\n")
            file.flush()

        async with serve(handler) as url:
            task = asyncio.create_task(
                run_mode(
                    Config(url, 5, 1, 2, 0), "bm25", ["complete", "pending"], event_sink=record
                )
            )
            await asyncio.wait_for(waiting.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["pending", "result", "pending"]
    assert events[1]["outcome"] == "success"
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[2]["request_id"] != events[0]["request_id"]
    assert events[2]["outcome"] == "pending"


def test_cli_refuses_existing_ledger(tmp_path: Path, monkeypatch) -> None:
    queries, out = cli_paths(tmp_path)
    ledger_path = out.with_suffix(".requests.jsonl")
    ledger_path.write_text("pending request evidence", encoding="utf-8")
    monkeypatch.setattr(bench_concurrent, "code_metadata", lambda root: {})

    async def unexpected(*args, **kwargs):
        pytest.fail("existing ledger must be rejected before requests")

    monkeypatch.setattr(bench_concurrent, "run_mode", unexpected)
    with pytest.raises(SystemExit) as error:
        bench_concurrent.main(["--queries", str(queries), "--out", str(out)])
    assert error.value.code == 2
    assert ledger_path.read_text(encoding="utf-8") == "pending request evidence"
