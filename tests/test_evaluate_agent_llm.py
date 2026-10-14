from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from scripts.evaluate_agent import evaluate as evaluate_deterministic
from scripts.evaluate_agent import render as render_deterministic
from scripts.evaluate_agent_llm import main

from searchpilot.agent.demo import build_tasks
from searchpilot.agent.llm_eval import (
    MAX_RESPONSE_BYTES,
    REFUSAL,
    EvaluationError,
    evaluate,
    execute_plan,
    expected_call,
    metric_citations,
    parse_object,
    score,
)


def answer(result: dict) -> dict:
    citations = metric_citations(result)
    prose = " ".join(citations) or "Tool evidence attached."
    if result.get("source") == "evorec":
        prose = f"来自 EvoRec run-fixture，commit abc123。{prose}"
    if result.get("comparable") is False:
        prose = REFUSAL
    return {"answer": prose, "citations": citations, "evidence": result}


def response(content: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "offline-fake",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


@pytest.mark.asyncio
async def test_all_50_tasks_offline_and_exclusive_ledger(tmp_path: Path) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.headers["X-Request-ID"])
        return golden_response(request)

    path = tmp_path / "run.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        summary = await evaluate(
            client, endpoint="http://localhost/test", model="fake", ledger_path=path
        )
        with pytest.raises(FileExistsError):
            await evaluate(client, endpoint="http://localhost/test", model="fake", ledger_path=path)
    assert summary["structured_pass"] == summary["tasks"] == 50
    assert summary["citation_supported"] == summary["citation_needed"] == 32
    assert summary["trap_passed"] == summary["trap_needed"] == 8
    assert len(requests) == len(set(requests)) == 100
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[1]["status"] == "pending" and events[1]["usage"] is None
    assert events[2]["usage"]["total_tokens"] == 15


@pytest.mark.parametrize("kind", ["http", "json", "timeout", "slow_body", "truncated"])
@pytest.mark.asyncio
async def test_failures_stay_in_denominator_without_retries(tmp_path: Path, kind: str) -> None:
    count = 0

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(1)
            yield b"{}"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if kind == "timeout":
            raise httpx.ReadTimeout("private endpoint detail", request=request)
        if kind == "http":
            return httpx.Response(503, text="secret server error")
        if kind == "slow_body":
            return httpx.Response(200, stream=SlowBody())
        if kind == "truncated":
            return httpx.Response(200, json={"choices": [{"finish_reason": "length"}]})
        return httpx.Response(200, text="not json")

    path = tmp_path / "failed.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        summary = await evaluate(
            client, endpoint="http://localhost/test", model="fake", ledger_path=path, timeout=0.01
        )
    assert count == 50
    assert summary["tasks"] == summary["failed_requests_or_protocol"] == 50
    assert summary["structured_pass"] == 0
    assert summary["citation_needed"] == 32 and summary["trap_needed"] == 8
    content = path.read_text(encoding="utf-8")
    assert "private endpoint detail" not in content and "secret server error" not in content
    events = [json.loads(line) for line in content.splitlines()]
    assert all(e["usage"] is None for e in events if e["event"] == "request_result")


@pytest.mark.parametrize("change", ["wrong_source", "wrong_value", "wrong_task", "extra_cite"])
def test_valid_json_and_present_citation_do_not_imply_support(change: str) -> None:
    task = build_tasks()[0]
    plan = expected_call(task)
    if change == "wrong_task":
        plan["arguments"]["experiment_id"] = "sp-vector"
    result = execute_plan(plan)
    final = answer(result)
    if change == "wrong_source":
        final["citations"][0] = final["citations"][0].replace("searchpilot", "evorec")
    if change == "wrong_value":
        final["evidence"] = {**result, "value": 0.9999}
    if change == "extra_cite":
        final["answer"] += " [sp-vector | ndcg@10 | test | 1.0 | searchpilot]"
    assert score(task, plan, result, final)["structured_pass"] is False


def test_trap_cannot_refuse_then_claim_improvement() -> None:
    task = next(t for t in build_tasks() if t["kind"] == "trap")
    plan = expected_call(task)
    result = execute_plan(plan)
    final = answer(result)
    final["answer"] += " But the improvement is 200%."
    assert not score(task, plan, result, final)["trap_refused"]


@pytest.mark.parametrize(
    "plan",
    [
        {"tool": "execute_shell", "arguments": {"command": "noop"}},
        {"tool": "run_sql", "arguments": {"query": "select 1"}},
        {"tool": "get_failure_samples", "arguments": {"experiment_id": "sp-bm25", "limit": True}},
        {"tool": "get_failure_samples", "arguments": {"experiment_id": "sp-bm25", "limit": 21}},
        {"tool": "get_experiment_config", "arguments": {"experiment_id": "sp-bm25", "sql": "x"}},
    ],
)
def test_untrusted_plan_is_rejected(plan: dict) -> None:
    with pytest.raises(EvaluationError):
        execute_plan(plan)


def test_default_cli_never_reads_env_or_calls_network(monkeypatch, capsys, tmp_path: Path) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("default run must stay offline")

    monkeypatch.setattr(
        "scripts.evaluate_agent_llm.os", SimpleNamespace(environ=SimpleNamespace(get=forbidden))
    )
    monkeypatch.setattr(httpx.AsyncClient, "stream", forbidden)
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "UNRUN"
    existing = tmp_path / "measured.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--out", str(existing)])
    assert existing.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "raw",
    [
        {"choices": [None]},
        {"choices": None},
        {"choices": [{"finish_reason": "stop", "message": None}]},
        {"choices": [{"finish_reason": "stop", "message": []}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": 42}}]},
    ],
)
@pytest.mark.asyncio
async def test_malformed_envelope_is_counted_not_raised(tmp_path: Path, raw: dict) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))
    ) as client:
        summary = await evaluate(
            client,
            endpoint="http://localhost/test",
            model="fake",
            ledger_path=tmp_path / "run.jsonl",
        )
    assert summary["failed_requests_or_protocol"] == 50
    assert summary["usage_unknown_requests"] == 50
    assert summary["total_tokens"] is None


@pytest.mark.parametrize("raw", ['{"value": 1e999}', '{"value": NaN}', '{"x": 1, "x": 2}'])
def test_nonfinite_or_duplicate_json_is_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_object(raw)


def test_deterministic_report_does_not_claim_supported_citations() -> None:
    stats = evaluate_deterministic()
    assert stats["citation_nonempty"] == 27
    assert "citation_supported" not in stats
    report = render_deterministic(stats)
    assert "引用列表非空" in report
    assert "不代表工具参数" in report


@pytest.mark.asyncio
async def test_interrupted_request_is_unknown_and_never_resumed(tmp_path: Path) -> None:
    count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        raise asyncio.CancelledError

    path = tmp_path / "interrupted.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(asyncio.CancelledError):
            await evaluate(client, endpoint="http://localhost/test", model="fake", ledger_path=path)
        with pytest.raises(FileExistsError):
            await evaluate(client, endpoint="http://localhost/test", model="fake", ledger_path=path)
    assert count == 1
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["status"] == "pending" and events[-1]["usage"] is None
    assert all(event["event"] != "summary" for event in events)


@pytest.mark.asyncio
async def test_oversize_body_stops_stream_and_counts_failure(tmp_path: Path) -> None:
    requests = 0
    consumed_chunks = 0
    closed = 0
    chunk_size = 64 * 1024

    class OversizeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal consumed_chunks
            for _ in range(MAX_RESPONSE_BYTES // chunk_size + 1):
                consumed_chunks += 1
                yield b"x" * chunk_size
            raise AssertionError("must stop reading immediately after exceeding the limit")

        async def aclose(self) -> None:
            nonlocal closed
            closed += 1

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, stream=OversizeBody())

    path = tmp_path / "oversize.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        summary = await evaluate(
            client, endpoint="http://localhost/test", model="fake", ledger_path=path
        )
    assert requests == closed == 50
    assert consumed_chunks == 50 * (MAX_RESPONSE_BYTES // chunk_size + 1)
    assert summary["failed_requests_or_protocol"] == summary["tasks"] == 50
    assert summary["usage_unknown_requests"] == 50
    assert summary["structured_pass"] == 0 and summary["total_tokens"] is None
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    results = [event for event in events if event["event"] == "request_result"]
    assert all(event["status"] == "response_too_large" for event in results)
    assert all(event["usage"] is None for event in results)


def golden_response(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][-1]["content"]
    tasks = {task["question"]: task for task in build_tasks()}
    if content in tasks:
        return response(expected_call(tasks[content]))
    return response(answer(json.loads(content)["tool_result"]))


@pytest.mark.asyncio
async def test_selected_smoke_preserves_full_suite_denominators(tmp_path: Path) -> None:
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return golden_response(request)

    path = tmp_path / "smoke.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        summary = await evaluate(
            client,
            endpoint="http://localhost/test",
            model="fake",
            ledger_path=path,
            task_ids=["search-00"],
            max_requests=2,
            max_output_tokens=1024,
            max_tokens_per_request=512,
        )
    assert len(sent) == 2 and all(body["max_tokens"] == 512 for body in sent)
    assert summary["tasks"] == summary["suite_tasks"] == 50
    assert summary["selected_tasks"] == summary["attempted_tasks"] == 1
    assert summary["structured_pass"] == summary["completed_tasks"] == 1
    assert summary["unrun_tasks"] == 49 and summary["selected_unrun_tasks"] == 0
    assert summary["citation_needed"] == 32 and summary["trap_needed"] == 8
    assert summary["output_tokens_reserved"] == 1024
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    task_rows = [event for event in events if event["event"] == "task"]
    assert len(task_rows) == 50
    assert sum(row["status"] == "unrun" for row in task_rows) == 49


@pytest.mark.parametrize(
    "limits",
    [
        {"max_requests": 1, "max_output_tokens": 4096},
        {"max_requests": 2, "max_output_tokens": 2048},
    ],
)
@pytest.mark.asyncio
async def test_stage_two_requires_full_reservation_even_when_actual_usage_low(
    tmp_path: Path,
    limits: dict,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return golden_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        summary = await evaluate(
            client,
            endpoint="http://localhost/test",
            model="fake",
            ledger_path=tmp_path / "budget.jsonl",
            task_ids=["search-00"],
            **limits,
        )
    assert calls == summary["requests_sent"] == summary["requests_reserved"] == 1
    assert summary["known_total_tokens"] == 15
    assert summary["output_tokens_reserved"] == 2048  # no refund for the actual 5 output tokens
    assert summary["attempted_tasks"] == summary["budget_exhausted_tasks"] == 1
    assert summary["completed_tasks"] == summary["structured_pass"] == 0
    assert summary["unrun_tasks"] == 49


@pytest.mark.asyncio
async def test_budget_exhaustion_distinguishes_part_attempted_and_unrun(tmp_path: Path) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(golden_response)) as client:
        summary = await evaluate(
            client,
            endpoint="http://localhost/test",
            model="fake",
            ledger_path=tmp_path / "budget.jsonl",
            task_ids=["search-00", "search-01", "search-02"],
            max_requests=3,
            max_output_tokens=6144,
        )
    assert summary["selected_tasks"] == summary["requests_sent"] == 3
    assert summary["attempted_tasks"] == summary["budget_exhausted_tasks"] == 2
    assert summary["selected_unrun_tasks"] == summary["completed_tasks"] == 1
    assert summary["structured_pass"] == 1 and summary["unrun_tasks"] == 48
    assert summary["output_tokens_reserved"] == 6144


@pytest.mark.asyncio
async def test_unknown_usage_does_not_refund_reserved_budget(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("unknown usage", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        summary = await evaluate(
            client,
            endpoint="http://localhost/test",
            model="fake",
            ledger_path=tmp_path / "unknown.jsonl",
            task_ids=["search-00", "search-01"],
            max_requests=2,
            max_output_tokens=2048,
        )
    assert calls == summary["usage_unknown_requests"] == 1
    assert summary["total_tokens"] is None and summary["output_tokens_reserved"] == 2048
    assert summary["failed_requests_or_protocol"] == summary["selected_unrun_tasks"] == 1
    assert summary["structured_pass"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"max_requests": 0},
        {"max_requests": 101},
        {"max_output_tokens": -1},
        {"max_output_tokens": 204801},
        {"max_tokens_per_request": 2049},
        {"max_tokens_per_request": True},
        {"task_ids": ["missing"]},
        {"task_ids": ["search-00", "search-00"]},
        {"task_ids": []},
        {"timeout": float("nan")},
    ],
)
@pytest.mark.asyncio
async def test_invalid_run_options_make_no_file_and_no_calls(tmp_path: Path, options: dict) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid options must not call the model")

    path = tmp_path / "invalid.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(ValueError):
            await evaluate(
                client, endpoint="http://localhost/test", model="fake", ledger_path=path, **options
            )
    assert not path.exists()


def test_cli_budget_preview_and_invalid_options_never_read_env(monkeypatch, capsys) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("must not read config or call model")

    monkeypatch.setattr(
        "scripts.evaluate_agent_llm.os", SimpleNamespace(environ=SimpleNamespace(get=forbidden))
    )
    monkeypatch.setattr(httpx.AsyncClient, "stream", forbidden)
    assert (
        main(["--task-id", "search-00", "--max-requests", "2", "--max-output-tokens", "4096"]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "UNRUN"
    assert summary["selected_tasks"] == 1 and summary["unrun_tasks"] == 50
    assert summary["requests_sent"] == 0 and summary["max_output_tokens"] == 4096
    with pytest.raises(SystemExit):
        main(["--execute", "--max-requests", "0"])
