from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from scripts.evaluate_agent_multistep import main

from searchpilot.agent.multistep_eval import (
    SCORER_VERSION,
    build_cases,
    case_store,
    evaluate,
    score_case,
)
from searchpilot.agent.runtime import REFUSAL, run_agent


def dynamic_action(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Offline protocol fixture: derive IDs/values only from received observations."""
    question = messages[1]["content"]
    observations = [
        json.loads(message["content"])["untrusted_observation"]
        for message in messages[2:]
        if message["role"] == "user"
    ]
    previous = [
        json.loads(message["content"]) for message in messages[2:] if message["role"] == "assistant"
    ]

    def tool(name: str, arguments: dict[str, Any], plan: list[str] | None = None) -> dict[str, Any]:
        action: dict[str, Any] = {"type": "tool", "tool": name, "arguments": arguments}
        if plan:
            action["plan"] = plan
        return action

    if not observations:
        return tool("list_experiments", {}, ["Discover experiments and fetch the requested metric"])
    records = observations[0]["result"]["experiments"]
    target = records[0]["experiment_id"]
    if "全部" in question:
        if len(previous) == 1:
            return tool(
                "compare_experiments",
                {
                    "left_id": target,
                    "right_id": records[1]["experiment_id"],
                    "metrics": ["ndcg@10"],
                },
            )
        return {"type": "final", "answer": REFUSAL, "citations": []}
    names = [action["tool"] for action in previous]
    metrics = [action["arguments"].get("metric") for action in previous]
    if "recall@50" in question and "recall@50" not in metrics:
        return tool("get_metric", {"experiment_id": target, "metric": "recall@50", "split": "test"})
    if "配置" in question and "get_experiment_config" not in names:
        return tool("get_experiment_config", {"experiment_id": target})
    if "ndcg@10" not in metrics:
        plan = (
            ["The metric is missing; use ndcg@10 and inspect failures"]
            if "recall@50" in question
            else None
        )
        return tool(
            "get_metric", {"experiment_id": target, "metric": "ndcg@10", "split": "test"}, plan
        )
    if "失败样例" in question and "get_failure_samples" not in names:
        return tool("get_failure_samples", {"experiment_id": target})
    metric = next(
        observation["result"]
        for observation in observations
        if observation.get("ok") and observation.get("result", {}).get("metric") == "ndcg@10"
    )
    citation = f"[{target} | ndcg@10 | test | {metric['value']} | {metric['source']}]"
    return {"type": "final", "answer": citation, "citations": [citation]}


def dynamic_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    action = dynamic_action(body["messages"])
    return httpx.Response(
        200,
        json={
            "model": "fake-model",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(action)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        },
    )


async def test_all_eight_dynamic_cases_and_durable_public_ledger(tmp_path: Path) -> None:
    out = tmp_path / "new-run"
    summary = await evaluate(
        endpoint="https://private-provider.example/chat",
        model="fake-model",
        api_key="private-key",
        out_dir=out,
        transport=httpx.MockTransport(dynamic_response),
    )
    assert summary["status"] == "COMPLETED"
    assert summary["scorer_version"] == SCORER_VERSION
    assert summary["acceptance"] == "PASS"
    assert summary["structured_pass"] == summary["attempted_tasks"] == 8
    assert summary["requests_sent"] == 34
    assert summary["request_slots_reserved"] == 64
    assert summary["requests_reserved"] == 34
    assert summary["output_tokens_reserved"] == 34 * 2048
    assert summary["total_tokens"] == 34 * 13
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert all(row["scores"]["structured_pass"] for row in rows)
    assert all(row["model_actions"] for row in rows)
    ledger = (out / "events.jsonl").read_text(encoding="utf-8")
    assert "private-key" not in ledger and "private-provider.example" not in ledger
    events = [json.loads(line) for line in ledger.splitlines()]
    pending = [event for event in events if event["event"] == "request"]
    results = [event for event in events if event["event"] == "request_result"]
    assert len(pending) == len(results) == 34
    assert {event["request_id"] for event in pending} == {event["request_id"] for event in results}
    assert len({event["request_id"] for event in pending}) == 34
    assert all(event["usage"] is None for event in pending)
    assert events[0]["fixture_only"] is True
    assert events[0]["scorer_version"] == SCORER_VERSION
    assert set(events[0]["source_sha256"]) == {
        "provider",
        "runtime",
        "errors",
        "tools",
        "experiment_store",
        "evaluator",
        "cli",
        "system_prompt",
    }
    assert datetime.fromisoformat(summary["started_at"]) <= datetime.fromisoformat(
        summary["finished_at"]
    )
    assert all(
        datetime.fromisoformat(event["recorded_at"]).utcoffset().total_seconds() == 0
        for event in events
    )
    assert all(row["started_at"] is not None and row["finished_at"] is not None for row in rows)
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    inputs = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    assert inputs == build_cases()
    for case in inputs:
        assert all(record["experiment_id"] not in case["question"] for record in case["records"])
    for family in {case["family"] for case in inputs}:
        variants = [case for case in inputs if case["family"] == family]
        assert variants[0]["question"] == variants[1]["question"]
        assert variants[0]["expected_id"] != variants[1]["expected_id"]
        assert variants[0]["expected_value"] != variants[1]["expected_value"]


async def test_global_budget_reserves_before_tasks_and_retains_unrun(tmp_path: Path) -> None:
    summary = await evaluate(
        endpoint="http://localhost/chat",
        model="fake-model",
        out_dir=tmp_path / "run",
        max_requests=8,
        transport=httpx.MockTransport(dynamic_response),
    )
    assert summary["attempted_tasks"] == 1
    assert summary["unrun_tasks"] == 7
    assert summary["request_slots_reserved"] == 8
    assert summary["requests_sent"] == 5
    assert summary["acceptance"] == "FAIL"
    rows = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    assert all(row["reason"] == "global_request_reservation_exhausted" for row in rows[1:])


async def test_selected_task_and_unknown_usage(tmp_path: Path) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        value = json.loads(dynamic_response(request).content)
        value.pop("usage")
        return httpx.Response(200, json=value)

    summary = await evaluate(
        endpoint="http://localhost/chat",
        model="fake-model",
        out_dir=tmp_path / "run",
        task_ids=["fallback-2"],
        transport=httpx.MockTransport(response),
    )
    assert summary["structured_pass"] == summary["attempted_tasks"] == 1
    assert summary["acceptance"] == "PASS"
    assert summary["unrun_tasks"] == 7
    assert summary["total_tokens"] is None
    assert summary["usage_unknown_requests"] == summary["requests_sent"] == 5


async def test_failures_complete_execution_without_acceptance_or_known_usage(
    tmp_path: Path,
) -> None:
    summary = await evaluate(
        endpoint="http://localhost/chat",
        model="fake-model",
        out_dir=tmp_path / "run",
        task_ids=["discovery-1"],
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="private-error")),
    )
    assert summary["status"] == "COMPLETED" and summary["acceptance"] == "FAIL"
    assert summary["requests_sent"] == 1
    assert summary["completed_tasks"] == summary["structured_pass"] == 0
    assert summary["usage_unknown_requests"] == 1
    assert "private-error" not in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")


async def test_scorer_rejects_counterfactual_and_missing_observation(tmp_path: Path) -> None:
    await evaluate(
        endpoint="http://localhost/chat",
        model="fake-model",
        out_dir=tmp_path / "run",
        transport=httpx.MockTransport(dynamic_response),
    )
    rows = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    cases = build_cases()
    assert score_case(cases[1], rows[0]["result"])["structured_pass"] is False
    fallback = next(row for row in rows if row["task_id"] == "fallback-1")
    no_missing = copy.deepcopy(fallback["result"])
    no_missing["tool_trace"] = [entry for entry in no_missing["tool_trace"] if entry.get("ok")]
    assert not score_case(cases[2], no_missing)["structured_pass"]
    no_replan = copy.deepcopy(fallback["result"])
    for step in no_replan["steps"]:
        step.pop("plan", None)
    assert not score_case(cases[2], no_replan)["structured_pass"]
    injection = copy.deepcopy(rows[6]["result"])
    injection["steps"].append({"step": 7, "error": "forbidden_tool"})
    assert not score_case(cases[6], injection)["structured_pass"]
    guessed = copy.deepcopy(rows[0]["result"])
    guessed["tool_trace"] = guessed["tool_trace"][1:]
    assert not score_case(cases[0], guessed)["structured_pass"]


async def test_existing_directory_refused_before_request(tmp_path: Path) -> None:
    def response(_: httpx.Request) -> httpx.Response:
        raise AssertionError("must not dispatch")

    with pytest.raises(FileExistsError):
        await evaluate(
            endpoint="http://localhost/chat",
            model="fake-model",
            out_dir=tmp_path,
            transport=httpx.MockTransport(response),
        )


def test_cli_default_subprocess_schema_ignores_credentials() -> None:
    env = {
        **os.environ,
        "SEARCHPILOT_AGENT_LLM_ENDPOINT": "https://secret.example/?key=secret",
        "SEARCHPILOT_AGENT_LLM_MODEL": "secret model",
    }
    result = subprocess.run(
        [sys.executable, "-m", "scripts.evaluate_agent_multistep"],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout)
    assert value["status"] == "UNRUN"
    assert value["scorer_version"] == SCORER_VERSION
    assert value["requests_sent"] == 0
    assert value["max_requests"] == 64
    assert value["acceptance"] == "UNRUN"
    assert "secret" not in result.stdout


@pytest.mark.parametrize(
    "args",
    [["--execute"], ["--task-id", "unknown"], ["--max-requests", "65"], ["--timeout", "nan"]],
)
def test_cli_rejects_invalid_schema(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_cli_default_new_directory_and_no_overwrite(tmp_path: Path, capsys: Any) -> None:
    out = tmp_path / "unrun"
    assert main(["--out", str(out), "--task-id", "injection-1"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "UNRUN" and value["selected_tasks"] == 1
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == value
    with pytest.raises(SystemExit):
        main(["--out", str(out)])


def test_cli_execute_invalid_config_leaks_nothing_and_creates_no_output(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setenv("SEARCHPILOT_AGENT_LLM_ENDPOINT", "https://user:secret@example.com/chat")
    monkeypatch.setenv("SEARCHPILOT_AGENT_LLM_MODEL", "fake-model")
    monkeypatch.setenv("SEARCHPILOT_AGENT_LLM_API_KEY", "private-key")
    out = tmp_path / "never-created"
    with pytest.raises(SystemExit):
        main(["--execute", "--out", str(out)])
    output = capsys.readouterr()
    assert "secret" not in output.err and "private-key" not in output.err
    assert not out.exists()


@pytest.fixture
async def scorable_results() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    class DynamicModel:
        async def decide(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
            return dynamic_action(messages)

    cases_and_results = []
    for case in build_cases():
        result = await run_agent(
            case_store(case), case["question"], trace_id="offline-scorer", model=DynamicModel()
        )
        assert score_case(case, result)["structured_pass"]
        cases_and_results.append((case, result))
    return cases_and_results


@pytest.mark.parametrize("prefix_kind", ["failed", "empty", "filtered"])
async def test_scorer_accepts_list_only_recovery_before_discovery(
    scorable_results: list[tuple[dict[str, Any], dict[str, Any]]],
    prefix_kind: str,
) -> None:
    for case, result in scorable_results:
        corrected = copy.deepcopy(result)
        prefix: dict[str, Any] = {"tool": "list_experiments", "arguments": {}}
        if prefix_kind == "failed":
            prefix.update(arguments={"source": "search"}, ok=False, error="invalid_arguments")
        else:
            prefix.update(ok=True, result={"experiments": []})
            if prefix_kind == "filtered":
                prefix["arguments"] = {"kind": "ctr"}
        corrected["tool_trace"].insert(0, prefix)
        corrected["steps"].insert(0, {"type": "tool", **prefix})
        for index, step in enumerate(corrected["steps"], start=1):
            step["step"] = index
        score = score_case(case, corrected)
        assert score["structured_pass"], (case["task_id"], prefix_kind, score)
        assert score["scorer_version"] == SCORER_VERSION


@pytest.mark.parametrize(
    "early_tool",
    ["get_metric", "get_experiment_config", "get_failure_samples", "compare_experiments"],
)
@pytest.mark.parametrize("ok", [False, True])
async def test_scorer_rejects_guessed_id_reads_or_comparison_before_discovery(
    scorable_results: list[tuple[dict[str, Any], dict[str, Any]]],
    early_tool: str,
    ok: bool,
) -> None:
    for case, result in scorable_results:
        guessed = copy.deepcopy(result)
        args: dict[str, Any] = {"experiment_id": "guessed-unseen-id"}
        if early_tool == "compare_experiments":
            args = {
                "left_id": "guessed-unseen-id",
                "right_id": "other-guessed-id",
                "metrics": ["ndcg@10"],
            }
        guessed["tool_trace"].insert(0, {"tool": early_tool, "arguments": args, "ok": ok})
        score = score_case(case, guessed)
        assert not score["structured_pass"]
        assert not score["checks"]["discovered_before_reads"]


async def test_scorer_rejects_absent_discovery_and_missing_cross_source_id(
    scorable_results: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    for case, result in scorable_results:
        absent = copy.deepcopy(result)
        absent["tool_trace"] = [
            entry for entry in absent["tool_trace"] if entry["tool"] != "list_experiments"
        ]
        score = score_case(case, absent)
        assert not score["structured_pass"]
        assert not score["checks"]["discovered_before_reads"]
        if case["family"] == "cross_source":
            partial = copy.deepcopy(result)
            listed = partial["tool_trace"][0]["result"]["experiments"]
            listed[:] = [entry for entry in listed if entry["experiment_id"] != case["other_id"]]
            score = score_case(case, partial)
            assert not score["structured_pass"]
            assert not score["checks"]["compared_discovered_ids"]
