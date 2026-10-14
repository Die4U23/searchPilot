from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from scripts.summarize_agent_llm import LedgerError, aggregate, main, read_shard

from agent_llm_helpers import golden_response
from searchpilot.agent import llm_eval
from searchpilot.agent.demo import build_tasks
from searchpilot.agent.llm_eval import evaluate, summarize


@pytest.fixture(scope="module")
def shard_events(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sharded-ledgers")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(golden_response)) as client:
            for name, selected in (
                ("smoke", ["search-00"]),
                ("remaining", [t["task_id"] for t in build_tasks() if t["task_id"] != "search-00"]),
            ):
                await evaluate(
                    client,
                    endpoint="http://localhost/test",
                    model="offline-fake",
                    ledger_path=directory / f"{name}.jsonl",
                    task_ids=selected,
                    max_requests=len(selected) * 2,
                    max_output_tokens=len(selected) * 4096,
                )

    asyncio.run(run())
    return [
        [
            json.loads(line)
            for line in (directory / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in ("smoke", "remaining")
    ]


def write_shards(tmp_path: Path, events: list) -> list[Path]:
    paths = []
    for index, shard in enumerate(events):
        path = tmp_path / f"shard-{index}.jsonl"
        path.write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in shard) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def test_complete_disjoint_shards_reproduce_full_suite(tmp_path: Path, shard_events) -> None:
    paths = write_shards(tmp_path, shard_events)
    before = [path.read_bytes() for path in paths]
    result = aggregate(paths)
    assert result["tasks"] == result["attempted_tasks"] == result["structured_pass"] == 50
    assert result["requests_sent"] == result["requests_reserved"] == 100
    assert result["output_tokens_reserved"] == 204800
    assert result["total_tokens"] == result["known_total_tokens"] == 1500
    assert result["usage_unknown_requests"] == result["unrun_tasks"] == 0
    assert result["citation_supported"] == result["citation_needed"] == 32
    assert result["trap_passed"] == result["trap_needed"] == 8
    assert [s["summary"]["requests_sent"] for s in result["shards"]] == [2, 98]
    assert all(len(s["ledger_sha256"]) == 64 for s in result["shards"])
    assert [path.read_bytes() for path in paths] == before
    assert "http://localhost" not in json.dumps(result)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("unpaired", "unpaired pending"),
        ("unfinished", "manifest/final summary"),
        ("pending", "pending/non-terminal"),
        ("manifest", "manifest versions"),
        ("stage", "task-stage mismatch"),
        ("summary", "summary does not match"),
    ],
)
def test_rejects_incomplete_or_inconsistent_ledgers(tmp_path: Path, shard_events, mutation, match):
    events = copy.deepcopy(shard_events)
    if mutation == "unpaired":
        events[0] = [
            e for e in events[0] if not (e["event"] == "request_result" and e["stage"] == "answer")
        ]
    elif mutation == "unfinished":
        events[1].pop()
    elif mutation == "pending":
        next(e for e in events[0] if e["event"] == "request_result")["status"] = "pending"
    elif mutation == "manifest":
        events[1][0]["endpoint_sha256"] = "0" * 64
    elif mutation == "stage":
        next(e for e in events[0] if e["event"] == "request_result")["stage"] = "answer"
    else:
        events[1][-1]["structured_pass"] = 0
    with pytest.raises(LedgerError, match=match):
        aggregate(write_shards(tmp_path, events))


def test_rejects_duplicate_global_request_id(tmp_path: Path, shard_events) -> None:
    events = copy.deepcopy(shard_events)
    first = next(e["request_id"] for e in events[0] if e["event"] == "request")
    replaced = next(e["request_id"] for e in events[1] if e["event"] == "request")
    for event in events[1]:
        if event.get("request_id") == replaced:
            event["request_id"] = first
    with pytest.raises(LedgerError, match="duplicate request ID across shards"):
        aggregate(write_shards(tmp_path, events))


def test_rejects_duplicate_attempted_task_even_with_fresh_request_ids(tmp_path: Path, shard_events):
    duplicate = copy.deepcopy(shard_events[0])
    for event in duplicate:
        if "request_id" in event:
            event["request_id"] += "_fresh"
    with pytest.raises(LedgerError, match="duplicate attempted task"):
        aggregate(write_shards(tmp_path, [shard_events[0], duplicate, shard_events[1]]))


def test_missing_task_cannot_be_hidden_by_unrun_rows(tmp_path: Path, shard_events) -> None:
    with pytest.raises(LedgerError, match="incomplete attempted fixed suite"):
        aggregate(write_shards(tmp_path, shard_events[:1]))


def test_unknown_usage_is_not_replaced_with_known_subtotal(tmp_path: Path, shard_events) -> None:
    events = copy.deepcopy(shard_events)
    next(e for e in events[0] if e["event"] == "request_result")["usage"] = None
    events[0][-1].update(
        usage_known_requests=1, usage_unknown_requests=1, known_total_tokens=15, total_tokens=None
    )
    result = aggregate(write_shards(tmp_path, events))
    assert result["known_total_tokens"] == 1485
    assert result["total_tokens"] is None
    assert result["usage_known_requests"] == 99 and result["usage_unknown_requests"] == 1


def test_cli_output_is_exclusive_and_input_unchanged(tmp_path: Path, shard_events, capsys) -> None:
    paths = write_shards(tmp_path, shard_events)
    out = tmp_path / "summary.json"
    args = ["--ledger", str(paths[0]), "--ledger", str(paths[1]), "--out", str(out)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["requests_sent"] == 100
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert out.read_bytes() == before


def test_historical_evaluator_hash_does_not_require_current_source_identity(
    tmp_path: Path, shard_events
):
    events = copy.deepcopy(shard_events)
    for shard in events:
        shard[0]["evaluator_sha256"] = "a" * 64
    result = aggregate(write_shards(tmp_path, events))
    assert result["structured_pass"] == 50
    assert (
        result["audit_code"]["scoring_helper_sha256"]
        == hashlib.sha256(Path(llm_eval.__file__).read_bytes()).hexdigest()
    )
    assert all(s["manifest"]["evaluator_sha256"] == "a" * 64 for s in result["shards"])


def test_source_evaluator_hashes_must_still_agree(tmp_path: Path, shard_events) -> None:
    events = copy.deepcopy(shard_events)
    events[0][0]["evaluator_sha256"] = "a" * 64
    with pytest.raises(LedgerError, match="manifest versions"):
        aggregate(write_shards(tmp_path, events))


@pytest.mark.parametrize(
    "field,match",
    [
        ("task_sha256", "current task protocol"),
        ("prompt_sha256", "current prompt protocol"),
        ("fixture_sha256", "current fixture protocol"),
    ],
)
def test_current_protocol_must_match_historical_ledgers(tmp_path: Path, shard_events, field, match):
    events = copy.deepcopy(shard_events)
    for shard in events:
        shard[0][field] = "a" * 64
    with pytest.raises(LedgerError, match=match):
        aggregate(write_shards(tmp_path, events))


def test_recorded_score_must_match_current_rescoring(tmp_path: Path, shard_events) -> None:
    events = copy.deepcopy(shard_events)
    next(e for e in events[0] if e["event"] == "task" and e["attempted"])["scores"][
        "structured_pass"
    ] = False
    with pytest.raises(LedgerError, match="task scores do not reproduce"):
        aggregate(write_shards(tmp_path, events))


@pytest.mark.parametrize("hidden_status", ["unrun", "budget_exhausted"])
def test_recomputed_terminal_summary_cannot_hide_request_failure(
    tmp_path: Path,
    shard_events,
    hidden_status: str,
) -> None:
    events = copy.deepcopy(shard_events)
    failed_result = next(
        e for e in events[0] if e["event"] == "request_result" and e["stage"] == "answer"
    )
    failed_result["status"] = "invalid_response"
    task = next(e for e in events[0] if e["event"] == "task" and e["attempted"])
    task.update(status=hidden_status, error="invalid_response")
    task.pop("scores")
    task.pop("final")
    # The terminal summary is forged consistently; validation must use the actual request result.
    events[0][-1].update(summarize([e for e in events[0] if e["event"] == "task"]))
    with pytest.raises(LedgerError, match="unrun task must|request failure must remain"):
        aggregate(write_shards(tmp_path, events))


@pytest.mark.parametrize(
    "limits",
    [
        {"max_requests": 1, "max_output_tokens": 4096},
        {"max_requests": 4, "max_output_tokens": 2048},
        {"max_requests": 2, "max_output_tokens": 1024},
    ],
)
def test_real_budget_exhaustion_before_plan_or_answer_is_valid(
    tmp_path: Path, limits: dict
) -> None:
    path = tmp_path / "budget.jsonl"

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(golden_response)) as client:
            await evaluate(
                client,
                endpoint="http://localhost/test",
                model="offline-fake",
                ledger_path=path,
                task_ids=["search-00", "search-01"],
                **limits,
            )

    asyncio.run(run())
    shard, _, _ = read_shard(path)
    assert shard["summary"]["budget_exhausted_tasks"] == 2
    assert shard["summary"]["failed_requests_or_protocol"] == 0
    assert shard["summary"]["requests_sent"] == (0 if limits["max_output_tokens"] < 2048 else 1)
    assert shard["summary"]["structured_pass"] == 0


def test_budget_exhausted_state_requires_actual_exhaustion(tmp_path: Path) -> None:
    path = tmp_path / "budget.jsonl"

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(golden_response)) as client:
            await evaluate(
                client,
                endpoint="http://localhost/test",
                model="offline-fake",
                ledger_path=path,
                task_ids=["search-00"],
                max_requests=1,
                max_output_tokens=4096,
            )

    asyncio.run(run())
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[0]["max_requests"] = events[-1]["max_requests"] = 2
    rewritten = write_shards(tmp_path, [events])[0]
    with pytest.raises(LedgerError, match="still has request/output allowance"):
        read_shard(rewritten)
