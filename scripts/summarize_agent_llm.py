"""Read-only validation and aggregation of completed, non-overlapping Agent ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from searchpilot.agent import llm_eval
from searchpilot.agent.demo import build_tasks, demo_store
from searchpilot.agent.llm_eval import digest, execute_plan, parse_object, score, summarize

VERSION_FIELDS = (
    "endpoint_sha256",
    "model",
    "evaluator_sha256",
    "task_sha256",
    "prompt_sha256",
    "fixture_sha256",
    "temperature",
    "max_tokens_per_request",
    "max_response_bytes",
    "timeout_seconds_per_request",
    "fixture_only",
    "retry_count",
)
MANIFEST_FIELDS = (
    *VERSION_FIELDS,
    "task_count",
    "selected_task_ids",
    "max_requests",
    "max_output_tokens",
    "recorded_at",
    "status",
)
RESULT_STATUSES = {"completed", "http_error", "timeout", "invalid_response", "response_too_large"}


class LedgerError(ValueError):
    """The ledger cannot support a complete, reproducible combined result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LedgerError(message)


def integer(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"invalid {name}")
    return value


def usage_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in results:
        usage = result.get("usage")
        require(usage is None or isinstance(usage, dict), "invalid usage object")
        if usage is not None:
            require(
                set(usage) <= {"prompt_tokens", "completion_tokens", "total_tokens"},
                "unknown usage fields",
            )
            for value in usage.values():
                integer(value, "usage token count")
            if set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}:
                require(
                    usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
                    "inconsistent usage total",
                )
    totals = [
        r["usage"]["total_tokens"]
        for r in results
        if r.get("usage") is not None and "total_tokens" in r["usage"]
    ]
    return {
        "requests_sent": len(results),
        "usage_known_requests": len(totals),
        "usage_unknown_requests": len(results) - len(totals),
        "known_total_tokens": sum(totals),
        "total_tokens": sum(totals) if len(totals) == len(results) else None,
    }


def read_shard(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = path.read_bytes()
    events = [parse_object(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    require(
        len(events) >= 2
        and events[0].get("event") == "manifest"
        and events[-1].get("event") == "summary",
        "ledger lacks initial manifest/final summary",
    )
    manifest, terminal = events[0], events[-1]
    require(
        manifest.get("status") == "RUNNING" and terminal.get("status") == "COMPLETED",
        "non-terminal ledger",
    )
    require(all(key in manifest for key in MANIFEST_FIELDS), "incomplete manifest")
    tasks = {task["task_id"]: task for task in build_tasks()}
    selected = manifest["selected_task_ids"]
    require(
        isinstance(selected, list)
        and all(isinstance(t, str) for t in selected)
        and len(selected) == len(set(selected))
        and set(selected) <= tasks.keys(),
        "invalid selected task IDs",
    )
    require(manifest["task_count"] == len(tasks), "suite size mismatch")
    per_request = integer(manifest["max_tokens_per_request"], "per-request reservation")
    require(1 <= per_request <= 2048, "invalid per-request reservation")
    request_limit = integer(manifest["max_requests"], "request budget")
    output_limit = integer(manifest["max_output_tokens"], "output budget")
    requests: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    rows: dict[str, dict[str, Any]] = {}
    stages: dict[str, list[str]] = {}
    for event in events[1:-1]:
        kind = event.get("event")
        task_id = event.get("task_id")
        require(isinstance(task_id, str) and task_id in tasks, "unknown task ID")
        assert isinstance(task_id, str)
        if kind in {"request", "request_result"}:
            require(
                task_id in selected and task_id not in rows, "request outside selected/open task"
            )
            request_id, stage = event.get("request_id"), event.get("stage")
            require(isinstance(request_id, str) and bool(request_id), "invalid request ID")
            require(stage in {"plan", "answer"}, "invalid request stage")
            assert isinstance(request_id, str) and isinstance(stage, str)
            if kind == "request":
                require(request_id not in requests, "duplicate request ID")
                prior = stages.setdefault(task_id, [])
                require(
                    prior == ([] if stage == "plan" else ["plan"]), "duplicate/out-of-order stage"
                )
                if stage == "answer":
                    plan_id = next(
                        k
                        for k, r in requests.items()
                        if r["task_id"] == task_id and r["stage"] == "plan"
                    )
                    require(
                        plan_id in results and results[plan_id]["status"] == "completed",
                        "answer precedes a completed plan",
                    )
                require(
                    event.get("status") == "pending" and event.get("usage") is None,
                    "invalid pending request",
                )
                require(
                    event.get("reserved_output_tokens") == per_request
                    and event.get("requests_reserved_total") == len(requests) + 1
                    and event.get("output_tokens_reserved_total")
                    == (len(requests) + 1) * per_request,
                    "reservation sequence mismatch",
                )
                requests[request_id] = event
                prior.append(stage)
            else:
                require(
                    request_id in requests and request_id not in results,
                    "unpaired/duplicate request result",
                )
                pending = requests[request_id]
                require(
                    all(
                        event.get(key) == pending.get(key)
                        for key in (
                            "task_id",
                            "stage",
                            "reserved_output_tokens",
                            "requests_reserved_total",
                            "output_tokens_reserved_total",
                        )
                    ),
                    "request/result task-stage mismatch",
                )
                require(
                    event.get("status") in RESULT_STATUSES, "pending/non-terminal request result"
                )
                results[request_id] = event
        elif kind == "task":
            require(task_id not in rows, "duplicate task row")
            require(event.get("kind") == tasks[task_id]["kind"], "task kind mismatch")
            task_requests = [r for r in requests.values() if r["task_id"] == task_id]
            require(
                all(r["request_id"] in results for r in task_requests), "unpaired pending request"
            )
            require(
                type(event.get("selected")) is bool
                and event["selected"] == (task_id in selected)
                and type(event.get("attempted")) is bool
                and event["attempted"] == bool(task_requests),
                "task attempt/selection mismatch",
            )
            elapsed = event.get("elapsed_seconds")
            require(type(elapsed) in {float, int}, "invalid task duration")
            assert isinstance(elapsed, int | float)
            require(math.isfinite(elapsed) and elapsed >= 0, "invalid task duration")
            status = event.get("status")
            require(
                status in {"completed", "failed", "unrun", "budget_exhausted"}, "invalid task state"
            )
            if status == "unrun":
                require(
                    not event["selected"] and not event["attempted"],
                    "unrun task must be unselected and unattempted",
                )
            else:
                require(event["selected"], "unselected task is not unrun")
                if status in {"completed", "failed"}:
                    require(event["attempted"], "completed/failed task without request")
            task_results = [results[r["request_id"]] for r in task_requests]
            request_failures = [r for r in task_results if r["status"] != "completed"]
            if request_failures:
                require(status == "failed", "request failure must remain a failed task")
                require(
                    event.get("error") == request_failures[-1]["status"],
                    "task error does not match request failure",
                )
            if status == "budget_exhausted":
                require(
                    stages.get(task_id, []) in ([], ["plan"]),
                    "budget-exhausted task already sent its answer",
                )
                require(
                    len(requests) >= request_limit
                    or (len(requests) + 1) * per_request > output_limit,
                    "budget-exhausted task still has request/output allowance",
                )
                require(
                    event.get("error") == "request_or_output_budget_exhausted",
                    "budget-exhausted task has invalid error",
                )
                if task_requests:
                    require(
                        digest(event.get("tool_result")) == digest(execute_plan(event.get("plan"))),
                        "budget-exhausted task lacks a valid completed plan",
                    )
            if status == "completed":
                require(
                    stages[task_id] == ["plan", "answer"]
                    and all(
                        results[r["request_id"]]["status"] == "completed" for r in task_requests
                    ),
                    "completed task lacks two completed stages",
                )
                require(
                    digest(event.get("tool_result")) == digest(execute_plan(event.get("plan"))),
                    "tool evidence does not reproduce",
                )
                rescored = score(
                    tasks[task_id], event["plan"], event["tool_result"], event.get("final")
                )
                require(event.get("scores") == rescored, "task scores do not reproduce")
            else:
                require(not event.get("scores"), "non-completed task carries scores")
            rows[task_id] = event
        else:
            raise LedgerError("unexpected ledger event")
    require(set(requests) == set(results), "unpaired pending request")
    require(set(rows) == set(tasks), "missing fixed-suite task row")
    reserved = len(requests) * per_request
    require(len(requests) <= request_limit and reserved <= output_limit, "shard budget exceeded")
    calculated = {
        **summarize(list(rows.values())),
        **usage_summary(list(results.values())),
        "requests_reserved": len(requests),
        "output_tokens_reserved": reserved,
        "max_requests": request_limit,
        "max_output_tokens": output_limit,
        "max_tokens_per_request": per_request,
    }
    require(
        all(terminal.get(key) == value for key, value in calculated.items()),
        "terminal summary does not match ledger events",
    )
    return (
        {
            "ledger": str(path),
            "ledger_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest": {key: manifest[key] for key in MANIFEST_FIELDS},
            "summary": calculated,
        },
        list(rows.values()),
        list(results.values()),
    )


def aggregate(paths: list[Path]) -> dict[str, Any]:
    require(bool(paths), "at least one ledger is required")
    require(len({p.resolve() for p in paths}) == len(paths), "duplicate ledger path")
    shards, attempted, requests = [], {}, {}
    reference: dict[str, Any] | None = None
    for path in paths:
        shard, rows, results = read_shard(path)
        version = {key: shard["manifest"][key] for key in VERSION_FIELDS}
        if reference is None:
            reference = version
        require(reference == version, "manifest versions/policies mismatch")
        for row in rows:
            if row["attempted"]:
                require(row["task_id"] not in attempted, "duplicate attempted task across shards")
                attempted[row["task_id"]] = row
        for result in results:
            require(result["request_id"] not in requests, "duplicate request ID across shards")
            requests[result["request_id"]] = result
        shards.append(shard)
    tasks = build_tasks()
    require(set(attempted) == {t["task_id"] for t in tasks}, "incomplete attempted fixed suite")
    assert reference is not None
    require(reference["task_sha256"] == digest(tasks), "current task protocol mismatch")
    require(
        reference["prompt_sha256"] == digest([llm_eval.PLAN_PROMPT, llm_eval.ANSWER_PROMPT]),
        "current prompt protocol mismatch",
    )
    require(
        reference["fixture_sha256"] == digest([r.to_dict() for r in demo_store().list_records()]),
        "current fixture protocol mismatch",
    )
    combined = summarize([attempted[t["task_id"]] for t in tasks])
    combined.update(usage_summary(list(requests.values())))
    combined.update(
        {
            "requests_reserved": sum(s["summary"]["requests_reserved"] for s in shards),
            "output_tokens_reserved": sum(s["summary"]["output_tokens_reserved"] for s in shards),
            "max_requests": sum(s["summary"]["max_requests"] for s in shards),
            "max_output_tokens": sum(s["summary"]["max_output_tokens"] for s in shards),
            "audit_code": {
                "scoring_helper_sha256": hashlib.sha256(
                    Path(llm_eval.__file__).read_bytes()
                ).hexdigest(),
                "aggregation_script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "validation": "source versions agree; current task/prompt/fixture and scores agree",
            },
            "shards": shards,
            "boundaries": {
                "fixture_only": True,
                "free_text_semantics": "UNSCORED",
                "usage_source": "provider-reported; unknown is not zero",
                "result_scope": "fixed-suite structured protocol, not general answer accuracy",
            },
        }
    )
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, help="Optional NEW output JSON; never overwritten")
    args = parser.parse_args(argv)
    if args.out is not None and args.out.exists():
        parser.error("output already exists")
    try:
        combined = aggregate(args.ledger)
        rendered = json.dumps(combined, ensure_ascii=False, indent=2) + "\n"
        if args.out is not None:
            with args.out.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
    except (ValueError, OSError, KeyError, TypeError, llm_eval.EvaluationError) as exc:
        parser.error(f"ledger validation failed: {type(exc).__name__}: {exc}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
