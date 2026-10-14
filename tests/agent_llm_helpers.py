"""Deterministic HTTP responses for offline external-evaluator tests."""

from __future__ import annotations

import json

import httpx

from searchpilot.agent.demo import build_tasks
from searchpilot.agent.llm_eval import REFUSAL, expected_call, metric_citations


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


def golden_response(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][-1]["content"]
    tasks = {task["question"]: task for task in build_tasks()}
    if content in tasks:
        return response(expected_call(tasks[content]))
    return response(answer(json.loads(content)["tool_result"]))
