from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from searchpilot.agent.provider import HttpDecisionModel
from searchpilot.agent.runtime import run_agent
from searchpilot.bootstrap import create_app
from searchpilot.config import Settings
from searchpilot.experiments.store import ExperimentRecord, InMemoryExperimentStore, MetricPoint


def make_store() -> InMemoryExperimentStore:
    store = InMemoryExperimentStore()
    store.put(
        ExperimentRecord(
            experiment_id="live-new-id",
            kind="search",
            config={},
            data_version="v1",
            protocol_version="online-v1",
            source="searchpilot",
            metrics=(MetricPoint("auc", "test", 0.81),),
        )
    )
    return store


class DiscoveryModel:
    async def decide(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if len(messages) == 2:
            return {
                "type": "tool",
                "tool": "list_experiments",
                "arguments": {},
                "plan": ["发现当前登记", "读取指标"],
            }
        obs = json.loads(messages[-1]["content"])["untrusted_observation"]["result"]
        if "experiments" in obs:
            return {
                "type": "tool",
                "tool": "get_metric",
                "arguments": {
                    "experiment_id": obs["experiments"][0]["experiment_id"],
                    "metric": "auc",
                    "split": "test",
                },
            }
        cite = f"[{obs['experiment_id']} | auc | test | {obs['value']} | searchpilot]"
        return {"type": "final", "answer": cite, "citations": [cite]}


def test_online_loop_uses_injected_store_and_returns_public_actions() -> None:
    app = create_app(experiments=make_store(), agent_model=DiscoveryModel())
    app.state.settings = Settings(agent_mode="llm")
    with TestClient(app) as client:
        response = client.post("/agent/analyze", json={"question": "查询当前实验的 auc"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "llm" and body["status"] == "completed"
    assert body["citations"] == ["[live-new-id | auc | test | 0.81 | searchpilot]"]
    assert [r["tool"] for r in body["tool_trace"]] == ["list_experiments", "get_metric"]
    assert len(body["steps"]) == 3
    assert app.state.agent_active_runs == 0


def test_deterministic_default_does_not_call_injected_model() -> None:
    class ForbiddenModel:
        async def decide(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
            raise AssertionError("default must not call a model")

    app = create_app(experiments=make_store(), agent_model=ForbiddenModel())
    with TestClient(app) as client:
        response = client.post("/agent/analyze", json={"question": "列出实验"})
    assert response.status_code == 200
    assert response.json()["mode"] == "deterministic"


@pytest.mark.parametrize("endpoint", [None, "http://external.invalid/v1/chat/completions"])
def test_missing_or_invalid_provider_config_is_safe_503(endpoint: str | None) -> None:
    app = create_app(experiments=make_store())
    app.state.settings = Settings(
        agent_mode="llm", agent_llm_endpoint=endpoint, agent_llm_model="test-model"
    )
    with TestClient(app) as client:
        response = client.post("/agent/analyze", json={"question": "列出实验"})
    assert response.status_code == 503
    assert "external.invalid" not in response.text
    assert app.state.agent_active_runs == 0


async def test_online_concurrency_rejects_without_starting_extra_model() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    class WaitingModel:
        async def decide(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            raise RuntimeError("unavailable")

    app = create_app(experiments=make_store(), agent_model=WaitingModel())
    app.state.settings = Settings(agent_mode="llm", agent_max_concurrent_runs=1, agent_max_steps=2)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        first = asyncio.create_task(c.post("/agent/analyze", json={"question": "first"}))
        await asyncio.wait_for(entered.wait(), 2)
        second = await c.post("/agent/analyze", json={"question": "second"})
        assert second.status_code == 429 and calls == 1
        release.set()
        response = await first
    assert response.json()["status"] != "completed"
    assert app.state.agent_active_runs == 0


def test_llm_mode_retains_caller_authentication() -> None:
    app = create_app(experiments=make_store(), agent_model=DiscoveryModel())
    app.state.settings = Settings(agent_mode="llm", api_token="test-only-token")
    with TestClient(app) as client:
        assert client.post("/agent/analyze", json={"question": "list"}).status_code == 401


def test_settings_hide_credentials_in_repr() -> None:
    settings = Settings(
        agent_llm_endpoint="https://private.invalid/route", agent_llm_api_key="private-test-secret"
    )
    assert "private-test-secret" not in repr(settings)
    assert "private.invalid" not in repr(settings)


@pytest.mark.parametrize(
    "status,code", [(401, "authentication_failed"), (429, "rate_limited"), (500, "http_error")]
)
async def test_provider_failure_stops_without_paid_retries(status: int, code: str) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="private provider response")

    async with HttpDecisionModel(
        "https://example.invalid/completions", "test-model", transport=httpx.MockTransport(respond)
    ) as provider:
        result = await run_agent(
            make_store(), "list experiments", trace_id="trace_failure", model=provider
        )
    assert calls == 1
    assert result["status"] == "failed" and result["stop_reason"] == code
    assert provider.summary()["requests_sent"] == 1
    assert "private provider response" not in json.dumps(result)
