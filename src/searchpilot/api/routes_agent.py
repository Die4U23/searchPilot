"""POST /agent/analyze。只读，不提供 SQL 或 Shell。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from searchpilot.agent.analyze import analyze
from searchpilot.agent.runtime import AgentLimits, run_agent
from searchpilot.agent.tools import new_trace_id
from searchpilot.api.deps import ExperimentsDep, SettingsDep
from searchpilot.contracts import AgentAnalyzeRequest, AgentAnalyzeResponse, ErrorResponse
from searchpilot.errors import NOT_READY, ApiError
from searchpilot.observability import get_request_id, new_request_id

router = APIRouter(tags=["agent"])


@router.post(
    "/agent/analyze",
    response_model=AgentAnalyzeResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def agent_analyze(
    body: AgentAnalyzeRequest, store: ExperimentsDep, settings: SettingsDep, request: Request
) -> AgentAnalyzeResponse:
    if store is None:
        raise ApiError(NOT_READY, "experiment store is not ready", 503, retryable=True)
    trace_id = new_trace_id()
    if settings.agent_mode == "deterministic":
        result = await run_in_threadpool(analyze, store, body.question, trace_id=trace_id)
    else:
        if request.app.state.agent_active_runs >= settings.agent_max_concurrent_runs:
            raise ApiError("AGENT_BUSY", "agent capacity is full", 429, retryable=True)
        request.app.state.agent_active_runs += 1
        try:
            limits = AgentLimits(
                max_steps=settings.agent_max_steps,
                total_timeout_seconds=settings.agent_timeout_seconds,
            )
            model = request.app.state.agent_model
            if model is not None:
                result = await run_agent(
                    store, body.question, trace_id=trace_id, model=model, limits=limits
                )
            else:
                if not settings.agent_llm_endpoint or not settings.agent_llm_model:
                    raise ApiError(NOT_READY, "agent model is not configured", 503, retryable=True)
                # Keep the optional HTTP dependency out of deterministic startup.
                try:
                    from searchpilot.agent.provider import HttpDecisionModel, ProviderError
                except ImportError:
                    raise ApiError(NOT_READY, "install the agent extra", 503) from None
                key = settings.agent_llm_api_key
                try:
                    provider = HttpDecisionModel(
                        settings.agent_llm_endpoint,
                        settings.agent_llm_model,
                        key.get_secret_value() if key else None,
                        max_requests=settings.agent_max_steps,
                    )
                except (ValueError, ProviderError):
                    raise ApiError(NOT_READY, "agent configuration is invalid", 503) from None
                async with provider:
                    result = await run_agent(
                        store, body.question, trace_id=trace_id, model=provider, limits=limits
                    )
                result["usage"] = provider.summary()
        finally:
            request.app.state.agent_active_runs -= 1
        result["mode"] = "llm"
    return AgentAnalyzeResponse(
        request_id=get_request_id() or new_request_id(),
        **result,
    )
