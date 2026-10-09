"""POST /agent/analyze。只读，不提供 SQL 或 Shell。"""

from __future__ import annotations

from fastapi import APIRouter

from searchpilot.agent.analyze import analyze
from searchpilot.agent.tools import new_trace_id
from searchpilot.api.deps import ExperimentsDep
from searchpilot.contracts import AgentAnalyzeRequest, AgentAnalyzeResponse, ErrorResponse
from searchpilot.errors import NOT_READY, ApiError
from searchpilot.observability import get_request_id, new_request_id

router = APIRouter(tags=["agent"])


@router.post(
    "/agent/analyze",
    response_model=AgentAnalyzeResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def agent_analyze(body: AgentAnalyzeRequest, store: ExperimentsDep) -> AgentAnalyzeResponse:
    if store is None:
        raise ApiError(NOT_READY, "experiment store is not ready", 503, retryable=True)
    result = analyze(store, body.question, trace_id=new_trace_id())
    return AgentAnalyzeResponse(
        request_id=get_request_id() or new_request_id(),
        trace_id=result["trace_id"],
        answer=result["answer"],
        citations=result["citations"],
        tool_trace=result["tool_trace"],
    )
