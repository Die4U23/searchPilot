"""GET /health/live 与 /health/ready。"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from searchpilot.api.deps import FeedbackDep, ProbesDep, RecommendDep, SearchDep
from searchpilot.contracts import LiveResponse, ReadyResponse
from searchpilot.errors import NOT_READY, ApiError, error_envelope

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=LiveResponse)
def live() -> LiveResponse:
    return LiveResponse()


@router.get("/health/ready", response_model=ReadyResponse)
def ready(
    probes: ProbesDep,
    search: SearchDep,
    recommend: RecommendDep,
    feedback: FeedbackDep,
) -> ReadyResponse | JSONResponse:
    # 端口缺失（启动时加载失败）本身就是未就绪；探针负责数据库等外部依赖。
    checks: dict[str, bool] = {
        "search": search is not None,
        "recommend": recommend is not None,
        "feedback": feedback is not None,
    }
    if search is not None:
        checks["vector_index"] = bool(search.vector_ready)
        checks["ltr"] = bool(search.ltr_ready)
    for probe in probes:
        try:
            checks.update({name: bool(ok) for name, ok in probe.check().items()})
        except Exception as exc:
            logger.warning("readiness probe failed: %s", type(exc).__name__)
            checks[type(probe).__name__] = False

    if all(checks.values()):
        return ReadyResponse(status="ready", checks=checks)

    error = ApiError(NOT_READY, "service is not ready", 503, retryable=True)
    body = error_envelope(error)
    body.update({"status": "not_ready", "checks": checks})
    return JSONResponse(status_code=503, content=body)
