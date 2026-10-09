"""POST /ctr/score。"""

from __future__ import annotations

from fastapi import APIRouter

from searchpilot.api.deps import CtrDep
from searchpilot.contracts import CtrScoreHitOut, CtrScoreRequest, CtrScoreResponse, ErrorResponse
from searchpilot.errors import NOT_READY, ApiError
from searchpilot.observability import get_request_id, new_request_id

router = APIRouter(tags=["ctr"])


@router.post(
    "/ctr/score",
    response_model=CtrScoreResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def ctr_score(body: CtrScoreRequest, runtime: CtrDep) -> CtrScoreResponse:
    if runtime is None:
        raise ApiError(NOT_READY, "ctr model is not ready", 503, retryable=True)

    hits = runtime.score(
        body.user_id,
        [(candidate.item_id, candidate.bid) for candidate in body.candidates],
        calibrated=body.calibrated,
    )
    return CtrScoreResponse(
        request_id=get_request_id() or new_request_id(),
        model_version=runtime.model_version,
        calibrator_version=runtime.calibrator_version,
        results=[
            CtrScoreHitOut(
                item_id=hit.item_id,
                pctr=hit.pctr,
                pctr_calibrated=hit.pctr_calibrated,
                ecpm=hit.ecpm,
            )
            for hit in hits
        ],
    )
