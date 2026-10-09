"""GET /experiments/{experiment_id}。没有数据库时读内存登记。"""

from __future__ import annotations

from fastapi import APIRouter

from searchpilot.api.deps import ExperimentsDep
from searchpilot.contracts import ErrorResponse, ExperimentResponse
from searchpilot.errors import ITEM_NOT_FOUND, NOT_READY, ApiError

router = APIRouter(tags=["experiments"])


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_experiment(experiment_id: str, store: ExperimentsDep) -> ExperimentResponse:
    if store is None:
        raise ApiError(NOT_READY, "experiment store is not ready", 503, retryable=True)
    record = store.get(experiment_id)
    if record is None:
        raise ApiError(ITEM_NOT_FOUND, "experiment does not exist", 404)
    return ExperimentResponse.model_validate(record.to_dict())
