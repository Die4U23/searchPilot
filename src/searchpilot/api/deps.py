"""FastAPI 依赖：从 app.state 取出注入的服务端口。缺失（None）由路由决定如何响应。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from fastapi import Depends, Request

from searchpilot.config import Settings
from searchpilot.ctr.runtime import CtrRuntime
from searchpilot.ports import (
    FeedbackStore,
    ItemStore,
    ReadinessProbe,
    RecommendPort,
    SearchPort,
)


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_search(request: Request) -> SearchPort | None:
    service: SearchPort | None = request.app.state.search
    return service


def get_recommend(request: Request) -> RecommendPort | None:
    service: RecommendPort | None = request.app.state.recommend
    return service


def get_feedback(request: Request) -> FeedbackStore | None:
    store: FeedbackStore | None = request.app.state.feedback
    return store


def get_items(request: Request) -> ItemStore | None:
    store: ItemStore | None = request.app.state.items
    return store


def get_probes(request: Request) -> Sequence[ReadinessProbe]:
    probes: Sequence[ReadinessProbe] = request.app.state.probes
    return probes


def get_ctr(request: Request) -> CtrRuntime | None:
    service: CtrRuntime | None = request.app.state.ctr
    return service


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SearchDep = Annotated[SearchPort | None, Depends(get_search)]
RecommendDep = Annotated[RecommendPort | None, Depends(get_recommend)]
FeedbackDep = Annotated[FeedbackStore | None, Depends(get_feedback)]
ItemsDep = Annotated[ItemStore | None, Depends(get_items)]
ProbesDep = Annotated[Sequence[ReadinessProbe], Depends(get_probes)]
CtrDep = Annotated[CtrRuntime | None, Depends(get_ctr)]
