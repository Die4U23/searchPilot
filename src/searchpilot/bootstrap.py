"""应用组装：create_app 供测试注入假对象，build_default_app 读配置并组装生产依赖。"""

from __future__ import annotations

import importlib
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from searchpilot.api.app import install_api
from searchpilot.config import Settings, get_settings
from searchpilot.observability import configure_logging
from searchpilot.ports import (
    FeedbackStore,
    ItemStore,
    ReadinessProbe,
    RecommendPort,
    SearchPort,
)

logger = logging.getLogger(__name__)


def create_app(
    *,
    search: SearchPort | None = None,
    recommend: RecommendPort | None = None,
    feedback: FeedbackStore | None = None,
    items: ItemStore | None = None,
    probes: Sequence[ReadinessProbe] = (),
    ctr: object | None = None,
    experiments: object | None = None,
) -> FastAPI:
    """构造应用；依赖放在 app.state，路由通过 Depends 读取。None 表示该能力未就绪（503）。"""
    closers: list[Callable[[], object]] = []

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        for close in closers:
            try:
                close()
            except Exception:
                logger.warning("shutdown hook failed", exc_info=True)

    app = FastAPI(title="SearchPilot", version="0.1.0", lifespan=lifespan)
    app.state.settings = get_settings()
    app.state.search = search
    app.state.recommend = recommend
    app.state.feedback = feedback
    app.state.items = items
    app.state.probes = tuple(probes)
    app.state.ctr = ctr
    app.state.experiments = experiments
    app.state.closers = closers
    install_api(app)
    return app


def _attr(module_name: str, attr: str) -> Any:
    return getattr(importlib.import_module(module_name), attr)


def _load[T](name: str, factory: Callable[[], T]) -> T | None:
    """运行可选组件的工厂；模块缺失或未就绪都只记警告，应用照常启动。"""
    try:
        return factory()
    except (ImportError, Exception) as exc:
        logger.warning("%s unavailable, starting without it: %s: %s", name, type(exc).__name__, exc)
        return None


def _load_database(
    settings: Settings,
) -> tuple[Any, FeedbackStore | None, ItemStore | None, list[Any]]:
    """返回 (pool, feedback, items, probes)；失败时全部为空，反馈接口将返回 503。"""
    assert settings.database_url is not None
    url = settings.database_url

    def make_pool() -> Any:
        return _attr("searchpilot.db.connection", "create_pool")(url)

    pool = _load("database pool", make_pool)
    if pool is None:
        return None, None, None, []
    feedback = _load(
        "postgres feedback store",
        lambda: _attr("searchpilot.db.feedback_store", "PostgresFeedbackStore")(pool),
    )
    items = _load(
        "postgres item store", lambda: _attr("searchpilot.db.items", "PostgresItemStore")(pool)
    )
    probe = _load(
        "database probe", lambda: _attr("searchpilot.db.readiness", "DatabaseProbe")(pool)
    )
    return pool, feedback, items, [probe] if probe is not None else []


def build_default_app() -> FastAPI:
    """生产入口（uvicorn --factory）。任何组件加载失败都以 None 启动，绝不让应用崩溃。"""
    settings = get_settings()
    configure_logging(settings.log_level)

    search: SearchPort | None = _load(
        "search service",
        lambda: _attr("searchpilot.search.service", "build_search_service")(settings.artifact_dir),
    )
    recommend: RecommendPort | None = _load(
        "recommend service",
        lambda: _attr("searchpilot.recommend.service", "build_recommend_service")(
            settings.data_dir
        ),
    )

    pool: Any = None
    feedback: FeedbackStore | None
    items: ItemStore | None = None
    probes: list[Any] = []
    if settings.database_url:
        pool, feedback, items, probes = _load_database(settings)
    else:
        feedback = _load(
            "in-memory feedback store",
            lambda: _attr("searchpilot.feedback.memory", "InMemoryFeedbackStore")(),
        )
    # 登记文件不存在则是空登记，不因此启动失败。有数据库时也先用这份文件，
    # 避免只配了连接串就把实验查询打成 503。
    registry = settings.artifact_dir / "experiments" / "registry.json"

    def load_registry() -> object:
        store_cls = _attr("searchpilot.experiments.store", "InMemoryExperimentStore")
        return store_cls.load(registry)

    experiments = _load("experiment registry", load_registry)

    ctr = _load(
        "ctr runtime",
        lambda: _attr("searchpilot.ctr.runtime", "load_ctr")(settings.artifact_dir),
    )

    app = create_app(
        search=search,
        recommend=recommend,
        feedback=feedback,
        items=items,
        probes=probes,
        ctr=ctr,
        experiments=experiments,
    )
    if pool is not None:
        app.state.closers.append(pool.close)
    return app
