"""把中间件、异常处理器与路由装配到 FastAPI 应用上。"""

from __future__ import annotations

from fastapi import FastAPI

from searchpilot.api import routes_feedback, routes_health, routes_recommend, routes_search
from searchpilot.api.middleware import RequestIdMiddleware
from searchpilot.errors import register_exception_handlers


def install_api(app: FastAPI) -> None:
    register_exception_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    for module in (routes_health, routes_search, routes_recommend, routes_feedback):
        app.include_router(module.router)
