"""可选的调用方令牌，以及反馈写入时的对象级核对。

未设置 ``SEARCHPILOT_API_TOKEN`` 时不检查。设置之后，除 ``/health/live`` 外都要带
``Authorization: Bearer <token>``。``POST /feedback`` 还要求 ``X-Actor`` 与 ``user_id`` 相同。
"""

from __future__ import annotations

import secrets

from fastapi import Request

from searchpilot.errors import FORBIDDEN, UNAUTHORIZED, ApiError

_BEARER = "Bearer "


def require_caller(request: Request) -> None:
    """令牌未配置时直接通过。``/health/live`` 始终开放，供进程探针使用。"""
    token = _configured_token(request)
    if token is None or request.url.path == "/health/live":
        return
    if not _bearer_matches(request.headers.get("authorization", ""), token):
        raise ApiError(UNAUTHORIZED, "authentication is required", 401)


def require_feedback_owner(request: Request, user_id: str | None) -> None:
    """令牌开启时，反馈只能写给与调用方相同的 user_id。"""
    if _configured_token(request) is None:
        return
    actor = request.headers.get("x-actor", "").strip()
    if not actor or user_id != actor:
        raise ApiError(FORBIDDEN, "feedback user does not match the caller", 403)


def _configured_token(request: Request) -> str | None:
    token = getattr(request.app.state.settings, "api_token", None)
    if not isinstance(token, str) or not token:
        return None
    return token


def _bearer_matches(header: str, token: str) -> bool:
    if not header.startswith(_BEARER):
        return False
    given = header.removeprefix(_BEARER)
    if len(given) != len(token):
        return False
    return secrets.compare_digest(given, token)
