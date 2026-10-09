"""可观测性：JSON 日志格式化与 request_id 上下文。"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

REQUEST_ID_PATTERN = re.compile(r"^req_[0-9a-f]{32}$")

_request_id_var: ContextVar[str | None] = ContextVar("searchpilot_request_id", default=None)

# LogRecord 自带属性；其余通过 extra= 传入的字段原样输出到 JSON。
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys() | {"message", "asctime"}
)
_HANDLER_MARK = "_searchpilot_json_handler"


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex


def is_valid_request_id(value: str) -> bool:
    return REQUEST_ID_PATTERN.fullmatch(value) is not None


def get_request_id() -> str | None:
    return _request_id_var.get()


def bind_request_id(request_id: str) -> Token[str | None]:
    return _request_id_var.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id_var.reset(token)


class JsonFormatter(logging.Formatter):
    """单行 JSON：ts、level、logger、msg、request_id，外加 extra 字段与异常文本。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or get_request_id(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _StderrJsonHandler(logging.Handler):
    """每次写入时再取 sys.stderr，避免测试捕获流关闭后报错。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def configure_logging(level: str | int = "INFO") -> None:
    """给根 logger 安装 JSON 处理器；重复调用只更新级别，不会重复安装。"""
    root = logging.getLogger()
    handler = next((h for h in root.handlers if getattr(h, _HANDLER_MARK, False)), None)
    if handler is None:
        handler = _StderrJsonHandler()
        handler.setFormatter(JsonFormatter())
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)
    resolved = level.upper() if isinstance(level, str) else level
    root.setLevel(resolved)
