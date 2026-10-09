"""PostgreSQL 连接池工厂。"""

from __future__ import annotations

import os

from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import ConnectionPool


def create_pool(database_url: str) -> ConnectionPool:
    """创建使用环境变量配置大小与超时的同步连接池。

    - ``SEARCHPILOT_DB_POOL_MIN`` / ``_MAX``：池大小（默认 1 / 4）。
    - ``SEARCHPILOT_DB_CONNECT_TIMEOUT_S``：建连超时（默认 5 秒）。
    - ``SEARCHPILOT_DB_STATEMENT_TIMEOUT_MS``：服务端 ``statement_timeout``（默认 5000 毫秒），
      超时的语句由 PostgreSQL 中止并抛错，事务随之回滚（PRD NFR：下游必须有超时）。

    URL 里显式给出的连接参数优先于环境变量：``options``（如 ``search_path``）会原样保留，
    ``statement_timeout`` 只在其未被显式设置时追加，绝不覆盖已有值。
    """
    minimum = int(os.environ.get("SEARCHPILOT_DB_POOL_MIN", "1"))
    maximum = int(os.environ.get("SEARCHPILOT_DB_POOL_MAX", "4"))
    if minimum < 0 or maximum < 1 or minimum > maximum:
        raise ValueError("invalid SEARCHPILOT_DB_POOL_MIN/MAX values")
    connect_timeout = int(os.environ.get("SEARCHPILOT_DB_CONNECT_TIMEOUT_S", "5"))
    statement_timeout_ms = int(os.environ.get("SEARCHPILOT_DB_STATEMENT_TIMEOUT_MS", "5000"))
    if connect_timeout < 1 or statement_timeout_ms < 1:
        raise ValueError("invalid SEARCHPILOT_DB_*_TIMEOUT values")
    info = conninfo_to_dict(database_url)
    info.setdefault("connect_timeout", str(connect_timeout))
    existing_options = str(info.get("options") or "")
    if "statement_timeout" not in existing_options:
        info["options"] = f"{existing_options} -c statement_timeout={statement_timeout_ms}".strip()
    # psycopg 的类型桩把 `**kwargs: ConnParam` 注成了单参数 `kwargs: ConnParam`，
    # 导致 **dict 展开被误报为 arg-type；运行时签名就是 **kwargs: ConnParam。
    return ConnectionPool(
        make_conninfo(**{k: v for k, v in info.items() if v is not None}),  # type: ignore[arg-type]
        min_size=minimum,
        max_size=maximum,
        timeout=float(connect_timeout),  # 从池中取连接的等待上限
    )
