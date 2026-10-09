"""数据库就绪探针。"""

from __future__ import annotations

from psycopg_pool import ConnectionPool


class DatabaseProbe:
    """执行最小查询并把所有数据库故障收敛为 false。"""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def check(self) -> dict[str, bool]:
        try:
            with self._pool.connection() as connection:
                connection.execute("SELECT 1").fetchone()
        except Exception:
            return {"database": False}
        return {"database": True}
