"""PostgreSQL 物品读取仓储。"""

from __future__ import annotations

from psycopg_pool import ConnectionPool

from searchpilot.ports import Document


class PostgresItemStore:
    """通过连接池按 item_id 读取搜索文档。"""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def get(self, item_id: str) -> Document | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT item_id, title, abstract, category, subcategory
                FROM items
                WHERE item_id = %s
                """,
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        return Document(
            item_id=row[0],
            title=row[1],
            abstract=row[2],
            category=row[3],
            subcategory=row[4],
        )

    def exists(self, item_id: str) -> bool:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM items WHERE item_id = %s", (item_id,)
            ).fetchone()
        return row is not None
