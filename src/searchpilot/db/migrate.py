"""带摘要校验与 advisory lock 的纯 SQL 迁移运行器。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg

MIGRATION_LOCK_ID = 5_343_145_049_576_079_796


class MigrationDriftError(RuntimeError):
    """已应用迁移的内容摘要与磁盘文件不一致。"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_migrations(database_url: str, migrations_dir: Path) -> list[str]:
    """按文件名应用迁移，拒绝修改任何已登记的迁移文件。"""
    migration_paths = sorted(migrations_dir.glob("*.sql"), key=lambda path: path.name)
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename text PRIMARY KEY,
                    sha256 text NOT NULL,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            for path in migration_paths:
                digest = _sha256(path)
                row = connection.execute(
                    "SELECT sha256 FROM schema_migrations WHERE filename = %s",
                    (path.name,),
                ).fetchone()
                if row is not None:
                    if row[0] != digest:
                        raise MigrationDriftError(f"applied migration was modified: {path.name}")
                    continue
                sql = path.read_text(encoding="utf-8")
                with connection.transaction():
                    connection.execute(sql)
                    connection.execute(
                        """
                        INSERT INTO schema_migrations (filename, sha256)
                        VALUES (%s, %s)
                        """,
                        (path.name, digest),
                    )
                applied.append(path.name)
        finally:
            connection.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))
    return applied
