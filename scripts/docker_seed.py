"""容器第一次启动时，把挂载的物品、目录和已测实验写入 PostgreSQL。

表里已经有行就跳过。曝光表不在这里导入。
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from searchpilot.recommend.service import resolve_data_version

STEPS = ("items", "catalog", "experiments")


def seed_steps(counts: Mapping[str, int]) -> tuple[str, ...]:
    """按空表决定要导入的步骤。缺计数视为不导入。"""
    return tuple(step for step in STEPS if counts.get(step, 1) == 0)


def table_has_rows(database_url: str, table: str) -> bool:
    from psycopg import connect

    statements = {
        "items": "SELECT EXISTS (SELECT 1 FROM items)",
        "queries": "SELECT EXISTS (SELECT 1 FROM queries)",
        "experiments": "SELECT EXISTS (SELECT 1 FROM experiments)",
    }
    statement = statements.get(table)
    if statement is None:
        raise ValueError(f"unexpected table {table}")
    with connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(statement)
        row = cursor.fetchone()
    return bool(row[0]) if row else False


def _run(argv: Sequence[str], database_url: str) -> None:
    env = os.environ.copy()
    env["SEARCHPILOT_DB_STATEMENT_TIMEOUT_MS"] = "120000"
    # Keep credentials out of process arguments and CalledProcessError's command text.
    env["SEARCHPILOT_DATABASE_URL"] = database_url
    subprocess.run([sys.executable, *argv], check=True, env=env)


def seed_mounted_data(database_url: str, data_dir: Path, artifact_dir: Path) -> None:
    """导入空表。物品 parquet 不在挂载里时跳过，让 API 仍能启动。"""
    try:
        version = resolve_data_version(data_dir / "processed")
    except Exception as exc:
        print(f"skip_seed={exc}")
        return
    items_path = data_dir / "processed" / version / "items.parquet"
    if not items_path.is_file():
        print(f"skip_seed=missing {items_path}")
        return
    counts = {
        "items": int(table_has_rows(database_url, "items")),
        "catalog": int(table_has_rows(database_url, "queries")),
        "experiments": int(table_has_rows(database_url, "experiments")),
    }
    pending = seed_steps(counts)
    print(f"seed_steps={','.join(pending) if pending else 'none'}")
    if "items" in pending:
        _run(["scripts/seed_items.py", str(items_path)], database_url)
    if "catalog" in pending:
        _run(
            [
                "scripts/seed_catalog.py",
                "--data-dir",
                str(data_dir),
                "--artifact-dir",
                str(artifact_dir),
            ],
            database_url,
        )
    if "experiments" in pending:
        from searchpilot.agent.demo import demo_store

        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from register_measured import upsert_postgres

        count = upsert_postgres(database_url, demo_store())
        print(f"upserted_experiments={count}")
