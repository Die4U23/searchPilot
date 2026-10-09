from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from searchpilot.db.migrate import MigrationDriftError, apply_migrations

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    base_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if not base_url:
        pytest.skip("SEARCHPILOT_DATABASE_URL is not set; PostgreSQL integration test skipped")
    schema = f"test_migrate_{uuid4().hex}"
    with psycopg.connect(base_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    options = conninfo_to_dict(base_url)
    options["options"] = f"-c search_path={schema}"
    scoped_url = make_conninfo(**options)
    try:
        yield scoped_url
    finally:
        with psycopg.connect(base_url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_migration_replay_and_tamper_rejection(database_url: str, tmp_path: Path) -> None:
    source = Path("db/migrations/0001_core.sql")
    migration = tmp_path / source.name
    migration.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    assert apply_migrations(database_url, tmp_path) == ["0001_core.sql"]
    assert apply_migrations(database_url, tmp_path) == []

    migration.write_text(
        migration.read_text(encoding="utf-8") + "\n-- tampered\n", encoding="utf-8"
    )
    with pytest.raises(MigrationDriftError, match="modified"):
        apply_migrations(database_url, tmp_path)
