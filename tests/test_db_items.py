from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import ConnectionPool

from searchpilot.db.connection import create_pool
from searchpilot.db.items import PostgresItemStore
from searchpilot.db.migrate import apply_migrations

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    base_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if not base_url:
        pytest.skip("SEARCHPILOT_DATABASE_URL is not set; PostgreSQL integration test skipped")
    schema = f"test_items_{uuid4().hex}"
    with psycopg.connect(base_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    options = conninfo_to_dict(base_url)
    options["options"] = f"-c search_path={schema}"
    try:
        yield make_conninfo(**options)
    finally:
        with psycopg.connect(base_url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture(scope="module")
def pool(database_url: str) -> Iterator[ConnectionPool]:
    apply_migrations(database_url, Path("db/migrations"))
    connection_pool = create_pool(database_url)
    try:
        yield connection_pool
    finally:
        connection_pool.close()


def test_item_store_get_and_exists(pool: ConnectionPool) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO items (
                item_id, title, abstract, category, subcategory, url
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            ("N12345", "Title", "Abstract", "news", "world", "https://example.test"),
        )
    store = PostgresItemStore(pool)

    document = store.get("N12345")

    assert document is not None
    assert document.item_id == "N12345"
    assert document.title == "Title"
    assert store.exists("N12345") is True
    assert store.get("N99999") is None
    assert store.exists("N99999") is False
