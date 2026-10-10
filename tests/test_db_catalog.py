from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.types.json import Json

from searchpilot.db.connection import create_pool
from searchpilot.db.migrate import apply_migrations

pytestmark = pytest.mark.integration


def _load():
    path = Path("scripts/seed_catalog.py")
    spec = importlib.util.spec_from_file_location("seed_catalog_db", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    base_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if not base_url:
        pytest.skip("SEARCHPILOT_DATABASE_URL is not set; PostgreSQL integration test skipped")
    schema = f"test_catalog_{uuid4().hex}"
    with psycopg.connect(base_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    options = conninfo_to_dict(base_url)
    options["options"] = f"-c search_path={schema}"
    try:
        yield make_conninfo(**options)
    finally:
        with psycopg.connect(base_url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_catalog_roundtrip(database_url: str) -> None:
    module = _load()
    apply_migrations(database_url, Path("db/migrations"))
    counts = module.upsert_catalog(
        database_url,
        queries=[("q1", "Hello", "hello", "exact_entity", "train")],
        labels=[("q1", "N1", 3, "grok-a")],
        models=[
            (
                "bm25-aaaaaaaa",
                "bm25",
                "search/bm25/meta.json",
                None,
                "ab" * 32,
                Json({"data_version": "d1"}),
                Json({"path": "search/bm25/meta.json"}),
            )
        ],
        tasks=[{"task_id": "search-00", "question": "what is ndcg@10", "kind": "searchpilot"}],
        bids=[("N1", 0.5, "synthetic-20261009")],
    )
    assert counts == {"queries": 1, "labels": 1, "models": 1, "tasks": 1, "bids": 1}

    pool = create_pool(database_url)
    try:
        with pool.connection() as connection:
            annotator = connection.execute(
                "SELECT annotator, grade FROM relevance_labels WHERE query_id = %s",
                ("q1",),
            ).fetchone()
            tasks = connection.execute("SELECT count(*) FROM agent_tasks").fetchone()
    finally:
        pool.close()
    assert annotator == ("grok-a", 3)
    assert tasks == (1,)
