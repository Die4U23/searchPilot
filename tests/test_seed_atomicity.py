"""Whole seed steps either commit all batches or leave an empty step retryable."""

from __future__ import annotations

import argparse
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts import seed_catalog, seed_items

from searchpilot.db.migrate import apply_migrations
from test_db_catalog import database_url as database_url

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def empty_seed_tables(database_url: str) -> None:
    apply_migrations(database_url, Path("db/migrations"))
    with psycopg.connect(database_url) as connection:
        connection.execute("TRUNCATE items, queries, model_versions, agent_tasks, bids CASCADE")


@pytest.mark.parametrize("fail_second_batch", [True, False])
def test_items_step_is_atomic(
    database_url: str, tmp_path: Path, monkeypatch, fail_second_batch: bool
) -> None:
    module = seed_items
    records = [
        {
            "item_id": f"N{index}",
            "title": None if fail_second_batch and index == 2 else "title",
            "abstract": "abstract",
            "category": "news",
            "subcategory": "tech",
            "url": "https://example.test/item",
            "first_seen_at": None,
        }
        for index in (1, 2)
    ]
    path = tmp_path / "items.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(items_parquet=path, database_url=database_url, batch_size=1),
    )
    if fail_second_batch:
        with pytest.raises(psycopg.errors.NotNullViolation):
            module.main()
    else:
        assert module.main() == 0
    # A separate connection observes committed state, not the seeder's transaction.
    with psycopg.connect(database_url) as connection:
        rows = connection.execute("SELECT item_id FROM items ORDER BY item_id").fetchall()
    assert rows == ([] if fail_second_batch else [("N1",), ("N2",)])


@pytest.mark.parametrize("fail_second_batch", [True, False])
def test_catalog_step_is_atomic_across_tables(database_url: str, fail_second_batch: bool) -> None:
    module = seed_catalog
    kwargs = {
        "queries": [("q1", "Hello", "hello", "exact_entity", "train")],
        "labels": [
            ("q1", "N1", 3, "fixture"),
            ("q1", "N2", 4 if fail_second_batch else 2, "fixture"),
        ],
        "models": [],
        "tasks": [{"task_id": "task1", "question": "what is ndcg@10", "kind": "searchpilot"}],
        "bids": [("N1", 0.5, "synthetic-test"), ("N2", 1.0, "synthetic-test")],
        "batch_size": 1,
    }
    if fail_second_batch:
        with pytest.raises(psycopg.errors.CheckViolation):
            module.upsert_catalog(database_url, **kwargs)
    else:
        assert module.upsert_catalog(database_url, **kwargs) == {
            "queries": 1,
            "labels": 2,
            "models": 0,
            "tasks": 1,
            "bids": 2,
        }
    with psycopg.connect(database_url) as connection:
        counts = connection.execute(
            "SELECT (SELECT count(*) FROM queries), (SELECT count(*) FROM relevance_labels), "
            "(SELECT count(*) FROM agent_tasks), (SELECT count(*) FROM bids)"
        ).fetchone()
    assert counts == ((0, 0, 0, 0) if fail_second_batch else (1, 2, 1, 2))
