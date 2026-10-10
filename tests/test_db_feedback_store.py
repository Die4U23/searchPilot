from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import ConnectionPool

from searchpilot.db.connection import create_pool
from searchpilot.db.feedback_store import PostgresFeedbackStore
from searchpilot.db.migrate import apply_migrations
from searchpilot.ports import FeedbackEvent, FeedbackWriteResult, IdempotencyConflictError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    base_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if not base_url:
        pytest.skip("SEARCHPILOT_DATABASE_URL is not set; PostgreSQL integration test skipped")
    schema = f"test_feedback_{uuid4().hex}"
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


def test_feedback_replay_and_conflict(pool: ConnectionPool) -> None:
    store = PostgresFeedbackStore(pool)
    event = FeedbackEvent(
        idempotency_key=str(uuid4()),
        request_id="req_feedback",
        item_id="N12345",
        kind="click",
        event_at=datetime(2026, 1, 2, tzinfo=UTC),
        user_id="U1",
        position=1,
    )

    first = store.write(event)
    replay = store.write(event)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.event_id == first.event_id
    assert replay.received_at == first.received_at
    with pytest.raises(IdempotencyConflictError):
        store.write(replace(event, request_id="req_different"))


def test_replay_keeps_first_model_version(pool: ConnectionPool) -> None:
    store = PostgresFeedbackStore(pool)
    event = FeedbackEvent(
        idempotency_key=str(uuid4()),
        request_id="req_feedback",
        item_id="N12345",
        kind="click",
        event_at=datetime(2026, 1, 2, tzinfo=UTC),
        model_version="bm25-aaaa1111",
    )

    first = store.write(event)
    replay = store.write(replace(event, model_version="bm25-bbbb2222"))

    assert replay.replayed is True
    assert replay.event_id == first.event_id
    with pool.connection() as connection:
        stored = connection.execute(
            "SELECT model_version FROM feedback_events WHERE event_id = %s",
            (first.event_id,),
        ).fetchone()
    assert stored is not None
    assert stored[0] == "bm25-aaaa1111"


def test_concurrent_same_key_writes_one_row(pool: ConnectionPool) -> None:
    store = PostgresFeedbackStore(pool)
    event = FeedbackEvent(
        idempotency_key=str(uuid4()),
        request_id="req_feedback",
        item_id="N12345",
        kind="click",
        event_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    barrier = threading.Barrier(8)
    results: list[FeedbackWriteResult] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        result = store.write(event)
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert sum(1 for result in results if not result.replayed) == 1
    assert len({result.event_id for result in results}) == 1
    with pool.connection() as connection:
        count = connection.execute(
            "SELECT count(*) FROM feedback_events WHERE idempotency_key = %s",
            (event.idempotency_key,),
        ).fetchone()
    assert count is not None
    assert count[0] == 1


def test_statement_timeout_rolls_back_insert(database_url: str) -> None:
    options = conninfo_to_dict(database_url)
    existing = str(options.get("options") or "")
    options["options"] = f"{existing} -c statement_timeout=100".strip()
    pool = create_pool(make_conninfo(**options))
    item_id = f"timeout-{uuid4().hex}"
    try:
        with pytest.raises(psycopg.errors.QueryCanceled), pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO items (item_id, title, abstract, category, subcategory, url)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (item_id, "Title", "Abstract", "news", "world", "https://example.test"),
            )
            connection.execute("SELECT pg_sleep(1)")
        with pool.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM items WHERE item_id = %s",
                (item_id,),
            ).fetchone()
    finally:
        pool.close()
    assert row is None
