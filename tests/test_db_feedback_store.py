from __future__ import annotations

import os
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
from searchpilot.ports import FeedbackEvent, IdempotencyConflictError

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
