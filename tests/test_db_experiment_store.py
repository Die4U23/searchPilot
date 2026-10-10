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
from searchpilot.db.experiment_store import PostgresExperimentStore
from searchpilot.db.migrate import apply_migrations
from searchpilot.experiments.store import ExperimentRecord, MetricPoint

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    base_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if not base_url:
        pytest.skip("SEARCHPILOT_DATABASE_URL is not set; PostgreSQL integration test skipped")
    schema = f"test_experiments_{uuid4().hex}"
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


def _record(experiment_id: str, value: float, source: str = "searchpilot") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        kind="search",
        config={"seed": 20261009},
        data_version="d3a904f41240",
        protocol_version="v1",
        source=source,
        code_commit="abc123",
        source_ref={"repo": "searchpilot", "sha256": "deadbeef"},
        metrics=(
            MetricPoint(name="ndcg@10", split="test", value=value, segment="all", source=source),
        ),
        failures=({"query_id": "q001", "reason": "zero_result"},),
    )


def test_put_get_replaces_metrics(pool: ConnectionPool) -> None:
    store = PostgresExperimentStore(pool)
    experiment_id = f"exp-{uuid4().hex}"
    store.put(_record(experiment_id, 0.2))
    store.put(_record(experiment_id, 0.4))

    loaded = store.get(experiment_id)

    assert loaded is not None
    assert loaded.code_commit == "abc123"
    assert loaded.source_ref == {"repo": "searchpilot", "sha256": "deadbeef"}
    assert loaded.failures == ({"query_id": "q001", "reason": "zero_result"},)
    assert [metric.value for metric in loaded.metrics] == [0.4]
    assert store.get(f"missing-{uuid4().hex}") is None


def test_list_records_filters_source(pool: ConnectionPool) -> None:
    store = PostgresExperimentStore(pool)
    left = f"exp-a-{uuid4().hex}"
    right = f"exp-b-{uuid4().hex}"
    store.put(_record(left, 0.1, source="searchpilot"))
    store.put(_record(right, 0.2, source="synthetic"))

    found = store.list_records(source="synthetic", limit=20)

    assert [item.experiment_id for item in found] == [right]
