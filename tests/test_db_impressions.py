from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from searchpilot.data.schemas import IMPRESSIONS_SCHEMA, USER_HISTORY_SCHEMA
from searchpilot.db.migrate import apply_migrations

pytestmark = pytest.mark.integration


def _load():
    path = Path("scripts/seed_impressions.py")
    spec = importlib.util.spec_from_file_location("seed_impressions_db", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    base_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if not base_url:
        pytest.skip("SEARCHPILOT_DATABASE_URL is not set; PostgreSQL integration test skipped")
    schema = f"test_impr_{uuid4().hex}"
    with psycopg.connect(base_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    options = conninfo_to_dict(base_url)
    options["options"] = f"-c search_path={schema}"
    try:
        yield make_conninfo(**options)
    finally:
        with psycopg.connect(base_url, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_load_impressions_and_user_segments(database_url: str, tmp_path: Path) -> None:
    module = _load()
    apply_migrations(database_url, Path("db/migrations"))
    shown = datetime(2019, 11, 9, tzinfo=UTC)
    impressions = tmp_path / "impressions.parquet"
    pq.write_table(
        pa.table(
            {
                "impression_id": ["1", "2"],
                "user_id": ["U1", "U2"],
                "shown_at": [shown, shown],
                "item_id": ["N1", "N2"],
                "clicked": pa.array([1, 0], type=pa.int8()),
                "split": ["train", "dev"],
                "source": ["mind", "mind"],
            },
            schema=IMPRESSIONS_SCHEMA,
        ),
        impressions,
    )
    history = tmp_path / "user_history.parquet"
    pq.write_table(
        pa.table(
            {"user_id": ["U1", "U2"], "history": [["N9"], []]},
            schema=USER_HISTORY_SCHEMA,
        ),
        history,
    )

    assert module.load_impressions(database_url, impressions, batch_size=10) == 2
    assert module.load_users(database_url, module.warm_user_ids(history)) == 2

    with psycopg.connect(database_url) as connection:
        segments = dict(
            connection.execute("SELECT user_id, segment FROM users ORDER BY user_id").fetchall()
        )
        position = connection.execute(
            "SELECT position, model_version FROM impressions WHERE impression_id = '1'"
        ).fetchone()
    assert segments == {"U1": "history_len_gt_0", "U2": "history_len_0"}
    assert position == (None, None)
