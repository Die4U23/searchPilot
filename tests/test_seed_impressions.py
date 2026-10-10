from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from searchpilot.data.schemas import USER_HISTORY_SCHEMA


def _load():
    path = Path("scripts/seed_impressions.py")
    spec = importlib.util.spec_from_file_location("seed_impressions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rows_from_batch_leaves_position_empty() -> None:
    module = _load()
    shown = datetime(2019, 11, 9, 0, 0, 19, tzinfo=UTC)
    batch = pa.record_batch(
        {
            "impression_id": ["1"],
            "user_id": ["U1"],
            "item_id": ["N1"],
            "clicked": pa.array([1], type=pa.int8()),
            "shown_at": [shown],
            "split": ["train"],
            "source": ["mind"],
        }
    )
    row = module.rows_from_batch(batch)[0]
    assert row[0:3] == ("1", "U1", "N1")
    assert row[3] is None
    assert row[4] == 1
    assert row[5] is None
    assert row[7] == "train"


def test_rows_from_batch_rejects_unknown_split() -> None:
    module = _load()
    batch = pa.record_batch(
        {
            "impression_id": ["1"],
            "user_id": ["U1"],
            "item_id": ["N1"],
            "clicked": pa.array([0], type=pa.int8()),
            "shown_at": [datetime(2019, 11, 9, tzinfo=UTC)],
            "split": ["test"],
            "source": ["mind"],
        }
    )
    with pytest.raises(ValueError, match="split"):
        module.rows_from_batch(batch)


def test_warm_user_ids_skips_empty_history(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "user_history.parquet"
    table = pa.table(
        {"user_id": ["U1", "U2"], "history": [["N1"], []]},
        schema=USER_HISTORY_SCHEMA,
    )
    pq.write_table(table, path)
    assert module.warm_user_ids(path) == ["U1"]
