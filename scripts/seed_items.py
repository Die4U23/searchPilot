"""从 items.parquet 分批 upsert PostgreSQL items。"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from searchpilot.db.connection import create_pool

UPSERT_SQL = """
    INSERT INTO items (
        item_id, title, abstract, category, subcategory, url, first_seen_at, source
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, 'mind')
    ON CONFLICT (item_id) DO UPDATE SET
        title = EXCLUDED.title,
        abstract = EXCLUDED.abstract,
        category = EXCLUDED.category,
        subcategory = EXCLUDED.subcategory,
        url = EXCLUDED.url,
        first_seen_at = EXCLUDED.first_seen_at,
        source = EXCLUDED.source
"""


def _batches(values: list[tuple[Any, ...]], size: int) -> Iterator[list[tuple[Any, ...]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items_parquet", type=Path)
    parser.add_argument("--database-url", default=os.environ.get("SEARCHPILOT_DATABASE_URL"))
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or SEARCHPILOT_DATABASE_URL must provide a PostgreSQL URL")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    rows = [
        (
            record["item_id"],
            record["title"],
            record["abstract"],
            record["category"],
            record["subcategory"],
            record["url"],
            record["first_seen_at"],
        )
        for record in pq.read_table(args.items_parquet).to_pylist()
    ]
    pool = create_pool(args.database_url)
    try:
        for batch in _batches(rows, args.batch_size):
            with pool.connection() as connection, connection.cursor() as cursor:
                cursor.executemany(UPSERT_SQL, batch)
    finally:
        pool.close()
    print(f"Upserted {len(rows)} items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
