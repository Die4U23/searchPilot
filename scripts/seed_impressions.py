"""把 MIND 曝光和用户分桶写入 PostgreSQL。

重跑会清空 ``impressions`` 与 ``users`` 再导入。MIND 没有展示位置和模型版本，这两列留空。
``history_len_gt_0`` 只来自 train 历史；其余用户是 ``history_len_0``。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from searchpilot.db.connection import create_pool

COPY_SQL = """
COPY impressions_load (
    impression_id, user_id, item_id, clicked, shown_at, split, source
) FROM STDIN
"""
_COLUMNS = (
    "impression_id",
    "user_id",
    "item_id",
    "clicked",
    "shown_at",
    "split",
    "source",
)


def rows_from_batch(batch: pa.RecordBatch) -> list[tuple[Any, ...]]:
    """一行一个（曝光, 物品）。位置和模型版本在 MIND 里不存在。"""
    data = batch.to_pydict()
    count = len(data["impression_id"])
    rows: list[tuple[Any, ...]] = []
    for index in range(count):
        split = str(data["split"][index])
        if split not in {"train", "dev"}:
            raise ValueError(f"impression split must be train or dev, got {split}")
        clicked = int(data["clicked"][index])
        if clicked not in {0, 1}:
            raise ValueError(f"clicked must be 0 or 1, got {clicked}")
        source = str(data["source"][index] or "mind")
        rows.append(
            (
                str(data["impression_id"][index]),
                str(data["user_id"][index]),
                str(data["item_id"][index]),
                None,
                clicked,
                None,
                data["shown_at"][index],
                split,
                source,
            )
        )
    return rows


def warm_user_ids(history_path: Path) -> list[str]:
    """train 历史非空的用户。"""
    table = pq.read_table(history_path, columns=["user_id", "history"])
    users = table.column("user_id").to_pylist()
    histories = table.column("history").to_pylist()
    return [str(user_id) for user_id, history in zip(users, histories, strict=True) if history]


def load_impressions(database_url: str, parquet_path: Path, *, batch_size: int = 100_000) -> int:
    """清空后按批 COPY，再按 (impression_id, item_id) 合并。返回合并后的行数。"""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(parquet_path)
    pool = create_pool(database_url)
    total = 0
    try:
        with pool.connection() as connection:
            connection.execute("TRUNCATE impressions, users")
            connection.execute("DROP TABLE IF EXISTS impressions_load")
            connection.execute(
                """
                CREATE UNLOGGED TABLE impressions_load (
                    impression_id text NOT NULL,
                    user_id text NOT NULL,
                    item_id text NOT NULL,
                    clicked smallint NOT NULL,
                    shown_at timestamptz NOT NULL,
                    split text NOT NULL,
                    source text NOT NULL
                )
                """
            )
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(_COLUMNS)):
            _copy_load_rows(pool, _load_rows(rows_from_batch(batch)))
            total += batch.num_rows
            print(f"impressions_raw={total}", flush=True)
        count = aggregate_impressions(database_url)
    finally:
        pool.close()
    print(f"impressions={count}", flush=True)
    return count


def aggregate_impressions(database_url: str) -> int:
    """把暂存表合并进 impressions。重复键保留最大 clicked。"""
    pool = create_pool(database_url)
    deduped: tuple[Any, ...] | None = None
    try:
        with pool.connection() as connection:
            connection.execute("SET statement_timeout = 0")
            connection.execute("ALTER TABLE impressions DROP CONSTRAINT IF EXISTS impressions_pkey")
            connection.execute("DROP INDEX IF EXISTS impressions_user_shown_idx")
            connection.execute(
                """
                INSERT INTO impressions (
                    impression_id, user_id, item_id, position, clicked,
                    model_version, shown_at, split, source
                )
                SELECT impression_id,
                       min(user_id),
                       item_id,
                       NULL,
                       max(clicked),
                       NULL,
                       min(shown_at),
                       min(split),
                       min(source)
                FROM impressions_load
                GROUP BY impression_id, item_id
                """
            )
            connection.execute("ALTER TABLE impressions ADD PRIMARY KEY (impression_id, item_id)")
            connection.execute(
                """
                CREATE INDEX impressions_user_shown_idx
                ON impressions (user_id, shown_at)
                """
            )
            deduped = connection.execute("SELECT count(*) FROM impressions").fetchone()
            connection.execute("DROP TABLE impressions_load")
    finally:
        pool.close()
    return int(deduped[0]) if deduped is not None else 0


def _load_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """去掉正式表里留空的位置和模型版本。"""
    return [(row[0], row[1], row[2], row[4], row[6], row[7], row[8]) for row in rows]


def _copy_load_rows(pool: Any, rows: list[tuple[Any, ...]]) -> None:
    with (
        pool.connection() as connection,
        connection.cursor() as cursor,
        cursor.copy(COPY_SQL) as copy,
    ):
        for row in rows:
            copy.write_row(row)


def load_users(database_url: str, warm_ids: Sequence[str]) -> int:
    """用曝光的最早时间建用户，再把有历史的用户标成 history_len_gt_0。"""
    pool = create_pool(database_url)
    try:
        with pool.connection() as connection, connection.cursor() as cursor:
            connection.execute("SET statement_timeout = 0")
            connection.execute(
                """
                INSERT INTO users (user_id, created_at, segment, source)
                SELECT user_id, min(shown_at), 'history_len_0', 'mind'
                FROM impressions
                GROUP BY user_id
                """
            )
            if warm_ids:
                cursor.execute(
                    "CREATE TEMP TABLE warm_users (user_id text PRIMARY KEY) ON COMMIT DROP"
                )
                with cursor.copy("COPY warm_users (user_id) FROM STDIN") as copy:
                    for user_id in warm_ids:
                        copy.write_row((user_id,))
                connection.execute(
                    """
                    UPDATE users
                    SET segment = 'history_len_gt_0'
                    FROM warm_users
                    WHERE users.user_id = warm_users.user_id
                    """
                )
            count = connection.execute("SELECT count(*) FROM users").fetchone()
    finally:
        pool.close()
    return int(count[0]) if count is not None else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("SEARCHPILOT_DATABASE_URL"))
    parser.add_argument("--impressions", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Merge an existing impressions_load table and skip the parquet copy.",
    )
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("--database-url or SEARCHPILOT_DATABASE_URL is required")
    if args.aggregate_only:
        written = aggregate_impressions(args.database_url)
        print(f"impressions={written}", flush=True)
    elif args.impressions is None:
        parser.error("--impressions is required unless --aggregate-only is set")
    else:
        written = load_impressions(args.database_url, args.impressions, batch_size=args.batch_size)
    warm = warm_user_ids(args.history) if args.history else []
    users = load_users(args.database_url, warm)
    print(f"upserted impressions={written} users={users} warm={len(warm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
