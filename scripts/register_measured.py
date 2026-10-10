"""把已测实验写入 registry.json，并可选 upsert 到 PostgreSQL。

没有数据库时 API 读 JSON。设置了 SEARCHPILOT_DATABASE_URL 时 API 只读数据库，
不会自动装载这份 JSON，需要带 --database-url 再跑一次。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from searchpilot.agent.demo import demo_store
from searchpilot.experiments.store import InMemoryExperimentStore


def upsert_postgres(database_url: str, store: InMemoryExperimentStore) -> int:
    """把内存登记写入 experiments / metrics。同一 id 会整份替换。"""
    from searchpilot.db.connection import create_pool
    from searchpilot.db.experiment_store import PostgresExperimentStore

    records = store.list_records(limit=1000)
    pool = create_pool(database_url)
    try:
        postgres = PostgresExperimentStore(pool)
        for record in records:
            postgres.put(record)
    finally:
        pool.close()
    return len(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the measured experiment registry")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/experiments/registry.json"),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Also upsert into this PostgreSQL URL. Omitted means JSON only.",
    )
    args = parser.parse_args(argv)
    store = demo_store()
    store.save(args.out)
    print(f"wrote {args.out}")
    if args.database_url:
        count = upsert_postgres(args.database_url, store)
        print(f"upserted_experiments={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
