"""应用 SearchPilot PostgreSQL 迁移。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from searchpilot.db.migrate import apply_migrations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("SEARCHPILOT_DATABASE_URL"))
    parser.add_argument("--migrations-dir", type=Path, default=Path("db/migrations"))
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or SEARCHPILOT_DATABASE_URL must provide a PostgreSQL URL")
    return args


def main() -> int:
    args = parse_args()
    applied = apply_migrations(args.database_url, args.migrations_dir)
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")
    else:
        print("No pending migrations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
