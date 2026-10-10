"""容器入口：先迁移，再启动 API。没有数据库连接串时直接启动。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    database_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if database_url:
        from searchpilot.db.migrate import apply_migrations

        migrations = Path(os.environ.get("SEARCHPILOT_MIGRATIONS_DIR", "db/migrations"))
        applied = apply_migrations(database_url, migrations)
        print(f"applied_migrations={applied}")
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "searchpilot.bootstrap:build_default_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--no-access-log",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
