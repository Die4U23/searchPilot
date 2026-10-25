"""容器入口：先迁移、补齐空库，再启动 API。没有数据库连接串时直接启动。"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

PR_SET_THP_DISABLE = 41
PR_GET_THP_DISABLE = 42


def configure_thp() -> dict[str, bool | str]:
    """Apply this process's Linux THP policy before imports and preserve it across exec.

    false clears an inherited disable flag; it does not override the host's THP
    policy or promise huge-page allocation. This does not trim existing memory.
    """
    raw = os.environ.get("SEARCHPILOT_DISABLE_THP", "true").strip().lower()
    if raw not in ("true", "false"):
        raise RuntimeError("SEARCHPILOT_DISABLE_THP must be true or false")
    requested = raw == "true"
    if sys.platform != "linux":
        return {"requested": requested, "applied": False, "policy": "unsupported_platform"}
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
    except (OSError, AttributeError):
        raise RuntimeError("Linux THP control is unavailable") from None
    # Linux UAPI linux/prctl.h: SET=41, GET=42. Explicit widths matter for varargs.
    if prctl(PR_SET_THP_DISABLE, int(requested), 0, 0, 0) != 0:
        raise RuntimeError(f"PR_SET_THP_DISABLE failed (errno={ctypes.get_errno()})")
    observed = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0)
    if observed == -1:
        raise RuntimeError(f"PR_GET_THP_DISABLE failed (errno={ctypes.get_errno()})")
    if observed != int(requested):
        raise RuntimeError("Linux THP policy verification failed")
    return {
        "requested": requested,
        "applied": True,
        "policy": "disabled" if requested else "system_default",
    }


def main() -> int:
    print(json.dumps(configure_thp()), flush=True)
    database_url = os.environ.get("SEARCHPILOT_DATABASE_URL")
    if database_url:
        from docker_seed import seed_mounted_data
        from searchpilot.db.migrate import apply_migrations

        migrations = Path(os.environ.get("SEARCHPILOT_MIGRATIONS_DIR", "db/migrations"))
        applied = apply_migrations(database_url, migrations)
        print(f"applied_migrations={applied}")
        data_dir = Path(os.environ.get("SEARCHPILOT_DATA_DIR", "data"))
        artifact_dir = Path(os.environ.get("SEARCHPILOT_ARTIFACT_DIR", "artifacts"))
        seed_mounted_data(database_url, data_dir, artifact_dir)
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
