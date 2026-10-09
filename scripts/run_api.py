"""启动 SearchPilot API：python scripts/run_api.py --host 127.0.0.1 --port 8000"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from searchpilot.config import get_settings
from searchpilot.observability import configure_logging


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SearchPilot API server.")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(get_settings().log_level)
    # factory 模式：每个进程启动时调用 build_default_app()；访问日志由应用中间件输出。
    # 不启用 --reload：压测与生产都应关闭热重载。
    uvicorn.run(
        "searchpilot.bootstrap:build_default_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
