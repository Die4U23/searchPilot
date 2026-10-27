"""Opt-in bounded multistep Agent evaluation; default runs make zero requests."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

from searchpilot.agent.multistep_eval import evaluate, select_cases, unrun_summary
from searchpilot.agent.provider import ProviderError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path, help="NEW directory; never overwritten or resumed")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--endpoint-env", default="SEARCHPILOT_AGENT_LLM_ENDPOINT")
    parser.add_argument("--model-env", default="SEARCHPILOT_AGENT_LLM_MODEL")
    parser.add_argument("--api-key-env", default="SEARCHPILOT_AGENT_LLM_API_KEY")
    args = parser.parse_args(argv)
    try:
        selected = select_cases(args.task_id)
        if not 1 <= args.max_requests <= 64:
            raise ValueError("max_requests must be in [1,64]")
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 300:
            raise ValueError("timeout must be finite in (0,300]")
        if args.out is not None and args.out.exists():
            raise ValueError("--out must be a NEW directory")
        if args.execute:
            if args.out is None:
                raise ValueError("--execute requires --out NEW_DIRECTORY")
            # Intentionally no dotenv loading or discovery of other configuration.
            result = asyncio.run(
                evaluate(
                    endpoint=os.environ.get(args.endpoint_env, ""),
                    model=os.environ.get(args.model_env, ""),
                    api_key=os.environ.get(args.api_key_env),
                    out_dir=args.out,
                    task_ids=selected,
                    max_requests=args.max_requests,
                    request_timeout_seconds=args.timeout,
                )
            )
        else:
            result = unrun_summary(selected, args.max_requests)
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=False)
                (args.out / "summary.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
    except (ValueError, ProviderError, OSError):
        parser.error(
            "invalid configuration or output path; no automatic retry; inspect any existing ledger"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
