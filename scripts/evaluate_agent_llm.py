"""Explicit opt-in external-model benchmark. Default execution is offline UNRUN."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from searchpilot.agent.llm_eval import evaluate, validate_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Allow up to 100 external requests")
    parser.add_argument("--endpoint-env", default="SEARCHPILOT_AGENT_LLM_ENDPOINT")
    parser.add_argument("--model-env", default="SEARCHPILOT_AGENT_LLM_MODEL")
    parser.add_argument("--api-key-env", default="SEARCHPILOT_AGENT_LLM_API_KEY")
    parser.add_argument("--ledger", type=Path, help="New JSONL path; existing files are refused")
    parser.add_argument("--out", type=Path, help="Optional NEW summary JSON; never overwritten")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--task-id", action="append", help="Select a fixed-suite ID; repeatable")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-output-tokens", type=int, default=204800)
    parser.add_argument("--max-tokens-per-request", type=int, default=2048)
    args = parser.parse_args(argv)
    try:
        selected_ids = validate_run(
            task_ids=args.task_id,
            max_requests=args.max_requests,
            max_output_tokens=args.max_output_tokens,
            max_tokens_per_request=args.max_tokens_per_request,
            timeout=args.timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.out is not None and args.out.exists():
        parser.error("--out already exists; no requests sent")
    if not args.execute:
        result = {
            "status": "UNRUN",
            "tasks": 50,
            "suite_tasks": 50,
            "selected_tasks": len(selected_ids),
            "selected_task_ids": selected_ids,
            "attempted_tasks": 0,
            "unrun_tasks": 50,
            "selected_unrun_tasks": len(selected_ids),
            "requests_sent": 0,
            "max_requests": args.max_requests,
            "max_output_tokens": args.max_output_tokens,
            "max_tokens_per_request": args.max_tokens_per_request,
            "reason": "explicit --execute and endpoint/model configuration required",
        }
    else:
        if args.ledger is None or args.ledger.exists():
            parser.error("--execute requires a NEW --ledger path; never resume an old run")
        if args.out is not None and args.out.resolve() == args.ledger.resolve():
            parser.error("--out and --ledger must be different paths")
        # Read only the named configuration; do not load .env or discover credentials.
        endpoint = os.environ.get(args.endpoint_env, "")
        model = os.environ.get(args.model_env, "")
        parsed = urlsplit(endpoint)
        if (
            not model.strip()
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (
                parsed.scheme != "https"
                and not (
                    parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                )
            )
        ):
            parser.error("configure a model and HTTPS endpoint (HTTP allowed only for loopback)")
        key = os.environ.get(args.api_key_env)

        async def run() -> dict:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            async with httpx.AsyncClient(
                headers=headers,
                timeout=args.timeout,
                follow_redirects=False,
                trust_env=False,
                transport=httpx.AsyncHTTPTransport(retries=0),
            ) as client:
                return await evaluate(
                    client,
                    endpoint=endpoint,
                    model=model,
                    ledger_path=args.ledger,
                    timeout=args.timeout,
                    task_ids=selected_ids,
                    max_requests=args.max_requests,
                    max_output_tokens=args.max_output_tokens,
                    max_tokens_per_request=args.max_tokens_per_request,
                )

        try:
            result = asyncio.run(run())
        except (ValueError, OSError):
            parser.error("invalid timeout or output path; inspect the ledger before any future run")
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
