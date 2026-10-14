"""Bounded, real-HTTP search benchmark; every issued request stays in the denominator."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from searchpilot.contracts import SearchResponse

MODES = ("bm25", "vector", "hybrid", "ltr")
MAX_BODY_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Config:
    base_url: str
    requests: int = 120
    concurrency: int = 4
    timeout: float = 10.0
    warmup: int = 4
    limit: int = 10

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("base URL must be HTTP(S), without credentials, query or fragment")
        if not 1 <= self.requests <= 100_000 or not 1 <= self.concurrency <= 256:
            raise ValueError("requests must be 1..100000 and concurrency 1..256")
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 300:
            raise ValueError("timeout must be finite and in (0, 300] seconds")
        if not 0 <= self.warmup <= 1000 or not 1 <= self.limit <= 100:
            raise ValueError("warmup must be 0..1000 and limit 1..100")


def latency_summary(samples: list[float]) -> dict[str, float | int | None]:
    """Nearest-rank percentiles: sorted[ceil(p * n) - 1], with explicit empty samples."""
    ordered = sorted(samples)
    result: dict[str, float | int | None] = {"count": len(ordered)}
    for name, q in (("p50_ms", 0.50), ("p95_ms", 0.95), ("p99_ms", 0.99)):
        result[name] = ordered[math.ceil(q * len(ordered)) - 1] if ordered else None
    result["max_ms"] = ordered[-1] if ordered else None
    return result


async def request_once(
    client: httpx.AsyncClient,
    config: Config,
    *,
    index: int,
    mode: str,
    query: str,
    phase: str = "measured",
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    request_id = "req_" + uuid.uuid4().hex
    record: dict[str, Any] = {
        "index": index,
        "request_id": request_id,
        "query": query,
        "mode": mode,
        "status_code": None,
        "outcome": "client_error",
        "response_bytes": 0,
        "model_version": None,
        "result_count": None,
    }
    if event_sink is not None:
        event_sink({"event": "pending", "phase": phase, **record, "outcome": "pending"})
    started = time.perf_counter()
    deadline = asyncio.timeout(config.timeout)
    cancelled = False
    try:
        # HTTPX timeouts are per operation; this outer timeout bounds the entire body too.
        async with (
            deadline,
            client.stream(
                "POST",
                config.base_url.rstrip("/") + "/search",
                json={"query": query, "limit": config.limit, "mode": mode},
                headers={"X-Request-Id": request_id},
            ) as response,
        ):
            record["status_code"] = response.status_code
            body = bytearray()
            async for chunk in response.aiter_bytes():
                record["response_bytes"] += len(chunk)
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    record["outcome"] = "response_too_large"
                    return record
                body.extend(chunk)
            if response.status_code != 200:
                record["outcome"] = "http_error"
                return record
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                record["outcome"] = "malformed_response"
                return record
            try:
                parsed = SearchResponse.model_validate(payload, strict=True)
            except ValidationError:
                record["outcome"] = "invalid_response"
                return record
            ids = [hit.item_id for hit in parsed.results]
            if (
                parsed.request_id != request_id
                or response.headers.get("X-Request-Id") != request_id
                or parsed.mode != mode
                or not parsed.model_version
                or len(ids) > config.limit
                or len(set(ids)) != len(ids)
                or any(
                    hit.rank != rank or not math.isfinite(hit.score)
                    for rank, hit in enumerate(parsed.results, 1)
                )
            ):
                record["outcome"] = "invalid_response"
                return record
            record.update(
                outcome="success",
                model_version=parsed.model_version,
                result_count=len(parsed.results),
                result_sha256=hashlib.sha256(
                    json.dumps(
                        [hit.model_dump(mode="json") for hit in parsed.results],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                ranking_sha256=hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
            )
    except (TimeoutError, httpx.TimeoutException):
        record["outcome"] = "timeout"
    except httpx.RemoteProtocolError:
        record["outcome"] = (
            "incomplete_response" if record["status_code"] is not None else "transport_error"
        )
    except httpx.HTTPError as exc:
        record.update(outcome="transport_error", error_type=type(exc).__name__)
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        # Preserve the request denominator even when the benchmark client fails.
        record.update(outcome="client_error", error_type=type(exc).__name__)
    finally:
        # Some transports surface a close/protocol error during timeout cancellation.
        if deadline.expired():
            record["outcome"] = "timeout"
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        if event_sink is not None and not cancelled:
            event_sink({"event": "result", "phase": phase, **record})
    return record


def summarize(records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    successes = [row for row in records if row["outcome"] == "success"]
    counts = Counter(row["outcome"] for row in records)
    return {
        "total": len(records),
        "successes": len(successes),
        "failures": len(records) - len(successes),
        "success_rate": len(successes) / len(records),
        "timeout_rate": counts["timeout"] / len(records),
        "outcomes": dict(sorted(counts.items())),
        "status_codes": dict(sorted(Counter(str(row["status_code"]) for row in records).items())),
        "elapsed_seconds": elapsed,
        "completed_requests_per_second": len(records) / elapsed,
        "successful_requests_per_second": len(successes) / elapsed,
        "success_latency": latency_summary([row["elapsed_ms"] for row in successes]),
        "all_latency": latency_summary([row["elapsed_ms"] for row in records]),
        "model_versions": sorted({row["model_version"] for row in successes}),
        "empty_successes": sum(row["result_count"] == 0 for row in successes),
    }


async def run_mode(
    config: Config,
    mode: str,
    queries: list[str],
    *,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if mode not in MODES or not queries or any(not query.strip() for query in queries):
        raise ValueError("a supported mode and nonempty queries are required")
    limits = httpx.Limits(
        max_connections=config.concurrency, max_keepalive_connections=config.concurrency
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(config.timeout),
        limits=limits,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        warmup = [
            await request_once(
                client,
                config,
                index=i,
                mode=mode,
                query=queries[i % len(queries)],
                phase="warmup",
                event_sink=event_sink,
            )
            for i in range(config.warmup)
        ]
        records: list[dict[str, Any]] = []
        indices = iter(range(config.requests))
        started = time.perf_counter()

        async def worker() -> None:
            for index in indices:
                dispatched_ms = (time.perf_counter() - started) * 1000
                row = await request_once(
                    client,
                    config,
                    index=index,
                    mode=mode,
                    query=queries[index % len(queries)],
                    event_sink=event_sink,
                )
                row["dispatched_ms"] = dispatched_ms
                records.append(row)

        await asyncio.gather(*(worker() for _ in range(min(config.concurrency, config.requests))))
        elapsed = time.perf_counter() - started
    records.sort(key=lambda row: row["index"])
    return {
        "mode": mode,
        "warmup": warmup,
        "summary": summarize(records, elapsed),
        "requests": records,
    }


def code_metadata(root: Path) -> dict[str, Any]:
    git = "C:/Program Files/Git/cmd/git.exe" if os.name == "nt" else "git"
    try:

        def read(*args: str) -> bytes:
            return subprocess.run(
                [git, *args], cwd=root, check=True, capture_output=True, timeout=10
            ).stdout

        return {
            "commit": read("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(read("status", "--porcelain")),
            "tracked_diff_sha256": hashlib.sha256(read("diff", "HEAD")).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {"status": "UNAVAILABLE"}


def package_versions() -> dict[str, str]:
    found = {}
    for name in ("searchpilot", "httpx", "pydantic", "uvicorn", "numpy", "torch"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "NOT_INSTALLED"
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["bm25"])
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--data-version", default="UNKNOWN")
    parser.add_argument("--server-description", default="UNKNOWN (not discovered by client)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = Config(
            args.base_url, args.requests, args.concurrency, args.timeout, args.warmup, args.limit
        )
        with args.queries.open(encoding="utf-8-sig", newline="") as file:
            queries = [row["query_text"] for row in csv.DictReader(file)]
        if not queries or any(not query.strip() or len(query) > 256 for query in queries):
            raise ValueError("queries must contain query_text values of 1..256 characters")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    ledger_path = args.out.with_suffix(".requests.jsonl")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "source": "searchpilot",
        "started_at": datetime.now(UTC).isoformat(),
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "config": asdict(config),
        "data_version_declared": args.data_version,
        "server_description_declared": args.server_description,
        "client_machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "packages": package_versions(),
        },
        "client_checkout_start": code_metadata(root),
        "queries": {
            "path": str(args.queries),
            "sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            "count": len(queries),
            "order": "CSV order, cyclic, all splits; relevance is not evaluated",
            "min_chars": min(map(len, queries)),
            "max_chars": max(map(len, queries)),
        },
        "method": {
            "workload": "closed-loop fixed worker count, modes sequential, no retry",
            "latency": "dispatch to complete body/validation or failure; excludes unsent queue",
            "percentile": "nearest rank: ceil(q*n)-1",
            "timeout": "total per request including response body; no server cancellation proof",
            "warmup": "sequential, separate raw ledger; first request is not proven cold start",
            "resource_sampling": "UNRUN: CPU utilization, peak RSS, GPU utilization",
            "server_version": "declared only; local client checkout is not server provenance",
        },
        "runs": [],
        "requested_modes": list(dict.fromkeys(args.modes)),
        "request_ledger": str(ledger_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve once before any request. An accidental rerun must preserve failure evidence.
    try:
        with args.out.open("x", encoding="utf-8") as file:
            file.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        parser.error(f"output already exists; choose a new run path: {args.out}")
    try:
        ledger = ledger_path.open("x", encoding="utf-8")
    except FileExistsError:
        parser.error(f"request ledger already exists; choose a new run path: {ledger_path}")

    def record_event(event: dict[str, Any]) -> None:
        event["at"] = datetime.now(UTC).isoformat()
        ledger.write(json.dumps(event, ensure_ascii=False) + "\n")
        ledger.flush()

    def save() -> None:
        report["updated_at"] = datetime.now(UTC).isoformat()
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    try:
        for mode in report["requested_modes"]:
            report["active_mode"] = mode
            save()
            run = asyncio.run(run_mode(config, mode, queries, event_sink=record_event))
            report["runs"].append(run)
            report["client_checkout_end"] = code_metadata(root)
            save()
            print(json.dumps({"mode": mode, **run["summary"]}, ensure_ascii=False), flush=True)
    except (KeyboardInterrupt, Exception) as exc:
        report["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        report["incomplete_mode_ledger"] = (
            "JSONL pending without result: UNKNOWN; may have reached server, do not retry blindly"
        )
        report["error_type"] = type(exc).__name__
        save()
        raise
    finally:
        ledger.close()
    report["active_mode"] = None
    report["status"] = "COMPLETED"
    report["finished_at"] = datetime.now(UTC).isoformat()
    save()
    return int(
        any(
            run["summary"]["failures"] or any(row["outcome"] != "success" for row in run["warmup"])
            for run in report["runs"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
