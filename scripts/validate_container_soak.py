"""Run an isolated, bounded container soak without reading the private .env file."""

from __future__ import annotations

import argparse
import asyncio
import csv
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psycopg
import pyarrow.parquet as pq
from scripts.bench_concurrent import MODES, Config, read_queries
from scripts.soak_memory import sample_windows_process

from searchpilot.recommend.history import load_user_history

ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "schema_migrations",
    "items",
    "queries",
    "relevance_labels",
    "model_versions",
    "agent_tasks",
    "bids",
    "experiments",
    "feedback_events",
)
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
NATIVE_ENV_KEYS = (
    "SEARCHPILOT_DISABLE_THP",
    "MALLOC_ARENA_MAX",
    "GLIBC_TUNABLES",
    "LD_PRELOAD",
    "PYTHONMALLOC",
    "PYTHONTRACEMALLOC",
)


def now():
    return datetime.now(UTC).isoformat()


def clean_environment():
    prefixes = ("SEARCHPILOT_", "OPENAI_", "ARCBENCH_", "POSTGRES", "COMPOSE_", "MALLOC_")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(prefixes) and k.upper() not in NATIVE_ENV_KEYS
    }
    env["POSTGRES_PASSWORD"] = "searchpilot"
    return env


def stop_owned_process(process, grace=5):
    """Reap the owned launcher and its explicitly registered Windows Python worker."""
    worker = getattr(process, "worker_handle", None)
    worker_error = None
    if worker is not None:
        kernel, handle = worker
        try:
            if kernel.WaitForSingleObject(handle, 0) == 258:
                if not kernel.TerminateProcess(handle, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
                if kernel.WaitForSingleObject(handle, int(grace * 1000)) != 0:
                    raise TimeoutError("owned Python worker did not exit")
        except Exception as exc:
            worker_error = exc
        finally:
            kernel.CloseHandle(handle)
            process.worker_handle = None
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=grace)
    if worker_error is not None:
        raise worker_error


def bind_windows_worker(process, pid):
    """Keep a process handle so PID reuse cannot target an unrelated process at cleanup."""
    if os.name != "nt":
        return
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x100000 | 0x0001, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    process.worker_handle = (kernel, handle)


async def wait_owned_process(process, deadline):
    try:
        async with asyncio.timeout(deadline):
            while process.poll() is None:
                await asyncio.sleep(0.2)
        return process.returncode
    finally:
        stop_owned_process(process)


async def wait_with_sampler(process, deadline, sampler):
    waiting = asyncio.create_task(wait_owned_process(process, deadline))
    try:
        done, _ = await asyncio.wait((waiting, sampler), return_when=asyncio.FIRST_COMPLETED)
        if sampler in done:
            sampler.result()
            raise RuntimeError("memory sampler stopped before benchmark completed")
        return await waiting
    finally:
        if not waiting.done():
            waiting.cancel()
            with suppress(asyncio.CancelledError):
                await waiting


async def probe_health(client, endpoint):
    started = time.perf_counter()
    try:
        async with asyncio.timeout(3):
            async with client.stream("GET", "/health/" + endpoint) as response:
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > 65536:
                        raise ValueError("health response exceeds 64 KiB")
                return {
                    "available": True,
                    "status": response.status_code,
                    "seconds": time.perf_counter() - started,
                    "body": json.loads(content),
                }
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }


class Harness:
    def __init__(self, args):
        self.args, self.out, self.env = args, args.out.resolve(), clean_environment()
        Config(
            f"http://127.0.0.1:{args.api_port}",
            concurrency=args.concurrency,
            duration_seconds=args.duration_seconds,
        )
        self.queries, self.query_raw = read_queries(args.queries)
        self.query_snapshot = self.out / "queries.csv"
        self.docker = (
            shutil.which("docker") or r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        )
        self.api, self.db = f"{args.project}-api-1", f"{args.project}-db-1"
        self.base = f"http://127.0.0.1:{args.api_port}"
        self.db_url = f"postgresql://searchpilot:searchpilot@127.0.0.1:{args.db_port}/searchpilot"
        self.compose = [
            self.docker,
            "compose",
            "--env-file",
            str(ROOT / ".env.example"),
            "-p",
            args.project,
            "-f",
            str(ROOT / "docker-compose.yml"),
            "-f",
            str(self.out / "compose.override.yml"),
        ]
        self.phase, self.process = "startup", None
        self.worker_pid = None
        self.owns_output = False
        self.started_containers = False
        self.stop = asyncio.Event()
        self.report = {
            "schema_version": 1,
            "status": "RUNNING",
            "started_at": now(),
            "config": {
                k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "queries": {
                "source_path": str(args.queries.resolve()),
                "path": str(self.query_snapshot),
                "sha256": hashlib.sha256(self.query_raw).hexdigest(),
                "count": len(self.queries),
            },
            "requests": [],
            "commands": [],
            "external_llm_calls": 0,
            "method": "Fresh isolated PostgreSQL; declared image, unchanged service entrypoint; "
            "no private .env; "
            "Docker exec observer briefly adds cgroup memory, not PID 1 RSS",
        }

    def save(self):
        (self.out / "summary.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def command(self, args, timeout=60):
        row = {"argv": args, "started_at": now()}
        self.report["commands"].append(row)
        try:
            result = subprocess.run(
                args,
                cwd=ROOT,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                creationflags=FLAGS,
            )
            row["exit_code"] = result.returncode
            return result.stdout
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["finished_at"] = now()

    def state(self):
        rows = json.loads(self.command([self.docker, "inspect", self.api, self.db]))
        states = []
        for row in rows:
            config = row.get("Config", {})
            env = dict(value.split("=", 1) for value in config.get("Env", []) if "=" in value)
            keys = {*NATIVE_ENV_KEYS, *(key for key in env if key.startswith("MALLOC_"))}
            states.append(
                {
                    **{k: row[k] for k in ("Id", "Name", "Image", "RestartCount", "State")},
                    "command": config.get("Cmd"),
                    "entrypoint": config.get("Entrypoint"),
                    "native_env": {key: env.get(key) for key in sorted(keys)},
                }
            )
        return states

    def counts(self):
        with psycopg.connect(
            self.db_url,
            connect_timeout=5,
            options="-c statement_timeout=10000 -c lock_timeout=3000",
        ) as connection:
            return {
                table: connection.execute("SELECT count(*) FROM " + table).fetchone()[0]
                for table in TABLES
            }

    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.owns_output = True
        with self.query_snapshot.open("xb") as file:
            file.write(self.query_raw)
        self.save()
        for resource in ("container", "volume", "network"):
            listing = [self.docker, resource, "ls", *(["-a"] if resource == "container" else [])]
            found = self.command(
                listing
                + [
                    "-q",
                    "--filter",
                    f"label=com.docker.compose.project={self.args.project}",
                ]
            )
            if found.strip():
                raise ValueError("Compose project already owns resources; choose a fresh project")
            names = self.command(
                listing + ["--format", "{{.Names}}" if resource == "container" else "{{.Name}}"]
            )
            reserved = {
                self.api,
                self.db,
                f"{self.args.project}_pgdata",
                f"{self.args.project}_default",
            }
            if reserved.intersection(names.splitlines()):
                raise ValueError("Reserved Compose resource name already exists")
        self.report["image"] = json.loads(
            self.command([self.docker, "image", "inspect", self.args.image])
        )[0]["Id"]
        self.report["engine"] = json.loads(
            self.command([self.docker, "info", "--format", "{{json .}}"])
        )
        paths = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")])
        self.report["local_source_sha256"] = {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        }
        sampler = (ROOT / "scripts/soak_memory.py").as_posix()
        (self.out / "compose.override.yml").write_text(
            f"services:\n  api:\n    image: {json.dumps(self.args.image)}\n"
            f"    ports: !override\n      - '127.0.0.1:{self.args.api_port}:8000'\n"
            "    environment:\n"
            f"      SEARCHPILOT_DATA_VERSION: {json.dumps(self.args.data_version)}\n"
            "    volumes:\n      - type: bind\n"
            f"        source: {json.dumps(sampler)}\n        target: /tmp/soak_memory.py\n"
            "        read_only: true\n  db:\n    ports: !override\n"
            f"      - '127.0.0.1:{self.args.db_port}:5432'\n",
            encoding="utf-8",
        )

    async def readiness(self, client):
        async with asyncio.timeout(240):
            while True:
                result = await probe_health(client, "ready")
                if result.get("status") == 200:
                    checks = result["body"]["checks"]
                    assert all(
                        checks.get(k) is True
                        for k in (
                            "search",
                            "recommend",
                            "feedback",
                            "vector_index",
                            "ltr",
                            "ctr",
                            "database",
                        )
                    )
                    return result["body"]
                await asyncio.sleep(0.5)

    async def sample_command(self, args, *, lines=False):
        try:
            text = await asyncio.to_thread(self.command, args, 8)
            return {
                "available": True,
                "data": [json.loads(line) for line in text.splitlines()]
                if lines
                else json.loads(text),
            }
        except Exception as exc:
            return {"available": False, "error_type": type(exc).__name__}

    async def samples(self, client):
        with (self.out / "samples.jsonl").open("x", encoding="utf-8") as sink:
            while not self.stop.is_set():
                started = time.perf_counter()
                row = {"at": now(), "phase": self.phase}
                server, stats, live, ready = await asyncio.gather(
                    self.sample_command(
                        [
                            self.docker,
                            "exec",
                            self.api,
                            "python",
                            "/tmp/soak_memory.py",
                            "--pid",
                            "1",
                        ]
                    ),
                    self.sample_command(
                        [
                            self.docker,
                            "stats",
                            "--no-stream",
                            "--format",
                            "{{json .}}",
                            self.api,
                            self.db,
                        ],
                        lines=True,
                    ),
                    probe_health(client, "live"),
                    probe_health(client, "ready"),
                )
                row.update(server=server.get("data", server), stats=stats, live=live, ready=ready)
                if self.worker_pid is not None and self.process.poll() is None:
                    try:
                        row["client"] = sample_windows_process(self.worker_pid)
                    except Exception as exc:
                        row["client"] = {"available": False, "error_type": type(exc).__name__}
                else:
                    row["client"] = {"available": False, "reason": "benchmark not running"}
                row["sample_seconds"] = time.perf_counter() - started
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.stop.wait(), max(0.01, 10 - row["sample_seconds"]))

    async def request(self, client, label, path, payload=None, expected=200):
        row = {"label": label, "status": "pending", "expected_status": expected}
        self.report["requests"].append(row)
        self.save()
        try:
            async with asyncio.timeout(30):
                response = (
                    await client.get(path)
                    if payload is None
                    else await client.post(path, json=payload)
                )
                body = response.json()
            row.update(http_status=response.status_code, response=body)
            assert response.status_code == expected
            row["status"] = "completed"
            return body
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            self.save()

    async def initial_checks(self, client):
        reference = {}
        for mode in MODES:
            body = await self.request(
                client,
                "initial_" + mode,
                "/search",
                {"query": "microsoft", "mode": mode, "limit": 5},
            )
            assert len(body["results"]) == 5
            reference[mode] = {k: v for k, v in body.items() if k != "request_id"}
        histories = load_user_history(
            ROOT / f"data/processed/{self.args.data_version}/user_history.parquet"
        )
        user = next(u for u, history in histories.items() if len(history) >= 3)
        known = await self.request(
            client, "known_recommend", "/recommend", {"user_id": user, "limit": 5}
        )
        cold = await self.request(
            client, "cold_recommend", "/recommend", {"user_id": "container-soak-user", "limit": 5}
        )
        assert known["channel"] == "itemcf" and not known["cold_start"]
        assert cold["channel"] == "popular" and cold["cold_start"]
        for body in (known, cold):
            assert len(body["results"]) == 5 and len({r["item_id"] for r in body["results"]}) == 5
        ctr = await self.request(
            client,
            "ctr",
            "/ctr/score",
            {
                "user_id": user,
                "candidates": [{"item_id": known["results"][0]["item_id"], "bid": 1.0}],
            },
        )
        assert ctr["model_version"] and ctr["calibrator_version"]
        assert all(
            math.isfinite(r[k]) and 0 <= r[k] <= 1
            for r in ctr["results"]
            for k in ("pctr", "pctr_calibrated")
        )
        experiment = await self.request(client, "experiment", "/experiments/sp-bm25")
        assert experiment["source"] == "searchpilot" and experiment["metrics"]
        feedback = {
            "idempotency_key": str(uuid.uuid4()),
            "request_id": "req_container_soak",
            "item_id": "N24742",
            "kind": "click",
            "event_at": now(),
            "user_id": "container-soak-user",
            "position": 1,
        }
        first = await self.request(client, "feedback_create", "/feedback", feedback, 201)
        replay = await self.request(client, "feedback_replay", "/feedback", feedback, 201)
        assert (
            not first["replayed"] and replay["replayed"] and first["event_id"] == replay["event_id"]
        )
        await self.request(
            client, "feedback_conflict", "/feedback", {**feedback, "kind": "like"}, 409
        )
        await self.request(
            client,
            "feedback_missing_item",
            "/feedback",
            {**feedback, "idempotency_key": str(uuid.uuid4()), "item_id": "missing_soak_item"},
            404,
        )
        return reference, feedback, first

    def provenance(self):
        code = (
            "import json,hashlib;from pathlib import Path;"
            "paths=sorted(Path('src').rglob('*.py'))+[Path('scripts')/n for n in "
            "['docker_entrypoint.py','docker_seed.py','seed_items.py','seed_catalog.py']];"
            "print(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() "
            "for p in paths}))"
        )
        hashes = json.loads(self.command([self.docker, "exec", self.api, "python", "-c", code]))
        self.report["container_source_sha256"] = hashes
        expected = {
            p: digest
            for p, digest in self.report["local_source_sha256"].items()
            if p.startswith("src/")
            or p
            in {
                "scripts/" + n
                for n in (
                    "docker_entrypoint.py",
                    "docker_seed.py",
                    "seed_items.py",
                    "seed_catalog.py",
                )
            }
        }
        assert hashes == expected, "Container source differs from local server source"
        code = (
            "import json,hashlib,searchpilot;from pathlib import Path;"
            "root=Path(searchpilot.__file__).parent;"
            "print(json.dumps({'root':str(root),'sha256':{"
            "'src/searchpilot/'+p.relative_to(root).as_posix():"
            "hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}}))"
        )
        imported = json.loads(self.command([self.docker, "exec", self.api, "python", "-c", code]))
        self.report["imported_source"] = imported
        assert imported["sha256"] == {
            p: digest for p, digest in expected.items() if p.startswith("src/searchpilot/")
        }, "Installed/imported Python package differs from local server source"
        code = (
            "import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in "
            "['searchpilot','torch','sentence-transformers','numpy','pandas',"
            "'fastapi','uvicorn','psycopg']}))"
        )
        self.report["container_packages"] = json.loads(
            self.command([self.docker, "exec", self.api, "python", "-c", code])
        )

    def benchmark_command(self):
        return [
            sys.executable,
            "-m",
            "scripts.bench_concurrent",
            "--base-url",
            self.base,
            "--duration-seconds",
            str(self.args.duration_seconds),
            "--concurrency",
            str(self.args.concurrency),
            "--timeout",
            "10",
            "--limit",
            "10",
            "--queries",
            str(self.query_snapshot),
            "--warmup",
            "4",
            "--modes",
            *MODES,
            "--data-version",
            self.args.data_version,
            "--server-description",
            "isolated Compose PG16 CPU embedding; source verified in harness summary",
            "--out",
            str(self.out / "soak.json"),
        ]

    async def run(self):
        sampler = None
        try:
            self.prepare()
            async with httpx.AsyncClient(base_url=self.base, trust_env=False, timeout=30) as client:
                sampler = asyncio.create_task(self.samples(client))
                started = time.perf_counter()
                self.started_containers = True
                await asyncio.to_thread(
                    self.command, self.compose + ["up", "-d", "--no-build", "api"], 180
                )
                self.report["ready"] = await self.readiness(client)
                self.report["startup_seconds"] = time.perf_counter() - started
                self.report["initial_state"], self.report["initial_counts"] = (
                    self.state(),
                    self.counts(),
                )
                counts = self.report["initial_counts"]
                assert (
                    counts["items"]
                    == pq.ParquetFile(
                        ROOT / f"data/processed/{self.args.data_version}/items.parquet"
                    ).metadata.num_rows
                )
                for table, name in (("queries", "queries"), ("relevance_labels", "labels")):
                    with (ROOT / f"data/queries/{name}.csv").open(encoding="utf-8-sig") as file:
                        assert counts[table] == sum(1 for _ in csv.DictReader(file))
                assert counts["feedback_events"] == 0
                self.provenance()
                reference, feedback, first = await self.initial_checks(client)
                self.phase = "baseline"
                self.save()
                await asyncio.sleep(self.args.baseline_seconds)
                bench = self.benchmark_command()
                self.phase = "load"
                self.report.update(soak_started_at=now(), benchmark_command=bench)
                pid_path = self.out / "benchmark.pid"
                go_path = self.out / "benchmark.start"
                wrapper = (
                    "import os,sys,time,runpy;from pathlib import Path;"
                    f"p=Path({str(pid_path)!r});g=Path({str(go_path)!r});"
                    "p.write_text(str(os.getpid()),encoding='ascii');"
                    "deadline=time.monotonic()+15;"
                    "\nwhile not g.exists():\n"
                    " if time.monotonic()>deadline: raise TimeoutError('worker registration')\n"
                    " time.sleep(0.02)\n"
                    "sys.argv=['scripts/bench_concurrent.py',*sys.argv[1:]];"
                    "runpy.run_module('scripts.bench_concurrent',run_name='__main__')"
                )
                launch = [sys.executable, "-c", wrapper, *bench[3:]]
                self.report["launcher_command"] = launch
                if sampler.done():
                    sampler.result()
                    raise RuntimeError("memory sampler stopped before benchmark launch")
                with (self.out / "soak-console.txt").open("x", encoding="utf-8") as console:
                    self.process = subprocess.Popen(
                        launch,
                        cwd=ROOT,
                        env=self.env,
                        stdout=console,
                        stderr=subprocess.STDOUT,
                        creationflags=FLAGS,
                    )
                    async with asyncio.timeout(10):
                        while not pid_path.exists() or not pid_path.read_text(encoding="ascii"):
                            if self.process.poll() is not None:
                                raise RuntimeError("benchmark exited before worker registration")
                            await asyncio.sleep(0.02)
                    self.worker_pid = int(pid_path.read_text(encoding="ascii"))
                    bind_windows_worker(self.process, self.worker_pid)
                    if sampler.done():
                        sampler.result()
                        raise RuntimeError("memory sampler stopped before benchmark dispatch")
                    go_path.write_text("registered", encoding="ascii")
                    self.report.update(benchmark_pid=self.worker_pid, launcher_pid=self.process.pid)
                    self.save()
                    self.report["soak_exit_code"] = await wait_with_sampler(
                        self.process, self.args.duration_seconds + 290, sampler
                    )
                self.report["soak_finished_at"] = now()
                self.phase = "recovery"
                for mode in MODES:
                    body = await self.request(
                        client,
                        "recovery_" + mode,
                        "/search",
                        {"query": "microsoft", "mode": mode, "limit": 5},
                    )
                    assert {k: v for k, v in body.items() if k != "request_id"} == reference[mode]
                await asyncio.sleep(self.args.recovery_seconds)
                self.report["post_soak_ready"] = await self.readiness(client)
                self.report["post_soak_state"] = self.state()
                self.phase = "restart"
                await asyncio.to_thread(self.command, self.compose + ["stop", "-t", "30", "api"])
                self.report["stopped_state"] = self.state()
                assert self.report["stopped_state"][0]["State"]["ExitCode"] == 0
                await asyncio.to_thread(self.command, self.compose + ["start", "api"])
                self.report["restart_ready"] = await self.readiness(client)
                replay = await self.request(
                    client, "feedback_after_restart", "/feedback", feedback, 201
                )
                assert replay["replayed"] and replay["event_id"] == first["event_id"]
                self.report["final_counts"], self.report["final_state"] = (
                    self.counts(),
                    self.state(),
                )
                assert self.report["final_counts"] == {**counts, "feedback_events": 1}
                assert not any(
                    r["State"]["OOMKilled"] or r["RestartCount"] for r in self.report["final_state"]
                )
                assert self.report["soak_exit_code"] == 0
                self.report["status"] = "COMPLETED"
                self.stop.set()
                await sampler
        except BaseException as exc:
            self.report.update(
                status="INTERRUPTED"
                if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                else "FAILED",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self.stop.set()
            if self.process is not None:
                try:
                    stop_owned_process(self.process)
                except Exception as exc:
                    self.report["cleanup_error"] = type(exc).__name__
                    self.report["status"] = "FAILED"
            if sampler is not None:
                try:
                    await sampler
                except BaseException as exc:
                    self.report["sampler_error"] = type(exc).__name__
                    self.report["status"] = "FAILED"
            if self.owns_output:
                for name in (self.api, self.db) if self.started_containers else ():
                    try:
                        output = subprocess.run(
                            [self.docker, "logs", name],
                            cwd=ROOT,
                            env=self.env,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            timeout=15,
                            creationflags=FLAGS,
                        )
                        (self.out / (name + ".log")).write_text(
                            output.stdout + output.stderr, encoding="utf-8"
                        )
                    except Exception as exc:
                        self.report.setdefault("log_errors", {})[name] = type(exc).__name__
                self.report["finished_at"] = now()
                self.save()
        if self.report["status"] != "COMPLETED":
            raise RuntimeError("validation cleanup or sampler failed; see summary.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project", default="sp-hour-20261023")
    parser.add_argument("--image", default="searchpilot-validation:2026-10-23-memory")
    parser.add_argument("--api-port", type=int, default=18082)
    parser.add_argument("--db-port", type=int, default=15433)
    parser.add_argument("--duration-seconds", type=float, default=3600)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries/queries.csv")
    parser.add_argument("--baseline-seconds", type=float, default=60)
    parser.add_argument("--recovery-seconds", type=float, default=120)
    parser.add_argument("--data-version", default="d3a904f41240")
    args = parser.parse_args(argv)
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.project)
        or not 0 < args.duration_seconds <= 3600
        or not 1 <= args.concurrency <= 256
        or any(
            not math.isfinite(v) or v < 0 for v in (args.baseline_seconds, args.recovery_seconds)
        )
        or not all(1024 <= port <= 65535 for port in (args.api_port, args.db_port))
        or args.api_port == args.db_port
    ):
        parser.error("invalid project, duration, concurrency, recovery interval, or ports")
    try:
        harness = Harness(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    asyncio.run(harness.run())
    return int(harness.report["status"] != "COMPLETED")


if __name__ == "__main__":
    raise SystemExit(main())
