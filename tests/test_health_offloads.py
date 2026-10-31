from __future__ import annotations

import asyncio
import functools
import socket
import threading
from collections import Counter
from types import SimpleNamespace

import anyio.to_thread
import httpx
import uvicorn
from fastapi import Request

from searchpilot.api import deps
from searchpilot.bootstrap import create_app


class Probe:
    def __init__(self, value: bool | None = True) -> None:
        self.value = value
        self.threads: list[int] = []

    def check(self) -> dict[str, bool]:
        self.threads.append(threading.get_ident())
        if self.value is None:
            raise RuntimeError("private diagnostic text")
        return {"database": self.value}


def sync_getter(field: str):
    # Baseline copy of the former synchronous app.state accessor.
    def getter(request: Request):
        return getattr(request.app.state, field)

    getter.__name__ = "baseline_" + field
    return getter


def test_http_state_offloads_semantics_and_db_thread(monkeypatch) -> None:
    async def exercise() -> None:
        probe = Probe()
        app = create_app(
            search=SimpleNamespace(vector_ready=True, ltr_ready=True),
            recommend=object(),
            feedback=object(),
            ctr=SimpleNamespace(ctr_ready=True),
            probes=[probe],
        )
        # Preserve token protection while avoiding private settings or env reads.
        app.state.settings = SimpleNamespace(api_token="test-token")
        fields = {
            deps.get_probes: "probes",
            deps.get_search: "search",
            deps.get_recommend: "recommend",
            deps.get_feedback: "feedback",
            deps.get_ctr: "ctr",
        }
        calls: list[str] = []
        original = anyio.to_thread.run_sync

        async def counted(func, *args, **kwargs):
            unwrapped = func
            while isinstance(unwrapped, functools.partial):
                unwrapped = unwrapped.func
            calls.append(getattr(unwrapped, "__name__", type(unwrapped).__name__))
            return await original(func, *args, **kwargs)

        monkeypatch.setattr(anyio.to_thread, "run_sync", counted)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        sock.setblocking(False)
        server = uvicorn.Server(
            uvicorn.Config(app, log_config=None, access_log=False, lifespan="off")
        )
        server.install_signal_handlers = lambda: None
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise AssertionError("server exited before startup")
                    await asyncio.sleep(0.01)
            loop_thread = threading.get_ident()
            url = f"http://127.0.0.1:{sock.getsockname()[1]}"
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=3) as client:
                app.dependency_overrides.update({k: sync_getter(v) for k, v in fields.items()})
                baseline = await client.get(
                    "/health/ready", headers={"Authorization": "Bearer test-token"}
                )
                baseline_calls = Counter(calls)
                assert {
                    k: baseline_calls[k] for k in ("baseline_" + v for v in fields.values())
                } == {
                    "baseline_probes": 1,
                    "baseline_search": 1,
                    "baseline_recommend": 1,
                    "baseline_feedback": 1,
                    "baseline_ctr": 1,
                }
                calls.clear()
                app.dependency_overrides.clear()
                candidate = await client.get(
                    "/health/ready", headers={"Authorization": "Bearer test-token"}
                )
                assert candidate.status_code == baseline.status_code == 200
                assert candidate.json() == baseline.json()
                assert not any(
                    k.startswith("baseline_") or k in {f.__name__ for f in fields} for k in calls
                )
                assert probe.threads and all(t != loop_thread for t in probe.threads)
                assert candidate.json()["checks"] == {
                    "search": True,
                    "recommend": True,
                    "feedback": True,
                    "vector_index": True,
                    "ltr": True,
                    "ctr": True,
                    "database": True,
                }
                # Live remains unauthenticated and requires no worker for endpoint.
                calls.clear()
                live = await client.get("/health/live")
                assert live.status_code == 200 and live.json() == {"status": "ok"}
                assert "live" not in calls
                denied = await client.get("/health/ready")
                assert denied.status_code == 401
                probe.value = False
                failed = await client.get(
                    "/health/ready", headers={"Authorization": "Bearer test-token"}
                )
                assert failed.status_code == 503 and failed.json()["checks"]["database"] is False
                app.dependency_overrides.update({k: sync_getter(v) for k, v in fields.items()})
                baseline_failed = await client.get(
                    "/health/ready", headers={"Authorization": "Bearer test-token"}
                )
                assert baseline_failed.status_code == failed.status_code
                assert baseline_failed.json()["checks"] == failed.json()["checks"]
                app.dependency_overrides.clear()
                probe.value = None
                error = await client.get(
                    "/health/ready", headers={"Authorization": "Bearer test-token"}
                )
                assert error.status_code == 503 and error.json()["checks"]["Probe"] is False
                assert "private diagnostic text" not in error.text
                app.dependency_overrides.update({k: sync_getter(v) for k, v in fields.items()})
                baseline_error = await client.get(
                    "/health/ready", headers={"Authorization": "Bearer test-token"}
                )
                assert baseline_error.status_code == error.status_code
                assert baseline_error.json()["checks"] == error.json()["checks"]
                assert "private diagnostic text" not in baseline_error.text
                app.dependency_overrides.clear()
                # FastAPI dependency overrides remain supported after conversion.
                app.dependency_overrides[deps.get_search] = lambda: None
                missing = await client.get(
                    "/health/ready", headers={"Authorization": "Bearer test-token"}
                )
                assert missing.status_code == 503 and missing.json()["checks"]["search"] is False
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(task, timeout=5)
            finally:
                sock.close()

    asyncio.run(exercise())
