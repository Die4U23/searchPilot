"""The diagnostic control plane must remain explicit, bounded and separate from the app."""

from __future__ import annotations

import http.client
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace

import pytest
from scripts import diagnose_allocator as module


class FakeAllocator:
    def __init__(self):
        self.trims = 0

    def mallinfo(self):
        return {"arena": 4096, "uordblks": 1024, "fordblks": 3072}

    def malloc_info(self):
        return {"heap_count": 1, "xml": '<malloc><heap nr="0" /></malloc>'}

    def pymalloc(self):
        return {"text": "fixture", "fields": {}, "missing_fields": ["arena_bytes"]}

    def trim(self):
        self.trims += 1
        return True


@pytest.fixture
def diagnostic(monkeypatch):
    allocator = FakeAllocator()
    monkeypatch.setattr(module, "sample_linux_process", lambda pid: {"pid": pid, "sources": {}})
    return module.Diagnostic(allocator)


def test_snapshot_observes_without_gc_trim_or_starting_trace(diagnostic, monkeypatch):
    monkeypatch.setattr(module.gc, "collect", lambda: pytest.fail("unexpected GC"))
    monkeypatch.setattr(module.tracemalloc, "start", lambda *a: pytest.fail("unexpected tracing"))
    result = diagnostic.snapshot()
    assert result["allocator"]["mallinfo2"]["data"]["uordblks"] == 1024
    assert result["allocator"]["malloc_info"]["data"]["heap_count"] == 1
    assert '<heap nr="0"' in result["allocator"]["malloc_info"]["data"]["xml"]
    assert diagnostic.allocator.trims == 0


def test_native_source_failure_is_not_reported_as_zero(diagnostic, monkeypatch):
    def unavailable():
        raise AttributeError("mallinfo2 unavailable")

    monkeypatch.setattr(diagnostic.allocator, "mallinfo", unavailable)
    snapshot = diagnostic.snapshot()
    failed = snapshot["allocator"]["mallinfo2"]
    assert failed["available"] is False
    assert failed["error_type"] == "AttributeError"
    assert "data" not in failed
    assert snapshot["allocator"]["malloc_info"]["available"] is True


def test_pymalloc_missing_symbol_does_not_hide_other_sources(diagnostic, monkeypatch):
    def unavailable():
        raise AttributeError("_PyObject_DebugMallocStats unavailable")

    monkeypatch.setattr(diagnostic.allocator, "pymalloc", unavailable)
    snapshot = diagnostic.snapshot()
    assert snapshot["pymalloc"]["available"] is False
    assert snapshot["pymalloc"]["error_type"] == "AttributeError"
    assert snapshot["allocator"]["mallinfo2"]["available"] is True


def test_pymalloc_rejects_unvalidated_internal_abi(monkeypatch):
    monkeypatch.setattr(module.sys, "version_info", (3, 13))
    with pytest.raises(OSError, match="only validated for CPython 3.12"):
        module.NativeAllocator().pymalloc()


def test_pymalloc_parses_byte_totals_without_confusing_arena_map_total():
    text = """# arenas allocated current         =                   50
50 arenas * 1048576 bytes/arena    =           52,428,800
# bytes in allocated blocks        =           50,936,448
# bytes in available blocks        =              325,376
12 unused pools * 16384 bytes      =              196,608
Total                              =           52,428,800
arena map counts
Total                              =              786,432
"""
    parsed = module.parse_pymalloc(text)
    assert parsed["fields"] == {
        "arenas_current": 50,
        "arena_bytes": 52428800,
        "allocated_blocks_bytes": 50936448,
        "available_blocks_bytes": 325376,
        "unused_pools_bytes": 196608,
    }
    assert parsed["missing_fields"] == []
    absent = module.parse_pymalloc("pymalloc unavailable or output format changed")
    assert absent["fields"] == {}
    assert len(absent["missing_fields"]) == 5


def test_gc_and_trim_happen_only_for_explicit_actions(diagnostic, monkeypatch):
    collections = []
    monkeypatch.setattr(module.gc, "collect", lambda: collections.append("gc") or 3)
    assert diagnostic.action("/gc", {})["result"] == {"collected": 3}
    assert collections == ["gc"]
    assert diagnostic.allocator.trims == 0
    assert diagnostic.action("/trim", {})["result"] == {"released": True}
    assert diagnostic.allocator.trims == 1
    with pytest.raises(ValueError):
        diagnostic.action("/trim", {"unexpected": True})
    assert diagnostic.allocator.trims == 1


def test_trace_control_and_overhead_are_explicit(diagnostic, monkeypatch):
    state = {"enabled": False, "frames": None}

    def start(frames):
        state.update(enabled=True, frames=frames)

    def stop():
        state["enabled"] = False

    monkeypatch.setattr(
        module,
        "tracemalloc",
        SimpleNamespace(
            is_tracing=lambda: state["enabled"],
            start=start,
            stop=stop,
            get_traced_memory=lambda: (100, 250),
            get_tracemalloc_memory=lambda: 75,
            get_traceback_limit=lambda: state["frames"],
        ),
    )
    assert diagnostic.snapshot()["tracemalloc"]["current_bytes"] is None
    traced = diagnostic.action("/trace/start", {"frames": 2})["snapshot"]["tracemalloc"]
    assert traced == {
        "enabled": True,
        "current_bytes": 100,
        "peak_bytes": 250,
        "overhead_bytes": 75,
        "traceback_limit": 2,
    }
    with pytest.raises(ValueError, match="already enabled"):
        diagnostic.action("/trace/start", {})
    stopped = diagnostic.action("/trace/stop", {})
    assert stopped["result"] == {"stopped": True}
    assert stopped["snapshot"]["tracemalloc"]["current_bytes"] is None
    for frames in (True, 0, 26, "1"):
        with pytest.raises(ValueError):
            diagnostic.action("/trace/start", {"frames": frames})
    assert state["enabled"] is False


@pytest.fixture
def control(diagnostic):
    server = module.control_server(diagnostic, 0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", diagnostic
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def request(base, path, *, data=None, headers=None):
    req = urllib.request.Request(base + path, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def test_control_get_cannot_run_mutating_action(control):
    base, diagnostic = control
    assert request(base, "/snapshot")[0] == 200
    assert request(base, "/trim")[0] == 404
    assert diagnostic.allocator.trims == 0
    code, body = request(base, "/trim", data=b"{}")
    assert code == 200 and body["result"]["released"]
    assert diagnostic.allocator.trims == 1
    assert request(base, "/missing", data=b"{}")[0] == 404


def test_control_rejects_large_or_malformed_body_before_action(control):
    base, diagnostic = control
    for body in (b"{broken", b"[]"):
        assert request(base, "/trim", data=body)[0] == 400
    # Send rejected headers without a body: early close with unread bytes may
    # reset the TCP connection on Windows instead of delivering the response.
    for key, value in (("Content-Length", "4097"), ("Transfer-Encoding", "chunked")):
        address = urllib.parse.urlsplit(base)
        connection = http.client.HTTPConnection(address.hostname, address.port, timeout=5)
        try:
            connection.putrequest("POST", "/trim")
            connection.putheader(key, value)
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 400
            response.read()
        finally:
            connection.close()
    assert diagnostic.allocator.trims == 0
    assert request(base, "/snapshot")[0] == 200


@pytest.mark.skipif(sys.platform != "linux", reason="glibc integration requires Linux")
def test_real_glibc_snapshot_has_raw_xml_and_live_allocation_counts():
    allocator = module.NativeAllocator()
    fields = allocator.mallinfo()
    info = allocator.malloc_info()
    assert fields["arena"] > 0 and fields["uordblks"] > 0
    assert fields["fordblks"] >= 0 and fields["hblkhd"] >= 0
    assert info["heap_count"] >= 1
    assert module.ET.fromstring(info["xml"]).tag == "malloc"
    python = module.observed(allocator.pymalloc)
    if python["available"]:
        assert python["data"]["return_code"] in (0, 1)
        assert isinstance(python["data"]["text"], str)
    else:
        assert python["error_type"] == "AttributeError"
