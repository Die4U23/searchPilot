"""Explicit Linux allocator diagnostic wrapper; never used by the normal API entrypoint.

The separate control server binds only container loopback. Do not publish its port.
GET /snapshot observes; POST /trace/start, /trace/stop, /gc and /trim explicitly
perturb this diagnostic process. Run interventions only after load has stopped.
No private .env is loaded, and no business API route is added.
This bypasses migration/seeding and the normal entrypoint's THP policy; prepare
the database first and apply any experimental process policy before this module.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import sys
import tempfile
import threading
import tracemalloc
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from scripts.soak_memory import sample_linux_process


class Mallinfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


class NativeAllocator:
    """glibc evidence; unavailable symbols remain explicit rather than zero-valued."""

    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None

    def _function(self, name, args, result):
        if self.libc is None:
            raise OSError("glibc diagnostics require Linux")
        function = getattr(self.libc, name)
        function.argtypes, function.restype = args, result
        return function

    def mallinfo(self):
        values = self._function("mallinfo2", [], Mallinfo2)()
        return {name: getattr(values, name) for name, _ in Mallinfo2._fields_}

    def _capture_native_output(self, writer):
        fdopen = self._function("fdopen", [ctypes.c_int, ctypes.c_char_p], ctypes.c_void_p)
        fclose = self._function("fclose", [ctypes.c_void_p], ctypes.c_int)
        with tempfile.TemporaryFile(mode="w+b") as output:
            duplicate = os.dup(output.fileno())
            stream = fdopen(duplicate, b"w")
            if not stream:
                os.close(duplicate)
                raise OSError(ctypes.get_errno(), "fdopen failed")
            try:
                result = writer(stream)
            finally:
                close_result = fclose(stream)
            if close_result:
                raise OSError(ctypes.get_errno(), "fclose failed")
            output.seek(0)
            raw = output.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("native statistics exceed 1 MiB")
        return raw, result

    def malloc_info(self):
        info = self._function("malloc_info", [ctypes.c_int, ctypes.c_void_p], ctypes.c_int)
        raw, result = self._capture_native_output(lambda stream: info(0, stream))
        if result:
            raise OSError("malloc_info failed")
        xml = raw.decode("utf-8")
        root = ET.fromstring(xml)
        return {"heap_count": len(root.findall("heap")), "xml": xml}

    def trim(self):
        return bool(self._function("malloc_trim", [ctypes.c_size_t], ctypes.c_int)(0))

    def pymalloc(self):
        # CPython 3.12 declares int _PyObject_DebugMallocStats(FILE *out).
        # PyDLL keeps the GIL; the optional internal symbol may not be exported.
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
            raise OSError("pymalloc internal ABI is only validated for CPython 3.12")
        read = ctypes.pythonapi._PyObject_DebugMallocStats
        read.argtypes, read.restype = [ctypes.c_void_p], ctypes.c_int
        raw, result = self._capture_native_output(read)
        text = raw.decode("utf-8", errors="replace")
        return {
            "source": "optional CPython internal _PyObject_DebugMallocStats(FILE*)",
            "return_code": result,
            "text": text,
            **parse_pymalloc(text),
        }


def parse_pymalloc(text):
    patterns = {
        "arenas_current": r"^# arenas allocated current\s*=\s*([\d,]+)\s*$",
        "arena_bytes": r"^\d+ arenas \* \d+ bytes/arena\s*=\s*([\d,]+)\s*$",
        "allocated_blocks_bytes": r"^# bytes in allocated blocks\s*=\s*([\d,]+)\s*$",
        "available_blocks_bytes": r"^# bytes in available blocks\s*=\s*([\d,]+)\s*$",
        "unused_pools_bytes": r"^\d+ unused pools \* \d+ bytes\s*=\s*([\d,]+)\s*$",
    }
    fields = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            fields[name] = int(match[1].replace(",", ""))
    return {"fields": fields, "missing_fields": sorted(patterns.keys() - fields.keys())}


def observed(call):
    try:
        return {"available": True, "data": call()}
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__, "message": str(exc)}


def trace_snapshot():
    enabled = tracemalloc.is_tracing()
    current, peak = tracemalloc.get_traced_memory() if enabled else (None, None)
    return {
        "enabled": enabled,
        "current_bytes": current,
        "peak_bytes": peak,
        "overhead_bytes": tracemalloc.get_tracemalloc_memory() if enabled else None,
        "traceback_limit": tracemalloc.get_traceback_limit() if enabled else None,
    }


class Diagnostic:
    def __init__(self, allocator=None):
        self.allocator = allocator if allocator is not None else NativeAllocator()

    def snapshot(self):
        return {
            "schema_version": 1,
            "pid": os.getpid(),
            "at": datetime.now(UTC).isoformat(),
            "method": "Diagnostic process only; sequential, non-atomic observations. Control "
            "thread and probe allocations are included. No automatic GC or trim. "
            "mallinfo2 fields have glibc-specific scopes; retain malloc_info XML.",
            "allocator": {
                "mallinfo2": observed(self.allocator.mallinfo),
                "malloc_info": observed(self.allocator.malloc_info),
            },
            "pymalloc": observed(self.allocator.pymalloc),
            "process": observed(lambda: sample_linux_process(os.getpid())),
            "tracemalloc": trace_snapshot(),
            "python_threads": threading.active_count(),
        }

    def action(self, path, payload):
        if path == "/trace/start":
            if set(payload) - {"frames"}:
                raise ValueError("trace/start accepts only frames")
            frames = payload.get("frames", 1)
            if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 25:
                raise ValueError("frames must be an integer from 1 to 25")
            if tracemalloc.is_tracing():
                raise ValueError("tracemalloc is already enabled; stop explicitly before restart")
            tracemalloc.start(frames)
            result = {"started": True, "frames": frames}
        elif path in ("/trace/stop", "/gc", "/trim"):
            if payload:
                raise ValueError("this action accepts an empty object only")
            if path == "/trace/stop":
                was_enabled = tracemalloc.is_tracing()
                tracemalloc.stop()
                result = {"stopped": was_enabled}
            elif path == "/gc":
                result = {"collected": gc.collect()}
            else:
                result = {"released": self.allocator.trim()}
        else:
            raise KeyError("unknown diagnostic action")
        return {"action": path, "result": result, "snapshot": self.snapshot()}


def control_server(diagnostic, port):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, *_args):
            pass

        def reply(self, status, body):
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/snapshot":
                self.reply(404, {"error": "only GET /snapshot is supported"})
                return
            self.reply(200, diagnostic.snapshot())

        def do_POST(self):
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, {"error": "transfer encoding is unsupported"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            try:
                if len(lengths) > 1:
                    raise ValueError("duplicate Content-Length")
                size = int(lengths[0]) if lengths else 0
                if not 0 <= size <= 4096:
                    raise ValueError("request body must be at most 4096 bytes")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("incomplete request body")
                body = json.loads(raw) if raw else {}
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                response = diagnostic.action(self.path, body)
            except KeyError:
                self.reply(404, {"error": "unknown diagnostic action"})
            except (ValueError, TimeoutError) as exc:
                self.reply(400, {"error_type": type(exc).__name__, "message": str(exc)})
            except Exception as exc:
                self.reply(500, {"error_type": type(exc).__name__, "message": str(exc)})
            else:
                self.reply(200, response)

    return HTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--control-port", type=int, default=8779)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("allocator diagnostic wrapper requires Linux")
    if not all(1 <= port <= 65535 for port in (args.api_port, args.control_port)):
        parser.error("ports must be in 1..65535")
    if args.api_port == args.control_port:
        parser.error("control and API ports must differ")
    import uvicorn

    from searchpilot.config import get_settings
    from searchpilot.observability import configure_logging

    configure_logging(get_settings().log_level)
    server = control_server(Diagnostic(), args.control_port)
    thread = threading.Thread(target=server.serve_forever, name="allocator-control", daemon=True)
    thread.start()
    try:
        uvicorn.run(
            "searchpilot.bootstrap:build_default_app",
            factory=True,
            host=args.api_host,
            port=args.api_port,
            log_config=None,
            access_log=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
