"""Container memory policy is applied before startup and verified across exec."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import docker_entrypoint as module

ENTRYPOINT = Path(__file__).resolve().parents[1] / "scripts/docker_entrypoint.py"


class Prctl:
    def __init__(self, returns):
        self.returns = iter(returns)
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return next(self.returns)


@pytest.mark.parametrize(
    ("configured", "disabled"), [(None, True), ("true", True), ("false", False), (" FALSE ", False)]
)
def test_verified_policy_and_native_abi(monkeypatch, configured, disabled):
    monkeypatch.delenv("SEARCHPILOT_DISABLE_THP", raising=False)
    if configured is not None:
        monkeypatch.setenv("SEARCHPILOT_DISABLE_THP", configured)
    native = Prctl([0, int(disabled)])
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda name, **kw: SimpleNamespace(prctl=native))
    result = module.configure_thp()
    assert result == {
        "requested": disabled,
        "applied": True,
        "policy": "disabled" if disabled else "system_default",
    }
    assert native.calls == [(41, int(disabled), 0, 0, 0), (42, 0, 0, 0, 0)]
    assert native.argtypes == [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    assert native.restype is ctypes.c_int


@pytest.mark.parametrize("bad", ["", "1", "0", "yes", "private-fixture-value"])
def test_invalid_setting_fails_without_echoing_value(monkeypatch, bad):
    monkeypatch.setenv("SEARCHPILOT_DISABLE_THP", bad)
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *a, **kw: pytest.fail("must not call libc"))
    with pytest.raises(RuntimeError) as caught:
        module.configure_thp()
    assert str(caught.value) == "SEARCHPILOT_DISABLE_THP must be true or false"


def test_unsupported_platform_never_claims_applied(monkeypatch):
    monkeypatch.delenv("SEARCHPILOT_DISABLE_THP", raising=False)
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *a, **kw: pytest.fail("must not call libc"))
    assert module.configure_thp() == {
        "requested": True,
        "applied": False,
        "policy": "unsupported_platform",
    }


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        ([-1], "PR_SET_THP_DISABLE"),
        ([0, -1], "PR_GET_THP_DISABLE"),
        ([0, 3], "verification failed"),
    ],
)
def test_syscall_failure_and_except_advised_are_not_success(monkeypatch, returns, expected):
    monkeypatch.delenv("SEARCHPILOT_DISABLE_THP", raising=False)
    monkeypatch.setattr(module.sys, "platform", "linux")
    native = Prctl(returns)
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=native))
    monkeypatch.setattr(module.ctypes, "get_errno", lambda: 1)
    with pytest.raises(RuntimeError, match=expected):
        module.configure_thp()


def test_missing_prctl_fails_explicitly(monkeypatch):
    monkeypatch.delenv("SEARCHPILOT_DISABLE_THP", raising=False)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace())
    with pytest.raises(RuntimeError, match="control is unavailable"):
        module.configure_thp()


def test_policy_applied_before_migration_seed_and_exec(monkeypatch, capsys):
    steps = []
    policy = {"requested": True, "applied": True, "policy": "disabled"}
    monkeypatch.setattr(module, "configure_thp", lambda: steps.append("policy") or policy)
    monkeypatch.setenv("SEARCHPILOT_DATABASE_URL", "postgresql://fixture/fixture")
    monkeypatch.setitem(
        sys.modules,
        "docker_seed",
        SimpleNamespace(seed_mounted_data=lambda *a: steps.append("seed")),
    )
    monkeypatch.setitem(
        sys.modules,
        "searchpilot.db.migrate",
        SimpleNamespace(apply_migrations=lambda *a: steps.append("migrate") or []),
    )

    def execvp(binary, args):
        steps.append("exec")
        assert args[:4] == [
            sys.executable,
            "-m",
            "uvicorn",
            "searchpilot.bootstrap:build_default_app",
        ]
        raise SystemExit(0)

    monkeypatch.setattr(module.os, "execvp", execvp)
    with pytest.raises(SystemExit):
        module.main()
    assert steps == ["policy", "migrate", "seed", "exec"]
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == policy


def test_policy_failure_prevents_api_start(monkeypatch):

    def fail():
        raise RuntimeError("policy failed")

    monkeypatch.setattr(module, "configure_thp", fail)
    monkeypatch.setattr(module.os, "execvp", lambda *a: pytest.fail("must not exec"))
    with pytest.raises(RuntimeError, match="policy failed"):
        module.main()


@pytest.mark.skipif(sys.platform != "linux", reason="real prctl/exec inheritance requires Linux")
@pytest.mark.parametrize("requested", ["true", "false"])
def test_linux_policy_survives_exec_in_isolated_child(requested):
    # Never mutate the pytest process. Start disabled so false proves clearing,
    # not merely skipping the request; an exec then checks the inherited result.
    child = """
import ctypes,importlib.util,json,os,sys
p=ctypes.CDLL(None,use_errno=True).prctl
p.argtypes=[ctypes.c_int]+[ctypes.c_ulong]*4
p.restype=ctypes.c_int
assert p(41,1,0,0,0)==0
spec=importlib.util.spec_from_file_location('candidate',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.configure_thp()
check=('import ctypes,json; p=ctypes.CDLL(None).prctl; '
       'p.argtypes=[ctypes.c_int]+[ctypes.c_ulong]*4; p.restype=ctypes.c_int; '
       'print(json.dumps({"observed":p(42,0,0,0,0)}))')
os.execv(sys.executable,[sys.executable,'-c',check])
"""
    env = {**os.environ, "SEARCHPILOT_DISABLE_THP": requested}
    completed = subprocess.run(
        [sys.executable, "-c", child, str(ENTRYPOINT.resolve())],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert json.loads(completed.stdout) == {"observed": int(requested == "true")}
