"""明确 0 分挡在第一篇相关文档前面时要数得准。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_ltr_gaps.py"
    name = "diagnose_ltr_gaps"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_zeros_before_ignores_weaker_positive_grades() -> None:
    script = _script()
    order = ["zero", "weak", "another_zero", "strong"]
    grades = {"zero": 0, "weak": 1, "another_zero": 0, "strong": 3}
    assert script.zeros_before(order, grades, 1) == 1
    assert script.zeros_before(order, grades, 2) == 2


def test_zeros_before_is_none_when_the_grade_never_appears() -> None:
    script = _script()
    assert script.zeros_before(["only"], {"only": 0}, 1) is None
