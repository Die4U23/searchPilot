"""名次对照只改指定列，不改原始矩阵和其他名次。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_ltr_rank_ablation.py"
    name = "diagnose_ltr_rank_ablation"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_zero_weight_clears_one_index() -> None:
    script = _script()
    original = (0.2, 0.6)
    assert script.zero_weight(original, 1) == (0.2, 0.0)
    assert original == (0.2, 0.6)


def test_fill_missing_rank_keeps_observed_ranks() -> None:
    script = _script()
    raw = np.array([[1.0, 101.0], [3.0, 4.0]])
    filled = script.fill_missing_rank(raw, 1, 101.0, 7.0)
    assert filled[0, 1] == 7.0
    assert filled[1, 1] == 4.0
    assert raw[0, 1] == 101.0
