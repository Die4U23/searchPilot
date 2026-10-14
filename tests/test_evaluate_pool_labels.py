"""查询类型均值会跳过没有相关文档的查询。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pool_labels.py"
    name = "evaluate_pool_labels"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_mean_for_type_skips_queries_without_a_defined_ndcg() -> None:
    script = _script()
    queries = [
        script.EvalQuery("q1", "synonym text", "synonym", "val"),
        script.EvalQuery("q2", "nothing", "no_answer", "val"),
    ]
    value, counted = script._mean_for_type(queries, {"q1": 0.5, "q2": None}, "synonym")
    assert value == 0.5
    assert counted == 1
    empty, skipped = script._mean_for_type(queries, {"q1": 0.5, "q2": None}, "no_answer")
    assert empty is None
    assert skipped == 0
