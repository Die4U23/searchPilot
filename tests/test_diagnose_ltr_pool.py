"""补标覆盖计数：向量前 50 被 LTR 挤出、以及两路都没召回。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_ltr_pool.py"
    spec = importlib.util.spec_from_file_location("diagnose_ltr_pool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vector_hit_past_ltr_cutoff_counts_missing_bm25() -> None:
    script = _script()
    row = script.Coverage()
    script._add_coverage(row, in_bm25=False, in_vector=True, vector_rank=8, ltr_rank=51)
    assert row.vector_only == 1
    assert row.vector50_not_ltr50 == 1
    assert row.vector50_not_ltr50_missing_bm25 == 1


def test_vector_hit_kept_inside_ltr_cutoff() -> None:
    script = _script()
    row = script.Coverage()
    script._add_coverage(row, in_bm25=True, in_vector=True, vector_rank=8, ltr_rank=3)
    assert row.vector50_not_ltr50 == 0
    assert row.neither == 0


def test_outside_both_retrieval_lists() -> None:
    script = _script()
    row = script.Coverage()
    script._add_coverage(row, in_bm25=False, in_vector=False, vector_rank=None, ltr_rank=None)
    assert row.neither == 1
    assert row.vector50 == 0


def test_rank_contribution_is_one_term_of_the_linear_score() -> None:
    script = _script()
    model = SimpleNamespace(weights=(0.0, 2.0), mean=(0.0, 10.0), std=(1.0, 2.0))
    assert script._contrib(model, 1, 14.0) == 4.0
