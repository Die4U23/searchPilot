from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path("scripts/diagnose_label_pool.py")
    spec = importlib.util.spec_from_file_location("diagnose_label_pool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coverage_counts_overlap() -> None:
    module = _load()
    assert module.coverage({"a", "b"}, {"b", "c"}) == (1, 2)
    assert module.coverage(set(), {"a"}) == (0, 0)


def test_summarize_adds_splits() -> None:
    module = _load()
    row = {"split": "test", "unlabeled_k10": 2}
    for kind in ("labeled", "positive"):
        for mode in module.MODES:
            for depth in (10, 50):
                row[f"{kind}_{mode}@{depth}"] = (1, 2)
    summary = module.summarize([row])
    assert summary["test"]["labeled:bm25@10"] == (1, 2)
    assert summary["all"]["unlabeled_k10"] == (2, 2)
