from __future__ import annotations

import math

import pandas as pd

from searchpilot.recommend.evaluate import (
    _binary_ndcg_at_k,
    _recall_at_k,
    evaluate_fallback,
)


def test_recall_and_ndcg_helpers() -> None:
    rel = frozenset({"b", "c"})
    ranked = ["a", "b", "d", "c"]
    assert _recall_at_k(rel, ranked, 20) == 1.0
    ndcg2 = _binary_ndcg_at_k(rel, ranked, 2)
    dcg = 1.0 / math.log2(3)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert abs(ndcg2 - dcg / idcg) < 1e-9


def test_evaluate_fallback_small_sample() -> None:
    impressions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u1"],
            "item_id": ["A", "B", "A", "C", "C"],
            "clicked": [1, 1, 1, 1, 1],
            "split": ["train", "train", "train", "dev", "dev"],
        }
    )
    histories = {"u1": ("A", "B"), "u2": ()}
    report = evaluate_fallback(
        impressions,
        histories,
        data_version="t1",
        model="popular",
    )
    assert report.source == "searchpilot"
    assert report.user_count == 2
    assert report.segments["cold_user"].user_count == 1
    assert report.segments["history_present"].user_count == 1
    assert 0.0 <= report.metrics.recall_at_20 <= 1.0
