"""排序指标的手算例子。grade 3 → gain 7，grade 1 → gain 1。"""

from __future__ import annotations

import math

import pytest

from searchpilot.eval.metrics import (
    judged_ranking,
    mean_defined,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# 名次：a(grade 3)、b(未标注)、c(grade 1)
RANKED = ["a", "b", "c"]
GRADES = {"a": 3, "c": 1}


def test_judged_ranking_drops_unlabeled_without_reordering() -> None:
    assert judged_ranking(["u", "a", "v", "c"], GRADES) == ["a", "c"]
    full = ndcg_at_k(["u", "a"], {"a": 3}, 10)
    condensed = ndcg_at_k(judged_ranking(["u", "a"], {"a": 3}), {"a": 3}, 10)
    assert full is not None and condensed is not None
    assert condensed == pytest.approx(1.0)
    assert full < condensed


def test_ndcg_at_3_hand_computed() -> None:
    # (7/log2(2) + 1/log2(4)) / (7/log2(2) + 1/log2(3))
    dcg = 7 / math.log2(2) + 1 / math.log2(4)
    idcg = 7 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, GRADES, 3) == pytest.approx(dcg / idcg)


def test_recall_precision_and_mrr_hand_computed() -> None:
    # 相关文档 {a, c}；b 不相关。
    assert recall_at_k(RANKED, GRADES, 1) == pytest.approx(0.5)
    assert recall_at_k(RANKED, GRADES, 2) == pytest.approx(0.5)
    assert recall_at_k(RANKED, GRADES, 3) == pytest.approx(1.0)
    assert precision_at_k(RANKED, GRADES, 1) == pytest.approx(1.0)
    assert precision_at_k(RANKED, GRADES, 2) == pytest.approx(0.5)
    assert precision_at_k(RANKED, GRADES, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(RANKED, GRADES, 3) == pytest.approx(1.0)

    later = ["b", "c", "a"]  # 首个相关在第 2 名
    assert reciprocal_rank(later, GRADES, 3) == pytest.approx(0.5)
    assert mrr([RANKED, later], [GRADES, GRADES], 3) == pytest.approx(0.75)


def test_k_larger_than_result_count() -> None:
    assert recall_at_k(RANKED, GRADES, 10) == pytest.approx(1.0)
    # 分母固定为 k，结果不足 k 条也不缩小。
    assert precision_at_k(RANKED, GRADES, 10) == pytest.approx(0.2)
    assert ndcg_at_k(RANKED, GRADES, 10) == pytest.approx(ndcg_at_k(RANKED, GRADES, 3))
    assert reciprocal_rank(RANKED, GRADES, 10) == pytest.approx(1.0)

    missed = ["b"]
    only_a = {"a": 3}
    assert recall_at_k(missed, only_a, 10) == pytest.approx(0.0)
    assert precision_at_k(missed, only_a, 10) == pytest.approx(0.0)
    assert reciprocal_rank(missed, only_a, 10) == pytest.approx(0.0)
    assert ndcg_at_k(missed, only_a, 10) == pytest.approx(0.0)


def test_no_relevant_documents_return_none() -> None:
    for grades in ({}, {"b": 0}):
        assert ndcg_at_k(RANKED, grades, 3) is None
        assert recall_at_k(RANKED, grades, 3) is None
        assert reciprocal_rank(RANKED, grades, 3) is None
    assert mrr([RANKED], [{"b": 0}], 3) is None
    # precision 对无相关文档有定义，恒为 0，不返回 None。
    assert precision_at_k(RANKED, {"b": 0}, 3) == pytest.approx(0.0)


def test_aggregation_skips_none_and_counts() -> None:
    rankings = [RANKED, ["b"], ["x"]]
    grades_list: list[dict[str, int]] = [GRADES, {"a": 3}, {"z": 0}]
    per_rr = [
        reciprocal_rank(ranked, grades, 3)
        for ranked, grades in zip(rankings, grades_list, strict=True)
    ]
    # 第三条没有相关文档 → None；前两条分别是 1 和 0，均值不把 None 当 0。
    assert per_rr[0] == pytest.approx(1.0)
    assert per_rr[1] == pytest.approx(0.0)
    assert per_rr[2] is None
    assert sum(value is None for value in per_rr) == 1
    assert mrr(rankings, grades_list, 3) == pytest.approx(0.5)

    per_ndcg = [
        ndcg_at_k(ranked, grades, 3) for ranked, grades in zip(rankings, grades_list, strict=True)
    ]
    assert per_ndcg[2] is None
    assert sum(value is None for value in per_ndcg) == 1
    defined = [value for value in per_ndcg if value is not None]
    assert mean_defined(per_ndcg) == pytest.approx(sum(defined) / len(defined))
    assert mean_defined([None, None]) is None
