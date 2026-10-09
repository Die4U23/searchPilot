"""排序评估指标（纯函数，手算可验）。

输入统一为 ``(ranked_item_ids, {item_id: grade})``：

- ``ranked`` 是系统返回的 item_id 序列（名次从 1 起）；
- ``grades`` 是该查询的标注，``grade ∈ 0–3``；**grade ≥ 1 视为相关**，缺失即 0。

「无相关文档」的查询（标注里没有任何 grade ≥ 1）对 nDCG / Recall / MRR 没有定义：
这些函数返回 ``None`` 而不是 0，由上层聚合时**跳过并计数**（``skipped_no_relevant``）。
这样做的原因是 ``no_answer`` 类查询本来就期望零结果，把它们算成 0 会系统性拉低指标、
把「正确地什么都没返回」记成失败；它们的正确性由零结果率单独衡量。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

RELEVANT_MIN_GRADE = 1


def gain(grade: int) -> float:
    """nDCG 的增益：``2^grade − 1``（grade 0 → 0，1 → 1，2 → 3，3 → 7）。"""
    return float(2**grade - 1) if grade > 0 else 0.0


def has_relevant(grades: Mapping[str, int]) -> bool:
    return any(g >= RELEVANT_MIN_GRADE for g in grades.values())


def relevant_items(grades: Mapping[str, int]) -> set[str]:
    return {item for item, g in grades.items() if g >= RELEVANT_MIN_GRADE}


def dcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """``Σ_{i=1..k} gain(grade_i) / log2(i + 1)``；``ranked`` 短于 ``k`` 时只累加实际存在的名次。"""
    if k <= 0:
        raise ValueError("k must be positive")
    total = 0.0
    for i, item_id in enumerate(ranked[:k], start=1):
        g = grades.get(item_id, 0)
        if g > 0:
            total += gain(g) / math.log2(i + 1)
    return total


def ideal_dcg_at_k(grades: Mapping[str, int], k: int) -> float:
    """理想 DCG：把所有标注按 grade 降序排列后的 DCG@k。"""
    if k <= 0:
        raise ValueError("k must be positive")
    ordered = sorted((g for g in grades.values() if g > 0), reverse=True)
    return sum(gain(g) / math.log2(i + 1) for i, g in enumerate(ordered[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float | None:
    """nDCG@k = DCG@k / IDCG@k；无相关文档返回 ``None``。"""
    idcg = ideal_dcg_at_k(grades, k)
    if idcg == 0.0:
        return None
    return dcg_at_k(ranked, grades, k) / idcg


def recall_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float | None:
    """前 k 名中命中的相关文档数 / 相关文档总数；无相关文档返回 ``None``。"""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = relevant_items(grades)
    if not relevant:
        return None
    hit = sum(1 for item_id in dict.fromkeys(ranked[:k]) if item_id in relevant)
    return hit / len(relevant)


def precision_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """前 k 名中相关文档数 / k（分母固定为 k，结果不足 k 条也不缩小分母）。

    无相关文档时自然为 0.0（此时若系统也返回空列表，precision 为 0 并不代表失败——
    请结合零结果率解读）。
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = relevant_items(grades)
    hit = sum(1 for item_id in dict.fromkeys(ranked[:k]) if item_id in relevant)
    return hit / k


def reciprocal_rank(
    ranked: Sequence[str], grades: Mapping[str, int], k: int | None = None
) -> float | None:
    """首个相关文档名次的倒数（限制在前 k 名内；``k=None`` 不限制）。

    前 k 名无相关文档返回 0.0；该查询根本没有相关文档返回 ``None``。
    """
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    relevant = relevant_items(grades)
    if not relevant:
        return None
    window = ranked if k is None else ranked[:k]
    for i, item_id in enumerate(window, start=1):
        if item_id in relevant:
            return 1.0 / i
    return 0.0


def mrr(
    rankings: Sequence[Sequence[str]],
    grades_list: Sequence[Mapping[str, int]],
    k: int | None = None,
) -> float | None:
    """多条查询的平均倒数名次；跳过没有相关文档的查询，全部被跳过时返回 ``None``。"""
    if len(rankings) != len(grades_list):
        raise ValueError("rankings and grades_list must have the same length")
    values = [
        rr
        for rr in (reciprocal_rank(r, g, k) for r, g in zip(rankings, grades_list, strict=True))
        if rr is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def mean_defined(values: Sequence[float | None]) -> float | None:
    """忽略 ``None`` 求均值；没有可用值返回 ``None``。"""
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)
