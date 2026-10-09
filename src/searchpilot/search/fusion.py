"""排名融合：Reciprocal Rank Fusion（RRF）。

RRF 只看名次不看原始分，因此天然规避「BM25 原始分与余弦分不能直接相加」的问题
（PRD FR-1）。本轮只有 BM25 一路可用，这里先把融合实现与测试准备好，
供 P1 的 hybrid 模式直接调用。
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """融合多路排名。

    对每一路排名 ``i`` 中名次为 ``r``（从 1 起）的 ``item_id``，累加
    ``weights[i] / (k + r)``；某一路中重复出现的 ``item_id`` 只按首次出现的名次计一次。
    返回按融合分降序、并列按 ``item_id`` 升序的 ``[(item_id, score), ...]``，已去重。

    ``weights`` 缺省为全 1；长度必须与 ``rankings`` 一致，且不允许负值。
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights length {len(weights)} does not match rankings length {len(rankings)}"
        )
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        rank = 0
        for item_id in ranking:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
