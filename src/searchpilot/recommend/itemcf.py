"""ItemCF：用户历史内物品共现的余弦相似度邻居。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from heapq import nsmallest
from typing import Any

NeighborMap = dict[str, list[tuple[str, float]]]


class ItemCFModel:
    """基于共现余弦相似度的 ItemCF。

    ``fit`` 共现计数复杂度为 ``O(Σ_u |H_u|²)``，逐物品处理倒排历史，
    不保留完整共现图。正数 k 的邻居选择为 ``O(Σ_i d_i log(k + 1))``，
    空间为 ``O(Σ_u |H_u| + max_i d_i + I*k)``；H 为去重历史，d 为邻居数。
    负数 k 保留 Python 切片语义，需要完整排序且可能保留每行大部分邻居。
    """

    def __init__(self) -> None:
        self._neighbors: NeighborMap = {}

    def fit(
        self,
        histories: Mapping[str, Sequence[str]],
        *,
        top_n_neighbors: int = 50,
        min_cooccurrence: int = 1,
    ) -> None:
        item_histories: dict[str, list[list[str]]] = {}
        for history in histories.values():
            unique_items = list(dict.fromkeys(history))
            for item in unique_items:
                item_histories.setdefault(item, []).append(unique_items)

        neighbors: NeighborMap = {}
        for item_id, containing_histories in item_histories.items():
            # 每份去重历史仅贡献一次；Counter 在当前行累加，避免全局 pair 字典。
            counts: Counter[str] = Counter()
            for history in containing_histories:
                counts.update(history)
            del counts[item_id]

            scores: list[tuple[str, float]] = []
            for neighbor_id, count in counts.items():
                if count < min_cooccurrence:
                    continue
                denom = (len(containing_histories) * len(item_histories[neighbor_id])) ** 0.5
                scores.append((neighbor_id, count / denom))
            if not scores:
                continue

            if top_n_neighbors > 0:
                neighbors[item_id] = nsmallest(top_n_neighbors, scores, key=lambda p: (-p[1], p[0]))
            else:
                neighbors[item_id] = sorted(scores, key=lambda p: (-p[1], p[0]))[:top_n_neighbors]
        self._neighbors = neighbors

    def predict(
        self,
        history: Sequence[str],
        exclude: frozenset[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        """用历史物品的近邻分数聚合候选；无历史时返回空列表。"""
        if not history:
            return []

        history_set = frozenset(history)
        aggregated: dict[str, float] = {}

        for seed in history:
            for neighbor_id, sim in self._neighbors.get(seed, ()):
                if neighbor_id in exclude or neighbor_id in history_set:
                    continue
                aggregated[neighbor_id] = aggregated.get(neighbor_id, 0.0) + sim

        ranked = sorted(aggregated.items(), key=lambda p: (-p[1], p[0]))
        return ranked[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "neighbors": {
                item_id: [[nbr, score] for nbr, score in pairs]
                for item_id, pairs in self._neighbors.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ItemCFModel:
        model = cls()
        raw = data.get("neighbors", {})
        model._neighbors = {
            str(item_id): [(str(nbr), float(score)) for nbr, score in pairs]
            for item_id, pairs in raw.items()
        }
        return model
