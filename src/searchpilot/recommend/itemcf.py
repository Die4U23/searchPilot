"""ItemCF：用户历史内物品共现的余弦相似度邻居。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

NeighborMap = dict[str, list[tuple[str, float]]]


class ItemCFModel:
    """基于共现余弦相似度的 ItemCF。

    ``fit`` 复杂度约 ``O(Σ_u |H_u|²)``（每用户历史内两两共现计数），
    在数万用户、数万物品、较短点击序列下可在单机完成；邻居截断为 ``top_n_neighbors``。
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
        item_user_count: dict[str, int] = {}
        cooc: dict[tuple[str, str], int] = {}

        for _user_id, history in histories.items():
            unique_items = list(dict.fromkeys(history))
            for item in unique_items:
                item_user_count[item] = item_user_count.get(item, 0) + 1
            n = len(unique_items)
            for i in range(n):
                for j in range(i + 1, n):
                    # 用局部变量做无序对规范化，避免改写外层循环变量导致后续 pair 错位
                    x, y = unique_items[i], unique_items[j]
                    key = (x, y) if x <= y else (y, x)
                    cooc[key] = cooc.get(key, 0) + 1

        neighbors: dict[str, dict[str, float]] = {}
        for (a, b), count in cooc.items():
            if count < min_cooccurrence:
                continue
            denom = (item_user_count[a] * item_user_count[b]) ** 0.5
            if denom <= 0:
                continue
            sim = count / denom
            neighbors.setdefault(a, {})[b] = sim
            neighbors.setdefault(b, {})[a] = sim

        self._neighbors = {}
        for item_id, nbr_scores in neighbors.items():
            ranked = sorted(nbr_scores.items(), key=lambda p: (-p[1], p[0]))
            self._neighbors[item_id] = ranked[:top_n_neighbors]

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
