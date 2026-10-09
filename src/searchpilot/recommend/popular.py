"""全局热门模型：按 train 切分内点击（或曝光）计数排序。"""

from __future__ import annotations

from typing import Any

import pandas as pd

ScoreMap = dict[str, float]


class PopularModel:
    """按物品热度排序的简单回退模型。"""

    def __init__(self) -> None:
        self._scores: ScoreMap = {}

    def fit(
        self,
        impressions: pd.DataFrame,
        *,
        use_clicks: bool = True,
    ) -> None:
        """在 impressions 上计分；默认只统计 ``clicked==1``，否则按曝光行数。"""
        train = impressions.loc[impressions["split"] == "train"]
        if use_clicks:
            clicked = train.loc[train["clicked"] == 1]
            counts = clicked.groupby("item_id", sort=False).size()
        else:
            counts = train.groupby("item_id", sort=False).size()
        self._scores = {str(item_id): float(count) for item_id, count in counts.items()}

    def predict(
        self,
        exclude: frozenset[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        """返回 ``(item_id, score)``，分数降序，并列按 ``item_id`` 升序。"""
        candidates: list[tuple[str, float]] = [
            (item_id, score) for item_id, score in self._scores.items() if item_id not in exclude
        ]
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        return candidates[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {"scores": dict(self._scores)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PopularModel:
        model = cls()
        raw = data.get("scores", {})
        model._scores = {str(k): float(v) for k, v in raw.items()}
        return model
