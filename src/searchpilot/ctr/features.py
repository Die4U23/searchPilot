"""CTR 特征。点击计数和用户历史只来自 train，当前行的 clicked 不是特征。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "log1p_history_len",
    "category_match_rate",
    "log1p_train_clicks",
    "log1p_title_chars",
)


def feature_vector(
    *,
    history_len: int,
    category_hits: int,
    train_clicks: int,
    title_chars: int,
) -> list[float]:
    length = max(int(history_len), 0)
    match = 0.0 if length == 0 else float(category_hits) / float(length)
    return [
        math.log1p(length),
        match,
        math.log1p(max(int(train_clicks), 0)),
        math.log1p(max(int(title_chars), 0)),
    ]


def rows_to_matrix(
    rows: list[tuple[str, str]],
    *,
    user_history_len: Mapping[str, int],
    user_category_counts: Mapping[str, Mapping[str, int]],
    item_clicks: Mapping[str, int],
    item_category: Mapping[str, str],
    item_title_chars: Mapping[str, int],
) -> np.ndarray:
    """``rows`` 是 ``(user_id, item_id)``。"""
    matrix = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for index, (user_id, item_id) in enumerate(rows):
        category = item_category.get(item_id, "")
        counts = user_category_counts.get(user_id, {})
        matrix[index] = feature_vector(
            history_len=user_history_len.get(user_id, 0),
            category_hits=int(counts.get(category, 0)),
            train_clicks=item_clicks.get(item_id, 0),
            title_chars=item_title_chars.get(item_id, 0),
        )
    return matrix


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(std) < 1e-12, 1.0, std)
    return (features - mean) / safe
