"""CTR 指标：AUC、LogLoss、等宽 ECE 与校准曲线。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def expected_calibration_error(
    labels: Sequence[float], probabilities: Sequence[float], *, bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """10 桶 ECE。空桶跳过。返回 (ece, 曲线点)。"""
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(y.size)
    if total == 0.0:
        return 0.0, []
    error = 0.0
    curve: list[dict[str, float]] = []
    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]
        right_closed = index == bins - 1
        mask = (p >= left) & (p <= right) if right_closed else (p >= left) & (p < right)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        error += (count / total) * abs(accuracy - confidence)
        curve.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": float(count),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return error, curve


def univariate_separation(
    labels: Sequence[float], feature: Sequence[float]
) -> dict[str, float | int | None]:
    """把单个特征当作分数。AUC 低于 0.5 表示数值越大越不容易点击。"""
    y = np.asarray(labels, dtype=np.float64)
    x = np.asarray(feature, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError("labels and feature must have the same shape")
    auc = binary_auc(y, x)
    clicked = y == 1
    zero = x <= 0
    return {
        "n": int(y.size),
        "n_clicked": int(clicked.sum()),
        "click_rate": float(y.mean()) if y.size else None,
        "auc": auc,
        "mean_when_clicked": float(x[clicked].mean()) if clicked.any() else None,
        "mean_when_unclicked": float(x[~clicked].mean()) if (~clicked).any() else None,
        "n_feature_zero": int(zero.sum()),
        "click_rate_when_zero": float(y[zero].mean()) if zero.any() else None,
        "click_rate_when_positive": float(y[~zero].mean()) if (~zero).any() else None,
    }


def binary_auc(
    labels: Sequence[float] | np.ndarray, probabilities: Sequence[float] | np.ndarray
) -> float | None:
    """两类都出现时返回 AUC，否则 None。"""
    y = np.asarray(labels, dtype=np.int64)
    if np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, np.asarray(probabilities, dtype=np.float64)))


def binary_log_loss(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(log_loss(np.asarray(labels, dtype=np.int64), p, labels=[0, 1]))
