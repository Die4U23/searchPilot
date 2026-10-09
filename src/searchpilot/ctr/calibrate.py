"""校准器只应在验证集上拟合。温度缩放是默认方法。"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, steps: int = 200) -> float:
    """最小化验证集 NLL，返回 T > 0。"""
    x = torch.tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    y = torch.tensor(np.asarray(labels, dtype=np.float64), dtype=torch.float64)
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=steps)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        temperature = log_t.exp().clamp(min=1e-3, max=100.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x / temperature, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(min=1e-3, max=100.0))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return sigmoid(np.asarray(logits, dtype=np.float64) / float(temperature))


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """对 logit 做逻辑回归，返回 (coef, intercept)。"""
    model = LogisticRegression(solver="lbfgs")
    model.fit(
        np.asarray(logits, dtype=np.float64).reshape(-1, 1), np.asarray(labels, dtype=np.int64)
    )
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(logits: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    return sigmoid(coef * np.asarray(logits, dtype=np.float64) + intercept)


def fit_isotonic(probabilities: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    model.fit(np.asarray(probabilities, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    xs = [float(v) for v in model.X_thresholds_]
    ys = [float(v) for v in model.y_thresholds_]
    return xs, ys


def apply_isotonic(
    probabilities: np.ndarray, thresholds_x: list[float], thresholds_y: list[float]
) -> np.ndarray:
    if len(thresholds_x) != len(thresholds_y) or not thresholds_x:
        raise ValueError("isotonic thresholds must be a non-empty aligned pair")
    return np.interp(
        np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0),
        np.asarray(thresholds_x, dtype=np.float64),
        np.asarray(thresholds_y, dtype=np.float64),
    )
