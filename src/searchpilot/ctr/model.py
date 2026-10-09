"""CTR 模型：LR 基线与一层 16 维的 MLP。两者都输出 logit。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

SEED = 20261009
EPOCHS = 15
LEARNING_RATE = 0.05


class LogisticModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class MlpModel(nn.Module):
    def __init__(self, width: int, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(features))).squeeze(-1)


@dataclass(frozen=True, slots=True)
class LinearWeights:
    weight: list[float]
    bias: float


@dataclass(frozen=True, slots=True)
class MlpWeights:
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]


def train_binary(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> nn.Module:
    torch.manual_seed(seed)
    x = torch.tensor(np.asarray(features, dtype=np.float32))
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_logits(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = model(torch.tensor(np.asarray(features, dtype=np.float32)))
    return values.detach().cpu().numpy().astype(np.float64)


def export_linear(model: LogisticModel) -> LinearWeights:
    weight = model.linear.weight.detach().cpu().numpy().reshape(-1)
    bias = float(model.linear.bias.detach().cpu().numpy().reshape(-1)[0])
    return LinearWeights(weight=[float(v) for v in weight], bias=bias)


def export_mlp(model: MlpModel) -> MlpWeights:
    w1 = model.hidden.weight.detach().cpu().numpy()
    b1 = model.hidden.bias.detach().cpu().numpy()
    w2 = model.output.weight.detach().cpu().numpy()
    b2 = model.output.bias.detach().cpu().numpy()
    return MlpWeights(
        w1=[[float(v) for v in row] for row in w1],
        b1=[float(v) for v in b1],
        w2=[[float(v) for v in row] for row in w2],
        b2=[float(v) for v in b2],
    )
