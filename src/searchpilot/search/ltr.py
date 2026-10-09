"""Pairwise 线性 LTR。

特征顺序固定为：

``bm25_score``、``vector_score``、``bm25_rank``、``vector_rank``、
``log1p_train_clicks``、``freshness_days``、``category_match``、``title_chars``。

训练是同一查询内 grade 不同的文档对，logistic pairwise（``softplus(-(s_hi - s_lo))``）。
超参写死，不在 val / test 上挑选。推理分数是标准化后的 ``w·x+b``。
产物只有 JSON：``weights.json``、``meta.json``、``popularity.json``。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from searchpilot.search.errors import SearchNotReadyError

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "bm25_score",
    "vector_score",
    "bm25_rank",
    "vector_rank",
    "log1p_train_clicks",
    "freshness_days",
    "category_match",
    "title_chars",
)
RETRIEVAL_DEPTH: Final[int] = 100
CATEGORY_MODE_K: Final[int] = 10
SEED: Final[int] = 20261009
EPOCHS: Final[int] = 40
LEARNING_RATE: Final[float] = 0.05

WEIGHTS_FILENAME: Final[str] = "weights.json"
META_FILENAME: Final[str] = "meta.json"
POPULARITY_FILENAME: Final[str] = "popularity.json"

__all__ = [
    "CATEGORY_MODE_K",
    "EPOCHS",
    "FEATURE_NAMES",
    "LEARNING_RATE",
    "LtrModel",
    "LtrPrep",
    "QuerySignals",
    "RETRIEVAL_DEPTH",
    "SEED",
    "assemble_ltr_model",
    "feature_matrix",
    "fit_standardizer",
    "freshness_days",
    "labels_for_queries",
    "load_ltr",
    "ltr_artifact_dir",
    "mode_category",
    "pairwise_indices",
    "rank_by_score",
    "save_ltr",
    "train_click_counts",
    "train_pairwise",
]


def ltr_artifact_dir(artifact_root: Path) -> Path:
    """``<artifact_root>/search/ltr``。"""
    return artifact_root / "search" / "ltr"


def mode_category(categories: Sequence[str]) -> str | None:
    """序列的众数（casefold）。并列时取最先出现的类目。空序列返回 ``None``。"""
    folded = [category.casefold() for category in categories]
    if not folded:
        return None
    return Counter(folded).most_common(1)[0][0]


def freshness_days(first_seen: datetime | None, reference: datetime | None) -> float:
    """相对 ``reference`` 的天数。缺时间、或文档不早于参考时刻时为 0。"""
    if first_seen is None or reference is None:
        return 0.0
    seen = _as_utc(first_seen)
    origin = _as_utc(reference)
    seconds = (origin - seen).total_seconds()
    if seconds <= 0.0:
        return 0.0
    return seconds / 86400.0


def train_click_counts(frame: pd.DataFrame) -> dict[str, int]:
    """只数 ``split == train`` 且 ``clicked == 1`` 的曝光。dev 与未点击行不进入热度。"""
    required = {"item_id", "clicked", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"impressions missing columns: {sorted(missing)}")
    split = frame["split"].astype(str)
    clicked = frame["clicked"].astype(int)
    chosen = frame.loc[(split == "train") & (clicked == 1), "item_id"].astype(str)
    counts = chosen.value_counts()
    return {str(item_id): int(count) for item_id, count in counts.items()}


def labels_for_queries(
    labels: Mapping[str, Mapping[str, int]], query_ids: set[str]
) -> dict[str, dict[str, int]]:
    """丢掉不在 ``query_ids`` 里的标注。训练传入 train 查询 id，避免 val/test 进入损失。"""
    return {
        query_id: {item_id: int(grade) for item_id, grade in grades.items()}
        for query_id, grades in labels.items()
        if query_id in query_ids
    }


@dataclass(frozen=True, slots=True)
class QuerySignals:
    """一条查询的两路分数与名次。名次只覆盖 top-``RETRIEVAL_DEPTH``，缺席不在字典里。"""

    bm25_score: Mapping[str, float]
    bm25_rank: Mapping[str, int]
    vector_score: Mapping[str, float]
    vector_rank: Mapping[str, int]
    category_mode: str | None


@dataclass(frozen=True, slots=True)
class LtrPrep:
    """``prepare_ltr`` 的输出：RRF 候选，以及这些候选和额外文档的特征信号。"""

    signals: QuerySignals
    candidates: tuple[str, ...]


def feature_matrix(
    item_ids: Sequence[str],
    signals: QuerySignals,
    *,
    categories: Mapping[str, str],
    clicks: Mapping[str, int],
    title_chars: Mapping[str, int],
    first_seen_at: Mapping[str, datetime],
    reference_time: datetime | None,
    depth: int = RETRIEVAL_DEPTH,
) -> np.ndarray:
    """按 ``FEATURE_NAMES`` 组行。不在 top-``depth`` 里的名次记为 ``depth + 1``。"""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    missing_rank = float(depth + 1)
    mode = signals.category_mode
    rows: list[list[float]] = []
    for item_id in item_ids:
        category = categories.get(item_id, "")
        matched = 1.0 if mode is not None and category.casefold() == mode else 0.0
        rows.append(
            [
                float(signals.bm25_score.get(item_id, 0.0)),
                float(signals.vector_score.get(item_id, 0.0)),
                float(signals.bm25_rank.get(item_id, missing_rank)),
                float(signals.vector_rank.get(item_id, missing_rank)),
                math.log1p(max(int(clicks.get(item_id, 0)), 0)),
                freshness_days(first_seen_at.get(item_id), reference_time),
                matched,
                float(title_chars.get(item_id, 0)),
            ]
        )
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用这批行的均值 / 标准差做 z-score。标准差为 0 的列改成 1，避免除零。"""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected shape (n, {len(FEATURE_NAMES)})")
    if array.shape[0] == 0:
        raise ValueError("no feature rows to fit")
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (array - mean) / std, mean, std


def pairwise_indices(grades: Sequence[int], groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """同一 ``groups`` 内 grade 不同的对。返回 ``(higher_index, lower_index)``。"""
    if len(grades) != len(groups):
        raise ValueError("grades and groups must have the same length")
    buckets: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        buckets.setdefault(group, []).append(index)
    higher: list[int] = []
    lower: list[int] = []
    for group in sorted(buckets):
        members = buckets[group]
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                i = members[left]
                j = members[right]
                grade_i = int(grades[i])
                grade_j = int(grades[j])
                if grade_i == grade_j:
                    continue
                if grade_i > grade_j:
                    higher.append(i)
                    lower.append(j)
                else:
                    higher.append(j)
                    lower.append(i)
    return np.asarray(higher, dtype=np.int64), np.asarray(lower, dtype=np.int64)


def train_pairwise(
    features: np.ndarray,
    higher_index: np.ndarray,
    lower_index: np.ndarray,
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[np.ndarray, float, float, float]:
    """在已标准化的特征上训练 ``nn.Linear``。返回权重、偏置、首步损失、末步损失。

    ``epochs`` / ``learning_rate`` / ``seed`` 使用固定默认值。调用方不要拿 test 指标改它们。
    """
    import torch

    array = np.asarray(features, dtype=np.float64)
    hi = np.asarray(higher_index, dtype=np.int64)
    lo = np.asarray(lower_index, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected shape (n, {len(FEATURE_NAMES)})")
    if hi.shape != lo.shape or hi.ndim != 1:
        raise ValueError("pair index arrays must be 1-d and aligned")
    if hi.size == 0:
        raise ValueError("no pairwise training pairs")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    inputs = torch.tensor(array, dtype=torch.float64)
    hi_t = torch.tensor(hi, dtype=torch.long)
    lo_t = torch.tensor(lo, dtype=torch.long)
    layer = torch.nn.Linear(array.shape[1], 1).double()
    optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate)

    def _loss() -> torch.Tensor:
        scores = layer(inputs).squeeze(-1)
        return torch.nn.functional.softplus(-(scores[hi_t] - scores[lo_t])).mean()

    first = float(_loss().detach())
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss()
        loss.backward()
        optimizer.step()
    last = float(_loss().detach())
    with torch.no_grad():
        weight = layer.weight.detach().cpu().numpy().reshape(-1).astype(np.float64).copy()
        bias = float(layer.bias.detach().cpu().numpy().reshape(-1)[0])
    return weight, bias, first, last


@dataclass(frozen=True, slots=True)
class LtrModel:
    """线性重排器。``score_rows`` 先做训练时记下的 z-score，再算 ``w·x+b``。"""

    weights: tuple[float, ...]
    bias: float
    mean: tuple[float, ...]
    std: tuple[float, ...]
    reference_time: datetime | None
    clicks: Mapping[str, int]
    title_chars: Mapping[str, int]
    first_seen_at: Mapping[str, datetime]
    model_version: str

    def score_rows(self, raw: np.ndarray) -> np.ndarray:
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.weights):
            raise ValueError(f"expected shape (n, {len(self.weights)})")
        if array.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        safe = np.where(np.abs(std) < 1e-12, 1.0, std)
        scaled = (array - mean) / safe
        weights = np.asarray(self.weights, dtype=np.float64)
        return scaled @ weights + float(self.bias)


def rank_by_score(
    item_ids: Sequence[str], scores: Sequence[float] | np.ndarray, limit: int
) -> list[tuple[str, float]]:
    """分数降序，并列按 ``item_id`` 升序，截断到 ``limit``。"""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if len(scores) != len(item_ids):
        raise ValueError("scores must align with item_ids")
    order = sorted(range(len(item_ids)), key=lambda i: (-float(scores[i]), item_ids[i]))
    return [(item_ids[i], float(scores[i])) for i in order[:limit]]


def assemble_ltr_model(
    weights: Sequence[float],
    bias: float,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    reference_time: datetime | None,
    clicks: Mapping[str, int],
    title_chars: Mapping[str, int],
    first_seen_at: Mapping[str, datetime],
) -> LtrModel:
    """由训练结果组装模型，并按评分内容计算 ``ltr-`` 版本号。"""
    seen = {str(item_id): _as_utc(stamp) for item_id, stamp in first_seen_at.items()}
    payload = _scoring_payload(
        weights=weights,
        bias=bias,
        mean=mean,
        std=std,
        reference_time=_iso(reference_time),
        clicks=clicks,
        title_chars=title_chars,
        first_seen_at={item_id: _iso_required(stamp) for item_id, stamp in seen.items()},
    )
    return LtrModel(
        weights=tuple(payload["weights"]),
        bias=float(payload["bias"]),
        mean=tuple(payload["mean"]),
        std=tuple(payload["std"]),
        reference_time=_optional_time(payload["reference_time"]),
        clicks=dict(payload["clicks"]),
        title_chars=dict(payload["title_chars"]),
        first_seen_at=seen,
        model_version=ltr_model_version(payload),
    )


def save_ltr(directory: Path, model: LtrModel, *, stats: Mapping[str, Any] | None = None) -> None:
    """写入三份 JSON。``stats`` 只进 meta，不参与 ``model_version``。"""
    payload = _payload_from_model(model)
    version = ltr_model_version(payload)
    if version != model.model_version:
        raise ValueError("model_version does not match scoring payload")
    directory.mkdir(parents=True, exist_ok=True)
    weights_doc = {
        "feature_names": list(FEATURE_NAMES),
        "weights": payload["weights"],
        "bias": payload["bias"],
        "mean": payload["mean"],
        "std": payload["std"],
    }
    popularity_doc = {"split": "train", "clicked": 1, "counts": payload["clicks"]}
    meta_doc: dict[str, Any] = {
        "model_version": version,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference_time": payload["reference_time"],
        "retrieval_depth": RETRIEVAL_DEPTH,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "feature_names": list(FEATURE_NAMES),
        "title_chars": payload["title_chars"],
        "first_seen_at": payload["first_seen_at"],
    }
    if stats is not None:
        meta_doc["train"] = {str(key): value for key, value in stats.items()}
    _write_json(directory / WEIGHTS_FILENAME, weights_doc)
    _write_json(directory / POPULARITY_FILENAME, popularity_doc)
    _write_json(directory / META_FILENAME, meta_doc)


def load_ltr(directory: Path) -> LtrModel:
    """读取 LTR 产物。缺文件、特征顺序不对或版本哈希对不上时抛 :class:`SearchNotReadyError`。"""
    paths = (
        directory / WEIGHTS_FILENAME,
        directory / META_FILENAME,
        directory / POPULARITY_FILENAME,
    )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise SearchNotReadyError(
            f"LTR artifacts missing in {directory}: {', '.join(missing)}. Run scripts/train_ltr.py."
        )
    try:
        weights_doc = json.loads(paths[0].read_text(encoding="utf-8"))
        meta_doc = json.loads(paths[1].read_text(encoding="utf-8"))
        popularity_doc = json.loads(paths[2].read_text(encoding="utf-8"))
        names = [str(name) for name in weights_doc["feature_names"]]
        if names != list(FEATURE_NAMES):
            raise SearchNotReadyError(f"LTR feature order mismatch in {directory}")
        if [str(name) for name in meta_doc["feature_names"]] != list(FEATURE_NAMES):
            raise SearchNotReadyError(f"LTR feature order mismatch in {directory}")
        clicks = {str(k): int(v) for k, v in dict(popularity_doc["counts"]).items()}
        title_chars = {str(k): int(v) for k, v in dict(meta_doc["title_chars"]).items()}
        first_seen_text = {str(k): str(v) for k, v in dict(meta_doc["first_seen_at"]).items()}
        payload = _scoring_payload(
            weights=[float(v) for v in weights_doc["weights"]],
            bias=float(weights_doc["bias"]),
            mean=[float(v) for v in weights_doc["mean"]],
            std=[float(v) for v in weights_doc["std"]],
            reference_time=(
                None if meta_doc["reference_time"] is None else str(meta_doc["reference_time"])
            ),
            clicks=clicks,
            title_chars=title_chars,
            first_seen_at=first_seen_text,
        )
        version = ltr_model_version(payload)
        if version != str(meta_doc["model_version"]):
            raise SearchNotReadyError(f"LTR model_version does not match contents: {directory}")
        if int(meta_doc["retrieval_depth"]) != RETRIEVAL_DEPTH:
            raise SearchNotReadyError(f"LTR retrieval_depth is not {RETRIEVAL_DEPTH}: {directory}")
        return LtrModel(
            weights=tuple(payload["weights"]),
            bias=float(payload["bias"]),
            mean=tuple(payload["mean"]),
            std=tuple(payload["std"]),
            reference_time=_optional_time(payload["reference_time"]),
            clicks=dict(payload["clicks"]),
            title_chars=dict(payload["title_chars"]),
            first_seen_at={item_id: _parse_time(text) for item_id, text in first_seen_text.items()},
            model_version=version,
        )
    except SearchNotReadyError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SearchNotReadyError(f"LTR artifacts in {directory} are corrupt: {exc}") from exc


def ltr_model_version(payload: Mapping[str, Any]) -> str:
    """``ltr-`` + 评分内容 JSON 的 SHA-256 前 8 位。不含 ``built_at`` 与训练统计。"""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "ltr-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def _payload_from_model(model: LtrModel) -> dict[str, Any]:
    return _scoring_payload(
        weights=model.weights,
        bias=model.bias,
        mean=model.mean,
        std=model.std,
        reference_time=_iso(model.reference_time),
        clicks=model.clicks,
        title_chars=model.title_chars,
        first_seen_at={
            item_id: _iso_required(stamp) for item_id, stamp in model.first_seen_at.items()
        },
    )


def _scoring_payload(
    *,
    weights: Sequence[float],
    bias: float,
    mean: Sequence[float],
    std: Sequence[float],
    reference_time: str | None,
    clicks: Mapping[str, int],
    title_chars: Mapping[str, int],
    first_seen_at: Mapping[str, str],
) -> dict[str, Any]:
    width = len(FEATURE_NAMES)
    if len(weights) != width or len(mean) != width or len(std) != width:
        raise ValueError(f"LTR vectors must have length {width}")
    positive_clicks = {
        str(item_id): int(count) for item_id, count in clicks.items() if int(count) > 0
    }
    return {
        "bias": float(bias),
        "clicks": {key: positive_clicks[key] for key in sorted(positive_clicks)},
        "feature_names": list(FEATURE_NAMES),
        "first_seen_at": {str(key): str(first_seen_at[key]) for key in sorted(first_seen_at)},
        "mean": [float(value) for value in mean],
        "reference_time": reference_time,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "std": [float(value) for value in std],
        "title_chars": {str(key): int(title_chars[key]) for key in sorted(title_chars)},
        "weights": [float(value) for value in weights],
    }


def _as_utc(stamp: datetime) -> datetime:
    aware = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    return datetime(
        aware.year,
        aware.month,
        aware.day,
        aware.hour,
        aware.minute,
        aware.second,
        aware.microsecond,
        tzinfo=UTC,
    )


def _iso(stamp: datetime | None) -> str | None:
    if stamp is None:
        return None
    return _as_utc(stamp).isoformat()


def _iso_required(stamp: datetime) -> str:
    text = _iso(stamp)
    if text is None:
        raise ValueError("missing timestamp")
    return text


def _optional_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    return _parse_time(value)


def _parse_time(value: str) -> datetime:
    return _as_utc(datetime.fromisoformat(value))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
