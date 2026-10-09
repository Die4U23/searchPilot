"""在线 CTR 打分：从 ``artifacts/ctr/`` 加载 LR + 温度校准，缺产物抛 ``CtrNotReadyError``。"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log1p_history_len",
    "category_match_rate",
    "log1p_train_clicks",
    "log1p_title_chars",
)
FEATURE_WIDTH: Final[int] = len(FEATURE_NAMES)
META_FILENAME: Final[str] = "meta.json"
LR_FILENAME: Final[str] = "lr.json"
TEMPERATURE_FILENAME: Final[str] = "temperature.json"
_STD_FLOOR: Final[float] = 1e-12


class CtrNotReadyError(Exception):
    """CTR 产物缺失或损坏，API 层映射为 503 ``NOT_READY``。"""


@dataclass(frozen=True, slots=True)
class CtrScoreHit:
    item_id: str
    pctr: float
    pctr_calibrated: float
    ecpm: float | None


def ctr_artifact_dir(artifact_root: Path) -> Path:
    """``<artifact_root>/ctr``。"""
    return Path(artifact_root) / "ctr"


def load_ctr(artifact_dir: Path) -> CtrRuntime:
    """从 ``artifact_dir/ctr/`` 读取 meta / lr / temperature；缺字段或损坏则抛错。"""
    directory = ctr_artifact_dir(artifact_dir)
    paths = (
        directory / META_FILENAME,
        directory / LR_FILENAME,
        directory / TEMPERATURE_FILENAME,
    )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise CtrNotReadyError(f"CTR artifacts missing in {directory}: {', '.join(missing)}")
    try:
        meta_doc = _read_json(paths[0])
        lr_doc = _read_json(paths[1])
        temperature_doc = _read_json(paths[2])
        names = [str(name) for name in meta_doc["feature_names"]]
        if names != list(FEATURE_NAMES):
            raise CtrNotReadyError(f"CTR feature_names must be {list(FEATURE_NAMES)}")
        mean = _as_float_vector(meta_doc["mean"], name="mean")
        std = _as_float_vector(meta_doc["std"], name="std")
        weight = _as_float_vector(lr_doc["weight"], name="weight")
        bias = _as_finite_float(lr_doc["bias"], name="bias")
        temperature = _as_finite_float(temperature_doc["T"], name="T")
        if temperature <= 0.0:
            raise CtrNotReadyError("temperature T must be positive")
        model_version = str(meta_doc["model_version"]).strip()
        calibrator_version = str(meta_doc["calibrator_version"]).strip()
        if not model_version or not calibrator_version:
            raise CtrNotReadyError("model_version and calibrator_version must be non-empty")
        tables = _load_feature_tables(directory / "feature_tables.json")
        mlp = _load_mlp(directory / "mlp.json")
    except CtrNotReadyError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise CtrNotReadyError(f"CTR artifacts in {directory} are corrupt: {exc}") from exc
    return CtrRuntime(
        model_version=model_version,
        calibrator_version=calibrator_version,
        mean=mean,
        std=std,
        weight=weight,
        bias=bias,
        temperature=temperature,
        feature_tables=tables,
        mlp=mlp,
    )


@dataclass(frozen=True, slots=True)
class CtrRuntime:
    """已加载的 LR + 温度校准器。特征侧车不在本产物包里时按全零特征打分。"""

    model_version: str
    calibrator_version: str
    mean: np.ndarray
    std: np.ndarray
    weight: np.ndarray
    bias: float
    temperature: float
    feature_tables: dict[str, object] | None = None
    mlp: dict[str, np.ndarray] | None = None

    @property
    def ctr_ready(self) -> bool:
        return True

    def score(
        self,
        user_id: str,
        candidates: Sequence[tuple[str, float | None]],
        *,
        calibrated: bool = True,
    ) -> list[CtrScoreHit]:
        if not user_id:
            raise ValueError("user_id must not be empty")
        if not candidates:
            return []
        features = self._feature_matrix(user_id, [item_id for item_id, _bid in candidates])
        std = np.where(np.abs(self.std) < _STD_FLOOR, 1.0, self.std)
        scaled = (features - self.mean) / std
        logits = self._logits(scaled)
        raw = _sigmoid(logits)
        calibrated_p = _sigmoid(logits / self.temperature)
        used = calibrated_p if calibrated else raw
        hits: list[CtrScoreHit] = []
        for index, (item_id, bid) in enumerate(candidates):
            pctr = float(raw[index])
            pctr_calibrated = float(calibrated_p[index])
            ecpm = None if bid is None else float(bid) * float(used[index])
            hits.append(
                CtrScoreHit(
                    item_id=item_id,
                    pctr=pctr,
                    pctr_calibrated=pctr_calibrated,
                    ecpm=ecpm,
                )
            )
        return hits

    def _feature_matrix(self, user_id: str, item_ids: list[str]) -> np.ndarray:
        from searchpilot.ctr.features import rows_to_matrix

        tables = self.feature_tables or {}
        return rows_to_matrix(
            [(user_id, item_id) for item_id in item_ids],
            user_history_len=tables.get("user_history_len", {}),  # type: ignore[arg-type]
            user_category_counts=tables.get("user_category_counts", {}),  # type: ignore[arg-type]
            item_clicks=tables.get("item_clicks", {}),  # type: ignore[arg-type]
            item_category=tables.get("item_category", {}),  # type: ignore[arg-type]
            item_title_chars=tables.get("item_title_chars", {}),  # type: ignore[arg-type]
        )

    def _logits(self, scaled: np.ndarray) -> np.ndarray:
        if self.mlp is None:
            return scaled @ self.weight + self.bias
        hidden = np.maximum(scaled @ self.mlp["w1"].T + self.mlp["b1"], 0.0)
        return (hidden @ self.mlp["w2"].T + self.mlp["b2"]).reshape(-1)


def _load_feature_tables(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    return {
        "item_clicks": {str(k): int(v) for k, v in dict(payload.get("item_clicks", {})).items()},
        "item_category": {
            str(k): str(v) for k, v in dict(payload.get("item_category", {})).items()
        },
        "item_title_chars": {
            str(k): int(v) for k, v in dict(payload.get("item_title_chars", {})).items()
        },
        "user_history_len": {
            str(k): int(v) for k, v in dict(payload.get("user_history_len", {})).items()
        },
        "user_category_counts": {
            str(user): {str(cat): int(count) for cat, count in dict(counts).items()}
            for user, counts in dict(payload.get("user_category_counts", {})).items()
        },
    }


def _load_mlp(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    return {
        "w1": np.asarray(payload["w1"], dtype=np.float64),
        "b1": np.asarray(payload["b1"], dtype=np.float64),
        "w2": np.asarray(payload["w2"], dtype=np.float64),
        "b2": np.asarray(payload["b2"], dtype=np.float64),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrNotReadyError(f"{path.name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise CtrNotReadyError(f"{path.name} must be a JSON object")
    return payload


def _as_float_vector(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != FEATURE_WIDTH:
        raise CtrNotReadyError(f"{name} must be a list of length {FEATURE_WIDTH}")
    try:
        array = np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CtrNotReadyError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise CtrNotReadyError(f"{name} must be finite")
    return array


def _as_finite_float(value: object, *, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CtrNotReadyError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise CtrNotReadyError(f"{name} must be finite")
    return number


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))
