"""官方目录切分和通用时间分位切分。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def split_for_directory(directory: Path) -> str:
    """把 MIND 官方 train/dev 目录名映射到稳定 split 值。"""
    split = directory.name.lower()
    if split not in {"train", "dev"}:
        raise ValueError(f"expected a train or dev directory, got: {directory}")
    return split


def split_by_shown_at_quantile(
    impressions: pd.DataFrame, train_fraction: float = 0.8
) -> pd.DataFrame:
    """按 shown_at 分位点切分；边界时刻归入 train，之后归入 dev。"""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    result = impressions.copy()
    if result.empty:
        result["split"] = pd.Series(dtype="string")
        return result
    shown_at = pd.to_datetime(result["shown_at"], utc=True)
    ordered = shown_at.sort_values(kind="stable")
    boundary_index = int((len(ordered) - 1) * train_fraction)
    boundary = ordered.iloc[boundary_index]
    result["split"] = pd.Series(
        ["train" if value <= boundary else "dev" for value in shown_at],
        index=result.index,
        dtype="string",
    )
    return result


def build_user_history(train_behaviors: pd.DataFrame) -> pd.DataFrame:
    """取每个用户在 train 内按时间排序后的最后一条点击历史。"""
    required = {"user_id", "shown_at", "history"}
    missing = required.difference(train_behaviors.columns)
    if missing:
        raise ValueError(f"missing behavior columns: {sorted(missing)}")
    if train_behaviors.empty:
        return pd.DataFrame(
            {
                "user_id": pd.Series(dtype="string"),
                "history": pd.Series(dtype="object"),
            }
        )
    ordered = train_behaviors.sort_values(["user_id", "shown_at", "impression_id"], kind="stable")
    latest = ordered.groupby("user_id", sort=True, as_index=False).tail(1)
    result = latest[["user_id", "history"]].sort_values("user_id", kind="stable")
    result = result.reset_index(drop=True)
    result["user_id"] = result["user_id"].astype("string")
    result["history"] = result["history"].map(list)
    return result
