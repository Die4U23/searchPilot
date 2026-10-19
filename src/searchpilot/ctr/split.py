"""Chronological evaluation partitions that never share an impression or timestamp."""

from __future__ import annotations

import pandas as pd


def split_dev_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the whole timestamp boundary nearest a balanced candidate-row split.

    Sampling may subsequently reduce either partition, but cannot place an
    impression in both. Fewer than two timestamps cannot form a temporal holdout.
    """
    required = ["impression_id", "shown_at"]
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("dev requires non-empty impression IDs and shown_at timestamps")
    ordered = frame.copy()
    ordered["shown_at"] = pd.to_datetime(ordered["shown_at"], errors="raise", utc=True)
    if ordered["shown_at"].isna().any():
        raise ValueError("dev shown_at must contain valid timestamps")
    if (ordered.groupby("impression_id")["shown_at"].nunique() != 1).any():
        raise ValueError("each impression must have exactly one shown_at timestamp")
    ordered = ordered.sort_values("shown_at", kind="mergesort")
    counts = ordered.groupby("shown_at", sort=True).size()
    if len(counts) < 2:
        raise ValueError("dev requires at least two distinct timestamps for val/test")
    # Exclude the last timestamp so both partitions remain non-empty. idxmin
    # selects the earlier boundary on a tie, making the choice deterministic.
    cumulative = counts.cumsum().iloc[:-1]
    boundary = (cumulative - len(ordered) / 2).abs().idxmin()
    before = ordered["shown_at"] <= boundary
    validation = ordered.loc[before].reset_index(drop=True)
    test = ordered.loc[~before].reset_index(drop=True)
    if set(validation["impression_id"]) & set(test["impression_id"]):
        raise ValueError("val/test impression overlap")
    return validation, test
