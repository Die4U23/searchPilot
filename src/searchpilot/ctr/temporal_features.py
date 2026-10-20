"""Training features use only positive events strictly earlier than the sample."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from searchpilot.ctr.features import FEATURE_NAMES, feature_vector

FEATURE_POLICY = "strictly-earlier-positive-events-v1"


def training_features(
    train: pd.DataFrame,
    positions: list[int],
    *,
    item_category: dict[str, str],
    item_title_chars: dict[str, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve sample order and return frozen end-of-training online tables.

    Histories start empty at the observation-window boundary. They are observed
    positive events, not reconstructed full lifetime or last-known user histories.
    """
    required = {"impression_id", "user_id", "item_id", "clicked", "shown_at"}
    if not required.issubset(train.columns):
        raise ValueError(f"training impressions require {sorted(required)}")
    frame = train.reset_index(drop=True)
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("training impressions must be nonempty with no missing required values")
    if not frame["clicked"].isin([0, 1]).all():
        raise ValueError("clicked must be binary")
    times = pd.to_datetime(frame["shown_at"], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("shown_at must not be missing")
    groups = pd.DataFrame({"impression_id": frame["impression_id"].astype(str), "time": times})
    if (groups.groupby("impression_id")["time"].nunique() > 1).any():
        raise ValueError("one impression must not span multiple timestamps")
    if any(
        isinstance(position, bool)
        or not isinstance(position, Integral)
        or not 0 <= position < len(frame)
        for position in positions
    ) or len(set(positions)) != len(positions):
        raise ValueError("sample positions must be unique valid row indices")

    history: dict[str, int] = {}
    categories: dict[str, dict[str, int]] = {}
    clicks: dict[str, int] = {}
    output = np.zeros((len(positions), len(FEATURE_NAMES)), dtype=np.float64)
    # Traverse only sampled rows and positive events; negatives never update state.
    selected = {int(position): offset for offset, position in enumerate(positions)}
    positive = frame["clicked"].to_numpy() == 1
    relevant = np.flatnonzero(positive | np.isin(np.arange(len(frame)), positions))
    order = relevant[np.argsort(times.astype("int64").to_numpy()[relevant], kind="stable")]
    users = frame["user_id"].astype(str).to_numpy()
    items = frame["item_id"].astype(str).to_numpy()
    ticks = times.astype("int64").to_numpy()
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ticks[order[end]] == ticks[order[start]]:
            end += 1
        batch = order[start:end]
        # Freeze the prefix for the ENTIRE timestamp, including distinct impressions.
        for position in batch:
            if position in selected:
                user, item = users[position], items[position]
                output[selected[position]] = feature_vector(
                    history_len=history.get(user, 0),
                    category_hits=categories.get(user, {}).get(item_category.get(item, ""), 0),
                    train_clicks=clicks.get(item, 0),
                    title_chars=item_title_chars.get(item, 0),
                )
        for position in batch:
            if positive[position]:
                user, item = users[position], items[position]
                history[user] = history.get(user, 0) + 1
                clicks[item] = clicks.get(item, 0) + 1
                counts = categories.setdefault(user, {})
                category = item_category.get(item, "")
                counts[category] = counts.get(category, 0) + 1
        start = end
    return output, {
        "user_history_len": history,
        "user_category_counts": categories,
        "item_clicks": clicks,
        "item_category": dict(item_category),
        "item_title_chars": dict(item_title_chars),
    }
