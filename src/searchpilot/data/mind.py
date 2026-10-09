"""MIND TSV 文件解析。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_COLUMNS = (
    "item_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)
BEHAVIOR_COLUMNS = ("impression_id", "user_id", "time", "history", "impressions")


def parse_news(path: Path) -> pd.DataFrame:
    """解析 news.tsv，并返回 items 契约所需的文本列。"""
    frame = pd.read_csv(
        path,
        sep="\t",
        names=list(NEWS_COLUMNS),
        header=None,
        dtype="string",
        keep_default_na=True,
    )
    selected = frame[["item_id", "title", "abstract", "category", "subcategory", "url"]].copy()
    for column in ("item_id", "title", "category", "subcategory", "url"):
        selected[column] = selected[column].fillna("").astype("string")
    selected["abstract"] = selected["abstract"].fillna("").astype("string")
    selected["first_seen_at"] = pd.Series(pd.NaT, index=selected.index, dtype="datetime64[us, UTC]")
    return selected


def parse_behaviors(path: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """解析 behaviors.tsv，展开候选曝光并保留每条行为的点击历史。"""
    if split not in {"train", "dev"}:
        raise ValueError(f"unsupported split: {split}")
    raw = pd.read_csv(
        path,
        sep="\t",
        names=list(BEHAVIOR_COLUMNS),
        header=None,
        dtype="string",
        keep_default_na=True,
    )
    raw["shown_at"] = pd.to_datetime(raw["time"], format="%m/%d/%Y %I:%M:%S %p", utc=True).astype(
        "datetime64[us, UTC]"
    )
    raw["history"] = raw["history"].fillna("").astype("string")

    rows: list[dict[str, object]] = []
    for record in raw.itertuples(index=False):
        candidates = str(record.impressions).split()
        for candidate in candidates:
            item_id, separator, clicked = candidate.rpartition("-")
            if separator != "-" or clicked not in {"0", "1"} or not item_id:
                raise ValueError(f"invalid impression token: {candidate!r}")
            rows.append(
                {
                    "impression_id": str(record.impression_id),
                    "user_id": str(record.user_id),
                    "shown_at": record.shown_at,
                    "item_id": item_id,
                    "clicked": int(clicked),
                    "split": split,
                    "source": "mind",
                }
            )

    impressions = pd.DataFrame(
        rows,
        columns=[
            "impression_id",
            "user_id",
            "shown_at",
            "item_id",
            "clicked",
            "split",
            "source",
        ],
    )
    for column in ("impression_id", "user_id", "item_id", "split", "source"):
        impressions[column] = impressions[column].astype("string")
    impressions["shown_at"] = pd.to_datetime(impressions["shown_at"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    impressions["clicked"] = impressions["clicked"].astype("int8")

    behaviors = raw[["impression_id", "user_id", "shown_at", "history"]].copy()
    behaviors["impression_id"] = behaviors["impression_id"].astype("string")
    behaviors["user_id"] = behaviors["user_id"].astype("string")
    behaviors["history"] = behaviors["history"].map(
        lambda value: str(value).split() if value else []
    )
    behaviors["split"] = pd.Series(split, index=behaviors.index, dtype="string")
    return impressions, behaviors
