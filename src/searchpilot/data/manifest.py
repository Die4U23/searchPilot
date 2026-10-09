"""构建版本化 Parquet 数据快照及 manifest。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from searchpilot.data.download import sha256_file
from searchpilot.data.mind import parse_behaviors, parse_news
from searchpilot.data.schemas import (
    IMPRESSIONS_SCHEMA,
    ITEMS_SCHEMA,
    USER_HISTORY_SCHEMA,
    DatasetManifest,
    ManifestFile,
    TimeBounds,
)
from searchpilot.data.split import build_user_history

INPUT_PATHS = (
    Path("train/news.tsv"),
    Path("train/behaviors.tsv"),
    Path("dev/news.tsv"),
    Path("dev/behaviors.tsv"),
)


def compute_data_version(paths: list[Path] | tuple[Path, ...]) -> str:
    """对按调用方顺序排列的输入摘要拼接值再次做 SHA-256。"""
    joined = "".join(sha256_file(path) for path in paths)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()[:12]


def _write_parquet(frame: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    pq.write_table(table, path)


def _iso_bound(frame: pd.DataFrame, split: str, operation: str) -> str | None:
    values = frame.loc[frame["split"] == split, "shown_at"]
    if values.empty:
        return None
    value = values.min() if operation == "min" else values.max()
    return pd.Timestamp(value).isoformat()


def build_dataset(raw_dir: Path, out_dir: Path) -> DatasetManifest:
    """解析 MIND 官方 train/dev 目录并写入不可变版本目录。"""
    inputs = tuple(raw_dir / relative for relative in INPUT_PATHS)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MIND inputs: {missing}")

    train_news = parse_news(raw_dir / "train/news.tsv")
    dev_news = parse_news(raw_dir / "dev/news.tsv")
    train_impressions, train_behaviors = parse_behaviors(raw_dir / "train/behaviors.tsv", "train")
    dev_impressions, _ = parse_behaviors(raw_dir / "dev/behaviors.tsv", "dev")

    impressions = pd.concat([train_impressions, dev_impressions], ignore_index=True).reset_index(
        drop=True
    )
    items = pd.concat([train_news, dev_news], ignore_index=True)
    items = items.drop_duplicates("item_id", keep="first").reset_index(drop=True)
    first_seen = impressions.groupby("item_id", sort=False)["shown_at"].min()
    items["first_seen_at"] = pd.to_datetime(items["item_id"].map(first_seen), utc=True).astype(
        "datetime64[us, UTC]"
    )
    user_history = build_user_history(train_behaviors)

    version = compute_data_version(inputs)
    version_dir = out_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    output_frames = (
        ("items.parquet", items, ITEMS_SCHEMA),
        ("impressions.parquet", impressions, IMPRESSIONS_SCHEMA),
        ("user_history.parquet", user_history, USER_HISTORY_SCHEMA),
    )
    for name, frame, schema in output_frames:
        _write_parquet(frame, schema, version_dir / name)

    source_files = tuple(
        ManifestFile(
            name=relative.as_posix(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
        )
        for relative, path in zip(INPUT_PATHS, inputs, strict=True)
    )
    outputs = tuple(
        ManifestFile(
            name=name,
            sha256=sha256_file(version_dir / name),
            rows=len(frame),
        )
        for name, frame, _ in output_frames
    )
    manifest = DatasetManifest(
        data_version=version,
        created_at=datetime.now(UTC).isoformat(),
        source_files=source_files,
        outputs=outputs,
        time_bounds=TimeBounds(
            train_start=_iso_bound(impressions, "train", "min"),
            train_end=_iso_bound(impressions, "train", "max"),
            dev_start=_iso_bound(impressions, "dev", "min"),
            dev_end=_iso_bound(impressions, "dev", "max"),
        ),
    )
    with (version_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest
