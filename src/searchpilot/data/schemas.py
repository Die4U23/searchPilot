"""离线 Parquet 快照及 manifest 的稳定数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa

ITEMS_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("abstract", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("first_seen_at", pa.timestamp("us", tz="UTC"), nullable=True),
    ]
)

IMPRESSIONS_SCHEMA = pa.schema(
    [
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("shown_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("clicked", pa.int8(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

USER_HISTORY_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("history", pa.list_(pa.string()), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class ManifestFile:
    """一个输入或输出文件的内容摘要。"""

    name: str
    sha256: str
    bytes: int | None = None
    rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """移除不适用于当前文件类型的空字段。"""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class TimeBounds:
    train_start: str | None
    train_end: str | None
    dev_start: str | None
    dev_end: str | None


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    data_version: str
    created_at: str
    source_files: tuple[ManifestFile, ...]
    outputs: tuple[ManifestFile, ...]
    time_bounds: TimeBounds

    def to_dict(self) -> dict[str, Any]:
        """转换为适合 JSON 序列化的结构。"""
        return {
            "data_version": self.data_version,
            "created_at": self.created_at,
            "source_files": [entry.to_dict() for entry in self.source_files],
            "outputs": [entry.to_dict() for entry in self.outputs],
            "time_bounds": asdict(self.time_bounds),
        }
