"""用户点击历史加载与内存存储。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from searchpilot.ports import HistoryStore


class InMemoryHistoryStore(HistoryStore):
    """``user_id -> 按时间升序的 item_id 序列``。"""

    def __init__(self, histories: Mapping[str, Sequence[str]]) -> None:
        self._histories: dict[str, tuple[str, ...]] = {
            user_id: tuple(history) for user_id, history in histories.items()
        }

    def get_history(self, user_id: str) -> tuple[str, ...]:
        return self._histories.get(user_id, ())


def load_user_history(parquet_path: Path) -> dict[str, tuple[str, ...]]:
    """读取 ``user_history.parquet``（列 ``user_id``, ``history`` list<str>）。"""
    table = pq.read_table(parquet_path)
    user_ids = table.column("user_id").to_pylist()
    histories = table.column("history").to_pylist()
    result: dict[str, tuple[str, ...]] = {}
    for user_id, history in zip(user_ids, histories, strict=True):
        if history is None:
            seq: tuple[str, ...] = ()
        else:
            seq = tuple(str(item_id) for item_id in history)
        result[str(user_id)] = seq
    return result
