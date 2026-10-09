"""回退推荐服务：从 Parquet 训练模型并在线推荐。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from searchpilot.ports import (
    RecommendationHit,
    RecommendationResult,
    RecommendChannel,
    RecommendPort,
)
from searchpilot.recommend.history import InMemoryHistoryStore, load_user_history
from searchpilot.recommend.itemcf import ItemCFModel
from searchpilot.recommend.popular import PopularModel


class RecommendNotReadyError(Exception):
    """Parquet 快照或 data_version 不可用。"""


def resolve_data_version(processed_dir: Path) -> str:
    """确定要加载的 ``data_version``：优先环境变量，否则取 processed 下唯一子目录。"""
    env_version = os.environ.get("SEARCHPILOT_DATA_VERSION")
    if env_version:
        version_dir = processed_dir / env_version
        if not version_dir.is_dir():
            msg = f"SEARCHPILOT_DATA_VERSION={env_version!r} but {version_dir} is not a directory"
            raise RecommendNotReadyError(msg)
        return env_version

    if not processed_dir.is_dir():
        raise RecommendNotReadyError(f"processed directory not found: {processed_dir}")

    subdirs = sorted(p for p in processed_dir.iterdir() if p.is_dir())
    if len(subdirs) == 0:
        raise RecommendNotReadyError(
            f"no data_version under {processed_dir}; set SEARCHPILOT_DATA_VERSION"
        )
    if len(subdirs) > 1:
        names = ", ".join(p.name for p in subdirs)
        raise RecommendNotReadyError(
            f"multiple data_version directories ({names}); set SEARCHPILOT_DATA_VERSION"
        )
    return subdirs[0].name


def build_recommend_service(data_dir: Path) -> RecommendPort:
    """从 ``data_dir/processed/<data_version>/`` 加载并训练 Popular 与 ItemCF。"""
    processed_dir = data_dir / "processed"
    data_version = resolve_data_version(processed_dir)
    version_dir = processed_dir / data_version

    impressions_path = version_dir / "impressions.parquet"
    history_path = version_dir / "user_history.parquet"
    if not impressions_path.is_file() or not history_path.is_file():
        missing = [
            name
            for name, path in (
                ("impressions.parquet", impressions_path),
                ("user_history.parquet", history_path),
            )
            if not path.is_file()
        ]
        raise RecommendNotReadyError(
            f"missing required Parquet under {version_dir}: {', '.join(missing)}"
        )

    impressions = pd.read_parquet(impressions_path)
    histories = load_user_history(history_path)

    popular = PopularModel()
    popular.fit(impressions)

    itemcf = ItemCFModel()
    itemcf.fit(histories)

    return _RecommendService(
        histories=histories,
        popular=popular,
        itemcf=itemcf,
        data_version=data_version,
    )


class InMemoryRecommendService(RecommendPort):
    """测试用：用给定 histories 同时训练 Popular（出现次数）与 ItemCF。"""

    def __init__(
        self,
        histories: Mapping[str, Sequence[str]],
        model_version: str,
    ) -> None:
        self._histories = {user_id: tuple(seq) for user_id, seq in histories.items()}
        self._store = InMemoryHistoryStore(self._histories)
        self._model_version_label = model_version

        rows: list[dict[str, object]] = []
        for user_id, seq in self._histories.items():
            for item_id in seq:
                rows.append(
                    {
                        "impression_id": f"imp-{user_id}-{item_id}",
                        "user_id": user_id,
                        "shown_at": pd.Timestamp("2020-01-01", tz="UTC"),
                        "item_id": item_id,
                        "clicked": 1,
                        "split": "train",
                        "source": "mind",
                    }
                )
        impressions = pd.DataFrame(rows)
        if impressions.empty:
            impressions = pd.DataFrame(
                columns=[
                    "impression_id",
                    "user_id",
                    "shown_at",
                    "item_id",
                    "clicked",
                    "split",
                    "source",
                ]
            )

        self._popular = PopularModel()
        self._popular.fit(impressions)

        self._itemcf = ItemCFModel()
        self._itemcf.fit(self._histories)

        self._data_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version_label

    def recommend(
        self,
        user_id: str,
        limit: int,
        exclude: frozenset[str],
    ) -> RecommendationResult:
        return _recommend_impl(
            user_id=user_id,
            limit=limit,
            exclude=exclude,
            history=self._store.get_history(user_id),
            popular=self._popular,
            itemcf=self._itemcf,
            data_version=self._data_version,
        )


class _RecommendService(RecommendPort):
    def __init__(
        self,
        *,
        histories: Mapping[str, tuple[str, ...]],
        popular: PopularModel,
        itemcf: ItemCFModel,
        data_version: str,
    ) -> None:
        self._store = InMemoryHistoryStore(histories)
        self._popular = popular
        self._itemcf = itemcf
        self._data_version = data_version

    @property
    def model_version(self) -> str:
        return f"itemcf-{self._data_version}"

    def recommend(
        self,
        user_id: str,
        limit: int,
        exclude: frozenset[str],
    ) -> RecommendationResult:
        return _recommend_impl(
            user_id=user_id,
            limit=limit,
            exclude=exclude,
            history=self._store.get_history(user_id),
            popular=self._popular,
            itemcf=self._itemcf,
            data_version=self._data_version,
        )


def _recommend_impl(
    *,
    user_id: str,
    limit: int,
    exclude: frozenset[str],
    history: tuple[str, ...],
    popular: PopularModel,
    itemcf: ItemCFModel,
    data_version: str,
) -> RecommendationResult:
    del user_id  # 契约保留参数；逻辑仅依赖 history
    blocked = exclude | frozenset(history)

    itemcf_pairs = itemcf.predict(history, blocked, limit)
    if history and itemcf_pairs:
        channel: RecommendChannel = "itemcf"
        cold_start = False
        pairs = itemcf_pairs
        model_version = f"itemcf-{data_version}"
    else:
        channel = "popular"
        cold_start = True
        pairs = popular.predict(blocked, limit)
        model_version = f"popular-{data_version}"

    hits = tuple(
        RecommendationHit(
            item_id=item_id,
            score=score,
            rank=rank,
            channel=channel,
        )
        for rank, (item_id, score) in enumerate(pairs, start=1)
    )
    return RecommendationResult(
        hits=hits,
        model_version=model_version,
        channel=channel,
        cold_start=cold_start,
    )
