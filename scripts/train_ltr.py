"""用 train 查询的 0–3 级标注训练线性 pairwise LTR。

只读 ``split=train``。val / test 的标注不进入标准化，也不进入损失。
超参固定为 ``ltr.SEED`` / ``EPOCHS`` / ``LEARNING_RATE``，没有用来扫 test 的开关。

用法::

    python scripts/train_ltr.py --data-dir ./data --artifact-dir ./artifacts

产物写到 ``<artifact-dir>/search/ltr/``（weights.json、meta.json、popularity.json）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from build_index import ITEMS_FILENAME, BuildIndexError, resolve_data_version
from searchpilot.eval.search_eval import load_labels, load_queries
from searchpilot.search.ltr import (
    EPOCHS,
    LEARNING_RATE,
    SEED,
    assemble_ltr_model,
    feature_matrix,
    fit_standardizer,
    labels_for_queries,
    ltr_artifact_dir,
    pairwise_indices,
    save_ltr,
    train_click_counts,
    train_pairwise,
)
from searchpilot.search.service import BM25SearchService, SearchNotReadyError, build_search_service

IMPRESSIONS_FILENAME = "impressions.parquet"
TRAIN_SPLIT = "train"


class TrainLtrError(Exception):
    """输入缺失或训练切分不合法；主函数把它转成非零退出码。"""


def _as_datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
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


def load_item_side(
    items_path: Path,
) -> tuple[dict[str, int], dict[str, datetime], datetime | None]:
    """标题长度、``first_seen_at``，以及它们的最大时间（写入产物的 reference_time）。"""
    if not items_path.is_file():
        raise TrainLtrError(f"items parquet not found: {items_path}")
    table = pq.read_table(items_path, columns=["item_id", "title", "first_seen_at"])
    title_chars: dict[str, int] = {}
    first_seen: dict[str, datetime] = {}
    latest: datetime | None = None
    for row in table.to_pylist():
        item_id = str(row["item_id"])
        title_chars[item_id] = len(str(row["title"] or ""))
        stamp = _as_datetime(row["first_seen_at"])
        if stamp is None:
            continue
        first_seen[item_id] = stamp
        if latest is None or stamp > latest:
            latest = stamp
    return title_chars, first_seen, latest


def train(
    data_dir: Path,
    artifact_dir: Path,
    *,
    queries_path: Path,
    labels_path: Path,
    data_version: str | None,
) -> tuple[Path, str, int]:
    try:
        version = resolve_data_version(data_dir / "processed", data_version)
    except BuildIndexError as exc:
        raise TrainLtrError(str(exc)) from exc
    version_dir = data_dir / "processed" / version
    queries = load_queries(queries_path, split=TRAIN_SPLIT)
    if any(query.split != TRAIN_SPLIT for query in queries):
        raise TrainLtrError("refusing to train on a non-train query")
    train_ids = {query.query_id for query in queries}
    labels = labels_for_queries(load_labels(labels_path), train_ids)
    if not set(labels) <= train_ids:
        raise TrainLtrError("refusing to train on labels outside split=train")
    if not queries:
        raise TrainLtrError("no train queries")

    try:
        service = build_search_service(artifact_dir)
    except SearchNotReadyError as exc:
        raise TrainLtrError(str(exc)) from exc
    if not isinstance(service, BM25SearchService):
        raise TrainLtrError("search service does not support LTR training")
    if not service.vector_ready:
        raise TrainLtrError("vector index is not loaded; build artifacts/search/vector first")

    impressions_path = version_dir / IMPRESSIONS_FILENAME
    if not impressions_path.is_file():
        raise TrainLtrError(f"impressions parquet not found: {impressions_path}")
    clicks = train_click_counts(
        pq.read_table(impressions_path, columns=["item_id", "clicked", "split"]).to_pandas()
    )
    title_chars, first_seen, reference = load_item_side(version_dir / ITEMS_FILENAME)
    categories = {
        item_id: category
        for item_id, category in zip(service.index.item_ids, service.index.categories, strict=True)
    }

    blocks: list[np.ndarray] = []
    grades: list[int] = []
    groups: list[str] = []
    in_bm25 = 0
    in_vector = 0
    print(f"scoring {len(queries)} train queries", flush=True)
    for index, query in enumerate(queries, start=1):
        usable = {
            item_id: int(grade)
            for item_id, grade in labels.get(query.query_id, {}).items()
            if item_id in categories
        }
        if not usable:
            continue
        prep = service.prepare_ltr(query.query_text, sorted(usable))
        item_ids = sorted(usable)
        block = feature_matrix(
            item_ids,
            prep.signals,
            categories=categories,
            clicks=clicks,
            title_chars=title_chars,
            first_seen_at=first_seen,
            reference_time=reference,
        )
        blocks.append(block)
        for item_id in item_ids:
            grades.append(usable[item_id])
            groups.append(query.query_id)
            if item_id in prep.signals.bm25_rank:
                in_bm25 += 1
            if item_id in prep.signals.vector_rank:
                in_vector += 1
        if index % 10 == 0 or index == len(queries):
            print(f"scored {index}/{len(queries)} train queries", flush=True)

    if not blocks:
        raise TrainLtrError("no labeled train documents in the index")
    features = np.vstack(blocks)
    normalized, mean, std = fit_standardizer(features)
    higher, lower = pairwise_indices(grades, groups)
    if higher.size == 0:
        raise TrainLtrError("no pairwise pairs in train; grades do not differ within queries")
    weights, bias, loss_first, loss_last = train_pairwise(normalized, higher, lower)
    model = assemble_ltr_model(
        weights,
        bias,
        mean,
        std,
        reference_time=reference,
        clicks=clicks,
        title_chars=title_chars,
        first_seen_at=first_seen,
    )
    out_dir = ltr_artifact_dir(artifact_dir)
    save_ltr(
        out_dir,
        model,
        stats={
            "data_version": version,
            "train_query_count": len(queries),
            "train_row_count": int(features.shape[0]),
            "pair_count": int(higher.size),
            "labeled_in_bm25_top": in_bm25,
            "labeled_in_vector_top": in_vector,
            "loss_first": loss_first,
            "loss_last": loss_last,
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
    )
    print(f"artifact_dir={out_dir}")
    print(f"model_version={model.model_version}")
    print(f"train_queries={len(queries)} rows={features.shape[0]} pairs={higher.size}")
    print(f"labeled_in_bm25_top={in_bm25} labeled_in_vector_top={in_vector}")
    print(f"loss_first={loss_first:.6f} loss_last={loss_last:.6f}")
    vector_meta = artifact_dir / "search" / "vector" / "meta.json"
    if vector_meta.is_file():
        payload = json.loads(vector_meta.read_text(encoding="utf-8"))
        print(f"vector_model_version={payload.get('model_version')}")
    print(f"bm25_model_version={service.model_version}")
    return out_dir, model.model_version, int(higher.size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a linear pairwise LTR model on train queries"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_DATA_DIR", "./data")),
    )
    parser.add_argument(
        "--data-version",
        default=os.environ.get("SEARCHPILOT_DATA_VERSION") or None,
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_ARTIFACT_DIR", "./artifacts")),
    )
    data_dir_default = Path(os.environ.get("SEARCHPILOT_DATA_DIR", "./data"))
    parser.add_argument(
        "--queries", type=Path, default=data_dir_default / "queries" / "queries.csv"
    )
    parser.add_argument("--labels", type=Path, default=data_dir_default / "queries" / "labels.csv")
    args = parser.parse_args(argv)
    for path in (args.queries, args.labels):
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
    try:
        train(
            args.data_dir,
            args.artifact_dir,
            queries_path=args.queries,
            labels_path=args.labels,
            data_version=args.data_version,
        )
    except TrainLtrError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
