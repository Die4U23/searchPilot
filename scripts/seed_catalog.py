"""把查询、标注、Agent 任务、模型版本和合成出价写入 PostgreSQL。

标注员字段原样保存。`annotator` 可以是人或外部工具的名字。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from psycopg.types.json import Json

from searchpilot.agent.demo import build_tasks
from searchpilot.ctr.ecpm import BID_SEED, synthetic_bid
from searchpilot.db.connection import create_pool
from searchpilot.search.normalize import normalize_query

QUERY_TYPES = {
    "exact_entity",
    "synonym",
    "multi_condition",
    "misspelling_or_abbrev",
    "no_answer",
    "long_tail_popular_distractor",
}
SPLITS = {"train", "val", "test"}

_MODEL_FILES: tuple[tuple[str, str], ...] = (
    ("search/bm25/meta.json", "bm25"),
    ("search/vector/meta.json", "encoder"),
    ("search/ltr/meta.json", "ltr"),
    ("ctr/meta.json", "ctr"),
)


def _batches(values: Sequence[tuple[Any, ...]], size: int) -> Iterator[list[tuple[Any, ...]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_queries(path: Path) -> list[tuple[str, str, str, str, str]]:
    """返回 (query_id, query_text, normalized_text, query_type, split)。"""
    rows: list[tuple[str, str, str, str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            query_id = (record.get("query_id") or "").strip()
            query_text = record.get("query_text") or ""
            query_type = (record.get("query_type") or "").strip()
            split = (record.get("split") or "").strip()
            if not query_id or not query_text.strip():
                raise ValueError(f"query row missing id or text: {record}")
            if query_id in seen:
                raise ValueError(f"duplicate query_id {query_id}")
            if query_type not in QUERY_TYPES:
                raise ValueError(f"query {query_id} has unknown query_type {query_type}")
            if split not in SPLITS:
                raise ValueError(f"query {query_id} has unknown split {split}")
            seen.add(query_id)
            rows.append((query_id, query_text, normalize_query(query_text), query_type, split))
    if not rows:
        raise ValueError(f"no queries in {path}")
    return rows


def read_labels(path: Path, query_ids: set[str]) -> list[tuple[str, str, int, str]]:
    """返回 (query_id, item_id, grade, annotator)。"""
    rows: list[tuple[str, str, int, str]] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            query_id = (record.get("query_id") or "").strip()
            item_id = (record.get("item_id") or "").strip()
            annotator = (record.get("annotator") or "").strip()
            if query_id not in query_ids:
                raise ValueError(f"label {query_id}/{item_id} has no query")
            if not item_id or not annotator:
                raise ValueError(f"label missing item or annotator: {record}")
            try:
                grade = int(record.get("grade") or "")
            except ValueError as exc:
                raise ValueError(f"label {query_id}/{item_id} has a non-integer grade") from exc
            if grade < 0 or grade > 3:
                raise ValueError(f"label {query_id}/{item_id} grade {grade} is outside 0..3")
            key = (query_id, item_id, annotator)
            if key in seen:
                raise ValueError(f"duplicate label {key}")
            seen.add(key)
            rows.append((query_id, item_id, grade, annotator))
    if not rows:
        raise ValueError(f"no labels in {path}")
    return rows


def collect_model_versions(artifact_dir: Path, data_dir: Path) -> list[tuple[Any, ...]]:
    """从已有产物收集模型版本。缺文件就跳过，不编造摘要。"""
    rows: list[tuple[Any, ...]] = []
    ctr_meta: dict[str, Any] = {}
    for relative, kind in _MODEL_FILES:
        path = artifact_dir / relative
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if kind == "ctr":
            ctr_meta = payload
        version = str(payload["model_version"])
        names = payload.get("feature_names")
        feature_version = None
        if isinstance(names, list) and names:
            joined = ",".join(str(name) for name in names).encode()
            feature_version = hashlib.sha256(joined).hexdigest()[:12]
        metadata = {
            "data_version": payload.get("data_version"),
            "path": relative.replace("\\", "/"),
        }
        rows.append(
            (
                version,
                kind,
                relative.replace("\\", "/"),
                feature_version,
                _sha256(path),
                Json(metadata),
                Json({"path": relative.replace("\\", "/")}),
            )
        )
    temperature = artifact_dir / "ctr" / "temperature.json"
    calibrator = ctr_meta.get("calibrator_version")
    if temperature.is_file() and isinstance(calibrator, str) and calibrator:
        rows.append(
            (
                calibrator,
                "calibrator",
                "ctr/temperature.json",
                None,
                _sha256(temperature),
                Json(
                    {
                        "data_version": ctr_meta.get("data_version"),
                        "path": "ctr/temperature.json",
                    }
                ),
                Json({"path": "ctr/temperature.json"}),
            )
        )
    data_version = ctr_meta.get("data_version")
    if isinstance(data_version, str) and data_version:
        history = data_dir / "processed" / data_version / "user_history.parquet"
        if history.is_file():
            digest = _sha256(history)
            uri = f"data/processed/{data_version}/user_history.parquet"
            for kind in ("popular", "itemcf"):
                rows.append(
                    (
                        f"{kind}-{data_version}",
                        kind,
                        uri,
                        None,
                        digest,
                        Json({"data_version": data_version}),
                        Json({"path": uri}),
                    )
                )
    return rows


def bid_rows(items_parquet: Path) -> list[tuple[str, float, str]]:
    if not items_parquet.is_file():
        return []
    column = pq.read_table(items_parquet, columns=["item_id"]).column("item_id")
    version = f"synthetic-{BID_SEED}"
    return [(str(item_id), synthetic_bid(str(item_id)), version) for item_id in column.to_pylist()]


def upsert_catalog(
    database_url: str,
    *,
    queries: Sequence[tuple[str, str, str, str, str]],
    labels: Sequence[tuple[str, str, int, str]],
    models: Sequence[tuple[Any, ...]],
    tasks: Sequence[dict[str, str]],
    bids: Sequence[tuple[str, float, str]],
    batch_size: int = 1000,
) -> dict[str, int]:
    pool = create_pool(database_url)
    try:
        _executemany(
            pool,
            """
            INSERT INTO queries (
                query_id, query_text, normalized_text, query_type, split, source
            )
            VALUES (%s, %s, %s, %s, %s, 'searchpilot')
            ON CONFLICT (query_id) DO UPDATE SET
                query_text = EXCLUDED.query_text,
                normalized_text = EXCLUDED.normalized_text,
                query_type = EXCLUDED.query_type,
                split = EXCLUDED.split,
                source = EXCLUDED.source
            """,
            queries,
            batch_size,
        )
        _executemany(
            pool,
            """
            INSERT INTO relevance_labels (query_id, item_id, grade, annotator, source)
            VALUES (%s, %s, %s, %s, 'searchpilot')
            ON CONFLICT (query_id, item_id, annotator) DO UPDATE SET
                grade = EXCLUDED.grade,
                source = EXCLUDED.source
            """,
            labels,
            batch_size,
        )
        _executemany(
            pool,
            """
            INSERT INTO model_versions (
                version, kind, artifact_uri, feature_version, sha256, metadata, source, source_ref
            )
            VALUES (%s, %s, %s, %s, %s, %s, 'searchpilot', %s)
            ON CONFLICT (version) DO UPDATE SET
                kind = EXCLUDED.kind,
                artifact_uri = EXCLUDED.artifact_uri,
                feature_version = EXCLUDED.feature_version,
                sha256 = EXCLUDED.sha256,
                metadata = EXCLUDED.metadata,
                source = EXCLUDED.source,
                source_ref = EXCLUDED.source_ref
            """,
            models,
            batch_size,
        )
        task_rows = [(task["task_id"], task["question"], task["kind"], Json([])) for task in tasks]
        _executemany(
            pool,
            """
            INSERT INTO agent_tasks (task_id, question, kind, expected_tools, source)
            VALUES (%s, %s, %s, %s, 'searchpilot')
            ON CONFLICT (task_id) DO UPDATE SET
                question = EXCLUDED.question,
                kind = EXCLUDED.kind,
                expected_tools = EXCLUDED.expected_tools,
                source = EXCLUDED.source
            """,
            task_rows,
            batch_size,
        )
        _executemany(
            pool,
            """
            INSERT INTO bids (item_id, bid, distribution_version, source)
            VALUES (%s, %s, %s, 'synthetic')
            ON CONFLICT (item_id) DO UPDATE SET
                bid = EXCLUDED.bid,
                distribution_version = EXCLUDED.distribution_version,
                source = EXCLUDED.source
            """,
            bids,
            batch_size,
        )
    finally:
        pool.close()
    return {
        "queries": len(queries),
        "labels": len(labels),
        "models": len(models),
        "tasks": len(tasks),
        "bids": len(bids),
    }


def _executemany(
    pool: Any, statement: str, rows: Sequence[tuple[Any, ...]], batch_size: int
) -> None:
    for batch in _batches(rows, batch_size):
        with pool.connection() as connection, connection.cursor() as cursor:
            cursor.executemany(statement, batch)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("SEARCHPILOT_DATABASE_URL"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("--database-url or SEARCHPILOT_DATABASE_URL is required")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    queries = read_queries(args.queries)
    labels = read_labels(args.labels, {row[0] for row in queries})
    version = _data_version(args.artifact_dir)
    items = args.data_dir / "processed" / version / "items.parquet" if version else Path()
    counts = upsert_catalog(
        args.database_url,
        queries=queries,
        labels=labels,
        models=collect_model_versions(args.artifact_dir, args.data_dir),
        tasks=build_tasks(),
        bids=bid_rows(items),
        batch_size=args.batch_size,
    )
    print(
        "upserted "
        f"queries={counts['queries']} labels={counts['labels']} "
        f"models={counts['models']} tasks={counts['tasks']} bids={counts['bids']}"
    )
    return 0


def _data_version(artifact_dir: Path) -> str | None:
    path = artifact_dir / "ctr" / "meta.json"
    if not path.is_file():
        path = artifact_dir / "search" / "bm25" / "meta.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("data_version")
    return version if isinstance(version, str) and version else None


if __name__ == "__main__":
    raise SystemExit(main())
