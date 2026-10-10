"""PostgreSQL 实验登记，表结构见 db/migrations。"""

from __future__ import annotations

import json
from typing import Any

from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from searchpilot.experiments.store import ExperimentRecord, MetricPoint


class PostgresExperimentStore:
    """把实验和指标写入 ``experiments`` / ``metrics``。同一 id 再次写入会替换指标。"""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def put(self, record: ExperimentRecord) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id, kind, config_json, code_commit, data_version,
                    protocol_version, source, source_ref, failures
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    config_json = EXCLUDED.config_json,
                    code_commit = EXCLUDED.code_commit,
                    data_version = EXCLUDED.data_version,
                    protocol_version = EXCLUDED.protocol_version,
                    source = EXCLUDED.source,
                    source_ref = EXCLUDED.source_ref,
                    failures = EXCLUDED.failures
                """,
                (
                    record.experiment_id,
                    record.kind,
                    Json(record.config),
                    record.code_commit,
                    record.data_version,
                    record.protocol_version,
                    record.source,
                    Json(record.source_ref) if record.source_ref is not None else None,
                    Json(list(record.failures)),
                ),
            )
            connection.execute(
                "DELETE FROM metrics WHERE experiment_id = %s",
                (record.experiment_id,),
            )
            for metric in record.metrics:
                connection.execute(
                    """
                    INSERT INTO metrics (experiment_id, name, split, segment, value, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.experiment_id,
                        metric.name,
                        metric.split,
                        metric.segment,
                        metric.value,
                        metric.source,
                    ),
                )

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT experiment_id, kind, config_json, code_commit, data_version,
                       protocol_version, source, source_ref, failures
                FROM experiments
                WHERE experiment_id = %s
                """,
                (experiment_id,),
            ).fetchone()
            if row is None:
                return None
            metrics = connection.execute(
                """
                SELECT name, split, segment, value, source
                FROM metrics
                WHERE experiment_id = %s
                ORDER BY metric_id
                """,
                (experiment_id,),
            ).fetchall()
        return _record_from_row(row, metrics)

    def list_records(
        self, *, source: str | None = None, kind: str | None = None, limit: int = 20
    ) -> list[ExperimentRecord]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        clauses: list[str] = []
        params: list[Any] = []
        if source is not None:
            clauses.append("source = %s")
            params.append(source)
        if kind is not None:
            clauses.append("kind = %s")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT experiment_id, kind, config_json, code_commit, data_version,
                       protocol_version, source, source_ref, failures
                FROM experiments
                {where}
                ORDER BY experiment_id
                LIMIT %s
                """,
                params,
            ).fetchall()
            found: list[ExperimentRecord] = []
            for row in rows:
                metrics = connection.execute(
                    """
                    SELECT name, split, segment, value, source
                    FROM metrics
                    WHERE experiment_id = %s
                    ORDER BY metric_id
                    """,
                    (row[0],),
                ).fetchall()
                found.append(_record_from_row(row, metrics))
        return found


def _record_from_row(row: tuple[Any, ...], metrics: list[tuple[Any, ...]]) -> ExperimentRecord:
    config = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
    source_ref = row[7]
    if isinstance(source_ref, str):
        source_ref = json.loads(source_ref)
    failures = row[8] if isinstance(row[8], list) else json.loads(row[8] or "[]")
    return ExperimentRecord(
        experiment_id=str(row[0]),
        kind=str(row[1]),
        config=dict(config),
        code_commit=None if row[3] is None else str(row[3]),
        data_version=str(row[4]),
        protocol_version=str(row[5]),
        source=str(row[6]),
        source_ref=None if source_ref is None else dict(source_ref),
        metrics=tuple(
            MetricPoint(
                name=str(item[0]),
                split=str(item[1]),
                value=float(item[3]),
                segment=None if item[2] is None else str(item[2]),
                source=str(item[4]),
            )
            for item in metrics
        ),
        failures=tuple(dict(item) for item in failures),
    )
