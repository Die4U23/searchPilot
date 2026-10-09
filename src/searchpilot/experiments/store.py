"""实验与指标的内存登记。PostgreSQL 表已在迁移里，没有库时用这份登记。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SOURCES: frozenset[str] = frozenset({"searchpilot", "mind", "synthetic", "evorec"})


@dataclass(frozen=True, slots=True)
class MetricPoint:
    name: str
    split: str
    value: float
    segment: str | None = None
    source: str = "searchpilot"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "segment": self.segment,
            "value": self.value,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    kind: str
    config: dict[str, Any]
    data_version: str
    protocol_version: str
    source: str
    code_commit: str | None = None
    source_ref: dict[str, Any] | None = None
    metrics: tuple[MetricPoint, ...] = ()
    failures: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "kind": self.kind,
            "config": self.config,
            "code_commit": self.code_commit,
            "data_version": self.data_version,
            "protocol_version": self.protocol_version,
            "source": self.source,
            "source_ref": self.source_ref,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "failures": [dict(item) for item in self.failures],
        }


class ExperimentStore(Protocol):
    def put(self, record: ExperimentRecord) -> None: ...

    def get(self, experiment_id: str) -> ExperimentRecord | None: ...

    def list_records(
        self, *, source: str | None = None, kind: str | None = None, limit: int = 20
    ) -> list[ExperimentRecord]: ...


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source}")


def compare_experiments(
    left: ExperimentRecord,
    right: ExperimentRecord,
    metric_names: Sequence[str],
) -> dict[str, Any]:
    """来源或 protocol_version 不同时 comparable 为 false，不给出提升结论。"""
    if left.source != right.source or left.protocol_version != right.protocol_version:
        return {
            "comparable": False,
            "reason": "source or protocol_version differs",
            "left_source": left.source,
            "right_source": right.source,
            "left_protocol_version": left.protocol_version,
            "right_protocol_version": right.protocol_version,
        }
    wanted = set(metric_names)

    def pick(record: ExperimentRecord) -> dict[str, float | None]:
        found = {metric.name: metric.value for metric in record.metrics if metric.name in wanted}
        return {name: found.get(name) for name in metric_names}

    return {"comparable": True, "reason": None, "left": pick(left), "right": pick(right)}


class InMemoryExperimentStore:
    """进程内登记。同一 experiment_id 再次 put 会整份替换。"""

    def __init__(self) -> None:
        self._records: dict[str, ExperimentRecord] = {}

    def put(self, record: ExperimentRecord) -> None:
        _check_source(record.source)
        for metric in record.metrics:
            _check_source(metric.source)
        self._records[record.experiment_id] = record

    def list_records(
        self, *, source: str | None = None, kind: str | None = None, limit: int = 20
    ) -> list[ExperimentRecord]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        rows = [
            record
            for record in self._records.values()
            if (source is None or record.source == source) and (kind is None or record.kind == kind)
        ]
        rows.sort(key=lambda record: record.experiment_id)
        return rows[:limit]

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._records.get(experiment_id)

    def save(self, path: Path) -> None:
        payload = [record.to_dict() for record in self._records.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> InMemoryExperimentStore:
        store = cls()
        if not path.is_file():
            return store
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"experiment registry must be a list: {path}")
        for item in raw:
            metrics = tuple(
                MetricPoint(
                    name=str(metric["name"]),
                    split=str(metric["split"]),
                    value=float(metric["value"]),
                    segment=None if metric.get("segment") is None else str(metric["segment"]),
                    source=str(metric.get("source", "searchpilot")),
                )
                for metric in item.get("metrics", [])
            )
            store.put(
                ExperimentRecord(
                    experiment_id=str(item["experiment_id"]),
                    kind=str(item["kind"]),
                    config=dict(item.get("config") or {}),
                    data_version=str(item["data_version"]),
                    protocol_version=str(item["protocol_version"]),
                    source=str(item["source"]),
                    code_commit=None
                    if item.get("code_commit") is None
                    else str(item["code_commit"]),
                    source_ref=_source_ref(item.get("source_ref")),
                    metrics=metrics,
                    failures=tuple(dict(item) for item in item.get("failures") or []),
                )
            )
        return store


def _source_ref(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("source_ref must be an object")
    return dict(value)
