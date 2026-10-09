"""只读实验分析工具。只有这六个名字可以调用。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from searchpilot.experiments.store import ExperimentRecord, ExperimentStore, compare_experiments

ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "list_experiments",
        "get_experiment_config",
        "get_metric",
        "compare_experiments",
        "get_segment_metrics",
        "get_failure_samples",
    }
)
METRIC_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ndcg@10",
        "mrr@10",
        "recall@50",
        "recall@20",
        "precision@10",
        "auc",
        "log_loss",
        "ece",
        "zero_result_rate",
    }
)
MAX_LIMIT = 20


class ToolError(Exception):
    """工具参数或权限错误。recoverable 为假时调用方应停止再试。"""

    def __init__(self, message: str, *, recoverable: bool) -> None:
        super().__init__(message)
        self.recoverable = recoverable


def new_trace_id() -> str:
    return "trace_" + uuid4().hex


def _limit(value: object, default: int = 20) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError("limit must be an integer", recoverable=True)
    if not 1 <= value <= MAX_LIMIT:
        raise ToolError(f"limit must be between 1 and {MAX_LIMIT}", recoverable=True)
    return value


def _text(value: object, name: str, *, max_len: int = 128) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} must be a non-empty string", recoverable=True)
    if len(value) > max_len:
        raise ToolError(f"{name} is too long", recoverable=True)
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def call_tool(store: ExperimentStore, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """按白名单调用工具。未知工具不可恢复。"""
    if name not in ALLOWED_TOOLS:
        raise ToolError(f"tool is not allowed: {name}", recoverable=False)
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be an object", recoverable=True)
    if name == "list_experiments":
        return _list_experiments(store, arguments)
    if name == "get_experiment_config":
        return _get_config(store, arguments)
    if name == "get_metric":
        return _get_metric(store, arguments)
    if name == "compare_experiments":
        return _compare(store, arguments)
    if name == "get_segment_metrics":
        return _segment(store, arguments)
    return _failures(store, arguments)


def _record(store: ExperimentStore, experiment_id: str) -> ExperimentRecord:
    record = store.get(experiment_id)
    if record is None:
        raise ToolError("experiment does not exist", recoverable=True)
    return record


def _list_experiments(store: ExperimentStore, arguments: dict[str, Any]) -> dict[str, Any]:
    source = _optional_text(arguments.get("source"), "source")
    kind = _optional_text(arguments.get("kind"), "kind")
    if source is not None and source not in {"searchpilot", "mind", "synthetic", "evorec"}:
        raise ToolError("unknown source", recoverable=True)
    rows = store.list_records(source=source, kind=kind, limit=_limit(arguments.get("limit")))
    return {
        "experiments": [
            {
                "experiment_id": row.experiment_id,
                "kind": row.kind,
                "source": row.source,
                "source_ref": row.source_ref,
                "protocol_version": row.protocol_version,
            }
            for row in rows
        ]
    }


def _get_config(store: ExperimentStore, arguments: dict[str, Any]) -> dict[str, Any]:
    record = _record(store, _text(arguments.get("experiment_id"), "experiment_id"))
    return {
        "experiment_id": record.experiment_id,
        "config": record.config,
        "source": record.source,
        "source_ref": record.source_ref,
        "protocol_version": record.protocol_version,
        "data_version": record.data_version,
        "code_commit": record.code_commit,
    }


def _metric_name(value: object) -> str:
    name = _text(value, "metric")
    if name not in METRIC_ALLOWLIST:
        raise ToolError("metric is not allowed", recoverable=True)
    return name


def _get_metric(store: ExperimentStore, arguments: dict[str, Any]) -> dict[str, Any]:
    record = _record(store, _text(arguments.get("experiment_id"), "experiment_id"))
    name = _metric_name(arguments.get("metric"))
    split = _text(arguments.get("split"), "split")
    segment = _optional_text(arguments.get("segment"), "segment")
    for metric in record.metrics:
        if metric.name == name and metric.split == split and metric.segment == segment:
            return {
                "experiment_id": record.experiment_id,
                "metric": name,
                "split": split,
                "segment": segment,
                "value": metric.value,
                "source": metric.source,
                "source_ref": record.source_ref,
            }
    raise ToolError("metric was not found", recoverable=True)


def _compare(store: ExperimentStore, arguments: dict[str, Any]) -> dict[str, Any]:
    left = _record(store, _text(arguments.get("left_id"), "left_id"))
    right = _record(store, _text(arguments.get("right_id"), "right_id"))
    raw_metrics = arguments.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ToolError("metrics must be a non-empty list", recoverable=True)
    names = [_metric_name(item) for item in raw_metrics]
    result = compare_experiments(left, right, names)
    result["left_id"] = left.experiment_id
    result["right_id"] = right.experiment_id
    result["left_source_ref"] = left.source_ref
    result["right_source_ref"] = right.source_ref
    return result


def _segment(store: ExperimentStore, arguments: dict[str, Any]) -> dict[str, Any]:
    record = _record(store, _text(arguments.get("experiment_id"), "experiment_id"))
    segment = _text(arguments.get("segment"), "segment")
    rows = [
        {
            "experiment_id": record.experiment_id,
            "metric": metric.name,
            "split": metric.split,
            "segment": metric.segment,
            "value": metric.value,
            "source": metric.source,
            "source_ref": record.source_ref,
        }
        for metric in record.metrics
        if metric.segment == segment
    ]
    return {"experiment_id": record.experiment_id, "segment": segment, "metrics": rows}


def _failures(store: ExperimentStore, arguments: dict[str, Any]) -> dict[str, Any]:
    record = _record(store, _text(arguments.get("experiment_id"), "experiment_id"))
    limit = _limit(arguments.get("limit"), default=10)
    samples = [dict(item) for item in record.failures[:limit]]
    return {
        "experiment_id": record.experiment_id,
        "source": record.source,
        "source_ref": record.source_ref,
        "failures": samples,
    }
