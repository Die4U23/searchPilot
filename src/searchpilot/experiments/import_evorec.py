"""只读导入 EvoRec 已发布实验结果。SHA-256 不一致则拒绝写入。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from searchpilot.experiments.store import ExperimentRecord, ExperimentStore, MetricPoint

SOURCE = "evorec"


def file_sha256(path: Path) -> str:
    """文件内容的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_evorec_result(
    path: Path,
    store: ExperimentStore,
    *,
    repo: str,
    commit: str,
    run_id: str,
    expect_sha256: str | None = None,
) -> ExperimentRecord:
    """读取结果 JSON，校验 SHA-256 后写入 ``source=evorec`` 的实验记录。

    先对文件字节计算 SHA-256。``expect_sha256`` 给出且不一致时抛出
    ``ValueError``，且不调用 ``store.put``。指标名与值原样保留，仅把
    ``source`` 标成 ``evorec``。
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expect_sha256 is not None and digest != expect_sha256.strip().lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expect_sha256}, got {digest}")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid EvoRec results JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"EvoRec results must be an object: {path}")

    record = ExperimentRecord(
        experiment_id=str(payload["experiment_id"]),
        kind=str(payload["kind"]),
        config={},
        data_version=str(payload["data_version"]),
        protocol_version=str(payload["protocol_version"]),
        source=SOURCE,
        code_commit=commit,
        source_ref={
            "repo": repo,
            "commit": commit,
            "path": path.as_posix(),
            "sha256": digest,
            "run_id": run_id,
        },
        metrics=_metrics(payload.get("metrics")),
    )
    store.put(record)
    return record


def _metrics(raw: object) -> tuple[MetricPoint, ...]:
    if not isinstance(raw, list):
        raise ValueError("metrics must be a list")
    points: list[MetricPoint] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("metric must be an object")
        segment = item.get("segment")
        points.append(
            MetricPoint(
                name=str(item["name"]),
                split=str(item["split"]),
                value=float(item["value"]),
                segment=None if segment is None else str(segment),
                source=SOURCE,
            )
        )
    return tuple(points)
