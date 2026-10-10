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


def read_evorec_records(
    path: Path,
    *,
    repo: str,
    commit: str,
    run_id: str,
    expect_sha256: str | None = None,
    source_path: str | None = None,
) -> list[ExperimentRecord]:
    """读取结果文件并返回实验记录，不写入 store。

    简单文件含 ``experiment_id`` 与 ``metrics`` 列表。EvoRec 已发布的
    ``results.json`` 则从 ``test_results`` 的 cohort 指标原样取出数值。
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
    recorded_path = source_path or path.as_posix()
    payload = _published_body(payload)
    if "experiment_id" in payload and "metrics" in payload:
        return [_simple_record(payload, recorded_path, repo, commit, run_id, digest)]
    return _published_records(payload, recorded_path, repo, commit, run_id, digest)


def import_evorec_file(
    path: Path,
    store: ExperimentStore,
    *,
    repo: str,
    commit: str,
    run_id: str,
    expect_sha256: str | None = None,
    source_path: str | None = None,
) -> list[ExperimentRecord]:
    """校验 SHA-256 后写入全部实验。不一致时不调用 ``store.put``。"""
    records = read_evorec_records(
        path,
        repo=repo,
        commit=commit,
        run_id=run_id,
        expect_sha256=expect_sha256,
        source_path=source_path,
    )
    for record in records:
        store.put(record)
    return records


def import_evorec_result(
    path: Path,
    store: ExperimentStore,
    *,
    repo: str,
    commit: str,
    run_id: str,
    expect_sha256: str | None = None,
) -> ExperimentRecord:
    """读取只有一条实验的结果 JSON，校验 SHA-256 后写入。"""
    records = read_evorec_records(
        path,
        repo=repo,
        commit=commit,
        run_id=run_id,
        expect_sha256=expect_sha256,
    )
    if len(records) != 1:
        raise ValueError(f"expected one experiment in {path}, got {len(records)}")
    store.put(records[0])
    return records[0]


def _simple_record(
    payload: dict[str, object],
    recorded_path: str,
    repo: str,
    commit: str,
    run_id: str,
    digest: str,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=str(payload["experiment_id"]),
        kind=str(payload["kind"]),
        config={},
        data_version=str(payload["data_version"]),
        protocol_version=str(payload["protocol_version"]),
        source=SOURCE,
        code_commit=commit,
        source_ref=_source_ref(repo, commit, recorded_path, digest, run_id),
        metrics=_metrics(payload.get("metrics")),
    )


def _published_body(payload: dict[str, object]) -> dict[str, object]:
    """有的结果把正式运行包在 ``series`` 里。"""
    series = payload.get("series")
    if "test_results" not in payload and isinstance(series, dict) and "test_results" in series:
        return series
    return payload


def _published_records(
    payload: dict[str, object],
    recorded_path: str,
    repo: str,
    commit: str,
    run_id: str,
    digest: str,
) -> list[ExperimentRecord]:
    """从已发布 ``test_results`` 取出 cohort 上的数值指标，不改名、不改值。"""
    trials = payload.get("test_results")
    if not isinstance(trials, list) or not trials:
        raise ValueError("EvoRec results have neither metrics nor test_results")
    configuration = payload.get("configuration")
    stage = ""
    if isinstance(configuration, dict) and configuration.get("stage"):
        stage = str(configuration["stage"])
    protocol = payload.get("protocol")
    protocol_version = str(payload.get("protocol_id") or "evorec")
    data_version = "evorec"
    if isinstance(protocol, dict) and protocol.get("catalog_sha256"):
        data_version = str(protocol["catalog_sha256"])
    elif isinstance(configuration, dict) and configuration.get("dataset_path"):
        data_version = str(configuration["dataset_path"])
    recorded_run = run_id or stage or "evorec"
    records: list[ExperimentRecord] = []
    seen: set[str] = set()
    for trial in trials:
        if not isinstance(trial, dict) or "name" not in trial:
            raise ValueError("test result is missing a name")
        name = str(trial["name"])
        experiment_id = _experiment_id(stage, name)
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment id {experiment_id}")
        seen.add(experiment_id)
        records.append(
            ExperimentRecord(
                experiment_id=experiment_id,
                kind="recommend",
                config={"stage": stage, "method": name},
                data_version=data_version,
                protocol_version=protocol_version,
                source=SOURCE,
                code_commit=commit,
                source_ref=_source_ref(repo, commit, recorded_path, digest, recorded_run),
                metrics=_cohort_metrics(trial.get("metrics")),
            )
        )
    return records


def _experiment_id(stage: str, name: str) -> str:
    parts = [_slug(part) for part in (stage, name) if part]
    if not parts:
        raise ValueError("experiment id is empty")
    return "-".join(parts)


def _slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in value.casefold())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _source_ref(
    repo: str, commit: str, recorded_path: str, digest: str, run_id: str
) -> dict[str, str]:
    return {
        "repo": repo,
        "commit": commit,
        "path": recorded_path,
        "sha256": digest,
        "run_id": run_id,
    }


def _cohort_metrics(raw: object) -> tuple[MetricPoint, ...]:
    if not isinstance(raw, dict):
        raise ValueError("published metrics must be an object")
    cohorts = raw.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("published metrics are missing cohorts")
    points: list[MetricPoint] = []
    for segment, block in cohorts.items():
        if not isinstance(block, dict):
            continue
        for name, value in block.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            points.append(
                MetricPoint(
                    name=str(name),
                    split="test",
                    value=float(value),
                    segment=str(segment),
                    source=SOURCE,
                )
            )
    if not points:
        raise ValueError("published results contain no numeric cohort metrics")
    return tuple(points)


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
