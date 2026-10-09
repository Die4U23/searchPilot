"""确定性只读分析。不调用外部模型，也不执行问题文本里的指令。"""

from __future__ import annotations

import re
from typing import Any

from searchpilot.agent.tools import ALLOWED_TOOLS, ToolError, call_tool
from searchpilot.experiments.store import ExperimentStore

CITATION = re.compile(r"\[([^|\]]+)\|([^|\]]+)\|([^|\]]+)\|([^|\]]+)\|([^|\]]+)\]")
_METRIC = re.compile(
    r"\b(ndcg@10|mrr@10|recall@50|recall@20|precision@10|auc|log_loss|ece|zero_result_rate)\b"
)


def analyze(store: ExperimentStore, question: str, *, trace_id: str) -> dict[str, Any]:
    """根据问题调用白名单工具，并只引用工具返回的数字。"""
    text = question.strip()
    calls = _plan(text)
    trace: list[dict[str, Any]] = []
    for name, arguments in calls:
        if name not in ALLOWED_TOOLS:
            trace.append(
                {
                    "tool": name,
                    "ok": False,
                    "recoverable": False,
                    "error": "tool is not allowed",
                }
            )
            break
        try:
            result = call_tool(store, name, arguments)
        except ToolError as exc:
            trace.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "ok": False,
                    "recoverable": exc.recoverable,
                    "error": str(exc),
                }
            )
            if not exc.recoverable:
                break
            continue
        trace.append({"tool": name, "arguments": arguments, "ok": True, "result": result})
    answer, citations = _render(text, trace)
    return {
        "trace_id": trace_id,
        "answer": answer,
        "citations": citations,
        "tool_trace": trace,
    }


def _plan(question: str) -> list[tuple[str, dict[str, Any]]]:
    """只看问题里的实验 id 和指标名。忽略 run_sql、execute_shell 和“忽略规则”。"""
    lowered = question.lower()
    ids = re.findall(r"\b((?:sp|evorec)-[a-z0-9_@.-]+)\b", lowered)
    metrics = _METRIC.findall(lowered)
    wants_compare = "compare" in lowered or "比较" in question or "提升" in question
    if wants_compare and len(ids) >= 2:
        return [
            (
                "compare_experiments",
                {
                    "left_id": ids[0],
                    "right_id": ids[1],
                    "metrics": metrics or ["ndcg@10"],
                },
            )
        ]
    if ids and ("失败" in question or "failure" in lowered):
        return [("get_failure_samples", {"experiment_id": ids[0], "limit": 5})]
    if ids and ("配置" in question or "config" in lowered):
        return [("get_experiment_config", {"experiment_id": ids[0]})]
    if ids and ("分组" in question or "segment" in lowered):
        segment = "synonym" if "synonym" in lowered or "同义" in question else "all"
        return [("get_segment_metrics", {"experiment_id": ids[0], "segment": segment})]
    if ids and metrics:
        return [
            (
                "get_metric",
                {
                    "experiment_id": ids[0],
                    "metric": metrics[0],
                    "split": "test",
                    "segment": None,
                },
            )
        ]
    source = "evorec" if "evorec" in lowered else None
    return [("list_experiments", {"source": source, "kind": None, "limit": 20})]


def _citation(
    experiment_id: str, metric: str, split: str, segment: str | None, value: float, source: str
) -> str:
    place = split if not segment else f"{split}/{segment}"
    return f"[{experiment_id} | {metric} | {place} | {value} | {source}]"


def _render(question: str, trace: list[dict[str, Any]]) -> tuple[str, list[str]]:
    citations: list[str] = []
    lines: list[str] = []
    for step in trace:
        if not step.get("ok"):
            continue
        result = step["result"]
        if step["tool"] == "compare_experiments" and result.get("comparable") is False:
            lines.append("无法判断 / 证据不足。来源或协议不同，拒绝比较优劣。")
            continue
        if step["tool"] == "get_metric":
            cite = _citation(
                result["experiment_id"],
                result["metric"],
                result["split"],
                result["segment"],
                result["value"],
                result["source"],
            )
            citations.append(cite)
            ref = result.get("source_ref") or {}
            if result["source"] == "evorec":
                lines.append(f"来自 EvoRec {ref.get('run_id')}，commit {ref.get('commit')}。{cite}")
            else:
                lines.append(cite)
        elif step["tool"] == "get_segment_metrics":
            if not result["metrics"]:
                lines.append("无法判断 / 证据不足。")
            for metric in result["metrics"]:
                cite = _citation(
                    metric["experiment_id"],
                    metric["metric"],
                    metric["split"],
                    metric["segment"],
                    metric["value"],
                    metric["source"],
                )
                citations.append(cite)
                lines.append(cite)
        elif step["tool"] == "get_failure_samples":
            if not result["failures"]:
                lines.append("无法判断 / 证据不足。")
            else:
                lines.append(
                    f"{result['experiment_id']} 失败样例 {len(result['failures'])} 条，"
                    f"source={result['source']}。"
                )
        elif step["tool"] == "list_experiments" and not result["experiments"]:
            lines.append("无法判断 / 证据不足。")
    if not lines:
        lines.append("无法判断 / 证据不足。")
    if "run_sql" in question.lower() or "execute_shell" in question.lower():
        lines.append("拒绝执行 SQL 或 Shell。")
    return " ".join(lines), citations
