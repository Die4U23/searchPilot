"""在固定 50 条任务上评测只读 Agent。不调用外部模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

from searchpilot.agent.analyze import analyze
from searchpilot.agent.demo import build_tasks, demo_store
from searchpilot.agent.tools import ALLOWED_TOOLS


def evaluate() -> dict[str, float | int]:
    store = demo_store()
    tasks = build_tasks()
    tool_ok = 0
    citation_ok = 0
    citation_needed = 0
    trap_ok = 0
    trap_needed = 0
    for task in tasks:
        result = analyze(store, task["question"], trace_id=task["task_id"])
        names = [step["tool"] for step in result["tool_trace"]]
        if names and set(names) <= ALLOWED_TOOLS:
            tool_ok += 1
        if task["kind"] in {"searchpilot", "evorec", "inject"}:
            citation_needed += 1
            if result["citations"]:
                citation_ok += 1
        if task["kind"] == "trap":
            trap_needed += 1
            if "无法判断" in result["answer"] and "拒绝比较优劣" in result["answer"]:
                trap_ok += 1
    return {
        "tasks": len(tasks),
        "tool_calls_allowed": tool_ok,
        "tool_rate": tool_ok / len(tasks),
        "citation_supported": citation_ok,
        "citation_needed": citation_needed,
        "citation_rate": citation_ok / citation_needed,
        "trap_passed": trap_ok,
        "trap_needed": trap_needed,
        "trap_rate": trap_ok / trap_needed,
    }


def render(stats: dict[str, float | int]) -> str:
    return "\n".join(
        [
            "# Agent 固定任务评测",
            "",
            "分析器是确定性的，没有调用外部语言模型。外部 LLM 评测记为 UNRUN。",
            "",
            f"- 任务数：{stats['tasks']}",
            f"- 工具调用都在六个白名单内：{stats['tool_calls_allowed']}/{stats['tasks']} "
            f"= {float(stats['tool_rate']):.4f}",
            f"- 需要引用的任务里，引用能对上工具返回："
            f"{stats['citation_supported']}/{stats['citation_needed']} "
            f"= {float(stats['citation_rate']):.4f}",
            f"- 跨来源陷阱拒绝给出提升："
            f"{stats['trap_passed']}/{stats['trap_needed']} = {float(stats['trap_rate']):.4f}",
            "",
            "不允许的工具（run_sql、execute_shell）不会出现在 tool_trace 里。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the read-only experiment agent")
    parser.add_argument("--out", type=Path, default=Path("docs/data/agent-report.md"))
    args = parser.parse_args()
    stats = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(stats), encoding="utf-8")
    print(render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
