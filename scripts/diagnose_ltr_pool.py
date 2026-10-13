"""用已保存的 LTR 权重检查并集标注，不重新训练。

相关文档来自 ``labels.csv`` 与 ``labels_tool.csv`` 里 grade > 0 的行。
名次只覆盖 BM25 / 向量各自的 top 100，缺席记为 101，与在线特征一致。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from searchpilot.eval.search_eval import EvalQuery, load_labels, load_queries, merge_label_maps
from searchpilot.search.ltr import (
    FEATURE_NAMES,
    RETRIEVAL_DEPTH,
    LtrModel,
    feature_matrix,
    load_ltr,
    ltr_artifact_dir,
    rank_by_score,
)
from searchpilot.search.service import BM25SearchService, build_search_service

SPLITS = ("test", "val", "train")
SOURCES = ("base", "tool", "pooled")
MISSING_RANK = RETRIEVAL_DEPTH + 1
TOP_K = 10
CUTOFF = 50


@dataclass
class Coverage:
    n: int = 0
    bm25_100: int = 0
    vector_100: int = 0
    neither: int = 0
    vector_only: int = 0
    vector50: int = 0
    vector50_not_ltr50: int = 0
    vector50_not_ltr50_missing_bm25: int = 0


@dataclass
class TopMix:
    slots: int = 0
    relevant: int = 0
    explicit_zero: int = 0
    unlabeled: int = 0


@dataclass
class Bundle:
    coverage: dict[tuple[str, str], Coverage] = field(default_factory=dict)
    mix: dict[tuple[str, str], TopMix] = field(default_factory=dict)

    def cover(self, split: str, source: str) -> Coverage:
        return self.coverage.setdefault((split, source), Coverage())

    def top(self, split: str, name: str) -> TopMix:
        return self.mix.setdefault((split, name), TopMix())


def _positives(labels: dict[str, dict[str, int]], query_id: str) -> set[str]:
    return {item_id for item_id, grade in labels.get(query_id, {}).items() if grade > 0}


def _rate(count: int, total: int) -> str:
    if total == 0:
        return "0"
    return f"{count}/{total} = {count / total:.4f}"


def _contrib(model: LtrModel, index: int, raw: float) -> float:
    std = model.std[index]
    safe = std if abs(std) >= 1e-12 else 1.0
    return model.weights[index] * (raw - model.mean[index]) / safe


def _categories(service: BM25SearchService) -> dict[str, str]:
    index = service.index
    return {
        item_id: category
        for item_id, category in zip(index.item_ids, index.categories, strict=True)
    }


def _positions(
    service: BM25SearchService,
    model: LtrModel,
    query: str,
    extra: set[str],
    categories: dict[str, str],
) -> tuple:
    prep = service.prepare_ltr(query, sorted(extra))
    if not prep.candidates:
        return prep, {}
    raw = feature_matrix(
        prep.candidates,
        prep.signals,
        categories=categories,
        clicks=model.clicks,
        title_chars=model.title_chars,
        first_seen_at=model.first_seen_at,
        reference_time=model.reference_time,
    )
    order = rank_by_score(prep.candidates, model.score_rows(raw), len(prep.candidates))
    return prep, {item_id: rank for rank, (item_id, _score) in enumerate(order, start=1)}


def _add_coverage(
    row: Coverage,
    *,
    in_bm25: bool,
    in_vector: bool,
    vector_rank: int | None,
    ltr_rank: int | None,
) -> None:
    row.n += 1
    row.bm25_100 += int(in_bm25)
    row.vector_100 += int(in_vector)
    row.neither += int(not in_bm25 and not in_vector)
    row.vector_only += int(in_vector and not in_bm25)
    in_vector50 = vector_rank is not None and vector_rank <= CUTOFF
    row.vector50 += int(in_vector50)
    dropped = in_vector50 and (ltr_rank is None or ltr_rank > CUTOFF)
    row.vector50_not_ltr50 += int(dropped)
    row.vector50_not_ltr50_missing_bm25 += int(dropped and not in_bm25)


def _add_mix(row: TopMix, item_id: str, grades: dict[str, int]) -> None:
    row.slots += 1
    if item_id not in grades:
        row.unlabeled += 1
    elif grades[item_id] > 0:
        row.relevant += 1
    else:
        row.explicit_zero += 1


def collect(
    service: BM25SearchService,
    model: LtrModel,
    queries: list[EvalQuery],
    base: dict[str, dict[str, int]],
    tool: dict[str, dict[str, int]],
    merged: dict[str, dict[str, int]],
) -> Bundle:
    bundle = Bundle()
    categories = _categories(service)
    for query in queries:
        base_ids = _positives(base, query.query_id)
        tool_ids = _positives(tool, query.query_id)
        pooled = base_ids | tool_ids
        prep, positions = _positions(service, model, query.query_text, pooled, categories)
        grades = merged.get(query.query_id, {})
        for item_id in pooled:
            in_bm25 = item_id in prep.signals.bm25_rank
            in_vector = item_id in prep.signals.vector_rank
            vector_rank = prep.signals.vector_rank.get(item_id)
            ltr_rank = positions.get(item_id)
            flags = dict(
                in_bm25=in_bm25,
                in_vector=in_vector,
                vector_rank=vector_rank,
                ltr_rank=ltr_rank,
            )
            _add_coverage(bundle.cover(query.split, "pooled"), **flags)
            source = "base" if item_id in base_ids else "tool"
            _add_coverage(bundle.cover(query.split, source), **flags)
        vector_top = [
            item_id for item_id, rank in prep.signals.vector_rank.items() if rank <= TOP_K
        ]
        ltr_top = [item_id for item_id, rank in positions.items() if rank <= TOP_K]
        for item_id in vector_top:
            _add_mix(bundle.top(query.split, "vector"), item_id, grades)
        for item_id in ltr_top:
            _add_mix(bundle.top(query.split, "ltr"), item_id, grades)
    return bundle


def _coverage_table(bundle: Bundle) -> list[str]:
    lines = [
        "| split | 来源 | 有关文档 | BM25 top 100 | 向量 top 100 | 两路都没有 | "
        "只在向量 top 100 | 向量 top 50 但不在 LTR top 50 | 其中 BM25 缺席 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        for source in SOURCES:
            row = bundle.cover(split, source)
            lines.append(
                f"| {split} | {source} | {row.n} | {_rate(row.bm25_100, row.n)} | "
                f"{_rate(row.vector_100, row.n)} | {_rate(row.neither, row.n)} | "
                f"{_rate(row.vector_only, row.n)} | {_rate(row.vector50_not_ltr50, row.n)} | "
                f"{_rate(row.vector50_not_ltr50_missing_bm25, row.vector50_not_ltr50)} |"
            )
    return lines


def _mix_table(bundle: Bundle) -> list[str]:
    lines = [
        "| split | 名单 | 位置数 | 有关 | 明确 0 分 | 两份标注都没有 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        for name in ("vector", "ltr"):
            row = bundle.top(split, name)
            lines.append(
                f"| {split} | {name} | {row.slots} | {_rate(row.relevant, row.slots)} | "
                f"{_rate(row.explicit_zero, row.slots)} | {_rate(row.unlabeled, row.slots)} |"
            )
    return lines


def _weight_lines(model: LtrModel) -> list[str]:
    lines = [
        "| 特征 | 权重 | 训练均值 | 训练标准差 |",
        "|---|---:|---:|---:|",
    ]
    for index, name in enumerate(FEATURE_NAMES):
        lines.append(
            f"| {name} | {model.weights[index]:.4f} | {model.mean[index]:.4f} | "
            f"{model.std[index]:.4f} |"
        )
    lines.extend(
        [
            "",
            "名次贡献 = 权重 * (名次 - 均值) / 标准差，与 `score_rows` 的一项相同。",
            "缺席 top 100 的名次是 101。",
            "",
            "| 特征 | 名次 1 | 名次 10 | 名次 50 | 名次 100 | 名次 101 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("bm25_rank", "vector_rank"):
        index = FEATURE_NAMES.index(name)
        cells = " | ".join(
            f"{_contrib(model, index, float(raw)):.4f}" for raw in (1, 10, 50, 100, 101)
        )
        lines.append(f"| {name} | {cells} |")
    return lines


def _finding(model: LtrModel, bundle: Bundle) -> list[str]:
    index = FEATURE_NAMES.index("bm25_rank")
    gap = _contrib(model, index, float(MISSING_RANK)) - _contrib(model, index, 1.0)
    direction = "更高" if gap > 0 else "更低"
    val_tool = bundle.cover("val", "tool")
    test_tool = bundle.cover("test", "tool")
    val_pooled = bundle.cover("val", "pooled")
    lines = [
        "## 解读",
        "",
        f"`bm25_rank` 权重是 {model.weights[index]:.4f}。名次 101 比名次 1 的贡献"
        f"{direction} {abs(gap):.4f}。",
        (
            f"val 的补标有关文档 {val_tool.n} 条里，"
            f"{val_tool.vector_only} 条在向量 top 100 但不在 BM25 top 100，"
            f"{val_tool.neither} 条两路 top 100 都没有。"
        ),
        (
            f"test 的补标有关文档 {test_tool.n} 条里，"
            f"{test_tool.vector_only} 条只在向量 top 100，"
            f"{test_tool.neither} 条两路都没有。"
        ),
        (
            f"val 并集有关文档里，向量 top 50 有 {val_pooled.vector50} 条，"
            f"其中 {val_pooled.vector50_not_ltr50} 条没有留在 LTR top 50；"
            f"这 {val_pooled.vector50_not_ltr50} 条里 "
            f"{val_pooled.vector50_not_ltr50_missing_bm25} 条 BM25 名次缺席。"
        ),
        _mix_line(bundle, "test"),
        _mix_line(bundle, "val"),
        "向量 top 10 落在已标注池内。LTR top 10 里没有分数的位置，nDCG 记为 0。",
        "名次 101 的贡献更高，所以缺席 BM25 的文档会被抬高，不是被挤出前 50。",
        "没有重新训练。train 行只说明这组权重见过的标注覆盖，不能当验收。",
        "",
    ]
    return lines


def _mix_line(bundle: Bundle, split: str) -> str:
    vector = bundle.top(split, "vector")
    ltr = bundle.top(split, "ltr")
    return (
        f"{split} 向量 top 10 的有关占比 {_rate(vector.relevant, vector.slots)}，"
        f"未标注占比 {_rate(vector.unlabeled, vector.slots)}；"
        f"LTR top 10 的有关占比 {_rate(ltr.relevant, ltr.slots)}，"
        f"未标注占比 {_rate(ltr.unlabeled, ltr.slots)}。"
    )


def render(
    service: BM25SearchService,
    model: LtrModel,
    queries_path: Path,
    labels_path: Path,
    tool_path: Path,
) -> str:
    queries = load_queries(queries_path)
    base = load_labels(labels_path)
    tool = load_labels(tool_path)
    merged = merge_label_maps(base, tool)
    bundle = collect(service, model, queries, base, tool, merged)
    lines = [
        "# 现有 LTR 在补标并集上的特征覆盖",
        "",
        f"2026-10-10。模型 `{model.model_version}`，没有重新训练。",
        "有关文档是 grade > 0。`base` 来自 `labels.csv`，`tool` 来自 `labels_tool.csv`。",
        "`pooled` 是两者之和。",
        f"BM25 与向量名次只记 top {RETRIEVAL_DEPTH}，缺席按 {MISSING_RANK} 进入线性分。",
        "LTR 只重排这两路 top 100 的 RRF 候选，深度之外的文档不会出现在 LTR 结果里。",
        "",
        "## 覆盖",
        "",
        *_coverage_table(bundle),
        "",
        f"## top {TOP_K} 的标注构成",
        "",
        "同一条查询的向量 top 10 与 LTR top 10，含没有相关文档的查询。",
        "明确 0 分是两份标注里 grade 为 0 的文档。",
        "",
        *_mix_table(bundle),
        "",
        "## 线性权重",
        "",
        *_weight_lines(model),
        "",
        *_finding(model, bundle),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries/queries.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/queries/labels.csv"))
    parser.add_argument("--tool-labels", type=Path, default=Path("data/queries/labels_tool.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/ltr-pool-features.md"))
    args = parser.parse_args()
    service = build_search_service(args.artifact_dir)
    if not isinstance(service, BM25SearchService):
        raise TypeError("search service does not expose prepare_ltr")
    if not service.vector_ready or not service.ltr_ready:
        raise RuntimeError("vector index and LTR artifact are required")
    model = load_ltr(ltr_artifact_dir(args.artifact_dir))
    text = render(service, model, args.queries, args.labels, args.tool_labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
