"""Replay fixed Codex text adjudications; never infer grades from retrieval output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq

from searchpilot.search.artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("data/queries/labels_adjudicated_train_val.csv")
ANNOTATOR = "codex-text-adjudication-2026-10-13"
INPUT_SHA256 = {
    "data/queries/queries.csv": "bf19a33e09c81963393d59d9b146b5425fe431a32fc14b7dd955a28f6ee7c49e",
    "data/queries/labels_tool.csv": (
        "4270981fcfed99d9c3e29e1695283077699de96a120ba7789aebe7d4e797a8e2"
    ),
    "data/queries/labels_claude.csv": (
        "3c97c5452c673439cafbdc037d553ba7417b2ec8f92687915ba14953176df9fd"
    ),
    "data/queries/labels.csv": "115d4fb12e6ca60c5b02638b28c6d699b60e5669e83d17accea088c3bde2c731",
    "data/processed/d3a904f41240/items.parquet": (
        "c58e6d4a8c57968454c24b911af7f8aaec82e837c280ae47b4a95e0562e791f9"
    ),
    "docs/data/annotation-guideline.md": (
        "5dba99bff939653b246e374fde2d691b0b30b1472fca0e68d691f52fc836d8b3"
    ),
}
# These are explicit text judgments made in the current Codex session, not model predictions.
DECISIONS = {
    ("q_022", "N18012"): (
        0,
        "查询要求北京获国际贸易机构许可惩罚华盛顿；摘要是华盛顿投诉后中国向美国等国支付的邮费上涨，"
        "机构为全球邮政联盟。行为主体、受益方向和措施均不同；共享中美及国际机构词面不足以建立实质关联。",
    ),
    ("q_024", "N47282"): (
        1,
        "标题摘要确实涉及出资建设可负担住房，提供主题层面的弱关联；"
        "但出资者为JP Morgan Chase、地点为Miami，均不匹配phone maker和Golden State，"
        "不能回答目标承诺。",
    ),
    ("q_029", "N50720"): (
        1,
        "车辆撞入敞开式园林拖车与查询中的车辆碰撞主题相近；"
        "标题摘要仅称vehicle及女性驾驶员，没有学生、校车或载学生车辆的证据，关键主体无法确认，不能判为同一事故。",
    ),
    ("q_031", "N21447"): (
        1,
        "标题摘要仅承诺让房屋洁净的专业清洁技巧，属于清洁方法的上位主题；"
        "未提餐具、洗碗或plate cleaner的效率。"
        "查询措辞也不足以确定指洗碗机还是洗涤剂，不能自行补全具体内容。",
    ),
    ("q_031", "N49925"): (
        1,
        "标题摘要说明低成本的老式家居清洁技巧，只有清洁方法这一上位主题关联；"
        "没有餐具或改善洗碗工具、洗涤剂效果的明确证据，不能假设未展示的33条技巧中含有目标方法。",
    ),
    ("q_059", "N18347"): (
        1,
        "新闻明确涉及火灾造成高速公路封闭，属于同类事件；"
        "但标题摘要定位Santa Barbara County及其西侧，"
        "没有LA地点匹配证据，而且报道的是山火影响高速公路，不能据此确认洛杉矶高速大火。",
    ),
    ("q_061", "N6773"): (
        0,
        "查询是GVSU学生会取消效忠誓词；文档是Drake University学生会拒绝Turning Point USA社团认证。"
        "学校和决策对象均不同，标题摘要未涉及誓词，"
        "共享student senate及政治语境不构成目标事件关联。",
    ),
    ("q_066", "N11915"): (
        1,
        "报道确实是煮水警告，但服务区域为Norristown和Plymouth Township，原因是水管破裂。"
        "没有McKeesport或共同供水事件的证据；只保留同类公共卫生通知的弱关联，不能作为目标地区的警告。",
    ),
    ("q_066", "N39875"): (
        1,
        "Reading煮水命令与查询同属饮水安全警告，摘要还包含煮水操作背景；"
        "明确地点为Reading而非McKeesport，未说明两地关联，不能当成目标地区的现行警告。",
    ),
    ("q_066", "N7953"): (
        1,
        "文本涉及煮水通知这一主题，但地点为Fort Lauderdale且状态为已解除；"
        "未提McKeesport，也没有两地事件关系。只具背景关联，不能回答目标地区是否需要煮水。",
    ),
    ("q_092", "N22801"): (
        1,
        "摘要涉及宾州司法选举广告资金并提Harrisburg，构成法院与政治资金背景；"
        "具体为Superior Court两个席位，未提供Supreme Court或gun donors证据，"
        "不能确认查询中的法院与捐赠者。",
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_text_bytes(path: Path) -> bytes:
    """Decode UTF-8 with universal newlines, then encode UTF-8 without trimming."""
    return path.read_text(encoding="utf-8").encode("utf-8")


def verify_inputs(root: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = root / name
        actual = (
            sha256(canonical_text_bytes(path))
            if path.suffix.lower() in {".csv", ".md"}
            else sha256_file(path)
        )
        if actual != digest:
            raise ValueError(f"Input SHA-256 changed: {name}; new adjudication required")


def read_csv(path: Path) -> list[dict[str, str]]:
    with io.StringIO(canonical_text_bytes(path).decode("utf-8"), newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = row
    return result


def disagreements(
    queries: list[dict[str, str]], tool: list[dict[str, str]], claude: list[dict[str, str]]
) -> tuple[dict, dict, dict, list, int]:
    q = keyed(queries, ("query_id",))
    a = keyed(tool, ("query_id", "item_id"))
    b = keyed(claude, ("query_id", "item_id"))
    for row in queries:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid query split")
    for labels in (a, b):
        for pair, row in labels.items():
            if (pair[0],) not in q or row["grade"] not in {"0", "1", "2", "3"}:
                raise ValueError(f"Invalid query or grade: {pair}")
    scoped_a = {p for p in a if q[(p[0],)]["split"] in {"train", "val"}}
    scoped_b = {p for p in b if q[(p[0],)]["split"] in {"train", "val"}}
    if scoped_a != scoped_b:
        raise ValueError("Train/val label pair coverage differs")
    selected = sorted(p for p in scoped_a if abs(int(a[p]["grade"]) - int(b[p]["grade"])) >= 2)
    return q, a, b, selected, len(scoped_a)


def build_rows(root: Path) -> tuple[list[dict], int]:
    verify_inputs(root, INPUT_SHA256)
    q, a, b, selected, count = disagreements(
        read_csv(root / "data/queries/queries.csv"),
        read_csv(root / "data/queries/labels_tool.csv"),
        read_csv(root / "data/queries/labels_claude.csv"),
    )
    if set(selected) != set(DECISIONS):
        raise ValueError("Current disputes differ from the fixed text decisions")
    item_ids = sorted({p[1] for p in selected})
    items = keyed(
        pq.read_table(
            root / "data/processed/d3a904f41240/items.parquet",
            columns=["item_id", "title", "abstract"],
            filters=[("item_id", "in", item_ids)],
        ).to_pylist(),
        ("item_id",),
    )
    rows = []
    for pair in selected:
        if (pair[1],) not in items:
            raise ValueError(f"Missing item: {pair[1]}")
        query, item = q[(pair[0],)], items[(pair[1],)]
        grade, reason = DECISIONS[pair]
        if grade not in range(4) or not reason.strip():
            raise ValueError(f"Invalid adjudication: {pair}")
        evidence = {"query": query, "item": item, "tool_label": a[pair], "claude_label": b[pair]}
        digest = sha256(
            json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        rows.append(
            {
                "query_id": pair[0],
                "item_id": pair[1],
                "grade": grade,
                "annotator": ANNOTATOR,
                "split": query["split"],
                "query_text": query["query_text"],
                "query_type": query["query_type"],
                "title": item["title"],
                "abstract": item["abstract"],
                "tool_grade": a[pair]["grade"],
                "claude_grade": b[pair]["grade"],
                "reason": reason,
                "input_sha256": digest,
            }
        )
    return rows, count


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def verify_output(path: Path, expected: bytes) -> None:
    if canonical_text_bytes(path) != expected:
        raise ValueError("Adjudication CSV differs from frozen decisions or evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Write only the independent adjudication CSV"
    )
    args = parser.parse_args()
    rows, count = build_rows(ROOT)
    expected = csv_bytes(rows)
    path = ROOT / OUTPUT
    if args.write:
        path.write_bytes(expected)
    verify_output(path, expected)
    verify_inputs(ROOT, INPUT_SHA256)
    print(
        json.dumps(
            {
                "verified": True,
                "train_val_pairs": count,
                "disputes": len(rows),
                "queries": len({r["query_id"] for r in rows}),
                "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
                "grades": {g: sum(r["grade"] == g for r in rows) for g in range(4)},
                "output_sha256": sha256(expected),
                "annotator": ANNOTATOR,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
