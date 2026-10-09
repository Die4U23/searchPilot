"""从 ``data/processed/<data_version>/items.parquet`` 构建 BM25 倒排索引产物。

用法::

    python scripts/build_index.py --data-dir ./data --artifact-dir ./artifacts [--data-version X]

产物写到 ``<artifact-dir>/search/bm25/{index.json,meta.json}``，并打印
``model_version`` / ``doc_count`` / ``vocab_size``。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

from searchpilot.ports import Document
from searchpilot.search.artifacts import bm25_artifact_dir, make_meta, save_index, sha256_file
from searchpilot.search.bm25 import DEFAULT_B, DEFAULT_K1
from searchpilot.search.inverted_index import InvertedIndex

ITEMS_FILENAME = "items.parquet"


class BuildIndexError(Exception):
    """输入缺失或不合法；主函数把它转成非零退出码。"""


def resolve_data_version(processed_dir: Path, requested: str | None) -> str:
    """``requested`` 为空时要求 ``processed/`` 下恰好有一个版本目录。"""
    if requested:
        if not (processed_dir / requested).is_dir():
            raise BuildIndexError(f"data version directory not found: {processed_dir / requested}")
        return requested
    if not processed_dir.is_dir():
        raise BuildIndexError(
            f"processed data directory not found: {processed_dir}."
            " Run scripts/build_dataset.py first."
        )
    versions = sorted(p.name for p in processed_dir.iterdir() if p.is_dir())
    if len(versions) != 1:
        raise BuildIndexError(
            f"expected exactly one data version under {processed_dir}, found {versions};"
            " pass --data-version explicitly."
        )
    return versions[0]


def load_documents(items_path: Path) -> list[Document]:
    """用 pyarrow 读 ``items.parquet`` 并转成 ``Document``（``abstract`` 为空值时置空串）。"""
    if not items_path.is_file():
        raise BuildIndexError(f"items parquet not found: {items_path}")
    columns = ["item_id", "title", "abstract", "category", "subcategory"]
    try:
        table = pq.read_table(items_path, columns=columns)
    except Exception as exc:  # pyarrow 抛出的异常类型不统一
        raise BuildIndexError(f"failed to read {items_path}: {exc}") from exc
    rows = table.to_pylist()
    docs: list[Document] = []
    for row in rows:
        docs.append(
            Document(
                item_id=str(row["item_id"]),
                title=str(row["title"] or ""),
                abstract=str(row["abstract"] or ""),
                category=str(row["category"] or ""),
                subcategory=str(row["subcategory"] or ""),
            )
        )
    if not docs:
        raise BuildIndexError(f"{items_path} contains no rows")
    return docs


def build(
    data_dir: Path,
    artifact_dir: Path,
    *,
    data_version: str | None,
    k1: float,
    b: float,
) -> tuple[Path, str, int, int]:
    processed_dir = data_dir / "processed"
    version = resolve_data_version(processed_dir, data_version)
    items_path = processed_dir / version / ITEMS_FILENAME
    docs = load_documents(items_path)
    index = InvertedIndex.build_from_documents(docs)
    meta = make_meta(
        index,
        data_version=version,
        input_sha256={ITEMS_FILENAME: sha256_file(items_path)},
        k1=k1,
        b=b,
    )
    out_dir = bm25_artifact_dir(artifact_dir)
    save_index(index, meta, out_dir)
    return out_dir, meta.model_version, meta.doc_count, meta.vocab_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the BM25 inverted index artifacts")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_DATA_DIR", "./data")),
    )
    parser.add_argument(
        "--data-version",
        default=os.environ.get("SEARCHPILOT_DATA_VERSION") or None,
        help="processed/<data_version>; defaults to the only directory present",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_ARTIFACT_DIR", "./artifacts")),
    )
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    args = parser.parse_args(argv)

    try:
        out_dir, model_version, doc_count, vocab_size = build(
            args.data_dir,
            args.artifact_dir,
            data_version=args.data_version,
            k1=args.k1,
            b=args.b,
        )
    except BuildIndexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"artifact_dir={out_dir}")
    print(f"model_version={model_version}")
    print(f"doc_count={doc_count}")
    print(f"vocab_size={vocab_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
