"""用 bge-small 为 items.parquet 构建精确向量索引。

用法::

    python scripts/build_vector_index.py --data-dir ./data --artifact-dir ./artifacts

需要 ``pip install -e ".[vector]"``。产物写到 ``<artifact-dir>/search/vector/``。
文档文本是标题与摘要，查询前缀只在检索时加。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from build_index import ITEMS_FILENAME, BuildIndexError, load_documents, resolve_data_version
from searchpilot.search.artifacts import sha256_file
from searchpilot.search.vector_index import (
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERY_PREFIX,
    VectorIndex,
    document_text,
    l2_normalize_rows,
    make_vector_meta,
    save_vector_index,
)


def build(
    data_dir: Path, artifact_dir: Path, *, data_version: str | None, batch_size: int
) -> tuple[Path, str, int]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise BuildIndexError(
            'sentence-transformers is not installed; pip install -e ".[vector]"'
        ) from exc
    version = resolve_data_version(data_dir / "processed", data_version)
    items_path = data_dir / "processed" / version / ITEMS_FILENAME
    docs = load_documents(items_path)
    texts = [document_text(doc.title, doc.abstract) for doc in docs]
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    array = l2_normalize_rows(np.asarray(matrix, dtype=np.float32))
    meta = make_vector_meta(
        matrix=array,
        model_name=DEFAULT_MODEL_NAME,
        data_version=version,
        input_sha256={ITEMS_FILENAME: sha256_file(items_path)},
        query_prefix=DEFAULT_QUERY_PREFIX,
    )
    index = VectorIndex(
        tuple(doc.item_id for doc in docs),
        tuple(doc.category for doc in docs),
        array,
        meta,
    )
    out_dir = artifact_dir / "search" / "vector"
    save_vector_index(out_dir, index)
    return out_dir, meta.model_version, meta.doc_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the bge-small vector index")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_DATA_DIR", "./data")),
    )
    parser.add_argument(
        "--data-version",
        default=os.environ.get("SEARCHPILOT_DATA_VERSION") or None,
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_ARTIFACT_DIR", "./artifacts")),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        out_dir, model_version, doc_count = build(
            args.data_dir,
            args.artifact_dir,
            data_version=args.data_version,
            batch_size=args.batch_size,
        )
    except BuildIndexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"artifact_dir={out_dir}")
    print(f"model_version={model_version}")
    print(f"doc_count={doc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
