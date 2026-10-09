"""向量索引：行归一化的 NumPy 矩阵，点积即余弦。

默认编码器是 ``BAAI/bge-small-en-v1.5``（384 维）。单元测试注入向量，不下载模型。
``sentence-transformers`` 只在真正编码时导入；没安装时向量模式返回 ``SearchNotReadyError``。
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from searchpilot.search.errors import SearchNotReadyError

EMBEDDINGS_FILENAME = "embeddings.npy"
IDS_FILENAME = "ids.json"
META_FILENAME = "meta.json"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QueryEncoder = Callable[[str], np.ndarray]


@dataclass(frozen=True, slots=True)
class VectorMeta:
    """写入 ``meta.json`` 的确定性描述。``model_version`` 不含 ``built_at``。"""

    model_name: str
    dim: int
    data_version: str
    doc_count: int
    input_sha256: dict[str, str]
    embeddings_sha256: str
    query_prefix: str
    normalized: bool
    built_at: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dim": self.dim,
            "data_version": self.data_version,
            "doc_count": self.doc_count,
            "input_sha256": self.input_sha256,
            "embeddings_sha256": self.embeddings_sha256,
            "query_prefix": self.query_prefix,
            "normalized": self.normalized,
            "built_at": self.built_at,
            "model_version": self.model_version,
        }


def document_text(title: str, abstract: str) -> str:
    """文档侧编码文本。查询前缀不加在文档上。"""
    return f"{title}\n{abstract}".strip()


def embeddings_sha256(matrix: np.ndarray) -> str:
    array = np.ascontiguousarray(matrix, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def vector_model_version(
    *,
    model_name: str,
    dim: int,
    data_version: str,
    doc_count: int,
    input_sha256: dict[str, str],
    embeddings_sha256: str,
    query_prefix: str,
) -> str:
    payload = {
        "model_name": model_name,
        "dim": dim,
        "data_version": data_version,
        "doc_count": doc_count,
        "input_sha256": input_sha256,
        "embeddings_sha256": embeddings_sha256,
        "query_prefix": query_prefix,
        "normalized": True,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "vector-" + hashlib.sha256(blob).hexdigest()[:8]


def hybrid_model_version(bm25_version: str, vector_version: str, rrf_k: int) -> str:
    raw = f"{bm25_version}|{vector_version}|rrf={rrf_k}".encode()
    return "hybrid-" + hashlib.sha256(raw).hexdigest()[:8]


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return (array / safe).astype(np.float32)


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return (array / norm).astype(np.float32)


class VectorIndex:
    """与 ``item_ids`` 行对齐的单位向量矩阵。"""

    def __init__(
        self,
        item_ids: tuple[str, ...],
        categories: tuple[str, ...],
        matrix: np.ndarray,
        meta: VectorMeta,
    ) -> None:
        if len(item_ids) != len(categories):
            raise ValueError("item_ids and categories must have the same length")
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(item_ids):
            raise ValueError("matrix rows must match item_ids")
        if array.shape[1] != meta.dim:
            raise ValueError("matrix dim does not match meta.dim")
        self.item_ids = item_ids
        self.categories = categories
        self.matrix = array
        self.meta = meta

    @property
    def model_version(self) -> str:
        return self.meta.model_version

    def search(
        self, query: np.ndarray, limit: int, *, category: str | None = None
    ) -> list[tuple[str, float]]:
        """余弦降序，并列按 ``item_id`` 升序。``limit < 1`` 抛 ``ValueError``。"""
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if self.matrix.shape[0] == 0:
            return []
        vector = l2_normalize_vector(query)
        if vector.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"query dim {vector.shape[0]} does not match index dim {self.matrix.shape[1]}"
            )
        if float(np.linalg.norm(vector)) == 0.0:
            return []
        scores = self.matrix @ vector
        eligible = np.ones(scores.shape[0], dtype=bool)
        if category is not None:
            wanted = category.casefold()
            eligible = np.fromiter(
                (value.casefold() == wanted for value in self.categories),
                dtype=bool,
                count=len(self.categories),
            )
        scores = np.where(eligible, scores, -np.inf)
        finite_count = int(np.isfinite(scores).sum())
        if finite_count == 0:
            return []
        k = min(limit, finite_count)
        candidates = [index for index in range(scores.shape[0]) if np.isfinite(scores[index])]
        chosen = heapq.nsmallest(
            k,
            candidates,
            key=lambda index: (-float(scores[index]), self.item_ids[index]),
        )
        return [(self.item_ids[index], float(scores[index])) for index in chosen]


def save_vector_index(directory: Path, index: VectorIndex) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / EMBEDDINGS_FILENAME, np.ascontiguousarray(index.matrix, dtype=np.float32))
    ids_payload = {"item_ids": list(index.item_ids), "categories": list(index.categories)}
    (directory / IDS_FILENAME).write_text(
        json.dumps(ids_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / META_FILENAME).write_text(
        json.dumps(index.meta.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_vector_index(directory: Path) -> VectorIndex:
    meta_path = directory / META_FILENAME
    ids_path = directory / IDS_FILENAME
    matrix_path = directory / EMBEDDINGS_FILENAME
    if not meta_path.is_file() or not ids_path.is_file() or not matrix_path.is_file():
        raise SearchNotReadyError(f"vector index is incomplete: {directory}")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    ids_raw = json.loads(ids_path.read_text(encoding="utf-8"))
    try:
        meta = VectorMeta(
            model_name=str(raw["model_name"]),
            dim=int(raw["dim"]),
            data_version=str(raw["data_version"]),
            doc_count=int(raw["doc_count"]),
            input_sha256={str(k): str(v) for k, v in dict(raw["input_sha256"]).items()},
            embeddings_sha256=str(raw["embeddings_sha256"]),
            query_prefix=str(raw["query_prefix"]),
            normalized=bool(raw["normalized"]),
            built_at=str(raw["built_at"]),
            model_version=str(raw["model_version"]),
        )
        item_ids = tuple(str(item) for item in ids_raw["item_ids"])
        categories = tuple(str(item) for item in ids_raw["categories"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SearchNotReadyError(f"vector index meta is invalid: {directory}") from exc
    matrix = np.load(matrix_path)
    digest = embeddings_sha256(matrix)
    if digest != meta.embeddings_sha256:
        raise SearchNotReadyError(f"vector embeddings hash mismatch: {directory}")
    if meta.doc_count != len(item_ids):
        raise SearchNotReadyError(f"vector doc_count does not match ids: {directory}")
    try:
        return VectorIndex(item_ids, categories, matrix, meta)
    except ValueError as exc:
        raise SearchNotReadyError(f"vector index shape is invalid: {directory}") from exc


def make_vector_meta(
    *,
    matrix: np.ndarray,
    model_name: str,
    data_version: str,
    input_sha256: dict[str, str],
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    built_at: datetime | None = None,
) -> VectorMeta:
    array = np.ascontiguousarray(matrix, dtype=np.float32)
    digest = embeddings_sha256(array)
    version = vector_model_version(
        model_name=model_name,
        dim=int(array.shape[1]) if array.ndim == 2 else 0,
        data_version=data_version,
        doc_count=int(array.shape[0]) if array.ndim == 2 else 0,
        input_sha256=input_sha256,
        embeddings_sha256=digest,
        query_prefix=query_prefix,
    )
    stamp = (built_at or datetime.now(tz=UTC)).astimezone(UTC).isoformat(timespec="seconds")
    return VectorMeta(
        model_name=model_name,
        dim=int(array.shape[1]),
        data_version=data_version,
        doc_count=int(array.shape[0]),
        input_sha256=input_sha256,
        embeddings_sha256=digest,
        query_prefix=query_prefix,
        normalized=True,
        built_at=stamp,
        model_version=version,
    )


class SentenceTransformerQueryEncoder:
    """懒加载 bge。模型对象缓存在实例上，避免每次查询重新加载。"""

    def __init__(self, model_name: str, query_prefix: str) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._model: Any = None

    def __call__(self, text: str) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise SearchNotReadyError(
                    "sentence-transformers is not installed; pip install -e .[vector]"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        vector = self._model.encode(
            self._query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        return array
