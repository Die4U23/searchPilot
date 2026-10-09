"""索引产物的落盘与加载（``artifacts/search/bm25/``）。

目录布局（build-plan 3.3）::

    <artifact_root>/search/bm25/
        index.json   # InvertedIndex.to_dict()
        meta.json    # IndexMeta.to_dict()

``model_version = "bm25-" + sha256(meta 中确定性字段的 JSON, sort_keys)[:8]``。
确定性字段 = 除 ``model_version`` 与 ``built_at`` 之外的全部 meta 字段
（k1、b、data_version、doc_count、vocab_size、输入 Parquet 的 SHA-256、字段权重、idf 变体、
分析器名、字段列表）。同样的数据与参数重建，一定得到同样的 ``model_version``。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from searchpilot.search.bm25 import DEFAULT_B, DEFAULT_FIELD_WEIGHTS, DEFAULT_K1, IdfVariant
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.inverted_index import InvertedIndex

INDEX_FILENAME = "index.json"
META_FILENAME = "meta.json"
DEFAULT_ANALYZER_NAME = "searchpilot.search.normalize.analyze"
MODEL_VERSION_PREFIX = "bm25-"


def bm25_artifact_dir(artifact_root: Path) -> Path:
    """``<artifact_root>/search/bm25``。"""
    return artifact_root / "search" / "bm25"


def sha256_file(path: Path) -> str:
    """文件内容的 SHA-256 十六进制摘要（流式读取）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_model_version(deterministic: Mapping[str, Any]) -> str:
    """``bm25-`` + 确定性字段 JSON（sort_keys、紧凑分隔符）的 SHA-256 前 8 位。"""
    blob = json.dumps(deterministic, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return MODEL_VERSION_PREFIX + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True, slots=True)
class IndexMeta:
    """``meta.json`` 的内容。"""

    model_version: str
    k1: float
    b: float
    data_version: str
    doc_count: int
    vocab_size: int
    built_at: str
    input_sha256: Mapping[str, str]
    field_weights: Mapping[str, float]
    idf_variant: str = "lucene"
    analyzer: str = DEFAULT_ANALYZER_NAME
    fields: tuple[str, ...] = ("abstract", "title")

    @staticmethod
    def deterministic_payload(
        *,
        k1: float,
        b: float,
        data_version: str,
        doc_count: int,
        vocab_size: int,
        input_sha256: Mapping[str, str],
        field_weights: Mapping[str, float],
        idf_variant: str,
        analyzer: str,
        fields: tuple[str, ...],
    ) -> dict[str, Any]:
        """参与 ``model_version`` 哈希的字段（不含 ``built_at`` 与 ``model_version`` 本身）。"""
        return {
            "k1": float(k1),
            "b": float(b),
            "data_version": data_version,
            "doc_count": int(doc_count),
            "vocab_size": int(vocab_size),
            "input_sha256": dict(sorted(input_sha256.items())),
            "field_weights": {name: float(w) for name, w in sorted(field_weights.items())},
            "idf_variant": idf_variant,
            "analyzer": analyzer,
            "fields": list(fields),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.deterministic_payload(
            k1=self.k1,
            b=self.b,
            data_version=self.data_version,
            doc_count=self.doc_count,
            vocab_size=self.vocab_size,
            input_sha256=self.input_sha256,
            field_weights=self.field_weights,
            idf_variant=self.idf_variant,
            analyzer=self.analyzer,
            fields=self.fields,
        )
        payload["model_version"] = self.model_version
        payload["built_at"] = self.built_at
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IndexMeta:
        try:
            return cls(
                model_version=str(payload["model_version"]),
                k1=float(payload["k1"]),
                b=float(payload["b"]),
                data_version=str(payload["data_version"]),
                doc_count=int(payload["doc_count"]),
                vocab_size=int(payload["vocab_size"]),
                built_at=str(payload["built_at"]),
                input_sha256={str(k): str(v) for k, v in payload["input_sha256"].items()},
                field_weights={str(k): float(v) for k, v in payload["field_weights"].items()},
                idf_variant=str(payload.get("idf_variant", "lucene")),
                analyzer=str(payload.get("analyzer", DEFAULT_ANALYZER_NAME)),
                fields=tuple(str(f) for f in payload.get("fields", ("abstract", "title"))),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SearchNotReadyError(f"meta.json is malformed: {exc}") from exc


def make_meta(
    index: InvertedIndex,
    *,
    data_version: str,
    input_sha256: Mapping[str, str],
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    field_weights: Mapping[str, float] | None = None,
    idf_variant: IdfVariant = "lucene",
    analyzer: str = DEFAULT_ANALYZER_NAME,
    built_at: datetime | None = None,
) -> IndexMeta:
    """根据索引与构建参数生成 meta，并计算 ``model_version``。

    ``field_weights`` 缺省时取 :data:`DEFAULT_FIELD_WEIGHTS` 中与索引字段相交的部分，
    索引里其余字段权重记为 1.0，这样 meta 对权重的描述是完整的。
    """
    base = DEFAULT_FIELD_WEIGHTS if field_weights is None else field_weights
    weights = {name: float(base.get(name, 1.0)) for name in sorted(index.fields)}
    fields = tuple(sorted(index.fields))
    deterministic = IndexMeta.deterministic_payload(
        k1=k1,
        b=b,
        data_version=data_version,
        doc_count=index.doc_count,
        vocab_size=index.vocab_size,
        input_sha256=input_sha256,
        field_weights=weights,
        idf_variant=idf_variant,
        analyzer=analyzer,
        fields=fields,
    )
    stamp = (built_at or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    return IndexMeta(
        model_version=compute_model_version(deterministic),
        k1=float(k1),
        b=float(b),
        data_version=data_version,
        doc_count=index.doc_count,
        vocab_size=index.vocab_size,
        built_at=stamp,
        input_sha256=dict(sorted(input_sha256.items())),
        field_weights=weights,
        idf_variant=idf_variant,
        analyzer=analyzer,
        fields=fields,
    )


def save_index(index: InvertedIndex, meta: IndexMeta, directory: Path) -> None:
    """把 ``index.json`` 与 ``meta.json`` 写入 ``directory``（通常是 ``bm25_artifact_dir(root)``）。

    会校验 meta 中的 ``doc_count`` / ``vocab_size`` / ``fields`` 与索引一致，
    防止写出自相矛盾的产物。
    """
    if meta.doc_count != index.doc_count or meta.vocab_size != index.vocab_size:
        raise ValueError("meta doc_count/vocab_size do not match the index")
    if tuple(meta.fields) != tuple(sorted(index.fields)):
        raise ValueError("meta fields do not match the index fields")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / INDEX_FILENAME).write_text(
        json.dumps(index.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (directory / META_FILENAME).write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_index(directory: Path) -> tuple[InvertedIndex, IndexMeta]:
    """从 ``directory`` 读回索引与 meta；缺文件或损坏时抛 :class:`SearchNotReadyError`。"""
    index_path = directory / INDEX_FILENAME
    meta_path = directory / META_FILENAME
    missing = [p.name for p in (index_path, meta_path) if not p.is_file()]
    if missing:
        raise SearchNotReadyError(
            f"BM25 index artifacts missing in {directory}: {', '.join(missing)}."
            " Run scripts/build_index.py first."
        )
    try:
        meta = IndexMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        index = InvertedIndex.from_dict(json.loads(index_path.read_text(encoding="utf-8")))
    except SearchNotReadyError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SearchNotReadyError(
            f"BM25 index artifacts in {directory} are corrupt: {exc}"
        ) from exc
    if index.doc_count != meta.doc_count:
        raise SearchNotReadyError(
            f"index.json has {index.doc_count} documents but meta.json says {meta.doc_count}"
        )
    return index, meta
