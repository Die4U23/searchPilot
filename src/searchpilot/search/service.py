"""在线搜索服务：实现 ``ports.SearchPort``。

契约（build-plan 第 2 节）：

- ``build_search_service(artifact_dir) -> SearchPort``：从 ``artifact_dir/search/bm25/`` 加载；
  缺产物抛 :class:`SearchNotReadyError`。
- ``InMemorySearchService(documents) -> SearchPort``：测试与小语料用，直接建索引。

``mode="bm25"`` 始终可用。``vector`` / ``hybrid`` 在加载了 ``artifacts/search/vector/``
之后可用；缺产物或未安装 ``sentence-transformers`` 时抛 :class:`SearchNotReadyError`。
``ltr`` 还要 ``artifacts/search/ltr/``。缺向量索引或缺 LTR 产物时抛
:class:`SearchNotReadyError`，不把故障伪装成空结果。

关于 ``filters``：``SearchPort.search`` 的签名固定为 ``(query, limit, mode)``，因此 category
过滤通过**额外方法** :meth:`BM25SearchService.search_with_filters` 提供；``search`` 等价于
``filters=None``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from searchpilot.ports import Document, SearchHit, SearchMode, SearchPort, SearchResult
from searchpilot.search.artifacts import IndexMeta, bm25_artifact_dir, load_index, make_meta
from searchpilot.search.bm25 import DEFAULT_B, DEFAULT_K1, BM25Scorer, DocFilter, IdfVariant
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from searchpilot.search.inverted_index import InvertedIndex
from searchpilot.search.ltr import (
    CATEGORY_MODE_K,
    FEATURE_NAMES,
    RETRIEVAL_DEPTH,
    LtrModel,
    LtrPrep,
    QuerySignals,
    feature_matrix,
    load_ltr,
    ltr_artifact_dir,
    mode_category,
    rank_by_score,
)
from searchpilot.search.normalize import Analyzer, analyze, normalize_query
from searchpilot.search.vector_index import (
    QueryEncoder,
    SentenceTransformerQueryEncoder,
    VectorIndex,
    hybrid_model_version,
    l2_normalize_vector,
    load_vector_index,
)

__all__ = [
    "BM25SearchService",
    "InMemorySearchService",
    "SearchNotReadyError",
    "build_search_service",
    "load_bm25_service",
]

SUPPORTED_MODES: Final[frozenset[str]] = frozenset({"bm25"})
KNOWN_MODES: Final[frozenset[str]] = frozenset({"bm25", "vector", "hybrid", "ltr"})
SUPPORTED_FILTER_KEYS: Final[frozenset[str]] = frozenset({"category"})
FUSION_DEPTH: Final[int] = 100


class BM25SearchService:
    """把分析器、倒排索引与 BM25 打分器组装成 ``SearchPort``。"""

    def __init__(
        self,
        index: InvertedIndex,
        scorer: BM25Scorer,
        model_version: str,
        *,
        analyzer: Analyzer = analyze,
    ) -> None:
        if scorer.index is not index:
            raise ValueError("scorer must be bound to the same index")
        self._index = index
        self._scorer = scorer
        self._model_version = model_version
        self._analyzer = analyzer
        self._item_pos = {item_id: i for i, item_id in enumerate(index.item_ids)}
        self._categories = {
            item_id: category
            for item_id, category in zip(index.item_ids, index.categories, strict=True)
        }
        self._vectors: VectorIndex | None = None
        self._vector_pos: dict[str, int] | None = None
        self._encode_query: QueryEncoder | None = None
        self._hybrid_model_version: str | None = None
        self._ltr: LtrModel | None = None

    def attach_vector_index(self, index: VectorIndex, encode_query: QueryEncoder) -> None:
        """挂上向量索引后，``vector`` 与 ``hybrid`` 才可用。``ltr`` 还要 LTR 产物。"""
        self._vectors = index
        self._vector_pos = {item_id: i for i, item_id in enumerate(index.item_ids)}
        self._encode_query = encode_query
        self._hybrid_model_version = hybrid_model_version(
            self._model_version, index.model_version, DEFAULT_RRF_K
        )

    def attach_ltr(self, model: LtrModel) -> None:
        """挂上 LTR 产物。没有向量索引时 ``mode="ltr"`` 仍然不可用。"""
        if len(model.weights) != len(FEATURE_NAMES):
            raise ValueError("LTR weight width does not match features")
        self._ltr = model

    @property
    def vector_ready(self) -> bool:
        """向量索引与查询编码器都已挂上。"""
        return self._vectors is not None and self._encode_query is not None

    @property
    def ltr_ready(self) -> bool:
        """LTR 产物已挂上。在线 ``mode="ltr"`` 还要求 :attr:`vector_ready`。"""
        return self._ltr is not None

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def index(self) -> InvertedIndex:
        return self._index

    @property
    def scorer(self) -> BM25Scorer:
        return self._scorer

    @property
    def doc_count(self) -> int:
        return self._index.doc_count

    def analyze(self, query: str) -> list[str]:
        """暴露分析器，便于评估脚本在失败样例里打印查询词项。"""
        return self._analyzer(query)

    def search(self, query: str, limit: int, mode: SearchMode) -> SearchResult:
        return self.search_with_filters(query, limit, mode, filters=None)

    def search_with_filters(
        self,
        query: str,
        limit: int,
        mode: SearchMode,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> SearchResult:
        """带可选过滤条件的检索。目前支持 ``{"category": <str>}``（casefold 后精确匹配）。

        - 空查询对 BM25 返回空 ``hits``。向量、混合与 LTR 在归一化结果为空字符串时同样返回空结果。
        - ``limit < 1`` 视为调用方错误，抛 ``ValueError``（API 层已经校验 1–100）。
        - 未加载向量索引时的 ``vector`` / ``hybrid`` / ``ltr``，以及未加载 LTR 产物时的 ``ltr``，
          抛 :class:`SearchNotReadyError`。完全未知的 ``mode`` 抛 ``ValueError``。
        """
        if mode not in KNOWN_MODES:
            raise ValueError(f"unknown search mode: {mode!r}")
        if mode in {"vector", "hybrid", "ltr"} and not self.vector_ready:
            raise SearchNotReadyError(
                f"search mode {mode!r} is not available; vector index is not loaded"
            )
        if mode == "ltr" and not self.ltr_ready:
            raise SearchNotReadyError(
                f"search mode {mode!r} is not available; LTR artifact is not loaded"
            )
        if limit < 1:
            raise ValueError("limit must be >= 1")

        normalized = normalize_query(query)
        if mode == "bm25":
            return self._search_bm25(query, normalized, limit, filters)
        if not normalized:
            version = self._result_version(mode)
            return SearchResult(
                normalized_query=normalized, hits=(), model_version=version, mode=mode
            )
        if mode == "vector":
            return self._search_vector(query, normalized, limit, filters)
        if mode == "hybrid":
            return self._search_hybrid(query, normalized, limit, filters)
        return self._search_ltr(query, normalized, limit, filters)

    def _result_version(self, mode: SearchMode) -> str:
        if mode == "vector":
            assert self._vectors is not None
            return self._vectors.model_version
        if mode == "hybrid":
            assert self._hybrid_model_version is not None
            return self._hybrid_model_version
        if mode == "ltr":
            assert self._ltr is not None
            return self._ltr.model_version
        return self._model_version

    def _category(self, filters: Mapping[str, str] | None) -> str | None:
        if not filters:
            return None
        unknown = set(filters) - SUPPORTED_FILTER_KEYS
        if unknown:
            raise ValueError(f"unsupported filter keys: {sorted(unknown)}")
        return filters.get("category")

    def _search_bm25(
        self,
        query: str,
        normalized: str,
        limit: int,
        filters: Mapping[str, str] | None,
    ) -> SearchResult:
        terms = self._analyzer(query)
        if not terms:
            return SearchResult(
                normalized_query=normalized, hits=(), model_version=self._model_version, mode="bm25"
            )
        doc_filter = self._build_doc_filter(filters)
        ranked = self._scorer.search(terms, limit, doc_filter=doc_filter)
        hits = tuple(
            SearchHit(item_id=self._index.item_ids[doc_idx], score=score, rank=rank, channel="bm25")
            for rank, (doc_idx, score) in enumerate(ranked, start=1)
        )
        return SearchResult(
            normalized_query=normalized, hits=hits, model_version=self._model_version, mode="bm25"
        )

    def _search_vector(
        self,
        query: str,
        normalized: str,
        limit: int,
        filters: Mapping[str, str] | None,
    ) -> SearchResult:
        assert self._vectors is not None and self._encode_query is not None
        ranked = self._vectors.search(
            self._encode_query(query), limit, category=self._category(filters)
        )
        hits = tuple(
            SearchHit(item_id=item_id, score=score, rank=rank, channel="vector")
            for rank, (item_id, score) in enumerate(ranked, start=1)
        )
        return SearchResult(
            normalized_query=normalized,
            hits=hits,
            model_version=self._vectors.model_version,
            mode="vector",
        )

    def _search_hybrid(
        self,
        query: str,
        normalized: str,
        limit: int,
        filters: Mapping[str, str] | None,
    ) -> SearchResult:
        assert self._vectors is not None and self._encode_query is not None
        depth = max(limit, FUSION_DEPTH)
        terms = self._analyzer(query)
        doc_filter = self._build_doc_filter(filters)
        if terms:
            bm25_ranked = self._scorer.search(terms, depth, doc_filter=doc_filter)
            bm25_ids = [self._index.item_ids[doc_idx] for doc_idx, _score in bm25_ranked]
        else:
            bm25_ids = []
        vector_ranked = self._vectors.search(
            self._encode_query(query), depth, category=self._category(filters)
        )
        vector_ids = [item_id for item_id, _score in vector_ranked]
        fused = reciprocal_rank_fusion([bm25_ids, vector_ids], k=DEFAULT_RRF_K)[:limit]
        hits = tuple(
            SearchHit(item_id=item_id, score=score, rank=rank, channel="rrf")
            for rank, (item_id, score) in enumerate(fused, start=1)
        )
        assert self._hybrid_model_version is not None
        return SearchResult(
            normalized_query=normalized,
            hits=hits,
            model_version=self._hybrid_model_version,
            mode="hybrid",
        )

    def prepare_ltr(
        self,
        query: str,
        extra_ids: Sequence[str] = (),
        *,
        filters: Mapping[str, str] | None = None,
    ) -> LtrPrep:
        """取 BM25 与向量各 top 100，RRF 得到候选，并给候选和 ``extra_ids`` 填分数。

        名次只来自这两份 top-100。不在名单里的文档分数用精确 BM25 / 余弦补上，名次保持缺席
        （特征里记为 101）。训练脚本用 ``extra_ids`` 传入标注文档，使训练和在线共用这一路径。
        """
        if not self.vector_ready:
            raise SearchNotReadyError("vector index is not loaded")
        assert self._vectors is not None and self._encode_query is not None
        terms = self._analyzer(query)
        doc_filter = self._build_doc_filter(filters)
        category = self._category(filters)
        if terms:
            ranked = self._scorer.search(terms, RETRIEVAL_DEPTH, doc_filter=doc_filter)
            bm25_ranked = [(self._index.item_ids[doc_idx], score) for doc_idx, score in ranked]
        else:
            bm25_ranked = []
        encoded = self._encode_query(query)
        vector_ranked = self._vectors.search(encoded, RETRIEVAL_DEPTH, category=category)
        fused = reciprocal_rank_fusion(
            [
                [item_id for item_id, _score in bm25_ranked],
                [item_id for item_id, _score in vector_ranked],
            ],
            k=DEFAULT_RRF_K,
        )
        candidates = tuple(item_id for item_id, _score in fused)
        bm25_score = {item_id: float(score) for item_id, score in bm25_ranked}
        bm25_rank = {item_id: rank for rank, (item_id, _score) in enumerate(bm25_ranked, start=1)}
        vector_score = {item_id: float(score) for item_id, score in vector_ranked}
        vector_rank = {
            item_id: rank for rank, (item_id, _score) in enumerate(vector_ranked, start=1)
        }
        query_vec = l2_normalize_vector(np.asarray(encoded, dtype=np.float32))
        seen: set[str] = set()
        focus: list[str] = []
        for item_id in (*candidates, *extra_ids):
            if item_id in seen:
                continue
            seen.add(item_id)
            focus.append(item_id)
        for item_id in focus:
            if item_id not in bm25_score:
                bm25_score[item_id] = self._bm25_score(terms, item_id)
            if item_id not in vector_score:
                vector_score[item_id] = self._vector_cosine(query_vec, item_id)
        top_categories = [
            self._categories[item_id]
            for item_id, _score in bm25_ranked[:CATEGORY_MODE_K]
            if item_id in self._categories
        ]
        signals = QuerySignals(
            bm25_score=bm25_score,
            bm25_rank=bm25_rank,
            vector_score=vector_score,
            vector_rank=vector_rank,
            category_mode=mode_category(top_categories),
        )
        return LtrPrep(signals=signals, candidates=candidates)

    def _search_ltr(
        self,
        query: str,
        normalized: str,
        limit: int,
        filters: Mapping[str, str] | None,
    ) -> SearchResult:
        assert self._ltr is not None
        prep = self.prepare_ltr(query, (), filters=filters)
        version = self._ltr.model_version
        if not prep.candidates:
            return SearchResult(
                normalized_query=normalized, hits=(), model_version=version, mode="ltr"
            )
        raw = feature_matrix(
            prep.candidates,
            prep.signals,
            categories=self._categories,
            clicks=self._ltr.clicks,
            title_chars=self._ltr.title_chars,
            first_seen_at=self._ltr.first_seen_at,
            reference_time=self._ltr.reference_time,
        )
        ranked = rank_by_score(prep.candidates, self._ltr.score_rows(raw), limit)
        hits = tuple(
            SearchHit(item_id=item_id, score=score, rank=rank, channel="ltr")
            for rank, (item_id, score) in enumerate(ranked, start=1)
        )
        return SearchResult(
            normalized_query=normalized, hits=hits, model_version=version, mode="ltr"
        )

    def _bm25_score(self, terms: Sequence[str], item_id: str) -> float:
        if not terms:
            return 0.0
        pos = self._item_pos.get(item_id)
        if pos is None:
            return 0.0
        return float(self._scorer.score(terms, pos))

    def _vector_cosine(self, query_vec: np.ndarray, item_id: str) -> float:
        if self._vectors is None or self._vector_pos is None:
            return 0.0
        row = self._vector_pos.get(item_id)
        if row is None or float(np.linalg.norm(query_vec)) == 0.0:
            return 0.0
        return float(self._vectors.matrix[row] @ query_vec)

    def _build_doc_filter(self, filters: Mapping[str, str] | None) -> DocFilter | None:
        if not filters:
            return None
        unknown = set(filters) - SUPPORTED_FILTER_KEYS
        if unknown:
            raise ValueError(f"unsupported filter keys: {sorted(unknown)}")
        category = filters.get("category")
        if category is None:
            return None
        categories = self._index.categories
        wanted = category.casefold()

        def _allowed(doc_idx: int) -> bool:
            return categories[doc_idx].casefold() == wanted

        return _allowed


class InMemorySearchService(BM25SearchService):
    """从文档序列直接建索引的 ``SearchPort`` 实现（测试与小语料）。

    ``model_version`` 由与落盘产物相同的规则计算（``data_version="in-memory"``，
    输入 SHA-256 为空），因此同一批文档与参数会得到稳定的版本号。
    """

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        field_weights: Mapping[str, float] | None = None,
        idf_variant: IdfVariant = "lucene",
        analyzer: Analyzer = analyze,
    ) -> None:
        index = InvertedIndex.build_from_documents(documents, analyzer)
        scorer = BM25Scorer(index, k1=k1, b=b, field_weights=field_weights, idf_variant=idf_variant)
        meta = make_meta(
            index,
            data_version="in-memory",
            input_sha256={},
            k1=k1,
            b=b,
            field_weights=scorer.field_weights,
            idf_variant=idf_variant,
        )
        super().__init__(index, scorer, meta.model_version, analyzer=analyzer)
        self._meta = meta

    @property
    def meta(self) -> IndexMeta:
        return self._meta


def service_from_artifacts(
    index: InvertedIndex, meta: IndexMeta, *, analyzer: Analyzer = analyze
) -> BM25SearchService:
    """用已加载的索引与 meta 组装服务（打分参数全部来自 meta）。"""
    if meta.idf_variant not in ("lucene", "okapi"):
        raise SearchNotReadyError(f"unsupported idf_variant in meta.json: {meta.idf_variant!r}")
    idf_variant: IdfVariant = "okapi" if meta.idf_variant == "okapi" else "lucene"
    scorer = BM25Scorer(
        index,
        k1=meta.k1,
        b=meta.b,
        field_weights=meta.field_weights,
        idf_variant=idf_variant,
    )
    return BM25SearchService(index, scorer, meta.model_version, analyzer=analyzer)


def load_bm25_service(artifact_dir: Path) -> BM25SearchService:
    """从 ``artifact_dir/search/bm25/`` 加载索引并组装具体的 :class:`BM25SearchService`。

    与 :func:`build_search_service` 相同，但返回具体类型，供评估脚本访问 ``analyze`` / ``scorer``。
    缺产物或产物损坏时抛 :class:`SearchNotReadyError`。
    """
    index, meta = load_index(bm25_artifact_dir(Path(artifact_dir)))
    return service_from_artifacts(index, meta)


def build_search_service(artifact_dir: Path) -> SearchPort:
    """契约工厂：从 ``artifact_dir/search/bm25/`` 加载索引并返回 ``SearchPort``。

    缺 BM25 产物或产物损坏时抛 :class:`SearchNotReadyError`。若同级
    ``search/vector/meta.json`` 存在，则挂上向量索引；向量文件损坏同样抛错。
    若 ``search/ltr/meta.json`` 存在，则挂上 LTR 产物。
    """
    service = load_bm25_service(artifact_dir)
    vector_dir = Path(artifact_dir) / "search" / "vector"
    if (vector_dir / "meta.json").is_file():
        index = load_vector_index(vector_dir)
        service.attach_vector_index(
            index, SentenceTransformerQueryEncoder(index.meta.model_name, index.meta.query_prefix)
        )
    ltr_dir = ltr_artifact_dir(Path(artifact_dir))
    if (ltr_dir / "meta.json").is_file():
        service.attach_ltr(load_ltr(ltr_dir))
    return service
