"""BM25 打分器（多字段加权，Lucene 风格 idf）。

打分公式（对每个查询词项 ``t``、每个字段 ``f``）：

    score(q, d) = Σ_f w_f · Σ_{t∈q} idf_f(t) · tf_{f,d,t}·(k1+1)
        / (tf_{f,d,t} + k1·(1 − b + b·|d_f| / avgdl_f))

- ``idf`` 默认是 Lucene 风格 ``ln(1 + (N − df + 0.5) / (df + 0.5))``，恒为正；
  ``variant="okapi"`` 切换成 rank_bm25.BM25Okapi 的 ``ln((N − df + 0.5) / (df + 0.5))``
  并对负值做 ``epsilon · 平均 idf`` 的下限处理，**只用于交叉验证测试对齐**。
- 每个字段用自己的 ``N``、``df``、``avgdl``（见 ``inverted_index`` 的说明）。
- 查询词项按序列原样累加：重复词项重复计分（与 Lucene 布尔查询 / rank_bm25 一致）。
- 并列分数按 ``item_id`` 升序打破，保证确定性。
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from searchpilot.search.inverted_index import FieldIndex, InvertedIndex

IdfVariant = Literal["lucene", "okapi"]

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
DEFAULT_FIELD_WEIGHTS: Mapping[str, float] = {"title": 2.0, "abstract": 1.0}
"""字段权重默认值：标题命中比摘要命中值钱一倍；索引里不在表内的字段权重为 1.0。"""

DocFilter = Callable[[int], bool]
"""按 ``doc_idx`` 判断文档是否允许进入结果（用于 category 过滤等）。"""


def lucene_idf(doc_count: int, df: int) -> float:
    """Lucene / Elasticsearch 的 BM25 idf：``ln(1 + (N − df + 0.5) / (df + 0.5))``，恒 ≥ 0。"""
    return math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))


def okapi_raw_idf(doc_count: int, df: int) -> float:
    """经典 Okapi / ATIRE idf：``ln((N − df + 0.5) / (df + 0.5))``，df > N/2 时为负。"""
    return math.log((doc_count - df + 0.5) / (df + 0.5))


class BM25Scorer:
    """绑定到一份 :class:`InvertedIndex` 的 BM25 打分器。"""

    def __init__(
        self,
        index: InvertedIndex,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        field_weights: Mapping[str, float] | None = None,
        idf_variant: IdfVariant = "lucene",
        okapi_epsilon: float = 0.25,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be within [0, 1]")
        weights = dict(DEFAULT_FIELD_WEIGHTS if field_weights is None else field_weights)
        unknown = set(weights) - set(index.fields)
        if field_weights is not None and unknown:
            raise ValueError(f"field_weights refer to unknown fields: {sorted(unknown)}")
        self.index = index
        self.k1 = float(k1)
        self.b = float(b)
        self.idf_variant: IdfVariant = idf_variant
        self.okapi_epsilon = float(okapi_epsilon)
        # 字段按名字排序遍历，保证内存建索引与从产物加载后的浮点累加顺序一致。
        self._field_names: tuple[str, ...] = tuple(sorted(index.fields))
        self.field_weights: dict[str, float] = {
            name: float(weights.get(name, 1.0)) for name in self._field_names
        }
        self._okapi_idf: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ idf

    def idf(self, term: str, field_name: str) -> float:
        """某字段上词项的 idf；词项不在该字段词表中时返回 0。"""
        fidx = self.index.fields[field_name]
        df = fidx.df(term)
        if df == 0:
            return 0.0
        if self.idf_variant == "lucene":
            return lucene_idf(fidx.doc_count, df)
        return self._okapi_table(field_name, fidx).get(term, 0.0)

    def _okapi_table(self, field_name: str, fidx: FieldIndex) -> dict[str, float]:
        """复刻 rank_bm25.BM25Okapi 的 idf 表：负值统一替换为 ``epsilon · 平均原始 idf``。"""
        table = self._okapi_idf.get(field_name)
        if table is not None:
            return table
        n = fidx.doc_count
        raw = {term: okapi_raw_idf(n, len(plist)) for term, plist in fidx.postings.items()}
        average = sum(raw.values()) / len(raw) if raw else 0.0
        floor = self.okapi_epsilon * average
        table = {term: (value if value >= 0 else floor) for term, value in raw.items()}
        self._okapi_idf[field_name] = table
        return table

    # ---------------------------------------------------------------- scoring

    def _term_weight(self, tf: int, doc_len: int, avgdl: float) -> float:
        if tf == 0:
            return 0.0
        norm = 1.0 - self.b + self.b * (doc_len / avgdl if avgdl > 0 else 0.0)
        return tf * (self.k1 + 1.0) / (tf + self.k1 * norm)

    def explain(self, query_terms: Sequence[str], doc_idx: int) -> dict[str, dict[str, float]]:
        """返回 ``{field: {term: 加权贡献}}``，用于失败样例诊断。重复词项的贡献会累加。"""
        out: dict[str, dict[str, float]] = {}
        for name in self._field_names:
            fidx = self.index.fields[name]
            weight = self.field_weights[name]
            avgdl = fidx.avg_doc_length
            doc_len = fidx.doc_lengths[doc_idx]
            contributions: dict[str, float] = {}
            for term in query_terms:
                tf = fidx.tf(term, doc_idx)
                if tf == 0:
                    continue
                part = weight * self.idf(term, name) * self._term_weight(tf, doc_len, avgdl)
                contributions[term] = contributions.get(term, 0.0) + part
            if contributions:
                out[name] = contributions
        return out

    def score(self, query_terms: Sequence[str], doc_idx: int) -> float:
        """单篇文档对查询的 BM25 分数（各字段加权求和）。"""
        if not 0 <= doc_idx < self.index.doc_count:
            raise IndexError(f"doc_idx {doc_idx} out of range")
        total = 0.0
        for name in self._field_names:
            fidx = self.index.fields[name]
            weight = self.field_weights[name]
            avgdl = fidx.avg_doc_length
            doc_len = fidx.doc_lengths[doc_idx]
            for term in query_terms:
                tf = fidx.tf(term, doc_idx)
                if tf == 0:
                    continue
                total += weight * self.idf(term, name) * self._term_weight(tf, doc_len, avgdl)
        return total

    def score_all(
        self, query_terms: Sequence[str], *, doc_filter: DocFilter | None = None
    ) -> dict[int, float]:
        """对所有至少命中一个查询词项的文档打分，返回 ``{doc_idx: score}``（无序）。"""
        scores: dict[int, float] = {}
        for name in self._field_names:
            fidx = self.index.fields[name]
            weight = self.field_weights[name]
            avgdl = fidx.avg_doc_length
            doc_lengths = fidx.doc_lengths
            for term in query_terms:
                plist = fidx.postings.get(term)
                if not plist:
                    continue
                idf = self.idf(term, name)
                if idf == 0.0:
                    continue
                for doc_idx, tf in plist:
                    part = weight * idf * self._term_weight(tf, doc_lengths[doc_idx], avgdl)
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + part
        if doc_filter is not None:
            scores = {d: s for d, s in scores.items() if doc_filter(d)}
        return scores

    def search(
        self,
        query_terms: Sequence[str],
        limit: int,
        *,
        doc_filter: DocFilter | None = None,
    ) -> list[tuple[int, float]]:
        """Top-k 检索：返回 ``[(doc_idx, score), ...]``，按分数降序、并列按 item_id 升序。

        用 ``heapq.nsmallest`` 做 top-k 选择，键为 ``(-score, item_id)``；
        空查询或无命中返回空列表。
        """
        if limit <= 0:
            return []
        scores = self.score_all(query_terms, doc_filter=doc_filter)
        if not scores:
            return []
        item_ids = self.index.item_ids
        top = heapq.nsmallest(limit, scores.items(), key=lambda kv: (-kv[1], item_ids[kv[0]]))
        return [(doc_idx, score) for doc_idx, score in top]
