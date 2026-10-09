"""多字段倒排索引。

设计选择：**每个字段单独建一张倒排表**（默认 ``title`` 与 ``abstract`` 两个字段），
每张表各自维护 posting list、文档长度、平均长度与 df。这样做而不是「单字段 + 词项前缀」
的原因：

- BM25 的长度归一化应该按字段做（标题普遍很短，若与摘要拼在一起会被摘要长度冲淡）；
- 字段权重可以在打分期自由调整而无需重建索引；
- 单字段交叉验证（与 rank_bm25 比对）只需把 ``title + " " + abstract`` 作为唯一字段建索引，
  复用同一套代码路径。

posting 用紧凑的 ``[(doc_idx, tf), ...]`` 列表存放（按 ``doc_idx`` 升序），持久化走 JSON，
不用 pickle。
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from searchpilot.ports import Document
from searchpilot.search.normalize import Analyzer, analyze

Posting = tuple[int, int]
"""``(doc_idx, tf)``。"""

FieldExtractor = Callable[[Document], str]

DEFAULT_FIELDS: Mapping[str, FieldExtractor] = {
    "title": lambda doc: doc.title,
    "abstract": lambda doc: doc.abstract,
}
"""默认的两字段配置。"""

SINGLE_TEXT_FIELDS: Mapping[str, FieldExtractor] = {
    "text": lambda doc: f"{doc.title} {doc.abstract}",
}
"""单字段配置（``title + " " + abstract``），用于与 rank_bm25 交叉验证。"""


@dataclass(slots=True)
class FieldIndex:
    """某一字段上的倒排表及长度统计。"""

    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: list[int] = field(default_factory=list)

    @property
    def doc_count(self) -> int:
        return len(self.doc_lengths)

    @property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths) / len(self.doc_lengths)

    def df(self, term: str) -> int:
        """文档频率：含有该词项的文档数。"""
        plist = self.postings.get(term)
        return len(plist) if plist is not None else 0

    def tf(self, term: str, doc_idx: int) -> int:
        """某文档中该词项的词频；不存在返回 0。posting 按 doc_idx 升序，用二分查找。"""
        plist = self.postings.get(term)
        if not plist:
            return 0
        pos = bisect_left(plist, doc_idx, key=lambda p: p[0])
        if pos < len(plist) and plist[pos][0] == doc_idx:
            return plist[pos][1]
        return 0

    def add_document(self, doc_idx: int, terms: Sequence[str]) -> None:
        """追加一篇文档；``doc_idx`` 必须严格递增以保持 posting 有序。"""
        if doc_idx != len(self.doc_lengths):
            raise ValueError(
                f"documents must be added in order: expected doc_idx {len(self.doc_lengths)},"
                f" got {doc_idx}"
            )
        self.doc_lengths.append(len(terms))
        for term, tf in sorted(Counter(terms).items()):
            self.postings.setdefault(term, []).append((doc_idx, tf))

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_lengths": list(self.doc_lengths),
            # 词项按字典序写出，保证产物字节级可复现。
            "postings": {
                term: [[doc_idx, tf] for doc_idx, tf in plist]
                for term, plist in sorted(self.postings.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FieldIndex:
        postings = {
            str(term): [(int(p[0]), int(p[1])) for p in plist]
            for term, plist in payload["postings"].items()
        }
        return cls(postings=postings, doc_lengths=[int(x) for x in payload["doc_lengths"]])


@dataclass(slots=True)
class InvertedIndex:
    """多字段倒排索引：``item_ids[doc_idx]`` 给出外部 ID，``fields[name]`` 给出各字段倒排表。

    ``categories`` 与 ``item_ids`` 等长，保留下来是为了让服务层做可选的 category 过滤
    而不必再去读 Parquet。
    """

    item_ids: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    fields: dict[str, FieldIndex] = field(default_factory=dict)

    @property
    def doc_count(self) -> int:
        return len(self.item_ids)

    @property
    def vocab_size(self) -> int:
        """所有字段词项的并集大小。"""
        vocab: set[str] = set()
        for fidx in self.fields.values():
            vocab.update(fidx.postings)
        return len(vocab)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields)

    def doc_idx_of(self, item_id: str) -> int | None:
        """按 item_id 查内部下标（线性扫描；只用于测试与失败样例诊断）。"""
        try:
            return self.item_ids.index(item_id)
        except ValueError:
            return None

    @classmethod
    def build_from_documents(
        cls,
        docs: Sequence[Document],
        analyzer: Analyzer = analyze,
        *,
        fields: Mapping[str, FieldExtractor] = DEFAULT_FIELDS,
    ) -> InvertedIndex:
        """从文档序列建索引。

        ``fields`` 决定索引几个字段以及每个字段的文本来自哪里；默认 ``title`` 与
        ``abstract`` 两个字段。文档顺序即 ``doc_idx``；重复的 ``item_id`` 视为输入错误。
        """
        seen: set[str] = set()
        index = cls(fields={name: FieldIndex() for name in fields})
        for doc_idx, doc in enumerate(docs):
            if doc.item_id in seen:
                raise ValueError(f"duplicate item_id in corpus: {doc.item_id!r}")
            seen.add(doc.item_id)
            index.item_ids.append(doc.item_id)
            index.categories.append(doc.category)
            for name, extract in fields.items():
                index.fields[name].add_document(doc_idx, analyzer(extract(doc)))
        return index

    @classmethod
    def from_tokenized(
        cls,
        item_ids: Sequence[str],
        field_tokens: Mapping[str, Sequence[Sequence[str]]],
        categories: Sequence[str] | None = None,
    ) -> InvertedIndex:
        """直接从已分词的字段 token 序列建索引（测试与交叉验证用）。"""
        n = len(item_ids)
        if len(set(item_ids)) != n:
            raise ValueError("item_ids must be unique")
        if categories is not None and len(categories) != n:
            raise ValueError("categories must align with item_ids")
        index = cls(
            item_ids=list(item_ids),
            categories=list(categories) if categories is not None else [""] * n,
            fields={name: FieldIndex() for name in field_tokens},
        )
        for name, docs_tokens in field_tokens.items():
            if len(docs_tokens) != n:
                raise ValueError(f"field {name!r} has {len(docs_tokens)} docs, expected {n}")
            for doc_idx, tokens in enumerate(docs_tokens):
                index.fields[name].add_document(doc_idx, tokens)
        return index

    def to_dict(self) -> dict[str, Any]:
        """可 JSON 序列化的紧凑表示。"""
        return {
            "format": "searchpilot.inverted_index.v1",
            "item_ids": list(self.item_ids),
            "categories": list(self.categories),
            "fields": {name: fidx.to_dict() for name, fidx in sorted(self.fields.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InvertedIndex:
        fmt = payload.get("format")
        if fmt != "searchpilot.inverted_index.v1":
            raise ValueError(f"unsupported index format: {fmt!r}")
        item_ids = [str(x) for x in payload["item_ids"]]
        categories_raw = payload.get("categories")
        categories = (
            [str(x) for x in categories_raw] if categories_raw is not None else [""] * len(item_ids)
        )
        index = cls(
            item_ids=item_ids,
            categories=categories,
            fields={
                str(name): FieldIndex.from_dict(fpayload)
                for name, fpayload in payload["fields"].items()
            },
        )
        for name, fidx in index.fields.items():
            if fidx.doc_count != index.doc_count:
                raise ValueError(
                    f"field {name!r} has {fidx.doc_count} docs but index has {index.doc_count}"
                )
        return index
