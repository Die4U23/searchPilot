from __future__ import annotations

import json

import pytest

from searchpilot.ports import Document
from searchpilot.search.inverted_index import (
    SINGLE_TEXT_FIELDS,
    FieldIndex,
    InvertedIndex,
)
from searchpilot.search.normalize import analyze

DOCS = [
    Document("N1", "Lakers beat Celtics", "Los Angeles Lakers win in Boston", "sports", "nba"),
    Document("N2", "Celtics trade rumors", "Boston Celtics eye a trade", "sports", "nba"),
    Document("N3", "Rain in Boston", "Heavy rain expected", "weather", "local"),
    Document("N4", "Empty abstract", "", "misc", "misc"),
]


def test_field_index_postings_df_tf() -> None:
    fidx = FieldIndex()
    fidx.add_document(0, ["a", "b", "a"])
    fidx.add_document(1, ["b", "c"])
    fidx.add_document(2, [])
    assert fidx.doc_count == 3
    assert fidx.doc_lengths == [3, 2, 0]
    assert fidx.avg_doc_length == pytest.approx(5 / 3)
    assert fidx.postings["a"] == [(0, 2)]
    assert fidx.postings["b"] == [(0, 1), (1, 1)]
    assert fidx.df("a") == 1
    assert fidx.df("b") == 2
    assert fidx.df("zzz") == 0
    assert fidx.tf("a", 0) == 2
    assert fidx.tf("a", 1) == 0
    assert fidx.tf("b", 1) == 1
    assert fidx.tf("zzz", 0) == 0


def test_field_index_requires_in_order_doc_idx() -> None:
    fidx = FieldIndex()
    fidx.add_document(0, ["a"])
    with pytest.raises(ValueError):
        fidx.add_document(2, ["a"])


def test_build_from_documents_two_fields() -> None:
    index = InvertedIndex.build_from_documents(DOCS, analyze)
    assert index.doc_count == 4
    assert index.item_ids == ["N1", "N2", "N3", "N4"]
    assert index.categories == ["sports", "sports", "weather", "misc"]
    assert set(index.field_names) == {"title", "abstract"}
    title = index.fields["title"]
    abstract = index.fields["abstract"]
    # "celtics" → "celtic" 出现在 N1、N2 的标题
    assert title.postings["celtic"] == [(0, 1), (1, 1)]
    # "boston" 只在摘要字段（N1、N2）和标题字段（N3）
    assert abstract.df("boston") == 2
    assert title.df("boston") == 1
    # 空摘要长度 0
    assert abstract.doc_lengths[3] == 0
    assert index.vocab_size == len(set(title.postings) | set(abstract.postings))


def test_build_single_text_field() -> None:
    index = InvertedIndex.build_from_documents(DOCS, analyze, fields=SINGLE_TEXT_FIELDS)
    assert index.field_names == ("text",)
    text = index.fields["text"]
    assert text.tf("laker", 0) == 2  # 标题 + 摘要各一次
    assert text.doc_lengths[0] == len(
        analyze("Lakers beat Celtics Los Angeles Lakers win in Boston")
    )


def test_duplicate_item_id_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        InvertedIndex.build_from_documents([DOCS[0], DOCS[0]], analyze)


def test_from_tokenized_and_doc_idx_of() -> None:
    index = InvertedIndex.from_tokenized(
        ["B", "A"], {"text": [["x", "y"], ["y"]]}, categories=["c1", "c2"]
    )
    assert index.doc_idx_of("A") == 1
    assert index.doc_idx_of("missing") is None
    assert index.fields["text"].df("y") == 2
    with pytest.raises(ValueError):
        InvertedIndex.from_tokenized(["A", "A"], {"text": [[], []]})
    with pytest.raises(ValueError):
        InvertedIndex.from_tokenized(["A", "B"], {"text": [[]]})


def test_to_dict_roundtrip_via_json() -> None:
    index = InvertedIndex.build_from_documents(DOCS, analyze)
    payload = json.loads(json.dumps(index.to_dict()))
    restored = InvertedIndex.from_dict(payload)
    assert restored.item_ids == index.item_ids
    assert restored.categories == index.categories
    assert restored.field_names == tuple(sorted(index.field_names))
    for name in index.field_names:
        assert restored.fields[name].postings == index.fields[name].postings
        assert restored.fields[name].doc_lengths == index.fields[name].doc_lengths
    # posting 以紧凑 list 存，不是 pickle
    assert payload["fields"]["title"]["postings"]["celtic"] == [[0, 1], [1, 1]]


def test_to_dict_is_deterministic() -> None:
    a = json.dumps(InvertedIndex.build_from_documents(DOCS, analyze).to_dict(), sort_keys=True)
    b = json.dumps(InvertedIndex.build_from_documents(DOCS, analyze).to_dict(), sort_keys=True)
    assert a == b


def test_from_dict_rejects_bad_format_or_mismatch() -> None:
    with pytest.raises(ValueError, match="format"):
        InvertedIndex.from_dict({"format": "nope", "item_ids": [], "fields": {}})
    bad = InvertedIndex.build_from_documents(DOCS, analyze).to_dict()
    bad["item_ids"] = bad["item_ids"][:-1]
    with pytest.raises(ValueError, match="docs"):
        InvertedIndex.from_dict(bad)
