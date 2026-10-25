"""Load fixtures preserve provenance and never become research labels."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts import build_load_queries as module

from searchpilot.ports import Document
from searchpilot.search.artifacts import make_meta, save_index, sha256_file
from searchpilot.search.inverted_index import InvertedIndex
from searchpilot.search.normalize import analyze, normalize_query


@pytest.fixture
def documents():
    return [
        Document(
            item_id=f"N{i:03d}",
            title=f"Reporter Person{chr(97 + i // 26)}{chr(97 + i % 26)} visits City "
            f"district {i}, {'café' if i % 2 else 'local'} news!",
            abstract="A detailed report about regional sports and markets with historical "
            "context, transport, education, and local community interviews. " * 3,
            category=("news", "sports", "travel")[i % 3],
            subcategory="fixture",
        )
        for i in range(90)
    ]


@pytest.fixture
def references():
    return [
        {
            "query_id": "original_1",
            "query_text": "reporter",
            "query_type": "exact_entity",
            "split": "test",
        }
    ]


def generate(documents, references, quotas=None):
    vocabulary = {term for doc in documents for term in analyze(doc.title + " " + doc.abstract)}
    return module.generate_queries(
        documents,
        references,
        vocabulary,
        indexed_ids={doc.item_id for doc in documents},
        quotas=quotas if quotas is not None else {family: 2 for family in module.QUOTAS},
    )


def test_generation_is_deterministic_and_independent_of_item_row_order(documents, references):
    first = generate(documents, references)
    assert first == generate(documents, references)
    assert first == generate(list(reversed(documents)), references)


def test_quotas_source_uniqueness_normalized_uniqueness_and_lengths(documents, references):
    rows, details = generate(documents, references)
    assert len(rows) == 15
    assert Counter(row["family"] for row in rows) == {
        "reference": 1,
        **dict.fromkeys(module.QUOTAS, 2),
    }
    assert len({row["normalized_query"] for row in rows}) == len(rows)
    assert len({tuple(row["analyzed_terms"]) for row in rows}) == len(rows)
    assert len({row["source_item_id"] for row in rows[1:]}) == 14
    assert details["synthetic_unique_items"] == 14
    assert all(1 <= row["char_length"] <= 256 for row in rows)
    assert all(128 <= row["char_length"] <= 256 for row in rows if row["family"] == "long")
    assert all(1 <= row["analyzed_token_count"] <= 3 for row in rows if row["family"] == "short")
    assert {key: rows[0][key] for key in references[0]} == references[0]


def test_natural_unicode_and_fullwidth_have_distinct_truthful_provenance(documents, references):
    rows, _ = generate(documents, references)
    source = {doc.item_id: doc for doc in documents}
    for row in rows:
        if row["family"] == "unicode_natural":
            assert row["query_text"] == source[row["source_item_id"]].title
            assert "é" in row["query_text"]
        elif row["family"] == "unicode_fullwidth":
            original = source[row["source_item_id"]].title
            assert original.isascii() and not row["query_text"].isascii()
            assert normalize_query(row["query_text"]) == normalize_query(original)
            assert row["transformation"].endswith("nfkc_equivalent")


def test_category_round_robin_does_not_follow_source_frequency(documents, references):
    rows, _ = generate(documents, references, {"title": 9})
    assert Counter(row["category"] for row in rows[1:]) == {"news": 3, "sports": 3, "travel": 3}


def test_duplicate_references_are_preserved_and_disclosed(documents, references):
    duplicate = {**references[0], "query_id": "original_2", "query_text": "REPORTER"}
    rows, details = generate(documents, references + [duplicate], {"title": 2})
    assert [row["query_text"] for row in rows[:2]] == ["reporter", "REPORTER"]
    assert details["reference_duplicates_preserved"] == {
        "normalized": [["original_1", "original_2"]],
        "analyzed_terms": [["original_1", "original_2"]],
    }
    assert all(tuple(row["analyzed_terms"]) != ("report",) for row in rows[2:])


def test_insufficient_quota_is_explicit_and_does_not_silently_shrink(documents, references):
    with pytest.raises(ValueError, match=r"insufficient unique candidates for title: 3/4"):
        generate(documents[:3], references, {"title": 4})
    with pytest.raises(ValueError, match="duplicate source item_id"):
        generate(documents + documents[:1], references, {"title": 1})


def test_no_indexed_term_cannot_enter_fixture(documents, references):
    with pytest.raises(ValueError, match="reference query is invalid or has no indexed term"):
        module.generate_queries(
            documents, references, set(), indexed_ids=set(), quotas={"title": 1}
        )


def test_files_are_reproducible_utf8_lf_exclusive_and_inputs_unchanged(
    tmp_path, documents, references
):
    items = tmp_path / "items.parquet"
    pq.write_table(pa.Table.from_pylist([asdict(doc) for doc in documents]), items)
    reference = tmp_path / "references.csv"
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=list(references[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(references)
    reference.write_bytes(content.getvalue().encode("utf-8"))
    index = InvertedIndex.build_from_documents(documents)
    meta = make_meta(
        index, data_version="unit-fixture", input_sha256={"items.parquet": sha256_file(items)}
    )
    index_dir = tmp_path / "index"
    save_index(index, meta, index_dir)
    protected = [items, reference, index_dir / "index.json", index_dir / "meta.json"]
    before = {path: path.read_bytes() for path in protected}
    first, second = tmp_path / "first", tmp_path / "second"
    quotas = {family: 1 for family in module.QUOTAS}
    module.build_fixture(items, reference, index_dir, first, quotas=quotas)
    module.build_fixture(items, reference, index_dir, second, quotas=quotas)
    for filename in ("queries.csv", "manifest.json"):
        raw = (first / filename).read_bytes()
        assert raw == (second / filename).read_bytes()
        assert b"\r\n" not in raw and not raw.startswith(b"\xef\xbb\xbf")
        raw.decode("utf-8")
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["queries_sha256"] == sha256_file(first / "queries.csv")
    assert manifest["input_sha256"]["items.parquet"] == sha256_file(items)
    assert manifest["reference_count"] == 1 and manifest["synthetic_count"] == 7
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        module.build_fixture(items, reference, index_dir, first, quotas=quotas)
    assert {path: path.read_bytes() for path in protected} == before
    items.write_bytes(b"modified input")
    with pytest.raises(ValueError, match="does not match the frozen BM25"):
        module.build_fixture(items, reference, index_dir, tmp_path / "bad", quotas=quotas)
    assert not (tmp_path / "bad").exists()
