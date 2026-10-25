"""Build a deterministic load-only fixture without modifying research queries or labels.

Run ``python -m scripts.build_load_queries --out artifacts/expanded-soak/fixture``.
Reference queries retain their original text and metadata. Generated queries are
synthetic workloads derived from frozen item text, not user queries or relevance labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from scripts.build_index import load_documents, resolve_data_version

from searchpilot.ports import Document
from searchpilot.search import normalize as normalization
from searchpilot.search.artifacts import load_index, sha256_file
from searchpilot.search.normalize import analyze, normalize_query, tokenize

SEED = "searchpilot-expanded-load-v1"
# Reserve scarce natural Unicode items before more general families consume them.
QUOTAS = {
    "unicode_natural": 75,
    "unicode_fullwidth": 75,
    "entity_heuristic": 200,
    "long": 200,
    "punctuation": 150,
    "short": 200,
    "title": 300,
}
CSV_FIELDS = (
    "query_id",
    "query_text",
    "query_type",
    "split",
    "family",
    "source_item_id",
    "category",
    "transformation",
)
ENTITY_SPAN = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b")


def order_key(seed: str, family: str, value: str) -> str:
    return hashlib.sha256(json.dumps([seed, family, value]).encode("utf-8")).hexdigest()


def candidate(document: Document, family: str) -> tuple[str, list[str], str] | None:
    title = document.title.strip()
    if family == "title":
        return title, ["title"], "title.strip"
    if family == "short":
        terms = [term for term in tokenize(normalize_query(title)) if analyze(term)]
        return " ".join(terms[:3]), ["title"], "first_up_to_three_nonstopword_tokens"
    if family == "entity_heuristic":
        match = ENTITY_SPAN.search(title)
        return (
            (match.group(), ["title"], "first_titlecase_span_2_to_4_words_not_ner")
            if match
            else None
        )
    if family == "long":
        combined = " ".join((title + " " + document.abstract).split())
        text = combined if len(combined) <= 256 else combined[:257].rsplit(" ", 1)[0]
        if not 128 <= len(text) <= 256:
            return None
        return text, ["title", "abstract"], "title_plus_abstract_word_boundary_128_to_256"
    if family == "punctuation":
        return (
            (title, ["title"], "original_title_with_unicode_punctuation")
            if any(unicodedata.category(char).startswith("P") for char in title)
            else None
        )
    if family == "unicode_natural":
        return (
            (title, ["title"], "original_title_with_non_ascii_no_replacement_char")
            if any(ord(char) > 127 for char in title) and "\ufffd" not in title
            else None
        )
    if family == "unicode_fullwidth":
        if not title.isascii() or not any(char.isalpha() for char in title):
            return None
        text = "".join(chr(ord(char) + 0xFEE0) if char.isalnum() else char for char in title)
        return text, ["title"], "ascii_letters_and_digits_to_fullwidth_nfkc_equivalent"
    raise ValueError(f"unknown family: {family}")


def _record(row: Mapping[str, str], *, fields: list[str], reference_row: int | None = None):
    text = row["query_text"]
    terms = analyze(text)
    return {
        **row,
        "source_fields": fields,
        "reference_csv_data_row": reference_row,
        "char_length": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "normalized_query": normalize_query(text),
        "analyzed_terms": terms,
        "analyzed_token_count": len(terms),
    }


def generate_queries(
    documents: Sequence[Document],
    references: Sequence[Mapping[str, str]],
    vocabulary: set[str],
    *,
    indexed_ids: set[str],
    seed: str = SEED,
    quotas: Mapping[str, int] | None = None,
) -> tuple[list[dict], dict]:
    """Pure selection; small explicit quotas support tests, not a new sampling API."""
    quotas = dict(QUOTAS if quotas is None else quotas)
    if set(quotas) - QUOTAS.keys() or any(type(n) is not int or n < 0 for n in quotas.values()):
        raise ValueError("quotas must use known families and nonnegative integer counts")
    if len({doc.item_id for doc in documents}) != len(documents):
        raise ValueError("duplicate source item_id")
    if (
        not references
        or any(not row["query_id"].strip() for row in references)
        or len({row["query_id"] for row in references}) != len(references)
    ):
        raise ValueError("reference query IDs must be nonempty and unique")
    records, generated = [], []
    seen_normalized: dict[str, list[str]] = defaultdict(list)
    seen_terms: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for number, row in enumerate(references, 1):
        text = row["query_text"]
        terms = tuple(analyze(text))
        if not text.strip() or not 1 <= len(text) <= 256 or not vocabulary.intersection(terms):
            raise ValueError(
                f"reference query is invalid or has no indexed term: {row['query_id']}"
            )
        record = _record(
            {
                "query_id": row["query_id"],
                "query_text": text,
                "query_type": row["query_type"],
                "split": row["split"],
                "family": "reference",
                "source_item_id": "",
                "category": "",
                "transformation": "original_reference_text_unchanged",
            },
            fields=["query_text"],
            reference_row=number,
        )
        records.append(record)
        seen_normalized[record["normalized_query"]].append(row["query_id"])
        seen_terms[terms].append(row["query_id"])
    reference_duplicates = {
        "normalized": [ids for ids in seen_normalized.values() if len(ids) > 1],
        "analyzed_terms": [ids for ids in seen_terms.values() if len(ids) > 1],
    }
    used_items: set[str] = set()
    selection = {}
    for family in QUOTAS:
        target = quotas.get(family, 0)
        if not target:
            continue
        bins: dict[str, deque[Document]] = defaultdict(deque)
        for doc in sorted(
            documents, key=lambda doc: (order_key(seed, family, doc.item_id), doc.item_id)
        ):
            bins[doc.category].append(doc)
        categories = sorted(bins)
        selected = 0
        rejected: Counter[str] = Counter()
        while selected < target and categories:
            next_categories = []
            for category in categories:
                queue = bins[category]
                while queue:
                    doc = queue.popleft()
                    if doc.item_id not in indexed_ids or doc.item_id in used_items:
                        rejected["not_indexed_or_item_already_selected"] += 1
                        continue
                    proposed = candidate(doc, family)
                    if proposed is None:
                        rejected["family_ineligible"] += 1
                        continue
                    text, fields, transformation = proposed
                    normalized, terms = normalize_query(text), tuple(analyze(text))
                    if not 1 <= len(text) <= 256 or not vocabulary.intersection(terms):
                        rejected["length_or_no_indexed_term"] += 1
                        continue
                    if normalized in seen_normalized or terms in seen_terms:
                        rejected["normalized_or_analyzed_duplicate"] += 1
                        continue
                    query_id = f"load_v1_{family}_{selected + 1:04d}"
                    if any(row["query_id"] == query_id for row in records):
                        raise ValueError(f"generated ID collides with reference: {query_id}")
                    generated.append(
                        _record(
                            {
                                "query_id": query_id,
                                "query_text": text,
                                "query_type": "synthetic_load",
                                "split": "load_only",
                                "family": family,
                                "source_item_id": doc.item_id,
                                "category": doc.category,
                                "transformation": transformation,
                            },
                            fields=fields,
                        )
                    )
                    seen_normalized[normalized].append(query_id)
                    seen_terms[terms].append(query_id)
                    used_items.add(doc.item_id)
                    selected += 1
                    break
                if queue:
                    next_categories.append(category)
                if selected == target:
                    break
            categories = next_categories
        if selected != target:
            raise ValueError(
                f"insufficient unique candidates for {family}: {selected}/{target}; "
                f"rejections={dict(sorted(rejected.items()))}"
            )
        selection[family] = {
            "quota": target,
            "selected": selected,
            "rejections": dict(sorted(rejected.items())),
        }
    # Mix synthetic families for the load schedule; reference order stays untouched.
    generated.sort(key=lambda row: (order_key(seed, "output", row["query_id"]), row["query_id"]))
    return records + generated, {
        "reference_duplicates_preserved": reference_duplicates,
        "synthetic_unique_items": len(used_items),
        "selection": selection,
    }


def build_fixture(
    items_path: Path,
    reference_path: Path,
    index_dir: Path,
    out: Path,
    *,
    seed: str = SEED,
    quotas: Mapping[str, int] | None = None,
) -> dict:
    if out.exists():
        raise FileExistsError(f"refuse to overwrite fixture directory: {out}")
    inputs = {
        "items.parquet": items_path,
        "reference_queries.csv": reference_path,
        "bm25/index.json": index_dir / "index.json",
        "bm25/meta.json": index_dir / "meta.json",
    }
    before = {name: sha256_file(path) for name, path in inputs.items()}
    index, meta = load_index(index_dir)
    if meta.input_sha256.get("items.parquet") != before["items.parquet"]:
        raise ValueError("items.parquet does not match the frozen BM25 index provenance")
    if meta.analyzer != "searchpilot.search.normalize.analyze":
        raise ValueError("fixture analyzer must match the frozen BM25 index analyzer")
    vocabulary = {term for field in index.fields.values() for term in field.postings}
    with reference_path.open(encoding="utf-8-sig", newline="") as stream:
        references = list(csv.DictReader(stream))
    documents = load_documents(items_path)
    records, selection = generate_queries(
        documents,
        references,
        vocabulary,
        indexed_ids=set(index.item_ids),
        seed=seed,
        quotas=quotas,
    )
    content = io.StringIO(newline="")
    writer = csv.DictWriter(
        content, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(records)
    query_bytes = content.getvalue().encode("utf-8")
    manifest = {
        "schema_version": 1,
        "fixture_version": "expanded-load-v1",
        "seed": seed,
        "purpose": "synthetic load-only workload; no relevance labels or retrieval-quality claims",
        "reference_policy": (
            "Original text/type/split/order retained as reference family; split is provenance only"
        ),
        "selection_policy": (
            "Family order reserves Unicode first; SHA256(seed,family,item_id) then category "
            "round-robin; one synthetic query per item; global normalized/analyzed deduplication"
        ),
        "output_order": (
            "Original references, then generated queries sorted by SHA256(seed,output,query_id)"
        ),
        "data_version": meta.data_version,
        "bm25_model_version": meta.model_version,
        "input_sha256": before,
        "generator_sha256": sha256_file(Path(__file__)),
        "analyzer_sha256": sha256_file(Path(normalization.__file__)),
        "versions": {
            "python": sys.version.split()[0],
            "unicode": unicodedata.unidata_version,
            "snowballstemmer": version("snowballstemmer"),
            "pyarrow": version("pyarrow"),
        },
        "queries_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "query_count": len(records),
        "reference_count": len(references),
        "synthetic_count": len(records) - len(references),
        "unique_raw_texts": len({row["query_text"] for row in records}),
        "unique_normalized_texts": len({row["normalized_query"] for row in records}),
        "unique_analyzed_sequences": len({tuple(row["analyzed_terms"]) for row in records}),
        "family_counts": dict(sorted(Counter(row["family"] for row in records).items())),
        "category_counts_synthetic": dict(
            sorted(
                Counter(row["category"] for row in records if row["family"] != "reference").items()
            )
        ),
        "family_category_counts": {
            family: dict(
                sorted(
                    Counter(row["category"] for row in records if row["family"] == family).items()
                )
            )
            for family in QUOTAS
        },
        "char_length_histogram": dict(
            sorted(Counter(row["char_length"] for row in records).items())
        ),
        "analyzed_token_count_histogram": dict(
            sorted(Counter(row["analyzed_token_count"] for row in records).items())
        ),
        "selection": selection,
        "limitations": [
            "Category round-robin is deliberately not traffic-frequency sampling.",
            "Entity spans are capitalization heuristics, not NER annotations.",
            "Indexed-term membership predicts lexical eligibility, "
            "not top-k relevance or vector success.",
            "Fullwidth text is explicitly generated; it is not original news text.",
            "HTTP outcomes and fingerprints require separate runtime verification.",
        ],
        "queries": records,
    }
    if before != {name: sha256_file(path) for name, path in inputs.items()}:
        raise RuntimeError("input changed while generating fixture")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "queries.csv").open("xb") as stream:
        stream.write(query_bytes)
    with (out / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-version")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    data_version = resolve_data_version(args.data_dir / "processed", args.data_version)
    report = build_fixture(
        args.data_dir / "processed" / data_version / "items.parquet",
        args.data_dir / "queries" / "queries.csv",
        args.artifact_dir / "search" / "bm25",
        args.out,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "query_count",
                    "reference_count",
                    "synthetic_count",
                    "family_counts",
                    "unique_normalized_texts",
                    "unique_analyzed_sequences",
                    "queries_sha256",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
