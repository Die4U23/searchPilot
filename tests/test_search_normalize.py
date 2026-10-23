from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from searchpilot.search import normalize
from searchpilot.search.normalize import (
    STOPWORDS,
    analyze,
    document_text,
    normalize_query,
    remove_stopwords,
    stem,
    tokenize,
)


def test_normalize_nfkc_casefold_whitespace_and_control_chars() -> None:
    raw = "\u3000Ｈｅｌｌｏ\x00\x07  World\t\nß\u200b!"
    assert normalize_query(raw) == "hello world ss!"


def test_normalize_keeps_punctuation_and_does_not_truncate() -> None:
    text = "a" * 1000 + ", b."
    out = normalize_query(text)
    assert out.startswith("a" * 1000)
    assert out.endswith(", b.")


def test_normalize_empty_and_whitespace_only() -> None:
    assert normalize_query("") == ""
    assert normalize_query("   \t\n ") == ""


def test_tokenize_keeps_internal_hyphen_and_dot() -> None:
    assert tokenize("covid-19 u.s. e-mail 3.5 a.b.c") == [
        "covid-19",
        "u.s",
        "e-mail",
        "3.5",
        "a.b.c",
    ]


def test_tokenize_drops_isolated_punctuation() -> None:
    assert tokenize('well -- known "quote" (paren) end.') == [
        "well",
        "known",
        "quote",
        "paren",
        "end",
    ]


def test_tokenize_is_unicode_friendly() -> None:
    assert tokenize("café naïve 北京 2024") == ["café", "naïve", "北京", "2024"]


def test_stopwords_table_is_small_and_lowercase() -> None:
    assert 30 <= len(STOPWORDS) <= 50
    assert all(w == w.casefold() for w in STOPWORDS)
    assert {"the", "a", "of", "and"} <= STOPWORDS
    # 可能携带意图的词不在表内
    assert "new" not in STOPWORDS
    assert "us" not in STOPWORDS


def test_remove_stopwords() -> None:
    assert remove_stopwords(["the", "lakers", "and", "celtics"]) == ["lakers", "celtics"]


def test_stem_only_alphabetic_tokens() -> None:
    assert stem(["running", "studies", "covid-19", "u.s", "n95", "3.5"]) == [
        "run",
        "studi",
        "covid-19",
        "u.s",
        "n95",
        "3.5",
    ]
    assert stem([]) == []


def test_concurrent_stemming_matches_serial_results(monkeypatch: pytest.MonkeyPatch) -> None:
    batches = [
        ["running", "studies", "u.s", "n95"],
        ["cats", "relational", "covid-19"],
        ["ponies", "happiness", "3.5"],
        ["jumping", "friendships", "e-mail"],
    ]
    expected = [stem(tokens) for tokens in batches]
    assert expected == [
        ["run", "studi", "u.s", "n95"],
        ["cat", "relat", "covid-19"],
        ["poni", "happi", "3.5"],
        ["jump", "friendship", "e-mail"],
    ]
    original_stem = normalize._STEMMER._stem

    def yielding_stem() -> bool:
        # Yield after Snowball set_current(), retaining its real algorithm and state.
        # No rendezvous inside the protected operation: serialization must be safe.
        time.sleep(0.003)
        return original_stem()

    monkeypatch.setattr(normalize._STEMMER, "_stem", yielding_stem)

    def worker(tokens: list[str]) -> list[list[str]]:
        return [stem(tokens) for _ in range(20)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, tokens) for tokens in batches]
        actual = [future.result(timeout=5) for future in futures]
    assert actual == [[result] * 20 for result in expected]


def test_non_alphabetic_tokens_do_not_acquire_stemmer_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-context-manager sentinel fails if this branch attempts to take the lock.
    monkeypatch.setattr(normalize, "_STEMMER_LOCK", None, raising=False)
    assert stem(["u.s", "n95", "3.5", "covid-19"]) == ["u.s", "n95", "3.5", "covid-19"]
    assert stem([]) == []


def test_analyze_pipeline() -> None:
    assert analyze("The U.S. COVID-19 Studies: running faster!") == [
        "u.s",
        "covid-19",
        "studi",
        "run",
        "faster",
    ]


def test_analyze_all_stopwords_returns_empty() -> None:
    assert analyze("the of and") == []
    assert analyze("") == []


def test_analyze_is_deterministic_and_keeps_duplicates() -> None:
    text = "Lakers lakers LAKERS"
    assert analyze(text) == ["laker", "laker", "laker"]
    assert analyze(text) == analyze(text)


def test_document_text_joins_title_and_abstract() -> None:
    assert document_text("Title", "Abstract") == "Title Abstract"
    assert document_text("Title", "") == "Title "
