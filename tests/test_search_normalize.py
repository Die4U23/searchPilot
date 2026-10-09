from __future__ import annotations

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
