from __future__ import annotations

import math

import pytest
from rank_bm25 import BM25Okapi

from searchpilot.ports import Document
from searchpilot.search.bm25 import BM25Scorer, lucene_idf, okapi_raw_idf
from searchpilot.search.inverted_index import SINGLE_TEXT_FIELDS, InvertedIndex
from searchpilot.search.normalize import analyze, document_text

CORPUS = [
    Document(
        "N01",
        "Lakers beat Celtics in overtime",
        "LeBron James scored 35 as the Los Angeles Lakers edged Boston.",
        "sports",
        "nba",
    ),
    Document(
        "N02",
        "Celtics trade rumors heat up",
        "Boston Celtics front office weighs a blockbuster trade before the deadline.",
        "sports",
        "nba",
    ),
    Document(
        "N03",
        "Heavy rain floods Boston streets",
        "Commuters faced delays as heavy rain hit the city overnight.",
        "weather",
        "local",
    ),
    Document(
        "N04",
        "Fed holds interest rates steady",
        "The Federal Reserve kept rates unchanged, citing inflation uncertainty.",
        "finance",
        "economy",
    ),
    Document(
        "N05",
        "Stocks rally after Fed decision",
        "Wall Street climbed as investors cheered the central bank's steady hand.",
        "finance",
        "markets",
    ),
    Document(
        "N06",
        "COVID-19 booster shots recommended",
        "Health officials urge older adults to get the updated COVID-19 vaccine.",
        "health",
        "covid",
    ),
    Document(
        "N07",
        "New study links sleep and memory",
        "Researchers found that deep sleep helps consolidate memories.",
        "health",
        "science",
    ),
    Document(
        "N08",
        "Apple unveils new iPhone lineup",
        "The tech giant introduced three phones with faster chips and better cameras.",
        "tech",
        "gadgets",
    ),
    Document(
        "N09",
        "Tesla recalls thousands of cars",
        "A software glitch prompted the recall of Model 3 and Model Y vehicles.",
        "autos",
        "recall",
    ),
    Document(
        "N10",
        "U.S. Senate passes budget bill",
        "The spending bill now heads to the House after a late-night Senate vote.",
        "news",
        "politics",
    ),
    Document(
        "N11",
        "Lakers sign veteran guard",
        "The Lakers added depth to their backcourt with a one-year deal.",
        "sports",
        "nba",
    ),
    Document(
        "N12",
        "Patriots win thriller over Jets",
        "A last-second field goal lifted New England past New York.",
        "sports",
        "nfl",
    ),
    Document(
        "N13",
        "Recipe: easy weeknight pasta",
        "A quick tomato pasta that comes together in twenty minutes.",
        "lifestyle",
        "food",
    ),
    Document(
        "N14",
        "Hurricane season outlook released",
        "Forecasters expect an above-average hurricane season this year.",
        "weather",
        "national",
    ),
    Document(
        "N15",
        "Boston Marathon registration opens",
        "Runners can now sign up for next spring's Boston Marathon.",
        "sports",
        "running",
    ),
    Document(
        "N16",
        "Oil prices climb on supply fears",
        "Crude rose after producers signalled deeper output cuts.",
        "finance",
        "energy",
    ),
    Document(
        "N17",
        "Celtics star out with injury",
        "Boston will be without its leading scorer for two weeks.",
        "sports",
        "nba",
    ),
    Document(
        "N18",
        "City council approves new park",
        "The downtown park will open to the public next summer.",
        "news",
        "local",
    ),
    Document(
        "N19",
        "Streaming service raises prices",
        "Subscribers will pay two dollars more per month starting in June.",
        "tech",
        "media",
    ),
    Document(
        "N20",
        "Scientists discover new exoplanet",
        "The planet orbits a nearby star every eleven days.",
        "science",
        "space",
    ),
]

QUERIES = [
    "lakers celtics",
    "boston rain",
    "fed interest rates",
    "covid-19 vaccine",
    "new study",
    "u.s. senate",
    "the",  # 全停用词 → 空
    "boston boston celtics",  # 重复词项
    "zzz unknown term",
]


def _tokenized_corpus() -> list[list[str]]:
    return [analyze(document_text(d.title, d.abstract)) for d in CORPUS]


def _single_field_scorer(**kwargs: object) -> BM25Scorer:
    index = InvertedIndex.build_from_documents(CORPUS, analyze, fields=SINGLE_TEXT_FIELDS)
    return BM25Scorer(index, field_weights={"text": 1.0}, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------- idf


def test_idf_formulas() -> None:
    assert lucene_idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5))
    assert okapi_raw_idf(10, 1) == pytest.approx(math.log(9.5 / 1.5))
    assert lucene_idf(10, 9) > 0
    assert okapi_raw_idf(10, 9) < 0


def test_lucene_idf_is_positive_and_monotonic() -> None:
    scorer = _single_field_scorer()
    fidx = scorer.index.fields["text"]
    idfs = {t: scorer.idf(t, "text") for t in fidx.postings}
    assert all(v > 0 for v in idfs.values())
    # df 越大 idf 越小
    assert idfs["boston"] < idfs["exoplanet"]
    assert scorer.idf("not-in-vocab", "text") == 0.0


# --------------------------------------------- rank_bm25 cross validation


@pytest.mark.parametrize("query", QUERIES)
def test_okapi_variant_matches_rank_bm25_per_document(query: str) -> None:
    """对齐方式：把 idf 切成 variant="okapi"（复刻 BM25Okapi 的 ln((N-df+0.5)/(df+0.5)) +
    epsilon 下限），单字段、无字段权重、同一分词，逐文档数值比对。"""
    corpus = _tokenized_corpus()
    reference = BM25Okapi(corpus, k1=1.2, b=0.75, epsilon=0.25)
    scorer = _single_field_scorer(k1=1.2, b=0.75, idf_variant="okapi", okapi_epsilon=0.25)
    terms = analyze(query)
    expected = reference.get_scores(terms)
    for doc_idx in range(len(CORPUS)):
        assert scorer.score(terms, doc_idx) == pytest.approx(expected[doc_idx], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("query", [q for q in QUERIES if analyze(q)])
def test_okapi_variant_topk_matches_rank_bm25(query: str) -> None:
    corpus = _tokenized_corpus()
    reference = BM25Okapi(corpus, k1=1.2, b=0.75)
    scorer = _single_field_scorer(k1=1.2, b=0.75, idf_variant="okapi")
    terms = analyze(query)
    expected = reference.get_scores(terms)
    ranked = scorer.search(terms, limit=50)
    # 我们只返回命中的文档；rank_bm25 对未命中的给 0 分
    assert all(expected[d] > 0 or math.isclose(s, 0.0) for d, s in ranked)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    for doc_idx, score in ranked:
        assert score == pytest.approx(expected[doc_idx], rel=1e-9)


@pytest.mark.parametrize("query", ["lakers", "boston", "celtics", "fed"])
def test_lucene_variant_single_term_ranking_matches_rank_bm25(query: str) -> None:
    """单词项查询下 idf 只是正的常数因子，Lucene 与 Okapi 变体排序必须一致。"""
    corpus = _tokenized_corpus()
    reference = BM25Okapi(corpus, k1=1.2, b=0.75)
    scorer = _single_field_scorer(k1=1.2, b=0.75)
    terms = analyze(query)
    expected = reference.get_scores(terms)
    ours = [d for d, _ in scorer.search(terms, limit=50)]
    ref_order = sorted(
        (d for d in range(len(CORPUS)) if expected[d] > 0),
        key=lambda d: (-expected[d], CORPUS[d].item_id),
    )
    assert ours == ref_order


# ----------------------------------------------------------- parameters


def _two_doc_index(tokens_a: list[str], tokens_b: list[str]) -> InvertedIndex:
    return InvertedIndex.from_tokenized(["A", "B"], {"text": [tokens_a, tokens_b]})


def test_higher_k1_rewards_term_frequency_more() -> None:
    # A 中 "x" 出现 3 次，B 出现 1 次，长度相同
    index = _two_doc_index(["x", "x", "x", "p"], ["x", "q", "r", "s"])
    low = BM25Scorer(index, k1=0.5, b=0.0, field_weights={"text": 1.0})
    high = BM25Scorer(index, k1=2.0, b=0.0, field_weights={"text": 1.0})
    gap_low = low.score(["x"], 0) - low.score(["x"], 1)
    gap_high = high.score(["x"], 0) - high.score(["x"], 1)
    assert gap_low > 0 and gap_high > gap_low


def test_k1_zero_makes_tf_irrelevant() -> None:
    index = _two_doc_index(["x", "x", "x", "p"], ["x", "q", "r", "s"])
    scorer = BM25Scorer(index, k1=0.0, b=0.0, field_weights={"text": 1.0})
    assert scorer.score(["x"], 0) == pytest.approx(scorer.score(["x"], 1))


def test_higher_b_penalizes_long_documents_more() -> None:
    # A 短文档、B 长文档，各含一次 "x"
    index = _two_doc_index(["x", "p"], ["x"] + [f"w{i}" for i in range(20)])
    no_norm = BM25Scorer(index, k1=1.2, b=0.0, field_weights={"text": 1.0})
    full_norm = BM25Scorer(index, k1=1.2, b=1.0, field_weights={"text": 1.0})
    assert no_norm.score(["x"], 0) == pytest.approx(no_norm.score(["x"], 1))
    assert full_norm.score(["x"], 0) > full_norm.score(["x"], 1)


def test_invalid_parameters_rejected() -> None:
    index = _two_doc_index(["x"], ["y"])
    with pytest.raises(ValueError):
        BM25Scorer(index, k1=-1.0)
    with pytest.raises(ValueError):
        BM25Scorer(index, b=1.5)
    with pytest.raises(ValueError, match="unknown fields"):
        BM25Scorer(index, field_weights={"title": 2.0})


# --------------------------------------------------------- field weights


def test_title_hit_ranks_above_abstract_hit() -> None:
    docs = [
        Document("A", "Nothing relevant here", "The lakers played well tonight overall", "s", "s"),
        Document("B", "Lakers win again", "Another game another victory for the team", "s", "s"),
    ]
    index = InvertedIndex.build_from_documents(docs, analyze)
    weighted = BM25Scorer(index)  # 默认 title 2.0 / abstract 1.0
    ranked = weighted.search(analyze("lakers"), limit=10)
    assert [index.item_ids[d] for d, _ in ranked] == ["B", "A"]
    explain = weighted.explain(analyze("lakers"), 1)
    assert set(explain) == {"title"}
    # 把标题权重压到很低，摘要命中反超（摘要更长，所以需要明显低于 1）
    inverted = BM25Scorer(index, field_weights={"title": 0.1, "abstract": 1.0})
    ranked2 = inverted.search(analyze("lakers"), limit=10)
    assert [index.item_ids[d] for d, _ in ranked2] == ["A", "B"]


def test_field_weight_scales_contribution_linearly() -> None:
    index = InvertedIndex.build_from_documents(CORPUS, analyze)
    w1 = BM25Scorer(index, field_weights={"title": 1.0, "abstract": 0.0})
    w3 = BM25Scorer(index, field_weights={"title": 3.0, "abstract": 0.0})
    terms = analyze("lakers celtics")
    for doc_idx in range(len(CORPUS)):
        assert w3.score(terms, doc_idx) == pytest.approx(3 * w1.score(terms, doc_idx))


def test_missing_field_weight_defaults_to_one() -> None:
    index = InvertedIndex.build_from_documents(CORPUS, analyze)
    scorer = BM25Scorer(index, field_weights={"title": 2.0})
    assert scorer.field_weights == {"abstract": 1.0, "title": 2.0}


# ------------------------------------------------------------ edge cases


def test_empty_query_and_unknown_terms() -> None:
    scorer = _single_field_scorer()
    assert scorer.search([], limit=10) == []
    assert scorer.search(["zzzz"], limit=10) == []
    assert scorer.search(["laker"], limit=0) == []
    assert scorer.score([], 0) == 0.0


def test_score_out_of_range() -> None:
    scorer = _single_field_scorer()
    with pytest.raises(IndexError):
        scorer.score(["laker"], 99)


def test_tie_break_by_item_id_ascending() -> None:
    # 三篇完全相同的文档，id 乱序插入
    index = InvertedIndex.from_tokenized(
        ["C", "A", "B", "D"], {"text": [["x", "y"], ["x", "y"], ["x", "y"], ["z"]]}
    )
    scorer = BM25Scorer(index, field_weights={"text": 1.0})
    ranked = scorer.search(["x"], limit=10)
    assert [index.item_ids[d] for d, _ in ranked] == ["A", "B", "C"]
    assert len({round(s, 12) for _, s in ranked}) == 1
    # 截断时也按同一规则
    assert [index.item_ids[d] for d, _ in scorer.search(["x"], limit=2)] == ["A", "B"]


def test_search_is_deterministic_across_calls() -> None:
    scorer = _single_field_scorer()
    terms = analyze("boston celtics lakers")
    assert scorer.search(terms, 10) == scorer.search(terms, 10)


def test_doc_filter_restricts_candidates() -> None:
    index = InvertedIndex.build_from_documents(CORPUS, analyze)
    scorer = BM25Scorer(index)
    terms = analyze("boston")
    only_weather = scorer.search(terms, 10, doc_filter=lambda d: index.categories[d] == "weather")
    assert [index.item_ids[d] for d, _ in only_weather] == ["N03"]


def test_repeated_query_terms_count_twice() -> None:
    scorer = _single_field_scorer()
    once = scorer.score(["boston"], 2)
    twice = scorer.score(["boston", "boston"], 2)
    assert twice == pytest.approx(2 * once)
