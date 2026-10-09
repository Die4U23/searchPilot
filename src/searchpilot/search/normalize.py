"""查询 / 文档文本的归一化、分词、停用词与词干化。

所有阶段都是纯函数且确定性的；文档与查询必须走**同一条** ``analyze`` 管线，
否则词项对不上。长度限制不在这里做（由 API 层负责）。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

import snowballstemmer

Analyzer = Callable[[str], list[str]]
"""把一段文本转成词项序列的可调用对象（文档与查询共用）。"""

# 一份小而明确的英文停用词表：只收功能词，不收可能携带检索意图的词
# （如 "new"、"first"、"us" 不在表内，因为在新闻语料里它们经常是实体的一部分）。
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
    }
)

# 词项正则：一个 \w+ 段，后面可以跟任意多个「连字符或点号 + \w+」段。
# 这样 "covid-19"、"u.s."（→ "u.s"）、"e-mail"、"3.5" 作为一个词项保留，
# 而句末的点、独立的标点、引号、破折号都被当作分隔符丢弃。
_TOKEN_RE = re.compile(r"\w+(?:[-.]\w+)*", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

_STEMMER = snowballstemmer.stemmer("english")


def normalize_query(text: str) -> str:
    """把原始文本归一化为可比较、可展示的形式。

    步骤（顺序固定）：

    1. Unicode NFKC 兼容分解再组合：全角字母数字 → 半角、连字 → 拆开、各类空格 → 普通空格。
    2. 去掉控制字符（Unicode 类别 Cc / Cf / Cs / Co），但把 ``\\t`` ``\\n`` 等空白控制字符
       保留为空白，交给下一步折叠。
    3. ``str.casefold()``（比 ``lower()`` 更激进，例如 ``ß`` → ``ss``）。
    4. 折叠连续空白为单个空格并去掉首尾空白。

    不截断、不去标点：标点由 :func:`tokenize` 处理，长度由 API 层限制。
    """
    text = unicodedata.normalize("NFKC", text)
    kept: list[str] = []
    for ch in text:
        if ch.isspace():
            kept.append(" ")
            continue
        if unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Co"}:
            continue
        kept.append(ch)
    text = "".join(kept).casefold()
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """按 Unicode 友好的正则切词，不做大小写或词干处理。

    规则：词项 = ``\\w+(?:[-.]\\w+)*``，即字母 / 数字 / 下划线段，允许用连字符或点号把
    多个段连成一个词项（``covid-19``、``u.s``、``3.5``）。只有**夹在两个 \\w 段中间**的
    连字符 / 点号才被保留；句末点号、孤立标点、引号、空白全部作为分隔符丢弃
    （``"u.s."`` → ``["u.s"]``，``"well -- known"`` → ``["well", "known"]``）。

    这个函数本身不做小写化，调用方应先 :func:`normalize_query`；:func:`analyze`
    已经把两步串起来。
    """
    return _TOKEN_RE.findall(text)


def remove_stopwords(tokens: list[str], stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """过滤停用词（大小写敏感；调用方应已 casefold）。"""
    return [tok for tok in tokens if tok not in stopwords]


def stem(tokens: list[str]) -> list[str]:
    """用 snowballstemmer 英文词干器处理一批词项。

    **只对纯字母词项词干化**（``str.isalpha()``）：``studies`` → ``studi``、``running`` → ``run``。
    含数字、连字符、点号的词项（``covid-19``、``u.s``、``3.5``、``n95``）原样保留——
    Porter2 会把 ``u.s`` 错误地截成 ``u.``（把尾部 ``s`` 当复数），对缩写与编号类
    精确实体查询（PRD 的 ``exact_entity`` 类）是有害的。
    """
    out: list[str] = []
    alpha_positions: list[int] = []
    alpha_tokens: list[str] = []
    for pos, tok in enumerate(tokens):
        out.append(tok)
        if tok.isalpha():
            alpha_positions.append(pos)
            alpha_tokens.append(tok)
    if alpha_tokens:
        for pos, stemmed in zip(alpha_positions, _STEMMER.stemWords(alpha_tokens), strict=True):
            out[pos] = stemmed
    return out


def analyze(text: str) -> list[str]:
    """默认分析器：``normalize_query`` → ``tokenize`` → 去停用词 → 词干。

    文档建索引与在线查询都用这个函数，保证词项空间一致。返回的列表保留重复与顺序
    （BM25 需要 tf；查询中重复词项按 Lucene 语义重复计分）。
    """
    return stem(remove_stopwords(tokenize(normalize_query(text))))


def document_text(title: str, abstract: str) -> str:
    """Document 的单字段可检索文本：``title + " " + abstract``。

    用于与 rank_bm25 做单字段交叉验证；多字段加权检索见 ``inverted_index`` / ``bm25``。
    """
    return f"{title} {abstract}"
