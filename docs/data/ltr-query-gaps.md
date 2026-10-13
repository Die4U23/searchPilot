# LTR 低于 BM25 的查询

2026-10-11。只评已标注：未标注文档移出名单，明确 0 分留在原位置。
比较的是并集标注上的 nDCG@10。只列出 LTR 低于 BM25 的 val / test 查询。
名次 101 表示不在该路 top 100。

共 12 条。

| split | query | type | bm25 | vector | hybrid | ltr |
|---|---|---|---:|---:|---:|---:|
| val | q_005 | exact_entity | 1.0000 | 1.0000 | 1.0000 | 0.5638 |
| val | q_039 | multi_condition | 0.7823 | 0.9522 | 0.8926 | 0.5997 |
| test | q_043 | multi_condition | 0.7967 | 0.8535 | 0.8644 | 0.3498 |
| val | q_044 | multi_condition | 0.8754 | 0.9790 | 1.0000 | 0.7917 |
| val | q_050 | multi_condition | 0.8316 | 0.9070 | 0.9132 | 0.7367 |
| val | q_052 | multi_condition | 0.9149 | 0.8679 | 0.8913 | 0.8628 |
| val | q_056 | misspelling_or_abbrev | 0.9380 | 0.9624 | 0.9812 | 0.2695 |
| val | q_058 | misspelling_or_abbrev | 0.8818 | 0.7293 | 0.7554 | 0.7712 |
| test | q_060 | misspelling_or_abbrev | 0.9402 | 1.0000 | 0.9317 | 0.6336 |
| val | q_068 | misspelling_or_abbrev | 0.8459 | 0.8262 | 0.9671 | 0.6652 |
| test | q_069 | misspelling_or_abbrev | 0.8704 | 0.9133 | 0.9247 | 0.8407 |
| test | q_093 | long_tail_popular_distractor | 0.8209 | 0.9528 | 0.9255 | 0.5260 |

12 条里，4 条的 LTR 前 50 里已标注文档少于 6 篇。
4 条的已标注第一名，BM25 名次大于或等于 50。
bm25_rank 权重为正，名次数字越大越靠前。
因此 BM25 名次很差的文档会被抬到 grade 3 前面。
把这项权重置 0 的对照见 `docs/data/ltr-rank-ablation.md`。

## q_005 val exact_entity

Kincade Fire

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N54624 g3 | N63244 g3 | N63244 g3 | 30 | 1 |
| 2 | N1208 g3 | N27270 g3 | N49221 g3 | 45 | 10 |
| 3 | N39081 g3 | N31878 g3 | N10002 g3 | 57 | 46 |
| 4 | N43547 g3 | N29335 g3 | N29335 g3 | 18 | 4 |
| 5 | N5493 g3 | N25340 g3 |  |  |  |
| 6 | N2239 g3 | N1804 g3 |  |  |  |

## q_039 val multi_condition

Yoshiteru Ikegawa brings yakitori to New York

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 1 | 1 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N53313 g3 | N53313 g3 | N32562 g0 | 101 | 2 |
| 2 | N42596 g0 | N32562 g0 | N53313 g3 | 1 | 1 |
| 3 | N59768 g0 | N54759 g1 | N49366 g0 | 101 | 7 |
| 4 | N41900 g0 | N12056 g1 | N28489 g0 | 101 | 5 |
| 5 | N1832 g0 | N28489 g0 | N1310 g0 | 101 | 6 |
| 6 | N11706 g0 | N1310 g0 | N3836 g0 | 101 | 8 |

## q_043 test multi_condition

Cleveland Browns versus Buffalo Bills at FirstEnergy Stadium

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N33788 g3 | N3835 g2 | N33788 g3 | 1 | 2 |
| 2 | N892 g3 | N33788 g3 | N3835 g2 | 5 | 1 |
| 3 | N29013 g2 | N892 g3 |  |  |  |
| 4 | N18983 g2 | N56265 g3 |  |  |  |
| 5 | N3835 g2 | N23843 g1 |  |  |  |
| 6 | N31764 g2 | N29013 g2 |  |  |  |

## q_044 val multi_condition

American Airlines ends Baltimore-Phoenix flights

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N17821 g3 | N1030 g3 | N17821 g3 | 1 | 2 |
| 2 | N57274 g1 | N17821 g3 | N34128 g1 | 101 | 5 |
| 3 | N58276 g1 | N49314 g1 | N43745 g1 | 101 | 6 |
| 4 | N61491 g1 | N61361 g1 | N4566 g0 | 101 | 9 |
| 5 | N59426 g1 | N34128 g1 | N50761 g1 | 101 | 8 |
| 6 | N59379 g1 | N43745 g1 | N1030 g3 | 7 | 1 |

## q_050 val multi_condition

country band Alabama plans a 2020 Indianapolis show after postponements

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N34507 g3 | N34507 g3 | N34507 g3 | 1 | 1 |
| 2 | N61843 g3 | N61843 g3 | N64484 g1 | 101 | 4 |
| 3 | N35735 g1 | N7773 g1 | N49430 g0 | 101 | 5 |
| 4 | N4528 g0 | N64484 g1 | N7773 g1 | 101 | 3 |
| 5 | N15103 g0 | N49430 g0 | N23878 g1 | 101 | 7 |
| 6 | N28136 g0 | N2874 g0 | N24170 g0 | 101 | 9 |

## q_052 val multi_condition

Carly Simon deathbed farewell to Jackie Kennedy

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N49904 g3 | N49904 g3 | N49904 g3 | 1 | 1 |
| 2 | N60993 g0 | N57913 g0 | N33318 g0 | 101 | 6 |
| 3 | N43529 g1 | N43593 g0 | N23054 g0 | 101 | 7 |
| 4 | N35581 g0 | N36693 g1 | N57913 g0 | 101 | 2 |
| 5 | N9543 g0 | N31734 g0 | N36693 g1 | 101 | 4 |
| 6 | N44540 g0 | N33318 g0 | N59051 g0 | 101 | 9 |

## q_056 val misspelling_or_abbrev

USDA flags metal in recalled poultry

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 1 | 1 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N3384 g3 | N11020 g3 | N18680 g0 | 94 | 10 |
| 2 | N11020 g3 | N3384 g3 | N28490 g1 | 101 | 7 |
| 3 | N48126 g1 | N48126 g1 | N11020 g3 | 2 | 1 |
| 4 | N326 g1 | N52115 g1 |  |  |  |
| 5 | N24391 g1 | N326 g1 |  |  |  |
| 6 | N38495 g1 | N12752 g2 |  |  |  |

## q_058 val misspelling_or_abbrev

Biz 40 shut down in Winston-Salem through Saturday night

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N52917 g3 | N52917 g3 | N52917 g3 | 1 | 1 |
| 2 | N21304 g1 | N22910 g0 | N2905 g1 | 91 | 6 |
| 3 | N42259 g0 | N13121 g0 | N47064 g1 | 101 | 7 |
| 4 | N1106 g0 | N8106 g0 | N22910 g0 | 69 | 2 |
| 5 | N59226 g1 | N58359 g0 | N58359 g0 | 58 | 5 |
| 6 | N32197 g1 | N2905 g1 | N61297 g0 | 101 | 9 |

## q_060 test misspelling_or_abbrev

GOP candidates sweep the Tamaqua school board election

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N18141 g3 | N18141 g3 | N46778 g1 | 87 | 8 |
| 2 | N35231 g0 | N41032 g1 | N24962 g1 | 101 | 6 |
| 3 | N59026 g1 | N7733 g1 | N41032 g1 | 101 | 2 |
| 4 | N53166 g1 | N1723 g1 | N1723 g1 | 101 | 4 |
| 5 | N28890 g1 | N43089 g1 | N43081 g1 | 101 | 7 |
| 6 | N59760 g1 | N24962 g1 | N18141 g3 | 1 | 1 |

## q_068 val misspelling_or_abbrev

networks seek Felicty Huffman's post-prison interview

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N33254 g3 | N33254 g3 | N33254 g3 | 1 | 1 |
| 2 | N2550 g2 | N2513 g1 | N14459 g1 | 101 | 8 |
| 3 | N43270 g1 | N50826 g1 | N35242 g1 | 101 | 4 |
| 4 | N59906 g1 | N35242 g1 | N9063 g0 | 101 | 5 |
| 5 | N50826 g1 | N9063 g0 | N2513 g1 | 87 | 2 |
| 6 | N54968 g0 | N2550 g2 | N11460 g0 | 101 | 10 |

## q_069 test misspelling_or_abbrev

Cristian Bale driving school in Chandler for Ford v Ferrari

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N5085 g3 | N5085 g3 | N5085 g3 | 1 | 1 |
| 2 | N42663 g1 | N62844 g1 | N57540 g1 | 3 | 6 |
| 3 | N57540 g1 | N15365 g1 | N42663 g1 | 2 | 4 |
| 4 | N43039 g1 | N42663 g1 | N62844 g1 | 5 | 2 |
| 5 | N62844 g1 | N55314 g2 | N15365 g1 | 7 | 3 |
| 6 | N5945 g1 | N57540 g1 | N54818 g0 | 101 | 10 |

## q_093 test long_tail_popular_distractor

Taste of Home Christmas Eve snacks

| 名单 | 第一篇相关文档前的 0 分 | 第一篇 grade>=2 前的 0 分 |
|---|---:|---:|
| bm25 | 0 | 0 |
| vector | 0 | 0 |
| ltr | 0 | 0 |

| # | bm25 | vector | ltr | ltr 的 BM25 名次 | ltr 的向量名次 |
|---:|---|---|---|---:|---:|
| 1 | N49664 g3 | N49664 g3 | N22351 g1 | 101 | 4 |
| 2 | N41752 g1 | N40402 g2 | N49664 g3 | 1 | 1 |
| 3 | N40402 g2 | N41752 g1 | N41752 g1 | 2 | 3 |
| 4 | N42459 g1 | N22351 g1 | N40402 g2 | 3 | 2 |
| 5 | N21368 g1 | N63580 g2 | N8873 g1 | 65 | 6 |
| 6 | N64304 g0 | N8873 g1 |  |  |  |
