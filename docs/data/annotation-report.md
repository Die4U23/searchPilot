# 查询集标注报告

2026-10-09。查询与候选来自 MIND Small（`data_version` `d3a904f41240`）的标题和摘要。标注员是两个互不看对方分数的 Grok 会话，不是两名人类。示例 CSV 已替换，不计入本报告。

## 规模

- 查询 100 条。切分种子 `20261009`，用 `random.Random` 打乱 `query_id` 后按 70 / 15 / 15 分配；每条查询自成一组，没有近义改写组。
- 候选是该查询的 BM25 前 10 条；种子新闻若不在前 10，额外加入。未出现在本文件中的文档，评估脚本按 0 分处理。
- 双标种子同样是 `20261009`，从全部 `query_id` 中抽取 20 条（20%），抽中查询的全部候选都由第二名标注员独立打分。

## 各类在三个 split 中的数量

| query_type | train | val | test |
|---|---:|---:|---:|
| exact_entity | 16 | 1 | 3 |
| synonym | 12 | 1 | 3 |
| multi_condition | 13 | 4 | 1 |
| misspelling_or_abbrev | 10 | 4 | 2 |
| no_answer | 12 | 2 | 1 |
| long_tail_popular_distractor | 7 | 3 | 5 |

## 双标一致率

- 双标查询数：20
- 候选对数：206
- 完全一致率：191/206 = 0.9272
- 相邻一致率（等级差绝对值不超过 1）：206/206 = 1.0000
- 二次加权 Cohen's kappa：0.9702

15 对不一致都只差 1 分，原始分数留在 `labels_grok_a.csv` 与 `labels_grok_b.csv`。最终 `labels.csv` 里这 15 对的 `annotator` 为 `adjudicated`：共享热门词但实体不对的干扰新闻记 0；Elton John 在 Bankers Life Fieldhouse（印第安纳波利斯）改期到 2020、乐队不是 Alabama，记 1；Christian Bale 与 Ford v Ferrari 匹配、但没有 Chandler 驾校，记 2。其余最终分数与 grok-a 相同。评估脚本对同一 `(query_id, item_id)` 会取最大 grade，所以最终文件每个候选只留一行。

## test 评估

`scripts/evaluate_search.py --split test`，模型 `bm25-4f46b3d8`。15 条 test 查询里 1 条 `no_answer` 没有相关文档，nDCG / MRR / Recall 不计入。有答案的查询没有零结果。

| segment | queries | skipped | nDCG@10 | MRR@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| all | 15 | 1 | 0.6396 | 0.7143 | 0.7500 |
| exact_entity | 3 | 0 | 1.0000 | 1.0000 | 1.0000 |
| long_tail_popular_distractor | 5 | 0 | 0.5894 | 0.6000 | 0.8000 |
| misspelling_or_abbrev | 2 | 0 | 0.9554 | 1.0000 | 1.0000 |
| multi_condition | 1 | 0 | 0.9657 | 1.0000 | 1.0000 |
| no_answer | 1 | 1 | n/a | n/a | n/a |
| synonym | 3 | 0 | 0.0437 | 0.3333 | 0.1667 |

同义改写是最弱的一类：查询故意不用标题里的原词，BM25 经常召不回种子新闻。完整失败样例在 `artifacts/search/eval/search_eval_test_bm25-4f46b3d8.md`（该目录不入库）。
