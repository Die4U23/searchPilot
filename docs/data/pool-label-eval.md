# 补标并集上的检索对照

2026-10-11。每一行检索只跑一次，再用两套标注打分。
基础标注是 `data/queries/labels.csv`，补标是 `data/queries/labels_tool.csv`。
两份文件没有相同的 `(query_id, item_id)`。并集不写回 `labels.csv`。
未出现在这两份文件里的文档仍按 0 分。nDCG / MRR / Recall 跳过没有相关文档的查询。
train 是 LTR 的训练切分，不能当验收。val 和 test 没有进入 LTR 损失。

`test` 有关文档：基础 50，并集 126。
`val` 有关文档：基础 56，并集 129。
`train` 有关文档：基础 284，并集 556。

## 总体

| split | mode | nDCG@10 基础 | nDCG@10 并集 | MRR@10 基础 | MRR@10 并集 | Recall@50 基础 | Recall@50 并集 | 跳过 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test | bm25 | 0.6396 | 0.5571 | 0.7143 | 0.7143 | 0.7500 | 0.5760 | 1 |
| test | vector | 0.7769 | 0.8796 | 0.8095 | 0.9643 | 0.9452 | 0.9483 | 1 |
| test | hybrid | 0.7214 | 0.7236 | 0.7679 | 0.8810 | 1.0000 | 1.0000 | 1 |
| test | ltr | 0.4596 | 0.5266 | 0.4595 | 0.8095 | 0.7175 | 0.6984 | 1 |
| val | bm25 | 0.9056 | 0.7465 | 0.9231 | 0.9231 | 0.9161 | 0.6536 | 2 |
| val | vector | 0.8580 | 0.9161 | 0.9327 | 1.0000 | 0.8343 | 0.8913 | 2 |
| val | hybrid | 0.8689 | 0.8424 | 0.9385 | 0.9385 | 0.9930 | 0.9962 | 2 |
| val | ltr | 0.5225 | 0.5055 | 0.6154 | 0.7289 | 0.5924 | 0.4803 | 2 |
| train | bm25 | 0.7663 | 0.6584 | 0.8049 | 0.8049 | 0.8241 | 0.6390 | 12 |
| train | vector | 0.8208 | 0.8985 | 0.8807 | 0.9828 | 0.8902 | 0.9246 | 12 |
| train | hybrid | 0.7988 | 0.7761 | 0.8541 | 0.8879 | 0.9745 | 0.9920 | 12 |
| train | ltr | 0.3928 | 0.4438 | 0.4728 | 0.7500 | 0.6522 | 0.6378 | 12 |

跳过列是没有相关文档的查询数。基础和并集不同时写成 `基础/并集`。
检索深度 50。

## 排序

- `test` 基础 nDCG@10：vector 0.7769 > hybrid 0.7214 > bm25 0.6396 > ltr 0.4596
- `test` 并集 nDCG@10：vector 0.8796 > hybrid 0.7236 > bm25 0.5571 > ltr 0.5266
- `val` 基础 nDCG@10：bm25 0.9056 > hybrid 0.8689 > vector 0.8580 > ltr 0.5225
- `val` 并集 nDCG@10：vector 0.9161 > hybrid 0.8424 > bm25 0.7465 > ltr 0.5055
- `train` 基础 nDCG@10：vector 0.8208 > hybrid 0.7988 > bm25 0.7663 > ltr 0.3928
- `train` 并集 nDCG@10：vector 0.8985 > hybrid 0.7761 > bm25 0.6584 > ltr 0.4438

test 和 val 都没有进入 LTR 损失。两边的并集 nDCG@10 名次相同：vector > hybrid > bm25 > ltr。
val 在基础标注上 nDCG@10 最高的是 bm25，并集后变成 vector。
train 名次只作诊断，不能用来声称 LTR 或某一路检索已通过验收。

## 并集上只评已标注文档

未标注文档从名单里拿掉，相对顺序不变，不另打分。明确 0 分留在原位置。
并集没有覆盖 LTR 自己挖出来的文档，所以这一列只说明已标注文档之间的顺序。

| split | mode | nDCG@10 未标注当 0 | nDCG@10 只评已标注 | MRR@10 未标注当 0 | MRR@10 只评已标注 |
|---|---|---:|---:|---:|---:|
| test | bm25 | 0.5571 | 0.5571 | 0.7143 | 0.7143 |
| test | vector | 0.8796 | 0.8796 | 0.9643 | 0.9643 |
| test | hybrid | 0.7236 | 0.7236 | 0.8810 | 0.8810 |
| test | ltr | 0.5266 | 0.7076 | 0.8095 | 0.9643 |
| val | bm25 | 0.7465 | 0.7465 | 0.9231 | 0.9231 |
| val | vector | 0.9161 | 0.9161 | 1.0000 | 1.0000 |
| val | hybrid | 0.8424 | 0.8424 | 0.9385 | 0.9385 |
| val | ltr | 0.5055 | 0.6959 | 0.7289 | 0.9231 |
| train | bm25 | 0.6584 | 0.6584 | 0.8049 | 0.8049 |
| train | vector | 0.8985 | 0.8985 | 0.9828 | 0.9828 |
| train | hybrid | 0.7761 | 0.7761 | 0.8879 | 0.8879 |
| train | ltr | 0.4438 | 0.6865 | 0.7500 | 0.9046 |

- `test` 只评已标注 nDCG@10：vector 0.8796 > hybrid 0.7236 > ltr 0.7076 > bm25 0.5571
- `val` 只评已标注 nDCG@10：vector 0.9161 > hybrid 0.8424 > bm25 0.7465 > ltr 0.6959

## 只评已标注，按查询类型

计入列是有相关文档的查询数。`no_answer` 不进入 nDCG。
val 里 exact_entity 和 synonym 各只有 1 条计入查询。

| split | query_type | 计入 | bm25 | vector | hybrid | ltr |
|---|---|---:|---:|---:|---:|---:|
| test | exact_entity | 3 | 0.9392 | 0.9935 | 0.9543 | 0.9776 |
| test | synonym | 3 | 0.0183 | 0.7411 | 0.3585 | 0.6476 |
| test | multi_condition | 1 | 0.7967 | 0.8535 | 0.8644 | 0.3498 |
| test | misspelling_or_abbrev | 2 | 0.9053 | 0.9566 | 0.9282 | 0.7372 |
| test | long_tail_popular_distractor | 5 | 0.4639 | 0.8687 | 0.6944 | 0.6414 |
| test | no_answer | 0 | n/a | n/a | n/a | n/a |
| val | exact_entity | 1 | 1.0000 | 1.0000 | 1.0000 | 0.5638 |
| val | synonym | 1 | 0.0000 | 0.9616 | 0.3961 | 0.7035 |
| val | multi_condition | 4 | 0.8511 | 0.9265 | 0.9243 | 0.7477 |
| val | misspelling_or_abbrev | 4 | 0.8562 | 0.8659 | 0.9030 | 0.6376 |
| val | long_tail_popular_distractor | 3 | 0.6250 | 0.9260 | 0.7486 | 0.7462 |
| val | no_answer | 0 | n/a | n/a | n/a | n/a |

test 上 LTR 低于向量的类型：exact_entity、synonym、multi_condition、misspelling_or_abbrev、long_tail_popular_distractor。
test 上 LTR 低于 BM25 的类型：multi_condition、misspelling_or_abbrev。
val 上 LTR 低于向量的类型：exact_entity、synonym、multi_condition、misspelling_or_abbrev、long_tail_popular_distractor。
val 上 LTR 低于 BM25 的类型：exact_entity、multi_condition、misspelling_or_abbrev。
test 上，未标注当 0 的名次是 vector > hybrid > bm25 > ltr，只评已标注变成 vector > hybrid > ltr > bm25。
val 上两种口径的名次相同：vector > hybrid > bm25 > ltr。
只评已标注不能当成某一路检索已经超过另一路的验收。
现有 LTR 权重的覆盖见 `docs/data/ltr-pool-features.md`。LTR 低于 BM25 的逐条对照见 `docs/data/ltr-query-gaps.md`。
