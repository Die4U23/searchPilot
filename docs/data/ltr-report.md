# LTR 与 BM25 / 向量 / RRF 的 test 对比

2026-10-09。数据版本 `d3a904f41240`。test 查询 15 条，其中 1 条没有相关文档，nDCG / MRR / Recall 不计入。四路用同一次 `scripts/compare_search_modes.py --split test`。

LTR 是 `ltr-4574f8f4`。训练只用 `split=train`：70 条查询、705 行标注、974 个 grade 不同的文档对。种子 `20261009`，线性层，logistic pairwise，40 个 epoch，学习率 0.05。没有用 val/test 标签更新权重，也没有按 test 改超参。训练对上的损失从 1.107 降到 0.183。705 行里 692 行落在 BM25 top 100，352 行落在向量 top 100。

## test 指标

| mode | model_version | nDCG@10 | MRR@10 | Recall@50 |
|---|---|---:|---:|---:|
| bm25 | bm25-4f46b3d8 | 0.6396 | 0.7143 | 0.7500 |
| vector | vector-b92663fd | 0.7769 | 0.8095 | 0.9452 |
| hybrid | hybrid-51d4dcfb | 0.7214 | 0.7679 | 1.0000 |
| ltr | ltr-4574f8f4 | 0.4596 | 0.4595 | 0.7175 |

## 结论

这是负结果。LTR 的 nDCG@10 是 0.4596，低于 hybrid（RRF）的 0.7214。MRR@10 和 Recall@50 也更低。

`exact_entity` 上四路 nDCG@10 都是 1.0000，其余类型 LTR 更低。学到的 `bm25_rank` 权重是正的（0.635）：训练行的名次均值约 7，名次数字更大反而更相关。按已保存权重，名次 101 的贡献高于名次 1，缺席 BM25 top 100 的文档会被抬高。val 上从向量 top 50 掉出 LTR top 50 的相关文档几乎都有 BM25 名次。详见 `docs/data/ltr-pool-features.md`。按 FR-5 保留该结果，不再加模型。
