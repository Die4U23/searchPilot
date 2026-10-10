# 分数归一化融合对照

每一路分数先缩放到 0–1 再平均。这不是默认融合；默认仍是 RRF。
test 查询与 `search_mode_compare_test.md` 相同。

- 查询数：15
- nDCG@10：0.7710
- MRR@10：0.7857
- Recall@50：1.0000
