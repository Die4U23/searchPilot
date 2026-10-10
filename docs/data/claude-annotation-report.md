# Claude 补标报告

2026-10-10。补标对象是 `candidate_pool_k10.csv`（#31 导出的三路前 10 并集中尚未标注的 938 对），由 Claude 依据 `annotation-guideline.md` 与 grok 已标样例校准后逐对标注，`annotator=claude`。原始 grok 标注（`labels.csv`）未被修改；本文件是独立补标，是否并入主标注集由项目所有者决定。

## 规模

- 查询 100 条，候选 938 对，全部有标题与摘要可依据。
- 与 grok 已标部分**零重叠**（候选池只含未标注文档），因此无法直接计算双标一致率。

## 等级分布

| grade | 对数 |
|---:|---:|
| 0 | 686 |
| 1 | 129 |
| 2 | 58 |
| 3 | 65 |

## 按 query_type 分布

| query_type | 0 | 1 | 2 | 3 | 合计 |
|---|---:|---:|---:|---:|
| exact_entity | 118 | 2 | 4 | 32 | 156 |
| synonym | 117 | 33 | 17 | 16 | 183 |
| multi_condition | 116 | 30 | 9 | 2 | 157 |
| misspelling_or_abbrev | 58 | 42 | 15 | 12 | 127 |
| no_answer | 179 | 0 | 0 | 0 | 179 |
| long_tail_popular_distractor | 98 | 22 | 13 | 3 | 136 |

`no_answer` 类 179 对全部为 0——没有把检索噪音标成相关，符合该类构造意图（测量系统是否把故障伪装成结果）。

## 校准口径

分级标准对照 grok 已标样例（q_004 / q_026 / q_040）：

- 不同实体但词面相近 → 0（如 Ricky Martin 之于 Ricky Gervais、Jim LeClair 之于 Jim Thorpe）
- 主要意图匹配但缺次要条件 → 2（如 Kincade Fire 相关但非目标火情的查询条件）
- 仅有背景关联 → 1（如"未标注当 0"类中同为该地区/品类但非目标的新闻）

## 已知限制

- 标注者是模型会话，不是人工；不能替代 PRD 要求的人工验收。
- 抽样自查发现 exact_entity 类 3 分多集中在实体名直接命中的新闻（al-Baghdadi、Kincade Fire 等），符合预期。
- 后续若并入 `labels.csv`，建议先按 `annotation-guideline.md` 的双标流程做人工抽样复核。
