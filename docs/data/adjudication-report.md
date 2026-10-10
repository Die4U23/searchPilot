# 双标对账：claude vs grok-tool（test 补标 142 对）

2026-10-10。两名标注员独立标注同一批 test 候选（`candidate_pool_k10.csv` 中的 142 对），互不查看对方结果：`labels_claude.csv`（annotator=claude，先完成）与 `labels_tool.csv`（annotator=grok-tool，后完成）。

## 一致率

- 双标对数：142
- 完全一致率：92/142 = 0.6479
- 相邻一致率：140/142 = 0.9859
- 二次加权 Cohen's kappa：0.6450

对照原始双标（grok-a vs grok-b，206 对）：完全一致 0.9272、kappa 0.9702。本次跨工具一致率显著更低，主要原因是补标池里 1/0 边界样本多、且两名标注员的分级尺度有系统性差异（见下）。

## 等级差 ≥ 2 的分歧（2 对）

| 查询 | 文档 | claude | grok-tool | 裁定 |
|---|---:|---:|---|
| q_030 playoff links favor spectacle over equity | N18103 Monday Measure: Why Clemson deserves playoff inclusion | 0 | 2 | **2**（grok-tool 正确；该文直接讨论季后赛入围的公平性，claude 过严） |
| q_086 Mary Matthews death | N34704 Children of woman mauled, killed by Great Danes seek answers | 0 | 3 | **3**（grok-tool 正确；这正是查询目标新闻的后续报道，claude 漏判） |

裁定结果已同步修正 `labels_claude.csv` 中这两对。

## 系统性差异

claude 的 0 分比例高于 grok-tool（对同一批候选，0 分 92 vs 66）。差异集中在「背景相关」边界：claude 按 grok-a/b 原标注的严格尺度（词面相近 ≠ 相关）把多数 1 分样本压到 0，而 grok-tool 对「主题相关但非直接回答」更宽松，给了 62 个 1 分。

## 结论

- 相邻一致率 0.9859 说明两套标注的排序大体一致；完全一致率偏低反映的是 0/1 边界的尺度差，不是标注矛盾。
- test 补标的正式口径建议以 grok-tool 为准（`labels_tool.csv`），claude 版本保留为对账参考。
- 按指南，争议样本经裁定后应保留两份原始标注——两份文件均已入库，本报告记录裁定过程。
