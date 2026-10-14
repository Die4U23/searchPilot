# 2026-10-13 接手与验收

本文保留首次接手完成时的本地验收快照。所有者随后授权自主提交、PR 与 CI，
权限以 [Git 交付约定](git-workflow.md) 为准；发布前新增修复与检查见
[发布说明](release-2026-10-13.md)。下文“未提交”及测试数量对应首次快照。

本轮由主代理协调三个执行子代理和一个独立只读复审子代理完成，过程中没有要求
项目所有者操作。工作范围来自 `handover.md` 的四项遗留工作：冻结模型，明确
标注口径，裁定 train/val 大分歧，补并发证据与外部模型评测入口。

接手基线是 commit `a4b2b8c624fb6b13588d4e5747ad7eceeb60e5de` 加已有脏工作区。
前人的补标、LTR 诊断、Docker 等未提交成果均保留。本轮不暂存、不提交、不推送，
不改依赖、端口契约、数据库迁移或原始标注，不训练或替换模型。

## 交付与决定

| 工作 | 本轮结果 | 证据 |
|---|---|---|
| 正式标注入口 | 保留 `labels.csv`，并集与裁定作为独立命名的补充协议 | [评估策略](../data/evaluation-policy.md) |
| train/val 裁定 | 796 对中大分歧 11 对；train 10、val 1、test 0；0/1/2/3 等级分别 2/9/0/0 | [裁定报告](../data/adjudication-train-val.md) |
| 并发 HTTP 基准 | 四模式各 120 请求，并发 4，正式段 480/480；预热 15/16 | [并发报告](../data/bench-concurrent.md) |
| 外部 Agent 入口 | 固定 50 题、受限工具计划、结构化证据与引用校验、总超时、2 MiB 响应上限、请求日志 | [外部 Agent 报告](../data/agent-llm-report.md) |
| 旧 Agent 结论修正 | 白名单 50/50、引用非空 27/27、拒绝短语 8/8；不再称为参数或引用支持正确率 | [确定性 Agent 报告](../data/agent-report.md) |
| 向量初始化 | 实例线程锁及锁内二次检查，防止并发首次请求重复构造编码器；推理不加串行锁 | `src/searchpilot/search/vector_index.py` |

裁定明确署名 `codex-text-adjudication-2026-10-13`，附查询原文、标题摘要、双方原分、
逐条理由与输入 SHA-256；这是知晓双方分数后的工具文本裁定，不是人工金标准或
独立模型盲标。裁定尚未用于训练或评估，其他 197 对相邻分歧也没有被悄悄裁定。
现有合并函数取最大等级，因此不能直接用它合并降低等级的裁定，具体覆盖规则见评估策略。

## 本次实际验证

以下均在本轮本机执行。Python 导入位置核对为
`E:\searchPilot\src\searchpilot\__init__.py`，不是其他 checkout。

| 检查 | 结果 | 边界 |
|---|---|---|
| 接手单元基线 | 317 passed、10 deselected | 修改前工作区 |
| 最终单元回归 | **388 passed、10 deselected，14.51 秒** | 含新增 71 个测试；默认不跑数据库集成 |
| PostgreSQL 集成 | **10 passed、317 deselected，9.17 秒** | 接手时真实 PostgreSQL 18.6；临时 schema，无数据库相关后续改动 |
| Ruff check | All checks passed | 最终整个工作区 |
| Ruff format --check | 160 files already formatted | 最终整个工作区 |
| Mypy src | 70 source files，无错误 | 最终源码 |
| Git diff --check | 无输出，成功 | 含原有工作区改动 |
| 裁定只读复现 | verified=true，11 对，输入哈希通过 | 原始输入未变 |
| 外部 Agent 默认入口 | UNRUN，requests_sent=0 | 未执行真实模型调用 |

单元测试有一条现有 FastAPI/Starlette TestClient 弃用警告；集成测试另有连接池
默认 open 参数弃用警告。本轮未更换依赖。以上是本机检查，不冒充 GitHub CI 运行。

主要命令：

```powershell
$env:TMP='E:\searchPilot\.tmp'; $env:TEMP=$env:TMP
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe scripts/adjudicate_train_val.py
.venv\Scripts\python.exe -m scripts.evaluate_agent_llm
git diff --check
```

集成测试在同一 Python 进程内用 `load_dotenv('.env')` 读取既有连接设置，先用
`psycopg.connect(..., connect_timeout=3)` 与 `select version()` 验证可达性，再调用
`pytest.main(['-m', 'integration'])`；没有打印或写入数据库凭据。

## 并发结果与初始化修复的关系

独立服务使用真实索引和缓存模型、单 worker、无 reload、无数据库，监听
`127.0.0.1:18080`。正式段成功请求的 p95/p99（毫秒）分别是：

| 模式 | 成功 | p95 | p99 |
|---|---:|---:|---:|
| BM25 | 120/120 | 47.033 | 56.120 |
| 向量 | 120/120 | 950.584 | 1099.964 |
| RRF | 120/120 | 1028.548 | 1331.922 |
| LTR | 120/120 | 1408.937 | 1583.321 |

首次向量预热在 10 秒截止时间超时，后续预热成功。包括预热在内是 **495/496**，
不是全部请求 100% 成功。恢复阶段 live/ready 均 200，额外 BM25 请求 1/1 成功，
单独记账。服务日志出现两次加载权重；代码复核确认懒初始化没有互斥，因此本轮
补初始化锁，避免超时请求的后台线程和后续请求同时构造编码器。

上述 480 次负载发生在锁修复之前，执行脚本快照与原始结果中的 SHA-256 已核对一致。
不能用它冒充最终源码的性能验证。修复后新增并发初始化、初始化失败后恢复测试；
`test_search_vector.py` 共 10 个测试通过。另以真实缓存的 `BAAI/bge-small-en-v1.5`
做两条同时进入同一编码器的调用，只构造 **1 次**，两条输出都是相同的 384 维有限向量。
该检查仅在模型构造器外增加计数包装，未替换真实模型；它验证功能，不提供 HTTP SLA。

原始证据：

- `artifacts/bench-concurrent/2026-10-13-local-c4/results.json`
- 同目录 `bench_concurrent.executed.py`、`recovery.json`、`resource-snapshot.json`
- `artifacts/bench-concurrent/encoder-init-check-2026-10-13.json`

这些文件位于忽略目录，尚未打包或发布。本轮自建服务已停止，18080 不再监听。

## 复核与收尾

独立复审分别覆盖质量、性能和复用，未要求泛化重构。确认并落实的补强包括：

- 压测输出独占创建，标记完整/中断/失败；当前脚本逐请求追加 pending/result 日志，
  中断时未配对的 pending 保持未知，不自动重发；预热失败也使退出码非零。
- LLM 响应 envelope 类型、重复键、非有限数、工具参数、错误证据和引用都有负向测试；
  2 MiB 流式响应上限避免无界累积。
- 初始化锁仅保护构造，失败时释放，推理保持锁外；独立复审已核对源码哈希和功能证据。

前两项在首次压测之后完成，最终全量回归覆盖修补后的代码，没有覆盖或重写旧性能记录。

## 未完成的外部验证

真实外部 LLM 评测仍为 **UNRUN**：本轮没有使用模型端点或 API 密钥，也没有外部调用费用。
新入口已完成本地自动验证；需要实际可用的模型服务才会产生真实模型结果。自由文本
语义为 UNSCORED，结构化通过不能证明所有论述无幻觉。CLI 退出 0 也仅表示评测执行结束。

Docker 引擎本轮未运行，未重跑镜像或容器验收；不能沿用 2026-10-12 的历史记录当作本轮通过。
并发测试只有本机一轮短负载，未包含数据库路径、CTR 并发、持续资源采样、多 worker、
跨机或长时稳定性；冷启动超时及服务器取消完成也未被证明解决。
因此本轮完成交接中的本地实现与证据补齐，不将整个 PRD 或稳定性/SLA 标为全部验收通过。
