# SearchPilot 技术选型

| 项目 | 内容 |
|---|---|
| 配套文档 | 《SearchPilot 产品需求文档》v2.0（`searchPilotPRD.md`），本文所有选型以其范围为准 |
| 文档版本 | v1.0 |
| 适用阶段 | MVP → V1 → V2（六周计划） |
| 状态 | 待首次安装验证；具体版本号在首次安装时写入 lock 文件，本文不预先钉死版本 |

**选型原则**（冲突时按此顺序取舍）：
1. **单机 CPU 可跑完整链路**：数万篇新闻、数十万条曝光、曝光级样本百万量级，不需要分布式组件。
2. **能解释到公式与数据结构**：BM25 的 k1 / b、RRF 的 k、ECE 的分箱、校准器的参数都必须可见、可改、可测；不选把这些藏在黑盒里的组件。
3. **可复现优先于性能**：锁文件、固定种子、确定性索引构建；近似方法必须先与精确方法对齐。
4. **与 EvoRec 的工程口径一致**：PostgreSQL、迁移方式、错误结构、幂等语义、lock 文件形式对齐，降低跨项目理解成本（只对齐约定，不复制代码）。
5. **最少活动部件**：每多一个服务进程，就多一个要在 README 里解释和在 CI 里拉起的东西；先测量再引入。

---

## 1. 总览

| 层 | 选择 | 备选（未选） | 选择理由 |
|---|---|---|---|
| 语言 / 运行时 | Python 3.12 | 3.11 / 3.13 | 与 EvoRec 服务环境一致；PyTorch、FastAPI、psycopg 均稳定支持 |
| 环境与锁定 | `uv` + `pyproject.toml` + `uv.lock`，另导出 `requirements-*.lock.txt` | pip-tools、poetry、conda | 安装快、锁定完整；导出 txt 供无 uv 环境与 CI 使用，形式上与 EvoRec 的 lock 文件一致 |
| Web 框架 | FastAPI + uvicorn + Pydantic v2 | Flask、Litestar、Django | 契约即代码（OpenAPI 可导出入库对账）、依赖注入适合注入假对象、与 EvoRec 一致 |
| 数据库 | PostgreSQL 16 | MySQL、SQLite | 事务 + 幂等约束 + advisory lock；与 EvoRec 的迁移脚本、临时 schema 测试夹具同一套路；SQLite 缺少真实并发与锁语义 |
| 数据库访问 | `psycopg` 3（binary + pool）+ 纯 SQL 迁移文件 | SQLAlchemy ORM、asyncpg | SQL 可见、便于解释事务边界与锁；迁移文件逐个 SHA-256 校验并用 advisory lock 串行执行（沿用 EvoRec 约定） |
| 离线数据处理 | pandas + pyarrow（Parquet） | polars、Spark | 规模不需要更重的工具；Parquet 快照便于校验值与版本化。DuckDB 作为可选的即席查询工具 |
| 词法检索 | 自实现倒排索引 + BM25（NumPy） | Elasticsearch / OpenSearch、Pyserini（Lucene）、Tantivy、PostgreSQL 全文检索 | 规模小、需要解释 posting list 与 BM25 参数、需要分阶段计时；用 `bm25s` 或 `rank_bm25` 做正确性交叉验证（单元测试内对分数容差比对） |
| 分词与归一化 | 正则分词 + 小写 + 全半角 + `snowballstemmer`；停用词表可配 | NLTK、spaCy | 英文新闻语料足够；不依赖运行时下载数据。P1 纠错用 `symspellpy` 或基于索引词表的编辑距离 ≤ 1 |
| 文本编码器 | `sentence-transformers` 加载 `BAAI/bge-small-en-v1.5`（384 维）为默认；`all-MiniLM-L6-v2` 为更轻备选 | 自训双塔、OpenAI Embedding API | CPU 可推理、许可宽松、离线可复现；自训双塔已在 EvoRec 做过，不重做；API 编码不可复现且引入外部依赖 |
| 向量索引 | NumPy 精确内积（默认）→ `faiss-cpu` `IndexFlatIP` / HNSW 作对照 | Milvus、Qdrant、Weaviate、pgvector | 数万条 × 384 维，精确检索毫秒级；近似索引只用于报告召回损失。Windows 上 `faiss-cpu` 轮子不可用时以 `hnswlib` 替代 |
| 融合 | RRF（自实现） | 分数归一化加权、学习融合 | 排名融合不受两路分数尺度影响；归一化加权保留为对照 |
| LTR 重排 | PyTorch pairwise（RankNet 式 MLP） | LightGBM `lambdarank`、XGBoost | 与 CTR 共用一套 PyTorch 训练与登记代码；LambdaMART 作为 OQ-2 对照，用于回答"树模型在小标注集上是否更稳" |
| CTR 模型 | PyTorch LR → 小 MLP | DeepFM / DCN、scikit-learn LR | 曝光级样本量支持 CPU 小批量训练；scikit-learn LR 仅作健全性对照 |
| 校准 | 温度缩放（自实现，默认）；Platt（scikit-learn LR on logits）与等渗（`IsotonicRegression`）作对照 | `CalibratedClassifierCV` 整体封装 | 需要分别暴露校准器参数并在验证集单独拟合；整体封装不便解释与登记 |
| 评估指标 | NDCG / MRR / Recall / 零结果率 / ECE 自实现并单测；AUC / LogLoss 用 scikit-learn | `ranx`、`pytrec_eval`、`torchmetrics` | 自实现才能在面向读者的报告里逐项解释；用 `ranx` 做一次交叉验证即可 |
| 实验登记 | PostgreSQL `experiments` / `metrics` 表 + JSON 配置文件 + 代码 commit | MLflow、Weights & Biases | 不引入新服务；`source / source_ref / protocol_version` 字段是 PRD 第 7 节的强制要求，自建表更直接 |
| EvoRec 结果导入 | 自写导入器：固定 commit 拉取结果文件 → SHA-256 校验 → 只读写入 | 运行时调用 EvoRec API | 静态导入可复现、不依赖对方服务；PRD 默认路径 |
| Agent 运行时 | 自写薄工具调用循环（约 300 行）+ Pydantic 校验工具参数；LLM 走 OpenAI 兼容 Chat API，供应商可配置（云端或本地 Ollama / vLLM） | LangChain、LlamaIndex、AutoGen | 窄工具与可审计是核心需求，框架会隐藏提示词与循环逻辑；本地模型保证评测可重复 |
| 日志 / 指标 / 追踪 | 标准库 `logging` JSON 格式 + `contextvars` 传递 `request_id` / `trace_id`（P0）；`prometheus_client` 暴露 `/metrics`（P1）；OpenTelemetry 延后（P2 可选） | loguru、structlog、全套 OTel | 先满足"日志 + 指标 + 追踪互补"，不先引入 collector |
| 压测 | 自写 `asyncio` + `httpx` 脚本，输出命令、机器、版本、原始结果 JSON | k6、Locust、wrk | PRD 要求压测记录可追溯；自写脚本跨平台（Windows 开发机）且输出格式可控；k6 作可选对照 |
| 测试 | pytest + `httpx.ASGITransport`；集成测试用真实 PostgreSQL 临时 schema | `pytest-postgresql`、SQLite 替身 | 与 EvoRec 一致：缺少数据库时集成测试明确失败，不静默跳过 |
| 代码质量 | ruff + mypy（PRD 13.3 命令） | flake8 + black + isort | 一个工具覆盖 lint 与格式化 |
| 容器与 CI | Dockerfile（`python:3.12-slim`）+ `docker compose`（服务 + PostgreSQL）；GitHub Actions 用 `services: postgres` | Kubernetes、Helm | 一键启动与 CI 复现；不引入编排 |
| 缓存 | 不引入（Redis 待测量后决定，OQ-7） | Redis、本地 LRU | PRD 6.2 的约束 |

---

## 2. 分层说明

### 2.1 数据层

**数据集：MIND Small。** 新闻标题 / 摘要 / 类目做搜索语料；behaviors 曝光日志自带点击与未点击标签，直接构成曝光级 CTR 样本和 ItemCF 回退所需的点击历史。MovieLens 没有文本、Amazon Reviews 没有负样本，均不满足"搜 + 广"两条线。MIND 遵循 Microsoft Research 许可条款（研究用途），原始文件不再分发：仓库只提供下载脚本、SHA-256 校验值与预处理说明。

**存储格式：Parquet 快照 + PostgreSQL。** 预处理后的新闻、曝光、查询集写成 Parquet（带校验值与 `data_version`），训练与评估从 Parquet 读；服务态内容、事件、标注、实验登记进 PostgreSQL。两者边界：离线不写库，在线不改 Parquet。

**PostgreSQL 而非 MySQL。** 三点理由：advisory lock 用于迁移与导入串行化；每个测试会话独立临时 schema 的夹具模式成熟；与 EvoRec 的 12 个迁移文件风格一致，便于互相阅读。不选 SQLite 是因为"数据库超时后事务回滚""并发幂等键"这两条 DoD 需要真实的锁与并发语义。

**访问方式：psycopg 3 + 纯 SQL。** 不用 ORM 的原因是 PRD 要求能解释事务边界、行锁与幂等写入；纯 SQL 文件审阅成本更低。连接池用 `psycopg_pool`，池大小在压测后再定，不预设数字。

### 2.2 搜索层

**倒排索引与 BM25 自实现。** 规模（数万篇）允许全部驻留内存；自实现的收益是可以逐阶段计时（分词 → 倒排取交并 → 打分 → 截断）、可以把 k1 / b 写进实验配置、可以在失败样例里打印命中词项。Elasticsearch 会把这些细节封装起来，并且多一个 JVM 进程要在 README 和 CI 里解释。正确性用 `bm25s` 或 `rank_bm25` 做交叉验证：同一语料、同一分词下，Top-50 分数在容差内一致，该测试进 CI。

**分词与归一化。** 正则分词、小写、全半角与标点清理、`snowballstemmer` 英文词干化，停用词表可配置并写入实验配置。P1 的拼写纠错优先用基于索引词表的编辑距离 ≤ 1（零依赖、可解释），`symspellpy` 作为更快的备选。

**编码器：bge-small-en-v1.5。** 384 维、CPU 单条推理毫秒级、MIT 许可；在英文检索基准上一般优于 MiniLM，MiniLM 作为更轻的备选（OQ-1）。向量在索引构建期一次性预计算并落盘（`.npy` + 校验值），服务启动时加载，不在请求期编码文档；只有 query 在请求期编码。

**向量索引：先精确。** 数万条 × 384 维的内积矩阵运算在 CPU 上是毫秒级，精确检索足够支撑评测与服务。`faiss-cpu` 的 `IndexFlatIP` 与 HNSW 只用于对照实验，报告相对精确检索的召回损失。Windows 开发机若无可用 `faiss-cpu` 轮子，用 `hnswlib` 替代近似索引，不影响默认路径。

**融合：RRF。** 排名融合天然规避"BM25 原始分与余弦分不能直接相加"的问题；分数归一化加权保留为对照，用来在报告里说明两者差异。k 值写入配置。

**LTR：PyTorch pairwise。** 特征只有十个左右（两路分数、两路排名、热度、新鲜度、类目匹配、标题长度），小 MLP 加 RankNet 式 pairwise 损失即可；与 CTR 模型共用训练循环、种子控制与实验登记代码。LightGBM `lambdarank` 作为 OQ-2 的对照，用于验证"小标注集上树模型是否更稳"；若对照赢了，报告如实写并解释。

### 2.3 CTR 与校准层

**模型：LR → 小 MLP。** 曝光级样本量在百万级，CPU 小批量训练可在分钟到十分钟量级完成（以实测为准）。LR 是必须保留的基线；MLP 只有两三层。特征为用户历史类目分布与历史长度、物品类目、物品热度、可选的标题向量。不选 DeepFM / DCN 的原因是本项目的证据重点在校准而不是模型结构。

**校准：温度缩放默认。** 单参数、在验证集上用负对数似然拟合、易于解释和登记；Platt 与等渗作为对照（OQ-3）。三种校准器都以独立对象登记参数与拟合集合，便于在报告中写明"只在验证集拟合"。

**ECE 自实现。** 固定分箱（默认 10 个等宽箱，写入配置），同时输出每箱样本数与校准曲线数据点；单元测试用构造数据验证 ECE = 0 与已知偏差的情况。AUC / LogLoss 用 scikit-learn，避免自实现引入错误。

**eCPM 对照。** 合成出价用固定分布与种子生成，`source=synthetic`；对照脚本只依赖 NumPy / pandas，不需要额外组件。

### 2.4 服务层

**FastAPI + Pydantic v2。** 输入契约（query 长度、候选列表上限、幂等键格式）在 Pydantic 模型里声明，OpenAPI 导出到仓库并由测试比对，避免文档与实现分叉。错误结构与 EvoRec 对齐：`{error:{code, message, retryable, request_id}}`。

**生命周期。** 索引、编码器、LTR / CTR 模型与校准器在应用启动的 lifespan 中加载并做自检（黄金样本打分），任何一项失败则 `/health/ready` 返回 503；请求期不加载模型。CPU 推理在线程池中执行，避免阻塞事件循环（这是 16.3 自检清单里的一问）。

**可观测性。** P0 用标准库 `logging` 输出 JSON 行日志，`request_id` / `trace_id` 通过 `contextvars` 贯穿中间件、用例与数据库层；P1 用 `prometheus_client` 暴露分阶段计时与错误计数；OpenTelemetry 作为 P2 可选，不先引入 collector。

### 2.5 实验登记与来源层

**自建表而非 MLflow。** PRD 第 7 节要求 `source / source_ref / protocol_version` 字段与跨来源不可比标记，这些是业务约束，放在自己的表里最直接；MLflow 还会多一个服务进程。配置用 JSON 文件（内容哈希入库），代码 commit 由脚本自动读取并拒绝脏工作区（可用 `--allow-dirty` 显式放行并记录）。

**EvoRec 导入器。** 以固定 commit 从公开仓库拉取 `docs/experiments/**/results.json` 等文件，逐文件 SHA-256 校验，写入 `experiments(source=evorec, source_ref)` 与 `metrics`，指标名原样保留；导入事务结束后记录只读（数据库层面用触发器或权限拒绝更新）。不在运行时调用 EvoRec 服务。

### 2.6 Agent 层（P2）

**自写薄循环。** 工具调用循环需要做的事情很少：发送消息、解析工具调用、用 Pydantic 校验参数、执行白名单工具、记录 `trace_id` 与每步耗时、限制最大步数、检测重复调用。这些写下来约 300 行，比引入 LangChain 更容易审计，也更容易在测试中注入假 LLM。

**LLM 接入：OpenAI 兼容 Chat API，供应商可配置。** 评测时优先用本地模型（Ollama 或 vLLM 的 OpenAI 兼容端点），固定模型名、温度 0、记录模型标识与提示词哈希，保证固定任务集可重复；云端 API 作为可选对照，并在报告里注明不可完全复现。

**评测器自写。** 读取 `agent_tasks` 表，逐任务比对工具选择、参数、最终答案与引用格式；跨来源陷阱题的期望答案为"协议不同，无法直接比较"。

### 2.7 工程与交付

- **环境**：`uv` 管理虚拟环境与锁定，`pyproject.toml` 分 `service`、`research`、`dev` 三组可选依赖；导出 `requirements-service.lock.txt` 等文本锁文件供 CI 与无 uv 环境使用。
- **测试**：单元测试不依赖网络与模型下载（编码器用假对象）；集成测试要求 `SEARCHPILOT_DATABASE_URL`，缺失时明确失败；向量与编码器相关测试用极小语料并标记为需要模型缓存。
- **CI**：GitHub Actions，`services: postgres`；流水线为 ruff → mypy → 单元 → 集成；不在 CI 下载 MIND 与编码器权重。
- **容器**：`Dockerfile` 基于 `python:3.12-slim`，`docker compose` 拉起服务与 PostgreSQL；Windows 开发机可不使用 Docker，直接连本地 PostgreSQL。
- **硬件**：默认 CPU 版 PyTorch 轮子；GPU 可选，仅加速编码器批量推理与 MLP 训练，不改变任何结论口径。

---

## 3. 依赖清单（按环境分组）

| 组 | 依赖 | 用途 |
|---|---|---|
| service | fastapi、uvicorn、pydantic、psycopg[binary,pool]、numpy、snowballstemmer、sentence-transformers、torch（CPU）、prometheus_client | 在线服务与模型加载 |
| research | pandas、pyarrow、scikit-learn、scipy、matplotlib、bm25s 或 rank_bm25（交叉验证）、ranx（交叉验证）、faiss-cpu 或 hnswlib（对照）、lightgbm（OQ-2 对照）、duckdb（可选） | 数据处理、训练、评估、报告图表 |
| agent | httpx、openai（兼容客户端）或直接 httpx 调用 | Agent 的 LLM 接入 |
| dev | pytest、pytest-asyncio、httpx、ruff、mypy、types-* 存根 | 测试与代码质量 |

所有版本在首次安装时由 `uv lock` 确定并提交；本文不预先指定版本号。

---

## 4. 与 EvoRec 的一致性对照

| 维度 | EvoRec | SearchPilot | 说明 |
|---|---|---|---|
| Python | 3.12 | 3.12 | 一致 |
| 数据库 | PostgreSQL，纯 SQL 迁移，SHA-256 校验 + advisory lock | 同 | 复用约定，不复制代码 |
| Web 框架 | FastAPI + Pydantic | 同 | 一致 |
| 错误结构 | `{error:{code,message,retryable,request_id}}` | 同 | 一致 |
| 幂等 | `Idempotency-Key` / `event_id` + 载荷哈希 | 同 | 一致 |
| 锁文件 | `requirements-*.lock.txt` | `uv.lock` + 导出的 `requirements-*.lock.txt` | 形式兼容 |
| 离线格式 | Parquet + DuckDB | Parquet + pandas，DuckDB 可选 | 兼容 |
| 模型产物 | 不可变 bundle + 清单校验 | 索引 / 模型 / 校准器各自版本化 + SHA-256 | 简化版，范围更小 |
| 实验登记 | 协议文档 + 结果 JSON | 数据库表 + 结果 JSON，带 `source` | SearchPilot 多了来源维度 |
| 搜索 / CTR / Agent 组件 | 无 | 本文第 2.2–2.6 节 | SearchPilot 的增量 |

---

## 5. 明确不选的组件及原因

| 组件 | 不选原因 |
|---|---|
| Elasticsearch / OpenSearch | 规模不需要；封装 BM25 细节；多一个 JVM 进程要维护与解释 |
| Milvus / Qdrant / pgvector | 数万条向量精确检索已足够；近似索引只需本地库对照 |
| MySQL | 失去与 EvoRec 的工程一致性；advisory lock 与临时 schema 夹具模式不如 PostgreSQL 顺手 |
| SQLAlchemy ORM | 需要在报告中解释事务与锁，纯 SQL 更直接；若后期表增多可再评估 Core |
| MLflow / W&B | 多一个服务；来源与协议字段需要自定义表 |
| LangChain / LlamaIndex | 隐藏提示词与循环逻辑，与"窄工具 + 可审计"冲突 |
| OpenAI Embedding API | 不可复现、引入外部依赖与成本；本地小模型足够 |
| Redis / 消息队列 / Kubernetes | PRD 非目标；测量出瓶颈前不引入 |
| Spark / polars | 数据规模不需要；pandas + pyarrow 足够且生态最通用 |

---

## 6. 风险与回退

| 风险 | 回退方案 |
|---|---|
| `faiss-cpu` 在 Windows 上安装失败 | 默认路径本就是 NumPy 精确检索；近似对照改用 `hnswlib` |
| 编码器在 CPU 上批量编码全部新闻过慢 | 一次性预计算并落盘；必要时换 MiniLM；GPU 可选加速 |
| 自实现 BM25 与参考实现分数不一致 | 交叉验证测试会在 CI 失败；以参考实现为准修正分词或公式 |
| 曝光级样本过大导致 CTR 训练慢 | 先按时间窗口下采样做开发集，正式实验用全量并记录耗时 |
| 本地 LLM 不支持可靠的工具调用 | 换支持工具调用的本地模型或云端 API，并在报告标注不可复现部分 |
| PostgreSQL 临时 schema 夹具在 CI 与本机行为不一致 | CI 用官方 `postgres` 镜像并固定大版本；本机安装同大版本 |

---

## 7. 待验证项（与 PRD 开放问题对应）

- OQ-1 编码器：bge-small 与 MiniLM 在查询集上的 NDCG@10 与 CPU 编码耗时对比后定稿。
- OQ-2 LTR：pairwise MLP 与 LightGBM `lambdarank` 在同一切分上的对比后决定默认实现。
- OQ-3 校准：三种校准器的 ECE 与分组 ECE 对比后决定默认方法。
- OQ-6 向量索引：精确检索延迟与近似索引召回损失测量后决定是否在服务中启用近似索引。
- OQ-7 Redis：压测显示热门结果或特征计算成为瓶颈时再评估。
