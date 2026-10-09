# SearchPilot 搭建计划（MVP 并行开发契约）

本文是并行开发的唯一协调文档。所有代理在动手前必须读完本文、`searchPilotPRD.md`（第 5、8、9、10 节）与 `searchPilot技术选型.md`。
与本文冲突的实现以本文为准；需要改契约时先停下来报告，不要单方面改。

## 0. 环境与硬性约束

- Python：`E:\searchPilot\.venv\Scripts\python.exe`（3.12）。依赖已由技术主管安装；**不要修改 `pyproject.toml`**，需要新依赖时在返回报告里列出。
- Shell：本机运行任何命令都需要 `required_permissions: ["all"]`。只在工作区内读写；不安装全局软件。
- Git：**禁止** `git add / commit / push / stash / reset / checkout` 等一切写操作（见 `.cursor/rules/git-workflow.mdc`）。所有改动留在工作区，由项目所有者审核提交。
- 网络：pip 可用。MIND 数据下载可能需要额外网络权限；下载失败不要阻塞，用合成小样本完成开发与测试并记录。
- 其他代理正在同一工作区并行写不同目录：**只写自己名下的文件**，不要"顺手"改别人的文件或格式化整个仓库。
- 文件编码一律 UTF-8（`open(..., encoding="utf-8")`），路径用 `pathlib`，CSV 用 `newline=""`。
- 质量门槛（各自名下文件必须通过）：
  - `.venv\Scripts\python.exe -m ruff check <自己的路径>`
  - `.venv\Scripts\python.exe -m ruff format <自己的路径>`
  - `.venv\Scripts\python.exe -m mypy src\searchpilot\<自己的包>`
  - `.venv\Scripts\python.exe -m pytest tests\test_<前缀>_*.py`
- 单元测试不访问网络、不下载模型、不需要数据库；总耗时控制在十几秒内。
- 临时文件一律用 pytest 的 `tmp_path` fixture，**不要**在仓库内自建临时目录。本机 `%TEMP%\pytest-of-7` 是一个权限损坏的残留目录，会让 `tmp_path` 报 `PermissionError`；运行 pytest 前把临时目录重定向到工作区内即可（已在 .gitignore）：
  `$env:TMP="E:\searchPilot\.tmp"; $env:TEMP=$env:TMP; New-Item -ItemType Directory -Force $env:TMP | Out-Null; .venv\Scripts\python.exe -m pytest ...`
- 确定性：所有排序对分数并列按 `item_id` 升序打破；随机过程显式传种子。

## 1. 文件归属

| 工作流 | 负责内容 | 独占写入路径 |
|---|---|---|
| **A 骨架与服务** | 配置、日志与 request_id、错误信封、Pydantic 契约、FastAPI 应用与路由、bootstrap 组装、CI、容器、README 骨架 | `src/searchpilot/config.py`、`observability.py`、`errors.py`、`contracts.py`、`bootstrap.py`、`src/searchpilot/api/**`、`tests/conftest.py`、`tests/test_api_*.py`、`tests/test_core_*.py`、`.github/workflows/ci.yml`、`Dockerfile`、`docker-compose.yml`、`.env.example`、`README.md`、`scripts/run_api.py` |
| **B 数据与数据库** | MIND 下载与解析、Parquet 快照与 manifest、时间切分、查询集模板与标注规范、PostgreSQL 迁移与运行器、仓储实现 | `src/searchpilot/data/**`、`src/searchpilot/db/**`、`db/migrations/**`、`scripts/download_mind.py`、`scripts/build_dataset.py`、`scripts/migrate_database.py`、`scripts/seed_items.py`、`tests/test_data_*.py`、`tests/test_db_*.py`、`docs/data/**`、`data/queries/**` |
| **C 搜索核心与评估** | 归一化与分词、倒排索引、BM25、RRF、搜索服务、搜索评估指标与脚本 | `src/searchpilot/search/**`、`src/searchpilot/eval/**`、`scripts/build_index.py`、`scripts/evaluate_search.py`、`tests/test_search_*.py`、`tests/test_eval_*.py` |
| **D 回退推荐与反馈** | 热门、ItemCF、用户历史加载、回退服务与评估、反馈幂等写入（内存实现） | `src/searchpilot/recommend/**`、`src/searchpilot/feedback/**`、`scripts/evaluate_fallback.py`、`tests/test_recommend_*.py`、`tests/test_feedback_*.py` |
| **技术主管（只读给各代理）** | 契约与协调 | `pyproject.toml`、`src/searchpilot/ports.py`、`src/searchpilot/__init__.py`、本文、`.gitignore` |

## 2. 模块间契约

类型全部在 `src/searchpilot/ports.py`，各模块只 import 那里的类型，不 import 彼此的实现。各工作流必须提供以下**工厂函数 / 类**（名字和签名固定，A 的 bootstrap 按此组装）：

| 提供方 | 符号 | 说明 |
|---|---|---|
| C | `searchpilot.search.service.build_search_service(artifact_dir: Path) -> SearchPort` | 从 `artifact_dir/search/bm25/` 加载索引；缺失时抛 `searchpilot.search.service.SearchNotReadyError` |
| C | `searchpilot.search.service.InMemorySearchService(documents: Sequence[Document]) -> SearchPort` | 测试与小语料用，直接从文档建索引 |
| D | `searchpilot.recommend.service.build_recommend_service(data_dir: Path) -> RecommendPort` | 从 `data_dir/processed/` 读取 Parquet 训练热门与 ItemCF；缺失时抛 `searchpilot.recommend.service.RecommendNotReadyError` |
| D | `searchpilot.recommend.service.InMemoryRecommendService(histories: Mapping[str, Sequence[str]], model_version: str) -> RecommendPort` | 测试用 |
| D | `searchpilot.feedback.memory.InMemoryFeedbackStore() -> FeedbackStore` | 幂等语义的参考实现 |
| B | `searchpilot.db.connection.create_pool(database_url: str) -> psycopg_pool.ConnectionPool` | 池大小从环境变量读，默认保守 |
| B | `searchpilot.db.feedback_store.PostgresFeedbackStore(pool) -> FeedbackStore` | 与内存实现行为一致，受同一组测试约束 |
| B | `searchpilot.db.items.PostgresItemStore(pool) -> ItemStore` | |
| B | `searchpilot.db.readiness.DatabaseProbe(pool) -> ReadinessProbe` | 返回 `{"database": bool}` |
| B | `searchpilot.db.migrate.apply_migrations(database_url: str, migrations_dir: Path) -> list[str]` | 返回本次应用的文件名；逐文件 SHA-256 入库，已应用文件被改动时拒绝 |
| A | `searchpilot.bootstrap.create_app(*, search: SearchPort | None = None, recommend: RecommendPort | None = None, feedback: FeedbackStore | None = None, items: ItemStore | None = None, probes: Sequence[ReadinessProbe] = ()) -> FastAPI` | 测试注入假对象；生产由 `bootstrap.build_default_app()` 读配置后调用上面各工厂 |

幂等语义（D 的内存实现与 B 的 PostgreSQL 实现必须一致）：
- 同一 `idempotency_key` + 相同规范化内容 → 返回首次结果，`replayed=True`，不重复计数。
- 同一 `idempotency_key` + 不同内容 → 抛 `ports.IdempotencyConflictError`（API 层映射为 409 `IDEMPOTENCY_CONFLICT`）。
- 命名提醒：ruff 规则 N818 要求异常类以 `Error` 结尾（如 `SearchNotReadyError`、`RecommendNotReadyError`）。
- 规范化内容 = `(request_id, item_id, kind, event_at 转 UTC ISO8601, user_id, position)`。

## 3. 数据契约

目录：`data/raw/`（原始下载，忽略不入库）、`data/processed/<data_version>/`（Parquet，忽略不入库）、`data/queries/`（查询集 CSV，入库）。`artifacts/`（索引与模型，忽略不入库）。

### 3.1 Parquet（B 产出；C、D 消费）

| 文件 | 列 | 说明 |
|---|---|---|
| `items.parquet` | `item_id: str`、`title: str`、`abstract: str`（可空→空串）、`category: str`、`subcategory: str`、`url: str`、`first_seen_at: timestamp[us, UTC]`（可空） | 来自 MIND `news.tsv`；`first_seen_at` 为该新闻在 behaviors 中首次出现的曝光时间 |
| `impressions.parquet` | `impression_id: str`、`user_id: str`、`shown_at: timestamp[us, UTC]`、`item_id: str`、`clicked: int8`（0/1）、`split: str`（`train`/`dev`）、`source: str`（固定 `mind`） | 来自 `behaviors.tsv`，一行一个（曝光, 候选）对；无位置信息，不伪造 |
| `user_history.parquet` | `user_id: str`、`history: list<str>` | 每个用户在 train 切分内最后一次曝光记录中的点击历史（按时间升序） |
| `manifest.json` | `data_version`、`created_at`、`source_files[{name, sha256, bytes}]`、`outputs[{name, sha256, rows}]`、`time_bounds{train_start, train_end, dev_start, dev_end}` | `data_version` = 四个输入文件按固定顺序（train/news, train/behaviors, dev/news, dev/behaviors）的 SHA-256 拼接成字符串后再做一次 SHA-256，取前 12 位 |

### 3.2 查询集（B 建模板与规范；标注由项目所有者完成）

- `data/queries/queries.csv`：`query_id, query_text, query_type, split`；`query_type` 取 `ports.QueryType` 的六个值；`split` ∈ `train/val/test`。
- `data/queries/labels.csv`：`query_id, item_id, grade, annotator`；`grade` ∈ 0–3。
- `docs/data/annotation-guideline.md`：分级定义、每类查询的构造方法、双标抽样与一致率计算方式。

### 3.3 索引与模型产物（C、D 产出）

- `artifacts/search/bm25/`：`index.json`（或二进制）+ `meta.json`（`model_version`、`k1`、`b`、`data_version`、`doc_count`、`vocab_size`、`built_at`、输入 Parquet 的 SHA-256）。
- `artifacts/recommend/<model_version>/`：热门表与 ItemCF 邻居表 + `meta.json`。
- `model_version` 命名：`bm25-<meta 内容 SHA-256 前 8 位>`、`popular-<data_version>`、`itemcf-<data_version>`。

## 4. API 契约（A 实现；其余工作流不碰路由）

统一错误信封（与 EvoRec 对齐）：

```json
{"error": {"code": "INVALID_INPUT", "message": "query is empty", "retryable": false, "request_id": "req_..."}}
```

错误码：`INVALID_INPUT`（400/422）、`ITEM_NOT_FOUND`（404）、`IDEMPOTENCY_CONFLICT`（409）、`NOT_READY`（503）、`INTERNAL`（500）。FastAPI 默认的 422 响应也要改写成该信封。

| 接口 | 请求 | 响应 |
|---|---|---|
| `GET /health/live` | — | `200 {"status":"ok"}` |
| `GET /health/ready` | — | `200 {"status":"ready","checks":{...}}` 或 `503 {"status":"not_ready","checks":{...}}` |
| `POST /search` | `{query: str, limit: int=10, mode: "bm25"\|"vector"\|"hybrid"\|"ltr" = "bm25", filters: {category?: str}}` | `200 {request_id, normalized_query, model_version, mode, results:[{item_id, score, rank, channel}]}`；空 query / 超长 query → 422；`mode` 未就绪 → 503 `NOT_READY`；无结果返回 `results: []` 且 200（空结果是合法结果） |
| `POST /recommend` | `{user_id: str, limit: int=10, context?: object}` | `200 {request_id, model_version, channel, cold_start, results:[{item_id, score, rank, channel}]}`；结果不重复 |
| `POST /feedback` | `{idempotency_key: uuid, request_id: str, item_id: str, kind: "impression"\|"click"\|"like"\|"hide", event_at: datetime(带时区), user_id?: str, position?: int}` | `201 {event_id, replayed, received_at}`；同键不同内容 → 409 |

输入限制（从配置读）：`query` 1–256 字符；`limit` 1–100；`user_id` 1–64 字符；`item_id` 匹配 `^[A-Za-z0-9_.:-]{1,128}$`；`event_at` 必须带时区。

`request_id`：中间件生成 `req_` + 32 位十六进制；若请求头 `X-Request-Id` 符合同一模式则沿用；写入响应头与所有日志行。

## 5. 配置（环境变量，前缀 `SEARCHPILOT_`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `SEARCHPILOT_DATABASE_URL` | 无 | 未设置时使用内存实现，`/health/ready` 的 `database` 检查项不出现 |
| `SEARCHPILOT_DATA_DIR` | `./data` | |
| `SEARCHPILOT_ARTIFACT_DIR` | `./artifacts` | |
| `SEARCHPILOT_DATA_VERSION` | 无 | 指定加载哪个 `processed/<data_version>`；未设置取目录中唯一的一个 |
| `SEARCHPILOT_MAX_QUERY_CHARS` | `256` | |
| `SEARCHPILOT_MAX_LIMIT` | `100` | |
| `SEARCHPILOT_LOG_LEVEL` | `INFO` | |
| `SEARCHPILOT_DB_POOL_MIN` / `_MAX` | `1` / `4` | |

## 6. 当前状态（2026-10-08 19:55 交接快照）

| 工作流 | 状态 | 备注 |
|---|---|---|
| A 骨架与服务 | 完成，验收通过 | 101 个测试过；`filters.category` 经 `ItemStore` 后过滤；`/health/ready` 的 checks 含 search/recommend/feedback 端口是否注入 |
| B 数据与数据库 | 完成，验收通过 | 单元测试过；`tests/test_db_*` 为 integration，未设 `SEARCHPILOT_DATABASE_URL` 时明确 skip；MIND 下载 URL 已从失效的 z20 镜像改为 Hugging Face `Recommenders/MIND` 的 MINDsmall_{train,dev}.zip（官方 blob 仍是 HTTP 409；z20 主机名已 NXDOMAIN） |
| C 搜索核心与评估 | 源码完成，测试写了一半，**四条质量命令一条都没跑** | 已有：`search/*`、`eval/*`、`scripts/build_index.py`、`scripts/evaluate_search.py`、`tests/test_search_{normalize,index,bm25,fusion}.py`。缺：`tests/test_search_artifacts.py`、`test_search_service.py`、`test_eval_metrics.py`、`test_eval_search_eval.py`。ruff 已知 6 处（E501、I001、4 个文件待 format）。设计选择：filters 走 `BM25SearchService.search_with_filters()` 额外方法；`BM25Scorer(idf_variant="okapi")` 用于与 rank_bm25 逐文档数值对齐，生产默认 Lucene idf；无相关文档查询 ndcg/recall/rr 返回 None 并计入 `skipped_no_relevant` |
| D 回退推荐与反馈 | 完成，验收通过 | 24 个测试过；`resolve_data_version` 已公开 |

A/B/D 合并后的门槛：ruff 全过、mypy 48 文件无问题、133 个单元测试全过（需按第 0 节把 TEMP 重定向到 `.tmp/`）。

## 7. 返回报告格式（每个代理结束时）

1. 改动文件清单（新增 / 修改）。
2. 四条质量命令的实际输出摘要（通过数、失败数；失败要给原因）。
3. 对契约的偏离或需要技术主管决定的事项。
4. 需要新增的依赖。
5. 未完成项与建议的下一步。
