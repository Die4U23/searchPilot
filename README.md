# SearchPilot

一个可评估的搜索与广告排序实验平台：**BM25 / 向量 / RRF 搜索 + 广告 CTR 预估与校准 + 回退推荐 + 实验来源登记**。
上游 EvoRec 的实验结果只会以**标注来源**（`source`、`source_ref`、commit、SHA-256）的方式被引用，不会被当作本项目的实测结论。

> 当前阶段：搜索（BM25 / 向量 / RRF / LTR）、回退推荐、反馈、CTR 校准、合成出价 eCPM、实验登记和只读 Agent 都已接到 API。查询标注不是人工双标；外部语言模型上的 Agent 评测未跑。

## 架构

```text
MIND Small                         本地 EvoRec 夹具（不是上游仓库的一次真实运行）
    |                                          |
    v                                          v
清洗 / 时间切分 / 查询标注 ---------> PostgreSQL（物品、反馈、实验、指标）
    |
    +--> BM25 倒排
    +--> bge-small 向量（精确余弦）
    +--> pairwise LTR
    +--> CTR（LR / MLP）+ 验证集上的温度缩放
    |
    v
FastAPI：/search  /recommend  /feedback  /ctr/score
         /experiments/{id}  /agent/analyze  /health/*
    |
只读 Agent：六个窄工具，引用带 source
```

没有数据库时，反馈在内存里，实验登记读 `artifacts/experiments/registry.json`。

## 数据

MIND Small，研究用途，原始文件不进 Git。下载脚本核对 SHA-256。`data_version` `d3a904f41240`：物品 65238，曝光 8584442，用户历史 50000。train 为 2019-11-09 至 2019-11-14，dev 为 2019-11-15。

查询集 100 条，种子 `20261009`，按查询切成 70 / 15 / 15，六类都有样本。双标 20 条、206 对：完全一致 0.9272，相邻一致 1.0，二次加权 kappa 0.9702。标注员是两个模型会话，不是人工。详见 `docs/data/annotation-report.md`。

仓库里没有真实的 EvoRec 结果文件。`evorec-itemcf` 是本地夹具，用来检查跨来源比较会被拒绝。

## 快速开始

需要 Python 3.12。

```powershell
# 1. 创建虚拟环境并安装（含开发与研究依赖）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,research]"

# 2. 启动 API（默认 127.0.0.1:8000）
.venv\Scripts\python.exe scripts\run_api.py --host 127.0.0.1 --port 8000

# 3. 质量命令
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest                 # 单元测试（默认不含 integration）
.venv\Scripts\python.exe -m pytest -m integration  # 需要 SEARCHPILOT_DATABASE_URL 指向 PostgreSQL
```

Linux / macOS 把 `.venv\Scripts\python.exe` 换成 `.venv/bin/python` 即可。

索引、推荐模型等产物缺失时 API 仍可启动：对应接口返回 `503 NOT_READY`，`/health/ready` 返回 503，不会静默返回空结果。

### 数据 → 索引 → 评估 → 服务

```powershell
# 1. 下载并解压 MIND Small（官方 blob 已关闭公共访问，脚本使用 Hugging Face 上的 Recommenders/MIND）
.venv\Scripts\python.exe scripts\download_mind.py --raw-dir data\raw\mind

# 2. 解析、时间切分、写 Parquet 与 manifest 到 data\processed\<data_version>\
.venv\Scripts\python.exe scripts\build_dataset.py --raw-dir data\raw\mind --out-dir data\processed

# 3. 建 BM25 倒排索引到 artifacts\search\bm25\（打印 model_version / doc_count / vocab_size）
.venv\Scripts\python.exe scripts\build_index.py --data-dir data --artifact-dir artifacts

# 3b.（可选，V1）建 bge-small 向量索引。需要先 pip install -e ".[vector]"
.venv\Scripts\python.exe scripts\build_vector_index.py --data-dir data --artifact-dir artifacts

# 3c.（可选，V1）训练线性 pairwise LTR（只读 split=train；需要向量索引）
.venv\Scripts\python.exe scripts\train_ltr.py --data-dir data --artifact-dir artifacts

# 4. 离线评估
#    搜索：按 query_type 分组的 nDCG@10 / MRR@10 / Recall@50 + 失败样例（查询集见 data\queries\，标注规范见 docs\data\annotation-guideline.md）
.venv\Scripts\python.exe scripts\evaluate_search.py --artifact-dir artifacts --queries data\queries\queries.csv --labels data\queries\labels.csv --split test --out artifacts\search\eval
#    多模式对比（bm25 / vector / hybrid / ltr），输出 search_mode_compare_<split>.md
.venv\Scripts\python.exe scripts\compare_search_modes.py --artifact-dir artifacts --queries data\queries\queries.csv --labels data\queries\labels.csv --split test --out artifacts\search\eval
#    回退推荐：Popular / ItemCF 在 dev 切分上的 Recall@20 / nDCG@10 / coverage@10，按冷启动分段
.venv\Scripts\python.exe scripts\evaluate_fallback.py --data-dir data --output-json artifacts\recommend\eval.json --output-md artifacts\recommend\eval.md

# 5.（可选）PostgreSQL：迁移、导入物品，并把已测实验写入数据库
#    设置 SEARCHPILOT_DATABASE_URL 后，反馈、就绪检查和实验查询走数据库。
#    数据库模式下 API 不读 registry.json，已测实验要用 --database-url 写入。
.venv\Scripts\python.exe scripts\migrate_database.py                       # 读 SEARCHPILOT_DATABASE_URL，或 --database-url
.venv\Scripts\python.exe scripts\seed_items.py data\processed\<data_version>\items.parquet
.venv\Scripts\python.exe scripts\seed_catalog.py
.venv\Scripts\python.exe scripts\seed_impressions.py --impressions data\processed\<data_version>\impressions.parquet --history data\processed\<data_version>\user_history.parquet
.venv\Scripts\python.exe scripts\register_measured.py --database-url $env:SEARCHPILOT_DATABASE_URL

# 6. 启动 API（读取 SEARCHPILOT_DATA_DIR / SEARCHPILOT_ARTIFACT_DIR，默认 .\data 与 .\artifacts）
.venv\Scripts\python.exe scripts\run_api.py
```

`data/raw/`、`data/processed/`、`artifacts/` 不入库；`data_version` 与 `model_version` 都是输入内容的哈希，同样输入可复现同样版本号。

### Docker

`docker compose up --build` 会先跑数据库迁移，再启动 API。物品导入仍需在容器外执行 `scripts/seed_items.py`。

配置全部来自环境变量，见 [`.env.example`](.env.example)（前缀 `SEARCHPILOT_`）。

## 目录结构

```text
src/searchpilot/
  ports.py          模块间 Protocol 与数据类（契约锚点，只读）
  config.py         Settings（SEARCHPILOT_ 前缀环境变量）
  observability.py  JSON 日志与 request_id
  errors.py         统一错误信封与异常处理器
  contracts.py      请求/响应 Pydantic 模型
  bootstrap.py      create_app / build_default_app
  api/              中间件与路由（health、search、recommend、feedback、ctr、experiments、agent）
  search/           BM25、RRF、搜索服务与评估
  recommend/        热门 / ItemCF 回退推荐
  feedback/         反馈幂等写入（内存实现）
  db/  data/        PostgreSQL 仓储、迁移；MIND 数据解析与快照
scripts/            run_api.py 及数据 / 索引 / 评估脚本
tests/              单元测试（不联网、不连库、不加载模型）
docs/dev/           并行开发契约（build-plan.md）
```

## API 一览

| 接口 | 说明 |
|---|---|
| `GET /health/live` | 进程存活 |
| `GET /health/ready` | 依赖就绪；任一未就绪返回 503 与 `checks` |
| `POST /search` | `query`、`limit`、`mode`（`bm25` 默认）、`filters.category` |
| `POST /recommend` | `user_id`、`limit`；仅回退通道，返回 `cold_start` |
| `POST /feedback` | 带 `idempotency_key` 的幂等写入，成功 201，冲突 409 |
| `POST /ctr/score` | CTR 打分；模型未加载时 503，不返回空列表 |
| `GET /experiments/{experiment_id}` | 实验配置、指标、`source` 与 `source_ref` |
| `POST /agent/analyze` | 只读实验分析；只用六个窄工具 |

所有响应带 `request_id`（同时在响应头 `X-Request-Id`）；错误统一为
`{"error": {"code", "message", "retryable", "request_id"}}`。字段、限制与错误码的完整定义见
[docs/dev/build-plan.md](docs/dev/build-plan.md) 第 4 节。

空查询：

```http
POST /search
{"query": ""}
```

```json
{"error": {"code": "INVALID_INPUT", "message": "body.query: String should have at least 1 character", "retryable": false, "request_id": "req_..."}}
```

有结果的搜索（本机 BM25，`microsoft`，`limit=1`）：

```json
{"request_id": "req_...", "normalized_query": "microsoft", "model_version": "bm25-4f46b3d8", "mode": "bm25", "results": [{"item_id": "N58995", "score": 24.661796673688574, "rank": 1, "channel": "bm25"}]}
```

模型未加载时对应接口返回 503 `NOT_READY`。

## Agent

`scripts/evaluate_agent.py` 跑 50 条固定任务。当前分析器是确定性的，不调用外部语言模型。工具都在白名单内 50/50，需要引用的任务 27/27 能对上工具返回，跨来源陷阱 8/8 拒绝比较优劣。外部语言模型评测是 UNRUN。详见 `docs/data/agent-report.md`。

## 已测结果（搜索 2026-10-09，CTR 2026-10-10）

同一套 15 条 test 查询，数据版本 `d3a904f41240`。标注由两个模型会话完成，不是人工双标，不能当作 PRD 要求的人工验收结论。

| 方案 | nDCG@10 | MRR@10 | Recall@50 |
|---|---:|---:|---:|
| BM25 `bm25-4f46b3d8` | 0.6396 | 0.7143 | 0.7500 |
| 向量 `vector-b92663fd` | 0.7769 | 0.8095 | 0.9452 |
| RRF `hybrid-51d4dcfb` | 0.7214 | 0.7679 | 1.0000 |
| LTR `ltr-4574f8f4` | 0.4596 | 0.4595 | 0.7175 |

LTR 低于 RRF，这是保留的负结果。分数归一化融合（先把 BM25 和向量分各自缩放到 0–1 再平均）在同一 test 集上 nDCG@10 为 0.7710、MRR@10 为 0.7857、Recall@50 为 1.0000，见 `docs/data/normalized-fusion.md`。它不是默认融合。CTR 抽样 test 上 LR AUC 0.4535，MLP 温度缩放后 ECE 0.0063，AUC 仍为 0.4836。同一 test 上 Platt ECE 0.0019、等渗回归 ECE 0.0014，都不是线上默认。合成出价 eCPM 的 Top-1 一致率 0.9180，不是真实广告收入。

## 已知限制

- 未设置 `SEARCHPILOT_DATABASE_URL` 时，实验登记读 `artifacts/experiments/registry.json`，反馈写在内存里。
- 查询标注不是人工双标。
- 外部语言模型上的 Agent 评测是 UNRUN。当前 Agent 是确定性工具调用。
- 分阶段计时见 `docs/data/bench-stages.md`（Windows 11，Python 3.12.6，无 `--reload`）。向量冷启动 15.3 s（含加载模型）；预热后 p50：BM25 1.2 ms，向量 87.7 ms，RRF 91.1 ms，LTR 127.4 ms，CTR 0.04 ms。更早的 BM25 100 次吞吐约 315 req/s，见 `docs/data/bench.md`。
- `docker compose up` 会先迁移再启动 API，不会自动导入物品。
- 没有真实的 EvoRec 结果文件时，导入器用本地夹具。跨来源比较返回不可比。

## 与 EvoRec 的边界

SearchPilot 自己负责搜索、回退推荐与线上服务；EvoRec 的数据、指标与实验只读导入并带来源标注，跨来源数值不直接比较。
