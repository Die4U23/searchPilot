# SearchPilot

一个可评估的搜索与广告排序实验平台：**BM25 / 向量 / RRF 搜索 + 广告 CTR 预估与校准 + 回退推荐 + 实验来源登记**。
上游 EvoRec 的实验结果只会以**标注来源**（`source`、`source_ref`、commit、SHA-256）的方式被引用，不会被当作本项目的实测结论。

> 当前阶段：MVP 骨架。已实现服务层（健康检查、搜索、回退推荐、反馈接口）；向量检索、LTR、CTR/校准、eCPM 对照、实验登记与 Agent 均为**规划中**。

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
# 1. 下载并解压 MIND Small（官方 blob 已关闭公共访问，脚本使用 Recommenders 团队镜像）
.venv\Scripts\python.exe scripts\download_mind.py --raw-dir data\raw\mind

# 2. 解析、时间切分、写 Parquet 与 manifest 到 data\processed\<data_version>\
.venv\Scripts\python.exe scripts\build_dataset.py --raw-dir data\raw\mind --out-dir data\processed

# 3. 建 BM25 倒排索引到 artifacts\search\bm25\（打印 model_version / doc_count / vocab_size）
.venv\Scripts\python.exe scripts\build_index.py --data-dir data --artifact-dir artifacts

# 4. 离线评估
#    搜索：按 query_type 分组的 nDCG@10 / MRR@10 / Recall@50 + 失败样例（查询集见 data\queries\，标注规范见 docs\data\annotation-guideline.md）
.venv\Scripts\python.exe scripts\evaluate_search.py --artifact-dir artifacts --queries data\queries\queries.csv --labels data\queries\labels.csv --split test --out artifacts\search\eval
#    回退推荐：Popular / ItemCF 在 dev 切分上的 Recall@20 / nDCG@10 / coverage@10，按冷启动分段
.venv\Scripts\python.exe scripts\evaluate_fallback.py --data-dir data --output-json artifacts\recommend\eval.json --output-md artifacts\recommend\eval.md

# 5.（可选）PostgreSQL：迁移并导入物品；设置 SEARCHPILOT_DATABASE_URL 后 API 的反馈写入与就绪检查走数据库
.venv\Scripts\python.exe scripts\migrate_database.py                       # 读 SEARCHPILOT_DATABASE_URL，或 --database-url
.venv\Scripts\python.exe scripts\seed_items.py data\processed\<data_version>\items.parquet

# 6. 启动 API（读取 SEARCHPILOT_DATA_DIR / SEARCHPILOT_ARTIFACT_DIR，默认 .\data 与 .\artifacts）
.venv\Scripts\python.exe scripts\run_api.py
```

`data/raw/`、`data/processed/`、`artifacts/` 不入库；`data_version` 与 `model_version` 都是输入内容的哈希，同样输入可复现同样版本号。

### Docker

```bash
docker compose up --build   # api + postgres:16，API 监听 8000
```

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
  api/              中间件与路由（health、search、recommend、feedback）
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

所有响应带 `request_id`（同时在响应头 `X-Request-Id`）；错误统一为
`{"error": {"code", "message", "retryable", "request_id"}}`。字段、限制与错误码的完整定义见
[docs/dev/build-plan.md](docs/dev/build-plan.md) 第 4 节。`/ctr/score`、`/experiments/{id}`、`/agent/analyze` 属于**规划中**，尚未实现。

## 与 EvoRec 的边界

SearchPilot 自己负责搜索、回退推荐与线上服务；EvoRec 的数据、指标与实验只读导入并带来源标注，跨来源数值不直接比较。
