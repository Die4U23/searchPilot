# SearchPilot

一个可评估的搜索与广告排序实验平台：**BM25 / 向量 / RRF 搜索 + 广告 CTR 预估与校准 + 回退推荐 + 实验来源登记**。
上游 EvoRec 的实验结果只会以**标注来源**（`source`、`source_ref`、commit、SHA-256）的方式被引用，不会被当作本项目的实测结论。

> 当前阶段：搜索（BM25 / 向量 / RRF / LTR）、回退推荐、反馈、CTR 校准、合成出价 eCPM、实验登记和只读 Agent 都已接到 API。查询标注可以由外部工具完成；外部语言模型已完成固定夹具评测，结构化检查通过 46/50；线上 Agent 仍为确定性工具调用。

2026-10-13 接手结果见 [本轮验收](docs/dev/acceptance-2026-10-13.md)：补齐并发压测、train/val 大分歧裁定与外部模型评测入口，并修复向量编码器并发初始化。正式标注入口与模型栈保持冻结，具体口径见 [评估策略](docs/data/evaluation-policy.md)。

2026-10-14 的向量候选扫描优化、真实 HTTP A–B–B–A 对照及资源采样见 [性能报告](docs/data/performance-2026-10-14.md)。外部模型已完成 50 题、100 次真实请求，结构化检查通过 46/50，详见 [外部 LLM 实测](docs/data/agent-llm-live-2026-10-14.md)。

2026-10-15 已优化 ItemCF 拟合内存：5,000 用户对照保持结果一致，无数据库配置的默认应用在 19.8 秒内就绪，8 个核心接口请求通过；详见 [启动验证](docs/data/itemcf-startup-2026-10-15.md)。首次向量请求仍超过 10 秒，未宣称 SLA 达标。

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

查询集 100 条，种子 `20261009`，按查询切成 70 / 15 / 15，六类都有样本。双标 20 条、206 对：完全一致 0.9272，相邻一致 1.0，二次加权 kappa 0.9702。当前 `annotator` 是 `grok-a` 和 `grok-b`。详见 `docs/data/annotation-report.md`。

Agent 固定任务中的 `evorec-itemcf` 是本地夹具，用来检查跨来源比较会被拒绝。另行导入的五份真实 EvoRec 结果及其固定 commit、SHA-256 见文末来源表，不能与夹具混用。

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

需要 Docker Desktop（Linux 容器）。`docker compose up --build` 在容器里的 PostgreSQL 上先跑迁移。物品表、查询目录和已测实验如果是空的，会从挂载的 `data/` 与 `artifacts/` 导入一次。曝光表不导入。然后启动 API，端口是 `8000`。`data/` 和 `artifacts/` 以只读方式挂进去，不打进镜像。镜像安装 `vector` 额外依赖，并在构建时下载 `BAAI/bge-small-en-v1.5`，所以向量和 LTR 检索可以在容器里运行。第一次构建需要网络，镜像比默认安装大。

配置来自环境变量，应用项使用 `SEARCHPILOT_` 前缀，容器与 glibc 项另行标注，见 [`.env.example`](.env.example)。

Docker/Compose 默认设置 `SEARCHPILOT_VECTOR_WARMUP=true`，在监听请求前加载向量模型并完成一次编码，避免把初始化延迟留给首个搜索请求。这会增加启动等待时间；预热失败时 BM25 仍可调用，`/health/ready` 返回 503，向量、混合与 LTR 搜索不可用，修复后需重启。容器健康检查仍检查 `/health/live`；流量准入应检查 `/health/ready`。直接运行 Python 默认不预热，可通过同名环境变量开启（Settings 不自动读取 `.env`）。

Linux/glibc 容器默认使用进程级 `SEARCHPILOT_DISABLE_THP=true` 和 `MALLOC_ARENA_MAX=2`，不执行定时 GC/trim。显式设置 `MALLOC_ARENA_MAX=0` 并重新创建 API 容器可恢复该 glibc 的自动 arena 上限策略；删除或留空会采用默认值 2。

最新[容器内存对照与一小时验证](docs/data/container-memory-arena-2026-10-25.md)定位了 THP 与 glibc 空闲内存保留的贡献。持续并发 4 的顺序 A→B 对照各完成 16,000 请求，400→16k RSS 增长分别为 45.953/3.609 MiB，结果指纹一致；背景负载差异保留，不据此宣称吞吐提升。最终正常入口一小时 **53,650/53,650** 请求成功，末 20 分钟 RSS 增长 **0.6875 MiB**、斜率 **0.0349 MiB/分钟**，通过预设内存门槛；17 项业务与重启检查通过。健康采样实际 361 轮全部成功，名义 360 轮的边界差异见报告。结论限于固定查询集、单进程、并发 4 和一小时观察，不代表无限运行无泄漏。

此前[一小时验证](docs/data/container-hour-2026-10-23.md)修复了共享 Snowball 词干器的并发状态污染，[十分钟验证](docs/data/container-soak-2026-10-22.md)修复了启动导入的事务原子性。

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

`scripts/evaluate_agent.py` 跑 50 条固定任务。当前分析器是确定性的，不调用外部语言模型。旧评测的 50/50 只检查工具名属于白名单，27/27 只检查引用非空，不能据此声称参数正确或引用受到证据支持；跨来源陷阱 8/8 检查固定拒绝短语。详见 `docs/data/agent-report.md`。独立的外部模型入口与协议见 [评测说明](docs/data/agent-llm-report.md)；[真实外部实测](docs/data/agent-llm-live-2026-10-14.md)为 46/50 结构化通过、引用 30/32、跨来源陷阱 8/8，保留 4 个失败，不等同于自由文本正确率或真实业务验证。

## 已测结果（搜索 2026-10-09，CTR 2026-10-10）

同一套 15 条 test 查询，数据版本 `d3a904f41240`。下表读的是 `labels.csv`，`annotator` 是 `grok-a` 和 `grok-b`。已标注文档里 97.6% 落在 BM25 前 10，只有 28.1% 落在向量前 10。三路前 10 的并集里还有 938 条不在这份文件中，分数在 `labels_tool.csv`，见 `docs/data/label-pool.md`。把未标注文档移出名单、不另打分后，test 上向量 nDCG@10 从 0.7769 升到 0.9432，LTR 从 0.4596 升到 0.8338，BM25 仍是 0.6396。见 `docs/data/judged-only.md`。把 `labels_tool.csv` 并进来之后，test 与 val 的 nDCG@10 都是向量高于 RRF、高于 BM25、高于 LTR。val 只读 `labels.csv` 时 BM25 最高（0.9056），并集后降到 0.7465，向量升到 0.9161。见 `docs/data/pool-label-eval.md`。

| 方案 | nDCG@10 | MRR@10 | Recall@50 |
|---|---:|---:|---:|
| BM25 `bm25-4f46b3d8` | 0.6396 | 0.7143 | 0.7500 |
| 向量 `vector-b92663fd` | 0.7769 | 0.8095 | 0.9452 |
| RRF `hybrid-51d4dcfb` | 0.7214 | 0.7679 | 1.0000 |
| LTR `ltr-4574f8f4` | 0.4596 | 0.4595 | 0.7175 |

LTR 低于 RRF，这是保留的负结果。分数归一化融合（先把 BM25 和向量分各自缩放到 0–1 再平均）在同一 test 集上 nDCG@10 为 0.7710、MRR@10 为 0.7857、Recall@50 为 1.0000，见 `docs/data/normalized-fusion.md`。它不是默认融合。CTR 抽样 test 上 LR AUC 0.4535，MLP 温度缩放后 ECE 0.0063，AUC 仍为 0.4836。同一 test 上 Platt ECE 0.0019、等渗回归 ECE 0.0014，都不是线上默认。合成出价 eCPM 的 Top-1 一致率 0.9180，不是真实广告收入。

## 已知限制

- 未设置 `SEARCHPILOT_DATABASE_URL` 时，实验登记读 `artifacts/experiments/registry.json`，反馈写在内存里。
- 当前 `labels.csv` 的 `annotator` 是 `grok-a` 和 `grok-b`。换工具时另存 CSV，不要覆盖这两份原始文件。
- 已保存的 LTR `ltr-4574f8f4` 在补标并集上仍低于向量和 RRF。它的 top 10 里大约六成文档两份标注都没有，nDCG 把这些位置记为 0。拿掉未标注文档后，test nDCG@10 从 0.5266 升到 0.7076，高于 BM25 的 0.5571，仍低于 RRF 的 0.7236。val 上 LTR 仍是最后（0.6959），低于 BM25 的类型是 exact_entity、multi_condition、misspelling_or_abbrev。见 `docs/data/pool-label-eval.md` 和 `docs/data/ltr-query-gaps.md`。
- 不重新训练，只把已保存的 `bm25_rank` 权重置 0 后，val 上未标注当 0 的 nDCG@10 从 0.5055 升到 0.7762，高于 BM25 的 0.7465。只替换缺席名次 101 做不到这一点。这不是新的线上模型。见 `docs/data/ltr-rank-ablation.md`。
- 外部语言模型已完成固定夹具验证（46/50 结构化通过）；自由文本语义未评分，EvoRec 为本地夹具，不能据此声称生产可用。线上 Agent 仍是确定性工具调用。
- 分阶段计时见 `docs/data/bench-stages.md`（Windows 11，Python 3.12.6，无 `--reload`）。向量冷启动 15.3 s（含加载模型）；预热后 p50：BM25 1.2 ms，向量 87.7 ms，RRF 91.1 ms，LTR 127.4 ms，CTR 0.04 ms。更早的 BM25 100 次吞吐约 315 req/s，见 `docs/data/bench.md`。
- 2026-10-13 独立本地 API 并发 4、每模式 120 请求，正式请求共 480/480 成功；预热 15/16 成功，首次向量预热在 10 秒超时。原始记录和 p95/p99 见 `docs/data/bench-concurrent.md`。这是初始化锁修复前的单轮测量，不代表最终版本冷启动、资源或稳定性验收。
- `docker compose up` 会先迁移，并为空表导入物品、查询目录与已测实验，再启动 API；曝光表不自动导入。
- Agent 固定任务里的 `evorec-itemcf` 仍是本地夹具。真实结果在下面的来源表里，协议不同，不能和本项目的 nDCG 放在一张提升表里。

## 与 EvoRec 的边界

SearchPilot 自己负责搜索、回退推荐与线上服务。EvoRec 公开仓库是 [Die4U23/EvoRec](https://github.com/Die4U23/EvoRec)，MIT 许可。下面五份 `results.json` 固定在 commit `5ce1d96b80ddaf3c205a5ef8a6a5e39df6ab0ff5`，指标名和数值原样导入，`source=evorec`。

| 路径 | SHA-256 |
|---|---|
| `docs/experiments/r03-content/results.json` | `8b975411c905429cc1e54d2ad81c0619c109d75db17f2333b1a43d734c9de3ac` |
| `docs/experiments/r04-gating/results.json` | `6c6fe2132a1cd9aac943a1a939b0aaff9859dcc3da0fd1196340cb0255039e84` |
| `docs/experiments/r05-cold-replication/results.json` | `8fe0efb6fe33fa1adc4e0493f85c72b65b4167ea705304414adcea38008debea` |
| `docs/experiments/r05-ranker/results.json` | `185286a00c86de52e9853be4219148bf4b9e0ab0e141a1b64c52a44077d889c8` |
| `docs/experiments/r06-multi-interest/results.json` | `a56097096c38317b7b5bdf9bff96a0cccaa386b6114b1e53211d2ed886f04774` |

这些是 EvoRec 自己的视频游戏推荐协议。例如 r05-ranker 的 RRF 在 `all_positive_events` 上 nDCG@10 为 0.006536741003757799（来源：EvoRec `r05-ranker`，commit `5ce1d96b80ddaf3c205a5ef8a6a5e39df6ab0ff5`）。它不能与本项目 MIND 搜索的 nDCG 比较。

设置 `SEARCHPILOT_API_TOKEN` 后，除 `GET /health/live` 外请求都要带 `Authorization: Bearer <token>`。`POST /feedback` 还要带 `X-Actor`，并且与 `user_id` 相同。未设置令牌时行为与以前一样。

