# SearchPilot

SearchPilot 在 MIND Small 新闻数据上比较 BM25 关键词检索、BGE 向量检索、RRF 排名融合和线性 LTR 重排四种方法。它们共用一个 FastAPI 接口和一套查询标注，可以对同一个问题比较返回结果，再用离线指标解释差异。

同一服务也提供热门与 ItemCF 回退推荐、点击反馈、广告点击率（CTR）预估与校准。实验登记保存指标、评估协议和来源，只读 Agent 可以据此回答“哪次实验、在哪个切分上、测到了什么”。

## 快速开始

第一次运行时，在仓库根目录用 Python 3.12 下载数据、构建索引，再启动 API。这个 BM25 示例使用本地文件，首次安装和下载需要联网。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .

python scripts/download_mind.py
python scripts/build_dataset.py
python scripts/build_index.py
python scripts/run_api.py
```

Linux / macOS 使用 `python3.12 -m venv .venv` 和 `source .venv/bin/activate`，后续 `python` 命令相同。已有 `data/processed/<data_version>/` 和 `artifacts/search/bm25/` 时，可以直接启动 API。

服务启动后，在另一个 PowerShell 窗口搜索 `microsoft`。

```powershell
$api = 'http://127.0.0.1:8000'
$body = @{ query = 'microsoft'; mode = 'bm25'; limit = 1 } | ConvertTo-Json
$result = Invoke-RestMethod "$api/search" -Method Post -ContentType 'application/json' -Body $body
$result.results
```

在数据版本 `d3a904f41240` 上，第一条结果的 `item_id` 是 `N58995`。浏览器打开 `http://127.0.0.1:8000/docs` 可以试用其他接口。只建 BM25 时，搜索可用，`/health/ready` 会返回 503，响应里的 `checks` 会列出尚未就绪的组件。

### 启用向量、融合与重排

在已激活的虚拟环境中构建向量索引和 LTR 模型，再重启 API 以加载新产物。请求中的 `mode` 可换为 `vector`、`hybrid`（RRF）或 `ltr`。

```powershell
python -m pip install -e ".[vector]"
python scripts/build_vector_index.py
python scripts/train_ltr.py
python scripts/compare_search_modes.py --split test
```

最后一条命令用同一批查询比较四种模式，写出 `artifacts/search/eval/search_mode_compare_test.md`。只有 BM25 时可运行 `python scripts/evaluate_search.py --split test`。

### 使用 Docker 和 PostgreSQL

准备好本地 `data/` 和 `artifacts/` 后，运行 `docker compose up --build`，API 监听端口 8000。Compose 启动 PostgreSQL 16，把两个目录只读挂载到 API 容器。入口先执行迁移，有数据快照时为空表导入物品、查询目录与已登记的实验。曝光数据需要单独导入。[Compose 配置](docker-compose.yml) · [数据导入脚本](scripts/seed_impressions.py)

镜像在构建时下载 `BAAI/bge-small-en-v1.5`，默认在服务启动时预热已有向量索引的编码器。容器健康检查使用 `/health/live`，接入依赖全部组件的流量时使用 `/health/ready`。[预热对照](docs/data/vector-startup-2026-10-22.md)

## API

| 接口 | 用途 |
| --- | --- |
| `POST /search` | 提交 `query`，可指定 `mode`、`limit` 和 `filters.category`。响应包含排名、分数和模型版本。 |
| `POST /recommend` | 按 `user_id` 请求回退推荐，响应标明推荐通道和冷启动状态。 |
| `POST /feedback` | 用 `idempotency_key` 幂等写入反馈，同一键对应不同事件时返回 409。 |
| `POST /ctr/score` | 返回候选物品的原始与校准点击率。传入 `bid` 时也返回 eCPM。 |
| `GET /experiments/{experiment_id}` | 查看实验配置、指标、来源 `source` 和引用 `source_ref`。 |
| `POST /agent/analyze` | 提交 `{"question":"..."}`，通过六个只读工具查询已登记实验。 |
| `GET /health/live`、`GET /health/ready` | 分别检查进程存活和组件就绪状态。 |

每个响应都带 `X-Request-Id` 请求头。设置 `SEARCHPILOT_API_TOKEN` 后，表中除存活探针外的接口须携带 `Authorization: Bearer <token>`。写反馈还须提供与 `user_id` 相同的 `X-Actor`。[请求与响应模型](src/searchpilot/contracts.py) · [配置示例](.env.example)

无数据库时，反馈保存在内存中，重启即清空。要在此模式下试用实验查询和 Agent，先运行 `python scripts/register_measured.py` 生成本地登记文件，再启动 API，即可查询 `GET /experiments/sp-bm25`。Agent 默认使用确定性规则。要让模型根据工具结果决定下一步，可在服务端显式启用有界多步 LLM 模式。[Agent 配置与调用示例](docs/dev/autonomous-agent.md)

## 数据与评测

MIND Small 数据快照 `d3a904f41240` 含 65,238 条新闻、8,584,442 条曝光和 50,000 份用户历史。仓库提供查询、标注和构建脚本，原始数据需另行下载，处理快照与索引由脚本在本地生成。

### 搜索

100 条查询按 70 / 15 / 15 切分为训练、验证和测试集。下表使用正式标注 `labels.csv`。15 条测试查询中，1 条没有已知相关文档，三个指标均在其余 14 条上计算。[四模式实验报告](docs/data/ltr-report.md)

| 搜索模式 | nDCG@10 | MRR@10 | Recall@50 |
| --- | ---: | ---: | ---: |
| BM25 · `bm25-4f46b3d8` | 0.6396 | 0.7143 | 0.7500 |
| 向量 · `vector-b92663fd` | **0.7769** | **0.8095** | 0.9452 |
| RRF · `hybrid-51d4dcfb` | 0.7214 | 0.7679 | **1.0000** |
| LTR · `ltr-4574f8f4` | 0.4596 | 0.4595 | 0.7175 |

向量检索在这组查询上的 nDCG@10 最高，RRF 在前 50 条找回了全部已标注相关文档。LTR 低于两者。诊断发现，模型给 BM25 名次数值学到了正权重，因而可能把 BM25 排名靠后的候选提前。[LTR 权重消融](docs/data/ltr-rank-ablation.md)

标注覆盖会改变读数。原始候选主要取自 BM25 前十，未标注文档在正式评估中计 0。两个独立 Grok 会话（`grok-a`、`grok-b`）对 206 对候选双标，191 对完全一致。最终 `labels.csv` 有 997 行署名 `grok-a`、15 行署名 `adjudicated`。另有 938 对候选补标。合并补标后，同一批测试查询的向量 nDCG@10 为 0.8796，BM25 为 0.5571。[标注报告](docs/data/annotation-report.md) · [补标对照](docs/data/pool-label-eval.md) · [评估策略](docs/data/evaluation-policy.md)

### CTR、Agent 与容器验证

| 实验 | 观测与适用范围 |
| --- | --- |
| CTR 校准 | 旧训练流程的抽样测试中，MLP 温度缩放将 ECE 从 0.0374 降至 0.0063，AUC 保持 0.4836。当前训练脚本仅使用曝光时刻之前的正反馈构建历史特征，新口径尚无全量训练指标。[历史指标](docs/data/ctr-report.md) · [时序特征定义](docs/dev/ctr-point-in-time-features.md) |
| 合成出价排序 | 20,831 次曝光上，原始与校准点击率产生相同首位广告的比例为 0.9180。该实验使用合成出价和日志点击衡量排序变化。[eCPM 对照](docs/data/ecpm-report.md) |
| 只读 Agent | 历史两阶段评测为 46/50。后续多步实测中，8 个合成任务全部正常结束，原结构化评分为 6/8。修正工具调用顺序的评分规则后，同一批轨迹复算为 8/8，新增模型调用为 0。自由文本语义未评分。[两阶段报告](docs/data/agent-llm-live-2026-10-14.md) · [多步验证](docs/data/agent-multistep-2026-10-27.md) |
| 容器持续负载 | 1,300 条查询、四种搜索模式、并发 32。三小时内 144,053 次正式请求均成功，5,200 个查询与模式组合各重复 27 至 28 次，每组返回的结果与排名均一致。候选配置 `MALLOC_ARENA_MAX=1` 在第二至第三小时的合并窗口内，RSS 和匿名内存各增长 5.8125 MiB，超过 2 MiB 门槛。该候选未通过长期内存稳定性验收，Compose 默认值为 2。[一小时对照](docs/data/container-expanded-soak-2026-10-25.md) · [三小时报告](docs/data/container-memory-three-hour-2026-10-27.md) |

### EvoRec 实验来源

只读 Agent 固定任务中的 `evorec-itemcf` 是本地测试夹具。另行导入的五份 EvoRec 实验结果使用视频游戏推荐协议，保留上游指标、`source=evorec`、提交和文件摘要，单独解读。[上游固定版本](https://github.com/Die4U23/EvoRec/tree/5ce1d96b80ddaf3c205a5ef8a6a5e39df6ab0ff5) · [导入脚本](scripts/import_evorec.py)

<details>
<summary>五份上游 results.json 的路径与 SHA-256</summary>

| 路径 | SHA-256 |
|---|---|
| `docs/experiments/r03-content/results.json` | `8b975411c905429cc1e54d2ad81c0619c109d75db17f2333b1a43d734c9de3ac` |
| `docs/experiments/r04-gating/results.json` | `6c6fe2132a1cd9aac943a1a939b0aaff9859dcc3da0fd1196340cb0255039e84` |
| `docs/experiments/r05-cold-replication/results.json` | `8fe0efb6fe33fa1adc4e0493f85c72b65b4167ea705304414adcea38008debea` |
| `docs/experiments/r05-ranker/results.json` | `185286a00c86de52e9853be4219148bf4b9e0ab0e141a1b64c52a44077d889c8` |
| `docs/experiments/r06-multi-interest/results.json` | `a56097096c38317b7b5bdf9bff96a0cccaa386b6114b1e53211d2ed886f04774` |

</details>

## 开发与复现

| 任务 | 入口 |
| --- | --- |
| 检索实现与服务组装 | [检索模块](src/searchpilot/search/) · [应用入口](src/searchpilot/bootstrap.py) |
| 评估热门与 ItemCF 推荐 | [回退推荐评估脚本](scripts/evaluate_fallback.py) · [ItemCF 启动验证](docs/data/itemcf-startup-2026-10-15.md) |
| 训练与加载 CTR 模型 | [训练脚本](scripts/train_ctr.py) · [推理产物格式](docs/dev/ctr-artifact-contract.md) |
| 复核压测与历史验收 | [并发延迟](docs/data/bench-concurrent.md) · [分阶段计时](docs/data/bench-stages.md) · [项目交接](docs/dev/handover.md) |

在虚拟环境中运行与 CI 对应的本地检查。下面的临时目录设置用于 Windows。最后一条集成测试需要 `SEARCHPILOT_DATABASE_URL` 指向可用的 PostgreSQL 测试库。

```powershell
python -m pip install -e ".[dev,research,vector]"
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TMP = (Resolve-Path .tmp).Path
$env:TEMP = $env:TMP
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
python -m pytest -m integration
```

