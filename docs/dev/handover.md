# SearchPilot 交接

2026-10-15 最新工作见 [ItemCF 与默认应用启动验证](../data/itemcf-startup-2026-10-15.md)：逐物品统计替代完整共现图，真实 5,000 用户对照结果相同；完整 50,000 用户拟合 14.5 秒，无数据库默认应用 ready 19.8 秒，8 个业务请求通过。冷向量仍超过 10 秒，数据库整站启动与稳定性尚未覆盖。

2026-10-14 外部 LLM 已执行，见 [真实外部评测与复核](../data/agent-llm-live-2026-10-14.md)：50 题、100 次请求、46/50 结构化通过。保留失败与夹具限制；私有 `.env` 不入库。

同日较早工作见 [性能与外部验证验收](acceptance-2026-10-14.md)，包含搜索隔离
HTTP 对照、默认应用启动阻塞与外部模型配置缺失边界。以下内容均为此前快照。

2026-10-13 接手更新见 [本轮验收](acceptance-2026-10-13.md)。以下保留为
2026-10-12 的历史快照；并发测试、11 对裁定及 Agent 评测入口的最新状态以新报告为准。
Git 交付权限随后已由所有者更新，以 [Git 交付约定](git-workflow.md) 为准；下文旧禁提交条款不再适用。

2026-10-12。读者是下一位继续这个仓库的人。`docs/dev/build-plan.md` 第 6 节是 2026-10-08 的旧快照，以本文为准。

模型栈已冻结：不加大 LTR，不加大 CTR。下一步是标注口径和还没跑的评测。

## 硬性约束

- 不要 `git add`、`git commit`、`git push`、改分支或丢弃工作区。改动留给项目所有者审核。Git 可执行文件是 `C:\Program Files\Git\cmd\git.exe`。
- 不要改 `pyproject.toml` 和 `src/searchpilot/ports.py`，除非先说明原因。向量依赖已经是可选额外项 `vector`，Docker 镜像安装的是它，不要把它挪进默认依赖。
- 不要改写 `db/migrations/0004_catalog.sql` 和 `db/migrations/0005_users_impressions.sql`。库里的迁移哈希对的是这两份文件的 CRLF 字节。
- 不要覆盖 `data/queries/labels.csv`、`labels_grok_a.csv`、`labels_grok_b.csv`。评估脚本默认读 `labels.csv`。
- 不要把数据库密码写进提交文件。`.env` 已忽略。
- 本机 pytest 前把临时目录指到仓库：`$env:TMP="E:\searchPilot\.tmp"; $env:TEMP=$env:TMP`。Python 是 `E:\searchPilot\.venv\Scripts\python.exe`。
- Shell 在这台机器上需要脱离沙箱才能跑。Docker 命令要先把 `C:\Program Files\Docker\Docker\resources\bin` 加进 `PATH`。

## 现在做到哪

PRD 第 13.2 节的 17 条验收里，16 条已有测试或报告。压测只做了单进程顺序请求，记下了机器 `Windows-11-10.0.26200-SP0`、Python 3.12.6、p95/p99。并发成功率没有测。PRD 里的勾选框仍是空的，以测试和 `docs/data/` 为准。

数据版本 `d3a904f41240`。物品 65238。查询 100，切分 70/15/15，种子 `20261009`。

宿主 PostgreSQL 18 与 compose 的 Postgres 16 是两套库。compose 卷在 2026-10-12 第一次启动时导入了物品 65238、查询 100、标注 1012、实验 6。曝光表没有导入。CTR 打分用挂载的 `artifacts/`，不依赖曝光表。

## 搜索结论

首页表仍是 `labels.csv`（`grok-a` / `grok-b`）的 test，nDCG@10：BM25 0.6396，向量 0.7769，RRF 0.7214，LTR 0.4596。这份标注里 97.6% 落在 BM25 前 10。

`labels.csv` 与 `labels_tool.csv` 没有重叠的 `(query_id, item_id)`。并上补标、未标注当 0 时，test 与 val 都是向量高于 RRF、高于 BM25、高于 LTR。val 只读 `labels.csv` 时 BM25 最高（0.9056），并集后降到 0.7465，向量升到 0.9161。数字在 `docs/data/pool-label-eval.md`。

只评已标注（未标注文档移出名单，明确 0 分留在原位）后，BM25、向量、RRF 的 nDCG 不变。LTR 在 test 升到 0.7076，高于 BM25 的 0.5571，仍低于 RRF 的 0.7236。val 上 LTR 仍是 0.6959，低于 BM25 的 0.7465。低于 BM25 的类型是 `exact_entity`、`multi_condition`、`misspelling_or_abbrev`。逐条在 `docs/data/ltr-query-gaps.md`，共 12 条。

已保存模型 `ltr-4574f8f4`。`bm25_rank` 权重 +0.6349，训练名次均值 7.3787。名次数字越大，分数越高，所以 BM25 名次 101 或 50 以上的文档会被抬到 grade 3 前面。只把 101 换成训练均值，val 未标注当 0 仍是 0.5451，低于 BM25。把整个 `bm25_rank` 权重置 0 后，val 升到 0.7762，高于 BM25 的 0.7465，仍低于向量的 0.9161。其他权重是在这项还在时学的，置 0 不是新的线上模型。见 `docs/data/ltr-rank-ablation.md`。

## 标注

| 文件 | 角色 |
|---|---|
| `labels.csv` | 正式评估入口。`grok-a` 997 行，`adjudicated` 15 行 |
| `labels_grok_a.csv`、`labels_grok_b.csv` | 原始双标。完全一致 191/206 = 0.9272，相邻一致 1.0000，二次加权 kappa 0.9702 |
| `labels_tool.csv` | `grok-tool`。候选池 938 对全部有分。0/1/2/3 为 517/304/57/60 |
| `labels_claude.csv` | 另一名工具对同一 938 对的独立标注 |

train/val 的 796 对没有裁定。完全一致 588/796 = 0.7387，相邻一致 785/796 = 0.9862，kappa 0.8125。等级差至少 2 的有 11 对，列在 `docs/data/adjudication-report.md`。test 的 142 对已对账，原始分歧见同一份报告的前半。

规则：人或外部工具都可以当标注员，名字写在 `annotator`。被评估的检索或排序不能给同一批 test 文档打分再拿来证明自己。`candidate_pool_k10.csv` 的 `in_*` 列不是分数。

## CTR 与 Agent

CTR 模型 `ctr-bb0917dc`，校准器 `temperature-9cc366a2`，T=2.6887。抽样 test：LR AUC 0.4535；MLP 温度缩放后 AUC 0.4836、ECE 0.0063。AUC 不变、ECE 下降是有效的校准结论，不证明排序有效。合成出价 eCPM 的 Top-1 一致率 0.9180，不是广告收入。

Agent 50 题：工具参数 50/50，引用 27/27，跨来源陷阱 8/8。分析器是确定性的。外部语言模型评测是 UNRUN。

EvoRec 五份 `results.json` 固定在 commit `5ce1d96b80ddaf3c205a5ef8a6a5e39df6ab0ff5`，SHA-256 在 README。视频游戏推荐协议，不能和本项目的 MIND 搜索 nDCG 放进同一张提升表。Agent 任务里的 `evorec-itemcf` 仍是本地夹具。

## Docker

`docker compose up --build` 使用镜像里的 `vector` 额外依赖，并在构建时下载 `BAAI/bge-small-en-v1.5`。运行时 `HF_HUB_OFFLINE=1`。第一次构建要网络，镜像比默认安装大，torch 用的是 CPU 轮子。

2026-10-12 在本机验证：`microsoft` 的 BM25 第一名是 `N58995`，分数 24.661796673688574。向量和 LTR 第一名都是 `N24742`。第一次向量请求约 7.3 秒，因为要加载模型。CTR `ctr-bb0917dc` 可以打分。

入口是 `scripts/docker_entrypoint.py`：先迁移，再按空表导入物品、目录和已测实验，然后启动 API。健康检查 `start_period` 是 240 秒。曝光不导入。

## 还没做

1. 决定正式评估是否改读并集。没有所有者点头，不要覆盖 `labels.csv`。
2. 裁定 train/val 那 11 对，保留两份原始文件。
3. 并发压测，记下命令、机器、版本、成功率和 p95/p99。
4. 外部语言模型 Agent 评测。现在不要把 UNRUN 写成已测。

不要为了追 nDCG 或 AUC 加模型。若要动 LTR，先读 `docs/data/ltr-rank-ablation.md`：问题在已保存的 `bm25_rank` 正权重，不是缺一个更大的网络。

## 工作区

最新已提交是 `a4b2b8c`（`labels_tool.csv` 当时只有 test 的 142 对）。其后的补标、并集评估、LTR 诊断和 Docker 修复还在工作区，未提交。建议所有者自己拆开提交，例如：

- `data(queries): grade the rest of the candidate pool with grok-tool`
- `eval(search): compare modes on the pooled labels`
- `eval(ltr): diagnose the saved bm25_rank weight`
- `build(docker): install the vector extra and seed an empty database`
- `docs: handover for 2026-10-12`
