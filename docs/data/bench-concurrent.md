# 真实 HTTP 并发压测

2026-10-13，来源：`searchpilot`。在本机 CPU、65238 篇 MIND 新闻、100 条固定查询、并发 4、每模式 120 个正式请求、每模式 4 次独立顺序预热条件下，BM25 / vector / hybrid / LTR 的正式请求均为 **120/120 成功**。向量模式第一条预热在 **10 秒截止时间超时**，不能把本轮写成冷启动全部通过。

这是单机、单 worker、一次闭环负载记录。未设置业务 SLA，未测容量上限或长期稳定性，也未使用 PostgreSQL。功能成功率与性能验收分开解释。

**版本边界：480 个正式请求来自编码器初始化锁修复前的服务。** 主代理随后为 `vector_index.py` 增加初始化锁与二次检查，另有真实缓存 BGE 两并发初始化验证，记录在 `artifacts/bench-concurrent/encoder-init-check-2026-10-13.json`。这里没有重跑修复后的性能，不作为最终代码的性能验收。

## 实测条件与版本

- 服务与压测客户端同机：`Windows-11-10.0.26200-SP0`，Intel Core i5-12450H，8 核 / 12 逻辑处理器，物理内存 16869351424 bytes。
- Python `3.12.6`；httpx `0.28.1`、pydantic `2.13.5`、uvicorn `0.54.0`、numpy `2.5.3`、torch `2.14.1`、searchpilot `0.1.0`。
- 服务地址 `http://127.0.0.1:18080`，PID `26772`。主代理为本轮启动并持有进程；单 worker、无 reload、关闭 access log。未加载 dotenv，明确清除数据库连接配置，使用绝对 data/artifacts 路径和离线模型缓存。`/health/ready` 不含 database 检查项。
- 数据版本 `d3a904f41240`，BM25 元数据记录文档数 `65238`。查询为 `data/queries/queries.csv` 全部 100 条、包含所有 split、按 CSV 顺序循环；120 次等于全部 100 条加前 20 条。query 长度 11–124 字符，limit=10。不据此评估相关性或做训练结论。
- 查询文件 SHA-256：`d5663200e2c7e98fbf283ef6333f53081725b44b947d70a3b2a8d94e6e65df69`。
- 本地 commit：`a4b2b8c624fb6b13588d4e5747ad7eceeb60e5de`，**dirty=true**。不是纯提交版本复现。执行开始与结束的 tracked diff SHA-256 均为 `0bb02e2bb9aa042692d7ee02327c7f829fea07f5d7f28df2fbfa0385b830960e`；该哈希不含未跟踪文件，不能替代完整 checkout 快照。
- 执行时间：`2026-10-13T14:30:10.556564+00:00` 至 `2026-10-13T14:31:26.719559+00:00`（北京时间 22:30–22:31）。

服务启动命令（由主代理执行）：

```powershell
.venv\Scripts\python.exe -m uvicorn searchpilot.bootstrap:build_default_app --factory --host 127.0.0.1 --port 18080 --no-access-log
```

本轮实际压测命令：

```powershell
.venv\Scripts\python.exe scripts/bench_concurrent.py --base-url http://127.0.0.1:18080 --requests 120 --concurrency 4 --timeout 10 --warmup 4 --limit 10 --modes bm25 vector hybrid ltr --data-version d3a904f41240 --server-description 'Parent-managed PID26772; local venv uvicorn single worker no reload; no DATABASE_URL; offline HF cache; absolute workspace data and artifacts; server source checkout is dirty' --out artifacts/bench-concurrent/2026-10-13-local-c4/results.json
```

## 正式请求结果

成功要求完整 HTTP 200 body、响应 schema 正确、请求与响应头/正文 request_id 一致、mode 一致、模型版本非空、结果不重复、rank 连续且 score 有限。空结果符合接口成功契约，但单独计数；本轮正式请求空结果为 0。HTTP 200 收到部分 body 并不算成功。

| 模式 | 成功 / 全部 | 超时 / 全部 | p50 ms | p95 ms | p99 ms | 最大 ms | 成功 req/s | 正式段秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 120/120 | 0/120 | 20.989 | 47.033 | 56.120 | 58.179 | 167.586 | 0.716 |
| vector | 120/120 | 0/120 | 608.911 | 950.584 | 1099.964 | 1167.205 | 6.636 | 18.083 |
| hybrid / RRF | 120/120 | 0/120 | 592.354 | 1028.548 | 1331.922 | 1388.954 | 6.535 | 18.363 |
| LTR | 120/120 | 0/120 | 802.985 | 1408.937 | 1583.321 | 1609.953 | 4.873 | 24.626 |

每行正式请求成功率 100%，失败率 0%，全部 HTTP 状态为 200。表中为成功请求延迟；由于本轮正式请求没有失败，**全体 120 请求延迟与成功延迟相同**。JSON 中两套分位数仍分开保存，未来有失败时不会丢掉失败样本。各段完成 req/s 与成功 req/s 本轮相同。

服务实际返回模型版本依次为：`bm25-4f46b3d8`、`vector-b92663fd`、`hybrid-51d4dcfb`、`ltr-4574f8f4`。

分位数采用 nearest rank：排序后 `ceil(q*n)-1`；延迟从 worker 分派请求开始，直到完整响应及校验完成或失败，包含连接和读取 body，排除尚未分派的客户端排队。固定 4 个 worker 闭环发压，模式依次运行，不重试、不跟随重定向、不使用环境代理。它不等同于固定到达率压测；慢响应会降低闭环施压速率。120 样本的 p99 只有有限尾部证据。

## 预热、首次向量请求与失败分母

每模式正式请求之前各发 4 次顺序预热，所有预热都有独立原始记录，不混入正式 120 请求分位数。

| 模式 | 预热成功 / 全部 | 第 1 次 ms | 第 2 次 ms | 第 3 次 ms | 第 4 次 ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 4/4 | 100.177 | 9.136 | 7.583 | 6.598 |
| vector | 3/4 | timeout，观测 9995.286 | 1041.752 | 101.029 | 124.656 |
| hybrid | 4/4 | 97.249 | 110.455 | 101.990 | 121.849 |
| LTR | 4/4 | 166.853 | 171.141 | 177.875 | 154.057 |

向量首次请求的 `status_code=null`、接收 body 字节数 0，配置总截止时间为 10 秒。预热总计 **15/16 成功，1/16 超时**；正式段 **480/480 成功**。如统计压测脚本所有请求，则是 **495/496 成功，1/496 超时**，不能删除首次失败后称所有请求 100% 成功。不要将各模式或冷热请求混在一起报告一个延迟分位数。

服务进程本轮首次启动后，主代理未事先调用 `/search`。第一条向量请求覆盖首次编码器使用，但 API 进程和索引此前已启动，且 BM25 已先跑。客户端没有模型阶段追踪，因此不能证明超时全部归因于编码器加载，也不能把该请求说成完整服务冷启动测量。客户端超时不证明服务器后台工作已取消；下一条预热可能与其剩余工作重叠。本报告只断言后续完整响应成功。

独立恢复检查在 `2026-10-13T14:32:18Z` 后执行：`/health/live=200`，`/health/ready=200`，search/recommend/feedback/vector_index/ltr/ctr 全为 true；额外 BM25 `microsoft` 请求 **1/1 成功，6.420 ms**。恢复检查不在 480 正式或 16 预热分母中。没有做强制中断、服务器取消完成或残留任务审计。

## 原始记录与执行脚本

本机原始文件保存在被 Git 忽略的 `artifacts/bench-concurrent/2026-10-13-local-c4/`：

- `results.json`：配置、命令参数、机器和依赖版本、代码与查询哈希、4 个模式的逐请求 ledger 和各自汇总。SHA-256：`8a45dfac8c14e99de32fb75620226f85c066fc17597e336dbfda757f8c62e87f`。
- `bench_concurrent.executed.py`：本轮实际执行脚本快照，SHA-256 `698bf3d42d510291d98e9d24e3a1e9206a409b49e107fa8dcd199e35c16e5c4d`，与 results 中开始/结束 benchmark 哈希一致。
- `recovery.json`：独立健康和恢复搜索探测。
- `resource-snapshot.json`：结束后资源快照，CPU 时间为进程启动以来累计，不是压测窗口 CPU 利用率。

原始 JSON 不写入响应 body 或错误正文，避免将服务错误中的敏感内容收进报告；失败保留类别、状态、请求 ID、输入、时长及异常类型。真实输出中的服务版本/数据版本声明来自运行协调信息；客户端自动获取的 checkout 与依赖版本不应冒充远程服务 provenance。

执行完后对 CLI 做了输出保护修补，**没有重复跑正式负载或改写上述 results**：当前脚本在发请求前用排他创建拒绝覆盖已有 `--out` 及其 `.requests.jsonl` 文件，记录 `RUNNING / COMPLETED / INTERRUPTED / FAILED` 与模式进度，退出码也将预热失败计入失败。每请求先将 pending 写入 JSONL 并 flush，再发请求；完成或失败后追加 result 并 flush，含 phase、request_id、输入、状态与延迟。当前执行代码与快照哈希不同是明确的后续修补，不是实测期间改动。原执行版本退出码为 0，仅说明正式段没有失败，不能掩盖已保存的预热超时。修补后的相同结果应退出 1。

重跑请用新的输出路径（例如 `--out artifacts/bench-concurrent/2026-10-13-local-c4-repeat/results.json`），保留之前的成功与失败证据。中断时 `runs` 只代表已完成模式；JSONL 仍保留当前模式的已完成结果，pending 没有匹配 result 的请求均是 UNKNOWN，可能已经到达服务器，不能删除或自动重发。原始已完成本轮快照尚无 JSONL 功能，完整逐请求数据在 results.json 中。若操作系统强制终止进程，可能只留下 RUNNING 或不完整文件，同样不能视作完整测试；flush 不保证断电后的磁盘持久性。新增 JSONL 的同步写入可能影响客户端施压速率，后续延迟不计自身 pending/result 写入耗时，但段吞吐会包含它们；因此不要把这次旧快照吞吐当成修补后脚本的再次实测。

## 资源与未测边界

结束后工作集 1862840320 bytes（约 1.735 GiB）；进程生命周期 peak working set 2456653824 bytes（约 2.288 GiB）；CPU 累计 582.375 秒。采样时间 `2026-10-13T14:32:19.2717055Z`，在恢复检查之后。**压测窗口 CPU 利用率、压测窗口内存峰值、GPU 利用率均 UNRUN**，这些快照不能替代持续采样。

本轮没有 PostgreSQL 路径、Docker 环境、CTR HTTP 并发、混合模式同时负载、多 worker、跨机、固定到达率、重复独立轮次、长时稳定性或 SLA 验收。BM25 正式窗口仅 0.716 秒，吞吐是本轮短突发观测。成功只表示完整接口契约检查，不是相关性指标提升，也不证明超时后的服务器资源已完全清理。分阶段进程内计时仍单独参考 `bench-stages.md`，不能当成本轮 HTTP 分阶段 trace。

## 脚本验证

`tests/test_bench_concurrent.py` 使用临时 loopback TCP 服务验证真实 HTTP 边界，不访问外部网络、不使用模型或数据库。覆盖总超时（等 header、持续滴流 body）、503、畸形 JSON、截断 body、连接提前关闭、HTTP 200 schema/请求 ID/mode/重复结果/非有限分数/错误排名、空结果成功、全部失败分母、并发上限、预热分离、参数限制、CLI 输出/JSONL 防覆盖、中断状态及当前模式逐请求证据保留。

修补后的实际检查：ruff check 通过，ruff format check 通过，脚本 mypy 通过，**31 tests passed**。存在现有 FastAPI/Starlette 提示 httpx TestClient 弃用的 1 条 warning，与本脚本真实 HTTP 客户端无关。

```powershell
$env:TMP='E:\searchPilot\.tmp'
$env:TEMP=$env:TMP
.venv\Scripts\python.exe -m ruff check scripts/bench_concurrent.py tests/test_bench_concurrent.py
.venv\Scripts\python.exe -m ruff format --check scripts/bench_concurrent.py tests/test_bench_concurrent.py
.venv\Scripts\python.exe -m mypy scripts/bench_concurrent.py
.venv\Scripts\python.exe -m pytest tests/test_bench_concurrent.py
```
