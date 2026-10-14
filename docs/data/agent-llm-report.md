# 外部语言模型 Agent 评测

最新结果：配置就绪后已完成 [2026-10-14 真实外部评测](agent-llm-live-2026-10-14.md)：50 题、100 次请求、46/50 结构化通过，79,030 tokens，4 个失败保留。默认 CLI 预览仍返回 UNRUN，不能与实际执行结果混淆。

以下为同日配置交付前的历史快照：真实模型评测当时为 **UNRUN**。本轮只核对仓库 `.env` 与进程环境中的模型配置键，未发现可用 endpoint、model、API key；GitHub 仓库的 secret/variable 名称列表也均为空（未读取 secret 值）。`artifacts/` 未发现此前外部 Agent 调用账本；这只是本地核对，不代表全账户历史调用审计。本轮没有探测模型端点或调用模型服务，没有外部模型任务正确率、延迟或 token 消耗的实测结果。离线假响应的测试通过不表示模型通过。

现有 `scripts/evaluate_agent.py` 仍运行确定性分析器，报告在 `agent-report.md`。新增入口为 `scripts/evaluate_agent_llm.py`，与线上 `/agent/analyze`、默认模型和既有 50 条题目分离。

## 评测协议与局限

- 固定输入沿用 `demo.build_tasks()` 的全部 50 题和 `demo_store()`。EvoRec 的 `evorec-itemcf` 是 `run-fixture` / `abc123` 本地夹具；此次入口不接真实导入结果，也不构成对真实 EvoRec 运行的验证。
- 每题至多两个顺序请求：模型给一个 JSON 工具计划；经过字段白名单和既有 `call_tool()` 参数二次校验后执行一个本地只读工具；模型根据返回值输出 JSON 答案。六个窄工具之外的名称、额外参数和多步工具调用均拒绝。计划失败不发送第二个请求。
- 预期工具与参数依据固定题目的种类和实验 ID 单独定义，不调用 deterministic `analyze()` 生成答案。每题的 `task_evidence_match` 要求最终 `evidence` 与**该题预期工具**的实际返回逐字段相等；模型选错实验，即使准确抄录其返回，也不通过。
- `tool_arguments_match`、`task_evidence_match`、`structured_pass` 的分母均为全部 50 题。HTTP 错误、完整响应体超时、无效 JSON、截断、schema 错误都计失败，不删出分母、不重试。子集验证保留完整分母，不把未选中或预算不足的任务标成成功；报告同时给出 `suite_tasks`、`selected_tasks`、`attempted_tasks`、`unrun_tasks`、`selected_unrun_tasks`、`completed_tasks` 与 `budget_exhausted_tasks`，不把仅验证一题称为完成整套模型验证。
- 引用逐项校验实验 ID、指标、split/segment、数值和来源，答案正文引用与引用列表均须匹配，EvoRec 还须出现准确的 run_id / commit。引用题为原有 27 道指标/注入题加 5 道 segment 题，分母 **32**；与旧 deterministic 的 27 题“引用列表非空”不同，不能直接比较比例。
- 8 道跨来源陷阱必须选择比较工具，并以严格的 `无法判断 / 证据不足。来源或协议不同，拒绝比较优劣。` 作为完整答案。附加提升结论不通过；同义但不完全相同的拒绝语也会被判失败，这是可复现自动判分的限制。
- 自由文本论述语义为 **UNSCORED**。结构化证据正确、引用正确都不能证明其余自然语言论述没有幻觉；本报告不把 `structured_pass` 称为完整答案正确率。config/failure 的结构化证据也会校验，但其解释质量仍待人工审阅。

## 本地使用

默认入口不读取配置环境变量、不发请求、不写报告：

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_agent_llm
```

返回 `status=UNRUN, tasks=50, requests_sent=0`。已有授权且运行好兼容 Chat Completions 的模型时，配置完整端点与实际模型名称，再显式执行：

```powershell
$env:SEARCHPILOT_AGENT_LLM_ENDPOINT='http://127.0.0.1:8001/v1/chat/completions'
$env:SEARCHPILOT_AGENT_LLM_MODEL='<已安装模型标识>'
.venv\Scripts\python.exe -m scripts.evaluate_agent_llm --execute --timeout 30 --ledger artifacts/agent-llm-run-001.jsonl --out artifacts/agent-llm-run-001-summary.json
```

建议首次先验证一题。以下第一条命令仅预览选择与预算，不读环境变量、不发请求；第二条命令只有在模型配置就绪且已授权时使用，最多 2 次请求、4096 个预留输出 token：

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_agent_llm --task-id search-00 --max-requests 2 --max-output-tokens 4096
.venv\Scripts\python.exe -m scripts.evaluate_agent_llm --execute --task-id search-00 --max-requests 2 --max-output-tokens 4096 --max-tokens-per-request 2048 --timeout 30 --ledger artifacts/agent-llm-smoke-001.jsonl --out artifacts/agent-llm-smoke-001-summary.json
```

`--task-id` 可重复指定，按原始固定套题顺序执行；未知或重复 ID 在读取配置前拒绝。`--max-requests` 范围 1–100，`--max-output-tokens` 范围 1–204800，`--max-tokens-per-request` 范围 1–2048。每次请求在发送前预留 1 次和完整的单次输出额度，不按实际较少的 token 使用量返还，也不因 usage 未知返还。第二阶段同样需要足额预算；额度不足记 `budget_exhausted`，不会再发请求，也不自动缩小单次额度。若一题第一阶段已发出但第二阶段预算不足，它属于 attempted 而非 unrun，仍不算完成或成功。所有预算都是本次运行的请求/输出上限，不含输入 token，也不是金额预算。

示例端点并未验证。父目录须已存在。需要认证的服务由调用者预先安全设置 `SEARCHPILOT_AGENT_LLM_API_KEY`；无认证 loopback 服务可不设置。脚本不自动读取 `.env`，不打印密钥，也不接受在 URL 中放凭据或查询参数。可通过 `--endpoint-env`、`--model-env`、`--api-key-env` 改用其他环境变量名。非 loopback 仅允许 HTTPS。

一次 `--execute` 最多 100 次请求，每次 `max_tokens=2048`、temperature=0；这是请求上限和输出上限，不是费用估算。客户端关闭重试、重定向及环境代理。每次请求有覆盖完整响应体的总超时（默认 30 秒，最大 300 秒），不会因收到 HTTP 200 就认为完成。响应体在同一总超时内流式读取，最多缓存 **2 MiB**；超量立即关闭响应并记 `response_too_large` 失败，usage 保持未知，不重试，也不从 50 题分母剔除。

CLI 退出码 0 表示评测执行完毕（默认离线调用则表示已输出 UNRUN），**不表示模型通过**；必须检查 summary 的 `structured_pass`、`failed_requests_or_protocol` 及逐题记录。

请求发出前，JSONL ledger 持久化唯一 request_id、pending 状态、累计预留请求/输出额度及 `usage=null`；完成后追加状态、耗时、提供方模型名和可用 usage。未返回合法 `total_tokens` 的请求总 token 保持未知，`known_total_tokens` 仅汇总已知项；reasoning token 不另行加到 completion token。manifest 记录任务/夹具/提示词/评测器代码哈希、端点哈希、模型、所选 task ID、预算和 UTC 时间。summary 保留全任务分母、失败数量、usage 已知/未知请求数、预算与已预留额度；任务级 p95 仅统计实际尝试的题（含失败），不混入未运行题的零耗时。

ledger 与 `--out` 均独占创建，拒绝覆盖或自动续跑；默认也不会改写本文。被取消或中断的请求可能已经计费，pending/usage 未知不是零消耗。中断后先核对已有 ledger 和服务端记录，不把换文件名重跑当作自动恢复；没有完整 summary 的日志只代表未完成运行。

后续请求日志还会记录白名单内的 `finish_reason` 和安全的 `validation_error` 阶段/原因码，帮助定位协议失败；不记录任意服务端错误正文。该诊断补丁晚于本次 100 次真实请求，无法补回旧日志缺失的原因。

## 历史离线验证

使用 `httpx.MockTransport`，不联网、不下载模型、不依赖数据库。覆盖 50 题完整循环及 100 个唯一请求 ID、全分母错误统计、HTTP 错误/无效 JSON/错误 envelope 类型/截断、HTTP 200 后响应体超时、错误实验/来源/数值/额外引用、陷阱“拒绝后又给提升”、危险工具/额外参数/范围校验、重复日志拒绝、默认 UNRUN 和输出不覆盖。

2026-10-13 历史验证为 34 passed，并重跑确定性报告得到白名单 50/50、引用非空 27/27、陷阱短语 8/8。2026-10-14 本轮新增预算/子集边界后，外部入口与既有 Agent 测试共 **50 passed**（2.16 秒），ruff check 通过，Agent 包 mypy **5 个文件通过**。新增覆盖参数无效零调用/不建文件、单题保留 50 题分母、跨阶段额度不足、实际低 usage 不返还额度、未知 usage 不返还额度、部分尝试与选中未运行区分、默认预览不读环境。原有中断账本、防覆盖、2 MiB 流式上限与错误分母测试继续通过。有一条已有 Starlette/httpx 弃用警告，没有测试失败。

```powershell
$env:TMP='E:\searchPilot\.tmp'; $env:TEMP=$env:TMP
.venv\Scripts\python.exe -m pytest tests/test_evaluate_agent_llm.py tests/test_agent.py
.venv\Scripts\python.exe -m ruff check src/searchpilot/agent/llm_eval.py scripts/evaluate_agent_llm.py scripts/evaluate_agent.py tests/test_evaluate_agent_llm.py
.venv\Scripts\python.exe -m mypy src/searchpilot/agent/llm_eval.py
```

历史配置阻塞（现已解除，实测见本文顶部链接）：`SEARCHPILOT_AGENT_LLM_ENDPOINT`、`SEARCHPILOT_AGENT_LLM_MODEL` 未配置，认证服务所需的 `SEARCHPILOT_AGENT_LLM_API_KEY` 也未配置；仓库 CI 没有可供调度的相关配置名称。已有运行入口与有界 smoke 命令无需进一步实现。待配置已授权、支持当前 JSON 输出协议的模型服务后，显式执行并审阅完整 ledger。该历史快照的结论为 **UNRUN**。
