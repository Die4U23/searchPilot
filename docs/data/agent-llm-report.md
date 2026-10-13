# 外部语言模型 Agent 评测

2026-10-13：**UNRUN**。本次没有探测模型端点、读取密钥或调用外部服务。没有外部模型任务正确率、延迟或 token 消耗的实测结果。离线假响应的测试通过不表示模型通过。

现有 `scripts/evaluate_agent.py` 仍运行确定性分析器，报告在 `agent-report.md`。新增入口为 `scripts/evaluate_agent_llm.py`，与线上 `/agent/analyze`、默认模型和既有 50 条题目分离。

## 评测协议与局限

- 固定输入沿用 `demo.build_tasks()` 的全部 50 题和 `demo_store()`。EvoRec 的 `evorec-itemcf` 是 `run-fixture` / `abc123` 本地夹具；此次入口不接真实导入结果，也不构成对真实 EvoRec 运行的验证。
- 每题至多两个顺序请求：模型给一个 JSON 工具计划；经过字段白名单和既有 `call_tool()` 参数二次校验后执行一个本地只读工具；模型根据返回值输出 JSON 答案。六个窄工具之外的名称、额外参数和多步工具调用均拒绝。计划失败不发送第二个请求。
- 预期工具与参数依据固定题目的种类和实验 ID 单独定义，不调用 deterministic `analyze()` 生成答案。每题的 `task_evidence_match` 要求最终 `evidence` 与**该题预期工具**的实际返回逐字段相等；模型选错实验，即使准确抄录其返回，也不通过。
- `tool_arguments_match`、`task_evidence_match`、`structured_pass` 的分母均为全部 50 题。HTTP 错误、完整响应体超时、无效 JSON、截断、schema 错误都计失败，不删出分母、不重试。
- 引用逐项校验实验 ID、指标、split/segment、数值和来源，答案正文引用与引用列表均须匹配，EvoRec 还须出现准确的 run_id / commit。引用题为原有 27 道指标/注入题加 5 道 segment 题，分母 **32**；与旧 deterministic 的 27 题“引用列表非空”不同，不能直接比较比例。
- 8 道跨来源陷阱必须选择比较工具，并以严格的 `无法判断 / 证据不足。来源或协议不同，拒绝比较优劣。` 作为完整答案。附加提升结论不通过；同义但不完全相同的拒绝语也会被判失败，这是可复现自动判分的限制。
- 自由文本论述语义为 **UNSCORED**。结构化证据正确、引用正确都不能证明其余自然语言论述没有幻觉；本报告不把 `structured_pass` 称为完整答案正确率。config/failure 的结构化证据也会校验，但其解释质量仍待人工审阅。

## 本地使用

默认入口不读取配置环境变量、不发请求、不写报告：

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_agent_llm
```

返回 `status=UNRUN, tasks=50, requests_sent=0`。未来已有授权且运行好兼容 Chat Completions 的模型时，配置完整端点与实际模型名称，再显式执行：

```powershell
$env:SEARCHPILOT_AGENT_LLM_ENDPOINT='http://127.0.0.1:8001/v1/chat/completions'
$env:SEARCHPILOT_AGENT_LLM_MODEL='<已安装模型标识>'
.venv\Scripts\python.exe -m scripts.evaluate_agent_llm --execute --timeout 30 --ledger artifacts/agent-llm-run-001.jsonl --out artifacts/agent-llm-run-001-summary.json
```

示例端点并未验证。父目录须已存在。需要认证的服务由调用者预先安全设置 `SEARCHPILOT_AGENT_LLM_API_KEY`；无认证 loopback 服务可不设置。脚本不自动读取 `.env`，不打印密钥，也不接受在 URL 中放凭据或查询参数。可通过 `--endpoint-env`、`--model-env`、`--api-key-env` 改用其他环境变量名。非 loopback 仅允许 HTTPS。

一次 `--execute` 最多 100 次请求，每次 `max_tokens=2048`、temperature=0；这是请求上限和输出上限，不是费用估算。客户端关闭重试、重定向及环境代理。每次请求有覆盖完整响应体的总超时（默认 30 秒，最大 300 秒），不会因收到 HTTP 200 就认为完成。响应体在同一总超时内流式读取，最多缓存 **2 MiB**；超量立即关闭响应并记 `response_too_large` 失败，usage 保持未知，不重试，也不从 50 题分母剔除。

CLI 退出码 0 表示评测执行完毕（默认离线调用则表示已输出 UNRUN），**不表示模型通过**；必须检查 summary 的 `structured_pass`、`failed_requests_or_protocol` 及逐题记录。

请求发出前，JSONL ledger 持久化唯一 request_id、pending 状态及 `usage=null`；完成后追加状态、耗时、提供方模型名和可用 usage。未返回合法 `total_tokens` 的请求总 token 保持未知，`known_total_tokens` 仅汇总已知项；reasoning token 不另行加到 completion token。manifest 记录任务/夹具/提示词/评测器代码哈希、端点哈希、模型、请求上限和 UTC 时间。summary 保留全任务分母、失败数量、任务级 p95 和 usage 已知/未知请求数。

ledger 与 `--out` 均独占创建，拒绝覆盖或自动续跑；默认也不会改写本文。被取消或中断的请求可能已经计费，pending/usage 未知不是零消耗。中断后先核对已有 ledger 和服务端记录，不把换文件名重跑当作自动恢复；没有完整 summary 的日志只代表未完成运行。

## 本次离线验证

使用 `httpx.MockTransport`，不联网、不下载模型、不依赖数据库。覆盖 50 题完整循环及 100 个唯一请求 ID、全分母错误统计、HTTP 错误/无效 JSON/错误 envelope 类型/截断、HTTP 200 后响应体超时、错误实验/来源/数值/额外引用、陷阱“拒绝后又给提升”、危险工具/额外参数/范围校验、重复日志拒绝、默认 UNRUN 和输出不覆盖。

本次验证：新增测试与既有 Agent 测试共 **34 passed**（1.83 秒），ruff check 通过，ruff format 后通过，Agent 包 mypy **5 个文件通过**。其中超量 body 测试确认在第一个超限 chunk 立即停止读取并关闭响应，50 题均保留为失败，usage 全部未知。有一条已有 Starlette/httpx 弃用警告，没有测试失败。另实际运行默认入口得到 UNRUN / 0 requests，并重跑确定性报告得到白名单 50/50、引用非空 27/27、陷阱短语 8/8。

```powershell
$env:TMP='E:\searchPilot\.tmp'; $env:TEMP=$env:TMP
.venv\Scripts\python.exe -m pytest tests/test_evaluate_agent_llm.py tests/test_agent.py
.venv\Scripts\python.exe -m ruff check src/searchpilot/agent/llm_eval.py scripts/evaluate_agent_llm.py scripts/evaluate_agent.py tests/test_evaluate_agent_llm.py
.venv\Scripts\python.exe -m mypy src/searchpilot/agent/llm_eval.py
```

真实外部评测剩余条件：提供已授权、可用且支持当前 JSON 输出协议的模型服务，再显式执行并审阅完整 ledger。当前结论保持 **UNRUN**。
