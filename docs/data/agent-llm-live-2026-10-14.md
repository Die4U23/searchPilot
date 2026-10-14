# 2026-10-14 外部 LLM 固定夹具验证

真实外部请求已执行，状态为 **COMPLETED，结构化检查 46/50（92%）**。
50 题各尝试一次，100 个唯一请求，没有重试、未配对 pending 或未知用量；
2 题响应协议失败、2 题严格引用格式失败全部保留。执行完毕不等于模型全部通过。
默认 CLI 的离线预览仍返回 UNRUN；此前配置缺失的 UNRUN 仅是历史快照。

证据：[联合 summary](agent-llm-live-2026-10-14-summary.json)、
[原始账本与执行源码 ZIP](agent-llm-live-2026-10-14-raw.zip)、
[协议与运行入口](agent-llm-report.md)。ZIP 仅含公开夹具的模型输出、账本、源码和校验值；
不含 `.env`、密钥或服务 URL。端点仅保留 SHA-256，配置继续本地忽略，不上传 CI。

## 执行与结果

执行基线为 `d48040fe780b9ffde3419f08df3715f2eed7e6ab`，模型栈、任务、提示词、
夹具与原始 `labels.csv` 未改。模型参数为 `deepseek-flash`；98 个已接受响应记录的
提供方 `model` 字段也为该名称，2 个无效响应未记录该字段。这是服务方声明，
不能据此独立确认后端模型身份。

| 分片 | UTC 开始—结束 | 选中且实际尝试 | 请求 | total_tokens |
|---|---|---:|---:|---:|
| smoke-001 | 03:16:23—03:16:27 | search-00，1 题 | 2 | 1,118 |
| remaining-049 | 03:17:05—03:21:53 | 其余 49 题 | 98 | 77,912 |
| 合并 | 2026-10-14，北京时间 11:16—11:21 | 50 个不同 task ID | **100** | **79,030** |

先执行单题 smoke，成功后仅执行剩余 49 题，没有重复 smoke。两份源账本各自包含
完整 50 题分母和未选中行；合并器只接纳每题唯一的实际尝试，不能直接相加子集分母。

| 指标 | 本轮结果 | 解释 |
|---|---:|---|
| 实际尝试 / 未运行 | 50 / 0 | 全部固定任务 |
| 完成任务 / 响应协议失败 | 48 / 2 | HTTP 200 不代表协议成功 |
| structured_pass | **46/50** | 结构化协议通过率，不是完整答案正确率 |
| tool_arguments_match | 48/50 | 两个答案阶段失败没有 scores，保留在分母 |
| task_evidence_match | 48/50 | 同上 |
| 引用支持检查 | **30/32** | 27 个指标/注入题及 5 个 segment 题 |
| 跨来源陷阱 | **8/8** | 必须严格拒绝比较 |
| 注入任务结构化检查 | 2/2 | 仅本套题范围 |
| 任务耗时 p95 | 11.703 秒 | nearest-rank，50 个实际尝试，含失败；每题最多两个请求 |
| usage 已知 / 未知请求 | 100 / 0 | 提供方报告，不是账单核验 |
| 输入 / 输出 tokens | 32,731 / 46,299 | 总计 79,030，reasoning 不重复相加 |

分组结构化通过：searchpilot 15/15、evorec 8/10、config 5/5、segment 5/5、
failure 3/5、trap 8/8、inject 2/2。额外只读审计发现 50 个已生成 plan 均匹配
expected_call；这不替换上述官方 `tool_arguments_match=48/50` 口径。

两分片请求预算分别为 2、98；输出预留预算分别为 4,096、200,704，总计 204,800。
每次请求发送前完整预留 2,048 输出 tokens，不因实际用量较少而返还额度。
temperature=0、单请求完整响应体超时 30 秒、响应体上限 2 MiB、不重定向、不重试。
这些上限不含输入 tokens，也不是金额预算；未查询或估算实际账单金额。

## 四个失败

| task_id | 原始状态 | 已证实原因与边界 |
|---|---|---|
| evorec-05 | completed，structured_pass=false | 工具、证据、数值 0.2 与引用列表正确。正文写“工具返回的是 EvoRec run-fixture，commit abc123”，未包含协议要求的“来自 EvoRec …”精确前缀，来源措辞检查失败。不能称为数值或来源错误。 |
| fail-00 | completed，structured_pass=false | 工具与证据正确，但 failure samples 按协议必须无引用；模型添加了 `[sp-bm25 \| ndcg \| q_026/synonym \| 0.0 \| searchpilot]`，还把 query_id 放入 split 位置。属于引用协议失败，不据此认定数值幻觉。 |
| evorec-09 | failed，invalid_response | answer 请求 HTTP 200，已知 usage；completion_tokens=2048。旧账本没有 finish_reason、原始无效响应或细分错误码，不能确定是截断还是其他格式问题。 |
| fail-04 | failed，invalid_response | 同样为 answer 阶段 HTTP 200、completion_tokens=2048、usage 已知；具体原因无法由旧账本还原。 |

本轮没有为提高分数而重试、修改提示词、放宽判分或更换任务。诊断补丁只影响后续执行：
在安全白名单内记录 `finish_reason`，并记录 `validation_error` 的解析阶段或内部原因码，
不保存任意服务端异常正文。补丁晚于这 100 次请求，不能回填旧失败的原因。

## 适用范围

- 这是外部 LLM 与本地只读工具夹具的真实 HTTP 验证，线上 `/agent/analyze` 仍是确定性分析器。
- 50 个 task ID 仅含 **33 条不同的字面问题、10 种不同的预期工具计划**，不能视为 50 个独立能力场景。
- EvoRec 题目写了 run 0—9，但预期调用未按 run 区分，统一返回 `run-fixture` / `abc123`。
  这是现有套题限制，本轮保留冻结协议；不证明能解析 10 次真实 EvoRec 实验。
- 引用与来源采用严格字符串合同，同义措辞、数字格式差异也可能失败。
- 自由文本语义仍为 **UNSCORED**；结构化证据一致不证明其余论述无幻觉。
- 单次有序执行的延迟与通过率不构成并发性能、稳定性、通用质量或生产 SLA 结论。

## 离线复核

`scripts/summarize_agent_llm.py` 不读配置、不访问网络、不续跑任务。它验证首尾事件、
完整任务行、唯一请求 ID、request/result 一一对应、阶段顺序、预算与终态 summary，
拒绝缺失或重复实际尝试；保留 unknown usage。两分片的模型、端点哈希、执行器、
任务、提示词、夹具和策略必须相同；当前任务/提示词/夹具必须匹配，并逐题重新评分。
执行器源码可有诊断更新，因此分别记录原执行哈希和当前审计代码哈希。
该工具核对账本内部一致性，不是服务提供方签名或独立账单证明。

在仓库安装依赖后，将 ZIP 解压到一个新建本地目录，例如 `.tmp/llm-evidence`：

```powershell
.venv\Scripts\python.exe -m scripts.summarize_agent_llm --ledger .tmp/llm-evidence/smoke-001.jsonl --ledger .tmp/llm-evidence/remaining-049.jsonl --out .tmp/llm-recomputed.json
```

`--out` 必须不存在，源账本不会改写。路径和当前审计源码哈希可随 checkout/换行格式变化；
任务分母、评分、用量、源账本字节哈希必须与提交 summary 一致。
ZIP 的 `SHA256SUMS.json` 校验每个证据成员，`source/` 保留实际执行时的源码字节。

| 对象 | SHA-256 |
|---|---|
| 实际执行的 llm_eval.py | `af260d0379599488c6005f89ca692dc17bc1a5eda0f59db2d1f2c80f3f5461a5` |
| smoke-001.jsonl | `ea1775b77a0139a5c4eda87faa3388aec9f6454dfa945f051369cca552aa644d` |
| remaining-049.jsonl | `c06e79aa3f5740b1bfc6217d3d9f3bf1b35150c87011a5370bc9bfcdc8f324a1` |
| 证据 ZIP（35,121 字节） | `12c6a478f631cde091be749b26a53c5e6c79fedf8f439241aa121cc67199c659` |

任务/提示词/夹具的完整哈希、预算、UTC 时间及各分片原始统计均见联合 summary。
已核对 ZIP CRC、各成员 SHA-256、执行源码哈希、私有配置值未出现在证据中，以及 `.env` 未跟踪。

## 本轮软件验证

本机完整单元回归 **461 passed、10 deselected**；外部入口与合并器专项 **74 passed**。
Ruff 全仓检查与格式检查通过，Mypy 对 `src` 和新增合并器共 71 个文件通过。
新增覆盖安全诊断、未知 usage、重复请求/任务、未完成账本、版本不一致、重评分不一致和输出防覆盖。
有一条既有 Starlette/httpx 弃用警告。本轮本机没有重跑 PostgreSQL 集成；CI 使用 PostgreSQL 16 单独验证。
这些离线测试通过只证明软件行为，不能替代上方真实模型的 46/50 结果。
