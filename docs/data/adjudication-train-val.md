# train / val 补标大分歧文本裁定

2026-10-13。数据版本 `d3a904f41240`。本次从两份当前 CSV 重新筛选，train/val 的 796 对中等级差绝对值至少 2 的确为 **11 对**，涉及 8 个查询：train 10 对、val 1 对、test 0 对。

裁定员是当前 Codex 会话，`annotator=codex-text-adjudication-2026-10-13`。这是已知双方原分后进行的工具文本裁定，不是人工复核，也不是第三个独立模型的盲标。未调用外部 API，未运行检索或排序模型，未使用候选通道、名次、模型分数、点击量或热度作为判断依据。

## 产物与边界

- `data/queries/labels_adjudicated_train_val.csv`：仅这 11 对的裁定补丁，不是覆盖 796 对的完整标签集。保留标准的 `query_id,item_id,grade,annotator`，附 split、查询原文、类型、标题、摘要、双方原分、理由和每对输入哈希。
- `scripts/adjudicate_train_val.py`：校验冻结输入、重新筛选大分歧、回放固定裁定并核对 CSV 统一为 LF 后的完整字节；默认只读，`--write` 仅写上述独立 CSV。脚本不自动作出新语义判断。
- `tests/test_adjudicate_train_val.py`：范围、重复、非法等级、覆盖缺失、输入变更的负向检查。
- 本报告：逐条理由、输入版本与复现证据。

本次未覆盖 `labels.csv`、`labels_tool.csv`、`labels_claude.csv` 或其他原始标注，未改正式评估入口、模型或测试集分数。尚未将此裁定应用于训练或评估，因此不产生新的 nDCG、排序有效性或性能结论。

正式入口和后续覆盖规则见[评估口径决定](evaluation-policy.md)。不能直接把裁定作为第三份标签传给 `merge_label_maps`，因为其冲突取最大等级，会吞掉降低等级的裁定；本轮没有添加合并或评估代码。

## 判定口径

依据 `annotation-guideline.md` 的标题与摘要规则。3 分要求直接完整满足意图；2 分要求主要意图匹配；1 分用于可辨认的主题或背景联系、但不能直接回答；0 分用于无实质关系、仅共享泛化词语的另一事件。

这里把明确的实体或事件矛盾与“仅缺一个次要条件”区分开。例如其他城市的煮水通知仍与饮水警告这一具体主题有弱关联，所以评 1；但不同学校针对不同议题的两项学生会决议，仅共享机构类别，评 0。没有将“同领域”一律升至 2，也没有因为实体不符一律压到 0。

查询以 `queries.csv` 的英文原文为准。旧对账表的中文短语只作线索：q_022 不包含“商务部”，q_031 的 `plate cleaner` 也无法单凭短语断定指洗碗剂还是洗碗机。没有补看新闻全文或推断标题摘要没有展示的内容。

## 11 对裁定

表中原分顺序为 grok-tool / claude。完整的原始标题、摘要及逐条输入 SHA-256 均附在 CSV 中，可脱离模型输出人工审阅。

| 查询 / 物品 | split | 原分 | 裁定 | 文本证据与理由 |
|---|---|---:|---:|---|
| q_022 / N18012 | train | 2 / 0 | **0** | 查询是北京获国际贸易机构许可惩罚华盛顿；新闻 `China says postal fees to rise after US complaint` 则是美国投诉后，中国支付的国际邮费上涨。机构为全球邮政联盟，措施与方向都不符；共享中美与国际机构词面不足以关联目标事件。 |
| q_024 / N47282 | train | 0 / 2 | **1** | `JP Morgan Chase Donates $5 Million To Build Affordable Housing In Miami` 与可负担住房出资主题相关；但 JP Morgan Chase 不是查询中的 phone maker，Miami 也不匹配 Golden State，不能回答目标承诺。 |
| q_029 / N50720 | train | 1 / 3 | **1** | `Vehicle ends up inside landscaping trailer in bizarre crash` 确有车辆与敞开式拖车碰撞；摘要只称 vehicle，没有学生或校车证据。缺失的是查询关键主体，不能视为同一事故。 |
| q_031 / N21447 | train | 1 / 3 | **1** | `Secret Cleaning Tips From the Pros` 摘要仅称让房屋洁净。家居清洁属于上位主题，未展示餐具或提升 plate cleaner 效率的具体方法；不能补全歧义。 |
| q_031 / N49925 | train | 1 / 3 | **1** | `33 Super Effective Old-School Cleaning Tricks You've Never Tried` 描述低成本的家居清洁技巧，没有洗碗效果证据；不能假设未展示的 33 条技巧必含目标方法。 |
| q_059 / N18347 | train | 2 / 0 | **1** | `101 Freeway reopened in Santa Barbara County after closure from brush fire` 是火灾影响高速公路的同类事件；摘要定位 Santa Barbara 西侧，没有 LA 地点匹配证据，不能据此确认洛杉矶高速大火。 |
| q_061 / N6773 | val | 2 / 0 | **0** | 查询为 GVSU 学生会取消效忠誓词；新闻讨论 Drake University 学生会拒绝 Turning Point USA 社团认证。学校与决策议题都不同，未涉及誓词，只有泛化的学生会及政治词面联系。 |
| q_066 / N11915 | train | 2 / 0 | **1** | Norristown、Plymouth Township 因水管破裂发布煮水警告，确属同类饮水安全通知；未提 McKeesport 或共同供水事件，不能作为目标地区警告。 |
| q_066 / N39875 | train | 2 / 0 | **1** | Reading 的煮水命令提供同类警告及煮水方法背景；明确城市不同，未说明与 McKeesport 的关系。 |
| q_066 / N7953 | train | 2 / 0 | **1** | Fort Lauderdale 的通知明确是解除状态；同属煮水通知主题，但地点与状态均不能回答 McKeesport 的警告。 |
| q_092 / N22801 | train | 2 / 0 | **1** | 宾州司法选举电视广告资金具有法院与政治捐资背景，并提 Harrisburg；实际讨论 Superior Court 两席，没有 Supreme Court 或 gun donors 的证据。 |

结果为 0/1/2/3 各 **2/9/0/0**。与 grok-tool 相同 3 对、与 claude 相同 2 对；另 6 对取双方原分之外的 1 分。这是依据文本重判，不是选择某一标注员作为默认赢家。

## 原始一致率保留

对当前文件重新计算，train/val 双标覆盖 85 个查询、796 对：完全一致 **588/796 = 0.7387**，相邻一致 **785/796 = 0.9862**，二次加权 Cohen's kappa **0.8125224774**。与旧报告数值一致。以上是裁定前的原始标注一致率，不将固定裁定复制到双方后报告虚假的一致率提升。

本次仅处理差至少 2 的 11 对。其他 **197 对相邻分歧**没有裁定；其余 588 对原本一致。因此本产物不表示全部 796 对已达成一致，也不解决候选池代表性问题。

## 哈希与证据重放

下表的 CSV/Markdown 哈希采用 **UTF-8 解码、universal newline 将 CRLF/CR 换为 LF、重新编码 UTF-8** 后的完整字节；不 trim、不忽略空格、不改变其他字符。这样 Git `core.autocrlf` 造成的换行转换不会令 Windows/Linux 复现失败。Parquet 仍按原始二进制字节计算，复用 `searchpilot.search.artifacts.sha256_file`。`labels.csv` 未用于决定等级，仅作为正式入口内容未被修改的保护检查。其余文件用于确定范围、对账、文本或规范。

| 文件 | SHA-256 |
|---|---|
| `data/queries/queries.csv` | `bf19a33e09c81963393d59d9b146b5425fe431a32fc14b7dd955a28f6ee7c49e` |
| `data/queries/labels_tool.csv` | `4270981fcfed99d9c3e29e1695283077699de96a120ba7789aebe7d4e797a8e2` |
| `data/queries/labels_claude.csv` | `3c97c5452c673439cafbdc037d553ba7417b2ec8f92687915ba14953176df9fd` |
| `data/queries/labels.csv` | `115d4fb12e6ca60c5b02638b28c6d699b60e5669e83d17accea088c3bde2c731` |
| `data/processed/d3a904f41240/items.parquet` | `c58e6d4a8c57968454c24b911af7f8aaec82e837c280ae47b4a95e0562e791f9` |
| `docs/data/annotation-guideline.md` | `5dba99bff939653b246e374fde2d691b0b30b1472fca0e68d691f52fc836d8b3` |

每对 `input_sha256` 对以下 JSON 的 UTF-8 字节做 SHA-256：

```python
evidence = {
    "query": query_csv_row,
    "item": {"item_id": item_id, "title": title, "abstract": abstract},
    "tool_label": original_tool_csv_row,
    "claude_label": original_claude_csv_row,
}
json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
```

CSV 在解析前按相同口径统一换行为 LF，包含引号内的多行字段；除此之外保留原始空值和字符串。Parquet 中标题摘要不做文本归一化。规则文件版本由上述文本哈希锁定。当前 11 对的 `input_sha256` 均未改变。

输出 CSV 生成时为 UTF-8、LF 换行；验证已有文件时也先统一换行为 LF，除此之外要求完整字节相同。当前文件原始字节与规范化内容的 SHA-256 都是 `6c16a186df9e9dc4605ad81aedf21ed0f966ca72deb8165b3388adbec09deb37`；换行修复没有改动裁定 CSV。其他 checkout 的 CRLF 原始字节哈希可能不同，但规范化后的输出哈希应相同。

## 复现命令与实测检查

在 `E:\searchPilot`，使用现有 `.venv`：

```powershell
$env:TMP = 'E:\searchPilot\.tmp'
$env:TEMP = $env:TMP
.venv\Scripts\python.exe scripts/adjudicate_train_val.py
.venv\Scripts\python.exe -m ruff check scripts/adjudicate_train_val.py tests/test_adjudicate_train_val.py
.venv\Scripts\python.exe -m ruff format --check scripts/adjudicate_train_val.py tests/test_adjudicate_train_val.py
.venv\Scripts\python.exe -m pytest tests/test_adjudicate_train_val.py -q
```

首次生成独立 CSV 可使用 `scripts/adjudicate_train_val.py --write`。除换行差异外，输入哈希、分歧集合或文件内容改变时，校验会失败；此时需要重新审阅，不能把旧判决静默套用到新输入。

本次实测：11 对 query/item 对应有效、无重复、全属 train/val、等级均在 0–3，11 个理由与逐条哈希完整；Windows 实际工作区的默认只读复现通过。Ruff check 通过，format 检查通过，targeted pytest **16 passed**，含 CSV/Markdown LF/CRLF 等价、文本内容变化拒绝、输出等级变化拒绝、不忽略尾部空格、引号内多行字段，以及二进制换行变化必须拒绝的负向检查。pytest 仅出现已有 FastAPI/Starlette 关于 httpx 的弃用警告，没有失败。没有修改 `src/`，本次未运行全仓测试、mypy、数据库测试或正式评估。

另在 `.tmp` 的独立临时目录构造 LF 输入镜像：只读 `git show HEAD:<path>` 的 queries、labels、labels_claude 和 guideline 四份 blob，它们与工作区规范化内容相同；HEAD 的 labels_tool 仍是旧内容，因此该文件取当前工作区内容统一为 LF。Parquet 使用只读硬链接。该镜像完整重建 11 行，与 Windows 实际输出逐行相同、11 个逐对哈希相同；镜像输出分别为 LF 与 CRLF 时均验证通过。前后再次比较 6 份源输入和裁定 CSV 的原始字节哈希，全部未变。这是换行兼容性模拟，不冒充在 Linux 主机实际执行。

语义等级仍是一次工具判断，尤其 0/1 边界值得所有者复核。可复现指冻结输入与裁定记录按上述换行口径重放，不承诺另一模型或重新采样必然作出同样判断。未验证摘要之外的事实，也未声明人工批准或正式采用。
