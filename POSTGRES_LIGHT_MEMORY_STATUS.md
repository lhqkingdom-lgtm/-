# PostgreSQL 轻量长期记忆 MVP 状态

本文档记录当前 Memory Lab 的实际进度。当前阶段已经从早期“完整审计链/版本链/Reviewer”方案，收敛为更接近 Claude Code 的轻量方案：

```text
CP SQLite 原始会话
  → turn 切分
  → 三层证据池 core / candidate / pointer
  → PostgreSQL 入库
  → Writer 固定输入包
  → 后续接单个 LLM Writer
  → 写入 memory_topics / memories
  → 生成 MEMORY.md + 主题 Markdown
```

当前尚未接入 LLM Writer，也尚未写正式 memory。本阶段已经完成的是基础数据链路和读取工具。

## 当前设计取舍

### 不做的复杂能力

当前明确不做：

- `memory_revisions` 版本链；
- `memory_evidence_links` 严格证据链；
- 独立 Reviewer LLM；
- 每条 memory 的原始 turn/message/part 强绑定；
- LangChain / LangGraph 编排。

原因：

- 先验证“能否从真实 CP turn 稳定沉淀有用记忆”；
- 贴近 Claude Code 的轻量 Markdown memory 体验；
- 降低表结构、回滚、审计和多模型分歧处理成本；
- 后续如发现误存、冲突、回滚需求强，再增量加复杂审计。

### 当前保留的约束

- CP SQLite 只读；
- LLM 后续只能输出 JSON 计划，不能直接写库；
- 程序负责校验计划并写 PostgreSQL；
- 原始敏感配置、本地数据库、导出包和报告不提交 GitHub；
- `turns / evidence_items` 是当前证据池事实源；
- `memory_topics / memories` 是后续长期记忆事实源。

## 已完成能力

### 1. PostgreSQL 专用本地实例

本地 Memory Lab PostgreSQL 连接参数：

```text
host     = localhost
port     = 55432
database = codepulse_memory_lab
user     = root
password = 123456
```

本地配置文件：

```text
pg-memory-lab.local.json
```

该文件已被 `.gitignore` 忽略，不会上传。

### 2. PostgreSQL schema

建表文件：

```text
db/postgres_schema_v1.sql
```

当前保留表：

```text
users
sources
turns
evidence_items
memory_topics
memories
memory_agent_runs
```

memory 轻量关系：

```text
memory_topics 1 ─── N memories
```

`memory_agent_runs` 仅用于后续 Writer 调试记录，不是正式记忆结构。

### 3. 三层证据池

每个 CP turn 解析为：

```text
core      = 本轮判断长期记忆最关键的证据
candidate = 规则清洗后的候选摘要
pointer   = 回查 CP SQLite 原始 part 的指针
```

真实 FanGuard/MJ60B 数据验证结果：

```text
turn_count = 9
core       = 19
candidate  = 174
pointer    = 193
```

质量检查结果：

```text
candidate_clean_turns = 9 / 9
pointer_integrity_ok  = true
sqlite_part_found     = 193 / 193
ready_for_next_chain  = true
```

说明：

- `candidate` 已经不是原始 `read completed {...}`，而是中文摘要；
- `core/candidate` 都带 `raw_pointer_id`；
- `pointer` 能回查到 CP SQLite 原始 `part.data`。

### 4. candidate 中文摘要清洗

示例：原始工具输出曾类似：

```text
read completed {"filePath":"<WORKSPACE>/adc_hw.c"} ...
```

现在 candidate 摘要为：

```text
读取文件 adc_hw.c：发现函数：adc_sample；命中重点：固定返回 2048（return 2048）、硬件桩/人工必须替换、涉及 ADC 采样、涉及 NTC 温度输入。
```

同时保留：

```text
raw_pointer_id = ptr_38b83dc47f0554d2
```

可继续回查原始 CP SQLite part。

### 5. Writer 读取工具

已实现并接入 CLI 的读取工具：

```text
pg-writer-context
pg-expand-pointer
pg-read-memory-detail
```

这些命令只是为了人工和开发调试。后续 Writer 流程会直接调用 Python 函数，不会让 LLM 执行命令行。

#### `pg-writer-context`

作用：组装 Writer 第一轮固定输入。

固定包含：

```text
turn_evidence = 当前 turn 的 core + candidate
memory_index  = 当前用户已有 memory_topics + memories 摘要
available_tools = expand_pointer, read_memory_detail
```

示例命令：

```powershell
.\memory-lab.ps1 pg-writer-context `
  -SourceDatabase 'C:\Users\19873\.local\share\codepulse\opencode-dev.db' `
  -LabDatabase 'data\fanguard_domain_auto_rerun_20260911.db' `
  -PgConfigPath 'pg-memory-lab.local.json' `
  -TurnId 'turn_e90cab8f66831ddd470536aae9e6f8b940d2faf01c0db74878705db0db20745e'
```

真实测试结果：

```text
core_count      = 2
candidate_count = 6
memory_topics   = 0
memory_items    = 0
available_tools = expand_pointer, read_memory_detail
```

#### `pg-expand-pointer`

作用：根据 `raw_pointer_id` 回查 CP SQLite 原始 part。

示例命令：

```powershell
.\memory-lab.ps1 pg-expand-pointer `
  -SourceDatabase 'C:\Users\19873\.local\share\codepulse\opencode-dev.db' `
  -LabDatabase 'data\fanguard_domain_auto_rerun_20260911.db' `
  -PgConfigPath 'pg-memory-lab.local.json' `
  -RawPointerId 'ptr_38b83dc47f0554d2' `
  -MaxChars 500
```

真实测试结果：

```text
status     = complete
part_id    = prt_08edf5757001nhEFNiM0TMcK61
source_type = tool_result
kind       = build_or_tool_result
truncated  = true
```

无效 pointer 返回：

```json
{
  "ok": false,
  "error": "pointer not found: ptr_not_exists"
}
```

#### `pg-read-memory-detail`

作用：读取已有 memory 的完整正文和所属 topic。

由于当前正式 memory 表为空，已用临时测试数据验证后删除。测试结果：

```text
memory_id  = mem_test_adc_stub
topic_key  = engineering_embedded_test
summary    = ADC stub fixed return can make runtime input constant
content    = Temporary memory used to test read_memory_detail...
confidence = 0.9
```

无效 memory 返回：

```json
{
  "ok": false,
  "error": "memory not found: mem_not_exists"
}
```

测试结束后确认：

```text
memory_topics = 0
memories      = 0
```

## 已修改/新增的关键文件

```text
db/postgres_schema_v1.sql
src/memory_lab/pg_store.py
src/memory_lab/evidence_pool.py
src/memory_lab/turn_evidence.py
src/memory_lab/cli.py
memory-lab.ps1
tests/test_evidence_pool.py
.gitignore
```

## 未完成能力

### 1. LLM Writer 尚未接入

还没有实现：

```text
turn_evidence + memory_index
  → DeepSeek Writer
  → create / merge / update / conflict / noop / need_more_evidence
```

### 2. 计划提交工具尚未实现

还没有实现：

```text
create_topic_if_needed
create_memory
merge_memory
update_memory
mark_conflict
noop
```

### 3. Markdown 投影尚未切到轻量 PostgreSQL 结构

还没有实现从：

```text
memory_topics / memories
```

生成：

```text
memory/
├── MEMORY.md
├── user_preferences.md
├── engineering_embedded.md
└── ...
```

### 4. 与 CP 集成尚未开始

当前不注入 CP、不改 CP 源码、不实现 sidecar API。

未来可能接口：

```text
POST /v1/recall
POST /v1/turns/notify
GET  /v1/health
```

## 下一步建议

下一步建议先做单 Writer 最小闭环：

```text
1. 定义 Writer 输出 JSON schema
2. 接 DeepSeek Writer
3. 支持 need_more_evidence：
   - expand_pointer
   - read_memory_detail
4. 支持最终计划：
   - create
   - merge
   - update
   - conflict
   - noop
5. 程序校验 JSON
6. 写入 memory_topics / memories
7. 再生成 Markdown 投影
```

第一条真实测试建议继续使用：

```text
turn_e90... 烧录后风扇不随温度变化，adc_sample 固定 return 2048
```

预期可沉淀：

```text
工程经验：硬件抽象桩函数固定返回值会导致编译通过但运行输入恒定；编译通过不代表硬件相关功能可用。
```

## 当前验收结论

截至当前状态：

```text
三层证据池链路：通过
candidate 中文清洗：通过
pointer 原始回查：通过
轻量 memory 表：已建
Writer 读取工具：通过
LLM Writer：未开始
正式 memory 写入：未开始
Markdown 投影：未开始
```

