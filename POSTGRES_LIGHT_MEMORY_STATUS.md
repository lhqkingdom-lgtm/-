# PostgreSQL 轻量长期记忆 MVP 状态

## 2026-09-15 当前最新状态

当前 Memory Lab 已经从“只读证据池”推进到“可独立运行的长期记忆沉淀 MVP”：

```text
CP SQLite 原始会话
  → turn 切分
  → 三层证据池 core / candidate / pointer
  → PostgreSQL 入库
  → Memory Agent 读取 turn 证据、主题索引、已有记忆摘要
  → DeepSeek 输出 memory-changes-v1
  → 程序校验 JSON、证据引用、scope、target、stable_key
  → commit_memory_changes 安全提交
  → 每轮生成结构化 JSON 日志
  → 支持多轮 batch dry-run / commit 观察
```

当前重点目标已经明确为三类长期记忆：

```text
1. 用户级协作习惯和偏好
2. 编译/构建失败或成功的原因、分析、解决方法、验证方式
3. 用户反馈运行现象后，定位到代码层面的通用原因、解决方法、验证方式
```

项目级记忆也支持，但只用于“以后回到该项目必须知道，且无法从代码稳定恢复”的隐性约束；当前项目的一次性需求参数默认不沉淀。

## 当前已实现能力

### 1. 三层证据池与 PostgreSQL 入库

已完成：

```text
pg-ingest-turn
pg-show-turn
pg-quality-report
```

每个 CP turn 会拆成：

```text
core      = 本轮最关键证据
candidate = 清洗后的候选摘要
pointer   = 可回查 CP SQLite 原始 part 的指针
```

真实 FanGuard/MJ60B 数据曾验证：

```text
turn_count = 9
core       = 19
candidate  = 174
pointer    = 193
pointer_integrity_ok = true
```

### 2. Agent 读取工具

已完成：

```text
pg-get-turn-evidence
pg-list-memory-topics
pg-read-topic-detail
pg-read-memory-detail
pg-expand-pointer
pg-writer-context
```

用途：

```text
get_turn_evidence   读取指定 turn 的 core/candidate/pointer
list_memory_topics  只读主题索引，不读正文
read_topic_detail   读取某个主题下的 memories
read_memory_detail  读取单条 memory
expand_pointer      根据 pointer 回查原始 CP SQLite part
writer_context      构造早期 Writer 输入包，保留用于调试
```

已验收边界：

```text
turn 不存在 → found=false，不崩流程
topic 不存在 → found=false
memory 不存在 → 安全错误
pointer 不存在 → 安全错误
内容过长 → 截断并返回 truncated 标记
```

### 3. 单条 Writer plan 链路

已完成：

```text
pg-writer-dry-run
pg-commit-plan
pg-writer-commit
```

能力：

```text
turn_id
  → DeepSeek 输出 memory-writer-plan-v1
  → 程序校验
  → create / merge / update / conflict / noop / need_more_evidence
  → 安全写入 PostgreSQL
```

该链路保留用于最小单计划测试，但后续主线已经升级为 `memory-changes-v1` 多变更链路。

### 4. 多变更 changes[] 校验和提交

已完成：

```text
pg-validate-memory-changes
pg-commit-memory-changes
```

支持 action：

```text
create_memory
merge_memory
update_memory
mark_conflict
noop
request_more_evidence
```

防幻觉边界：

```text
非法 JSON 拒绝
未知字段拒绝
伪造 evidence_id / raw_pointer_id 拒绝
伪造 memory_id / topic_id 拒绝
scope 非法拒绝
stable_key 重复 create 拒绝
confidence 越界拒绝
空正文拒绝
同一 turn 重复处理同一 memory 拒绝
conflict memory 默认不能作为 active target
```

每条 change 独立校验、独立事务提交：

```text
一条坏 change 不影响其他合法 change
noop / request_more_evidence 不写库
```

写入 memory metadata 的轻量来源：

```json
{
  "source_turn_id": "...",
  "evidence_refs": [...],
  "change_reason": "...",
  "last_action": "..."
}
```

### 5. Memory Agent 主链路

已完成：

```text
pg-memory-agent-run
```

链路：

```text
turn_id
  → get_turn_evidence
  → list_memory_topics
  → build memory_index
  → DeepSeek 输出 memory-changes-v1
  → validate_memory_changes
  → commit_memory_changes
  → 写结构化日志
```

支持：

```powershell
.\memory-lab.ps1 pg-memory-agent-run -TurnId <turn_id> -DryRun
.\memory-lab.ps1 pg-memory-agent-run -TurnId <turn_id>
```

已验收：

```text
不存在 turn：失败在 input 阶段，不调用模型、不写库
真实 turn dry-run：输出 changes[] 并校验通过
真实 turn commit：可 merge/create
同一 turn 重跑：程序边界拒绝重复写同一 memory
```

### 6. 多轮 batch 观察

已完成：

```text
pg-memory-agent-batch
```

支持：

```powershell
.\memory-lab.ps1 pg-memory-agent-batch -Limit 10 -DryRun
.\memory-lab.ps1 pg-memory-agent-batch -TurnIds <turn1>,<turn2> -DryRun
```

每轮独立执行，失败不影响下一轮。

已验收：

```text
Limit 2 dry-run:
turns = 2
complete = 2
failed = 0
dry_run_changes = 3
```

### 7. 结构化日志

已完成：

```text
reports/memory-agent-runs/<run_id>.json
reports/memory-agent-runs/latest.json
reports/memory-agent-runs/<batch_id>.json
reports/memory-agent-runs/latest-batch.json
```

日志记录：

```text
get_turn_evidence       core/candidate/pointer 数量
list_memory_topics      主题数量
build_memory_index      memory 数量
model_writer            模型是否返回 JSON
validate_changes        changes 数量、是否修复
commit_changes          applied/rejected/noop/request 统计
plan                    模型输出的 changes[]
commit                  程序提交结果
```

### 8. 当前提示词策略

已优化 Memory Agent prompt：

```text
优先沉淀用户级协作习惯
工程经验必须抽象为可复用模式
项目参数默认 noop
项目隐性约束用 project scope
运行现象经验按“症状/触发条件/排查路径/代码层根因/解决方法/验证方式/复用范围”组织
同症状不同根因不是冲突，应新增另一条 failure_pattern
同 scope、同条件、同版本下无法解释的相反结论才 mark_conflict
```

最近真实 dry-run 结果显示，模型开始把项目细节抽象为：

```text
目标芯片外设驱动不能套用模板寄存器定义；
必须按目标芯片数据手册确认外设寄存器、引脚复用和时钟源；
替换桩函数后要全量 rebuild，并结合烧录实测验证。
```

## 已验证的功能点

### 新增

受控测试通过：

```text
用户级协作习惯 → create_memory / scope=global / memory_type=collaboration_preference
项目级隐性约束 → create_memory / scope=project / memory_type=project_constraint
同症状不同根因 → create_memory / scope=domain embedded / memory_type=runtime_failure_pattern
```

### 合并

真实 turn 验证：

```text
已有 ADC 桩函数失败模式
  + 新的修复闭环和验证证据
  → merge_memory
```

### 冲突

受控测试通过：

```text
同一 turn 重复处理同一 memory → 拒绝
不同 turn 且 evidence 属于对应 turn → mark_conflict 可提交
伪造 evidence_id → 拒绝
```

### 跳过 / 不写库

已支持：

```text
noop
request_more_evidence
dry-run 模式
```

## 当前未实现能力

### 1. 自动补证据闭环

目前模型可以输出：

```text
request_more_evidence
```

但程序尚未自动执行：

```text
expand_pointer / read_memory_detail / read_topic_detail
  → 追加证据
  → 重新调用 Writer
```

### 2. Markdown 正式投影

尚未将 PostgreSQL 当前记忆投影成：

```text
memory/
├── MEMORY.md
├── user_preferences.md
├── engineering_embedded.md
└── ...
```

### 3. 中文 Markdown/HTML 报告

目前已有结构化 JSON 日志，但尚未生成展示用报告：

```text
单轮变化报告
多轮 batch 变化报告
```

### 4. Curator 主题整理

尚未实现低频整理工具：

```text
split_topic
merge_topics
move_memory
rename_topic
prune_stale_memory
```

当前只支持实时链路中的 create/merge/update/conflict/noop/request。

### 5. 更完整矛盾处理策略

已有实时防线：

```text
伪造 target/evidence 拒绝
同 turn 重复写拒绝
conflict 可标记
同症状不同根因提示词要求 create 而非 conflict
```

尚未实现：

```text
自动识别用户契约的“项目例外”与“全局冲突”
工程经验 verified / unverified 状态
同 scope 反向用户要求按最新长期要求自动 update
冲突解决历史
```

### 6. 召回链路

尚未实现：

```text
pg-memory-search
pg-memory-render
```

当前重点仍是沉淀质量，不急于接召回。

### 7. CP 接入

当前仍然不修改 CP 源码、不注入 CP Agent。

未来保留接口方向：

```text
POST /v1/recall
POST /v1/turns/notify
GET  /v1/health
```

未来 CP 插件/sidecar 接入点：

```text
chat.message
experimental.chat.system.transform
session.idle
```

## 下一阶段测试计划

### 1. 多轮 dry-run 质量测试

命令：

```powershell
.\memory-lab.ps1 pg-memory-agent-batch -Limit 10 -DryRun
```

检查：

```text
需求参数是否 noop
用户习惯是否 global collaboration_preference
编译/构建问题是否 build_failure_pattern/build_success_pattern
运行现象是否 runtime_failure_pattern
同症状不同根因是否 create 而非 conflict
同一 turn 重跑是否不重复写
```

### 2. 小批量 commit 测试

选择 3～5 个高价值 turn：

```powershell
.\memory-lab.ps1 pg-memory-agent-batch -TurnIds <turn1>,<turn2>,<turn3>
```

检查：

```text
memory 数量变化合理
metadata 记录 source_turn_id/evidence_refs
无重复 create
无项目细节误沉淀为通用经验
```

### 3. 冲突/矛盾专项测试

构造或选择真实 turn：

```text
用户全局偏好反向
项目例外
同症状不同根因
同 scope 同条件相反工程结论
伪造 evidence/target
```

检查：

```text
update / project scope / create separate pattern / mark_conflict / reject
```

### 4. 长期开放测试

目标：

```text
至少 3 个真实项目
至少 20～30 个 turn
持续多天 dry-run + 小批量 commit
每周检查误存、漏存、重复、范围污染、冲突处理
```

## 当前验收结论

截至 2026-09-15：

```text
三层证据池链路：通过
PostgreSQL 入库：通过
读取工具：通过
DeepSeek 单计划 Writer：通过
多变更 changes[] schema：通过
多变更安全提交：通过
Memory Agent 单轮 run：通过
Memory Agent batch dry-run：通过
结构化日志：通过
提示词抽象策略：初步通过
Markdown 投影：未实现
补证据闭环：未实现
报告生成：未实现
Curator 主题拆分/合并：未实现
召回链路：未实现
CP 接入：未实现
```

---

## 历史记录：早期 PostgreSQL 轻量长期记忆 MVP 状态

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
port     = 55512
database = codepulse_memory_lab
user     = root
password = 本机测试密码，保存在 pg-memory-lab.local.json
```

本地配置文件：

```text
pg-memory-lab.local.json
```

该文件已被 `.gitignore` 忽略，不会上传。

端口说明：

- 这套数据目录位于 `D:\codepulse\Codepulse-Memory-Lab\pgdata`；
- 历史配置曾使用 `55432`；
- 当前 Windows 将 `55432` 放入 TCP 排除端口段 `55412–55511`，PostgreSQL 直接绑定会报 `could not bind IPv4 address "127.0.0.1": Permission denied`；
- 因此当前本机使用同一套 `pgdata` 临时启动到 `55512`。

启动命令：

```powershell
& D:\PostgreSQL\17\bin\pg_ctl.exe start `
  -D D:\codepulse\Codepulse-Memory-Lab\pgdata `
  -o "-p 55512 -h localhost" `
  -l D:\codepulse\Codepulse-Memory-Lab\pgdata\memory-lab-postgres-55512.log
```

确认命令：

```powershell
netstat -ano | Select-String ':55512'
& D:\PostgreSQL\17\bin\psql.exe -h localhost -p 55512 -U root -d codepulse_memory_lab -c "\dt"
```

当前已确认该实例包含：

```text
turns = 9
evidence_items = 386
memory_topics = 1
memories = 1
```

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

当前路线已从“一次性 Writer”调整为 ReAct 长期记忆维护器：

```text
turn 三层证据
  ↓
已有 memory/topic 摘要索引
  ↓
模型判断是否需要补证据
  ↓
只读工具 read_memory_detail / read_topic_detail / expand_pointer
  ↓
模型输出 memory-changes-v1 计划
  ↓
程序校验并写入 PostgreSQL
```

新的近期计划：

```text
R1：重写模型提示词，明确长期记忆整理流程、沉淀标准、动作选择标准和 JSON 示例。
R2：优化 payload，显式区分 summary_only_memory_ids、observed_memory_ids、modifiable_memory_ids。
R3：保留硬限制，但对“未读详情直接修改已有 memory”优先自动补 read_memory_detail 后重判。
R4：用真实 turn dry-run 验收 noop / update / create / merge。
R5：dry-run 稳定后再 commit 验收。
R6：多 turn 连续跑，观察重复、合并、更新、范围污染。
R7：生成中文变更报告和可视化。
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
ReAct 读取工具：通过
ReAct 提示词 R1：已重写
ReAct payload R2：已加入 memory_access_state / action_decision_guide
ReAct dry-run：已通过 1 个真实运行现象 turn
正式 memory 写入：提交器可用，但本阶段暂以 dry-run 验证为主
Markdown 投影：未开始
```

最新真实 turn dry-run 验收：

```text
turn_id = turn_e90cab8f66831ddd470536aae9e6f8b940d2faf01c0db74878705db0db20745e
输入类型 = 用户反馈运行现象 + 代码层根因定位
结果 = ok=true
iterations = 2
observation_count = 1
流程 = 第 1 轮 read_memory_detail(mem_ab2f188c06b6481e)，第 2 轮 final_changes/noop
结论 = 已有 memory 覆盖本轮“桩函数固定返回值导致假数据、外设无响应”的失败模式，因此不重复写入。
```

R4 多类型 dry-run 验收：

```text
1. turn_b87025692cfaccf32d9af2b4cb5b077dd227c6d5f793262e77acaeab206d7296
   类型 = 既有 Keil C51 工程代码生成 + 编译成功
   结果 = ok=true
   iterations = 2
   observation_count = 2
   计划 = create_memory 用户偏好：既有 Keil C51 工程不重建结构、不改 STARTUP.A51、优先 C51 兼容写法、交付时说明模块划分与需人工确认硬件资源；同时对已有桩函数记忆 noop。

2. turn_0e6c9aa67c3a0776783748acfe5838a3e9e0f1a59e1ecc60d982d95ddfb442f6
   类型 = 用户确认修复成功
   结果 = ok=true
   iterations = 2
   observation_count = 1
   计划 = merge_memory 到 mem_ab2f188c06b6481e，补充“替换真实 ADC 驱动后，经用户烧录实测验证输入变化、占空比切换和失效安全有效”的验证证据。

3. turn_d8f25f9e4ae0d1f72838f3416fc47f2a47b5935e852866eb1ed132f512450f93
   类型 = 需求对齐
   结果 = ok=true
   iterations = 1
   observation_count = 0
   计划 = noop；温度阈值、滞回、PWM 频率、滤波、看门狗等属于当前项目需求参数，可由需求文档恢复，不沉淀为长期记忆。

4. turn_ae03bf8b31152f404c805fe70821d6e50707fe0517e84a0a8da7f9d7af6b59fc
   类型 = 真实驱动替换 + Keil 编译成功
   结果 = ok=true
   iterations = 2
   observation_count = 3
   计划 = update_memory 到 mem_ab2f188c06b6481e，把已有“桩函数假数据”经验扩展为“桩函数或不匹配外设寄存器/模块都会导致编译通过但运行无响应”，并补充 Keil 0 errors / 0 warnings 编译验证。

5. turn_64bbc76b5ecdea714511940a63629fe45357b2917a67af60000e42a3e853c88a
   类型 = Keil C51 一次编译通过确认
   结果 = ok=true
   iterations = 1
   observation_count = 0
   计划 = create_memory 编译成功经验：Keil C51 编译前做 storage modifier、code/const、ISR、static 函数定义顺序等自检，可减少编译返工。
```

程序级边界验收：

```text
update_without_read_should_fail = PASS
  未通过 read_memory_detail/read_topic_detail 读取目标 memory 前，update_memory 会被拦截。

update_after_read_should_pass = PASS
  读取过目标 memory 后，update_memory schema 可通过。

conflict_after_read_should_pass_schema = PASS
  读取过目标 memory 后，mark_conflict schema 可通过；真实业务冲突样本尚未出现，后续需用长期测试补足。
```
