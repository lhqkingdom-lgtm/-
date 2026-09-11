# Memory Lab LangGraph + PostgreSQL 重构计划

## 目标

将当前独立实验版 Memory Lab 重构为更清晰的长期记忆维护器：

```text
CP SQLite 只读输入
  → 三层证据池
  → LangGraph 编排语义处理
  → PostgreSQL 存证据、记忆、修订和审计
  → Markdown 生成可读投影
```

核心原则：

- CP SQLite 不修改，只作为原始会话输入源。
- PostgreSQL 作为事实源，保存 turn、证据、记忆、修订、冲突和主题。
- Markdown 只是投影，给人和未来 Agent 阅读，不保存完整原始证据。
- LangGraph 只做流程编排，不直接随意写库。
- DeepSeek 只输出结构化 JSON 判断或计划；最终提交由程序校验后执行。
- 当前实验阶段固定 `user_id = local-default-user`，数据库和配置预留未来真实用户主键。

## 总体数据流

```text
CP SQLite 原始会话
  ↓
Source Adapter
  ↓
Turn Builder
  ↓
Evidence Pool Builder
  ↓
PostgreSQL 证据层
  ↓
LangGraph Workflow
  ↓
PostgreSQL 正式记忆层
  ↓
Markdown Projection
```

## 阶段说明

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| Source Adapter | CP SQLite | 只读读取 message / part / tool / answer | 原始会话对象 |
| Turn Builder | 原始会话对象 | 按用户消息和稳定状态切一轮任务 | turn |
| Evidence Pool Builder | turn | 脱敏、裁剪、加 ref、分层 | core / candidate / pointer |
| PostgreSQL 证据层 | 三层证据 | 持久化、去重、可审计 | turns / evidence_items |
| Phase Router | core + candidate | 判断真实阶段，可修正规则初判 | final_phase |
| Evidence Request | 当前证据 | 判断是否缺证据 | evidence_requests |
| Evidence Fetcher | pointer | 回查局部原文或工具结果 | supplemental_evidence |
| Memory Writer | 证据 + 阶段 + 旧记忆 | 生成记忆更新计划 | update_plan |
| Reviewer | update_plan + 证据 | 审核、修订或拒绝 | review_decision |
| Committer | 审核通过计划 | 事务写入正式记忆 | memories / revisions / links |
| Markdown Projection | 正式记忆 | 生成索引和主题文件 | MEMORY.md / topic.md |

## 三层证据池

### core_evidence

直接给大模型看的关键证据。

应该包含：

- 用户关键原话；
- 助手最终结论；
- 真实编译结果；
- 用户确认修复成功。

不应该包含：

- 完整文件内容；
- 普通 `read/edit/write/glob/grep` 工具结果；
- 大段日志；
- 当前项目普通需求选项。

### candidate_evidence

可能有用但不直接展开的摘要。

应该包含：

- 文件读取摘要；
- 代码修改摘要；
- 中间排查过程；
- 具体需求选项；
- 长工具输出摘要。

后续模型可通过固定格式请求展开。

### raw_pointers

本地回查指针，不把正文直接给模型。

应该包含：

- `message_id`
- `part_id`
- `source_type`
- `summary`
- 顺序信息

用于补证据和审计。

## LangGraph 工作流

建议节点：

```text
scan_turn
  ↓
build_evidence_pool
  ↓
route_phase
  ↓
need_more_evidence?
  ├─ yes → fetch_more_evidence → route_phase
  └─ no
       ↓
write_memory_plan
  ↓
review_memory_plan
  ↓
commit_memory
  ↓
publish_markdown
```

状态对象建议：

```python
class MemoryState(TypedDict):
    user_id: str
    source_id: str
    turn_id: str
    project_key: str | None
    rule_phase: str | None
    final_phase: str | None
    evidence_pool: dict
    supplemental_evidence: list[dict]
    evidence_requests: list[dict]
    memory_plan: dict | None
    review_decision: dict | None
    commit_result: dict | None
```

## PostgreSQL 建议表

### users

当前先固定一个用户：

```text
user_id = local-default-user
```

后续可接邮箱、CP account id 或 UUID。

字段建议：

- `user_id text primary key`
- `email text unique null`
- `display_name text null`
- `created_at timestamptz`

### sources

记录 CP SQLite 数据源。

- `source_id`
- `user_id`
- `source_kind`
- `database_fingerprint`
- `created_at`

### turns

保存一轮任务。

- `turn_id`
- `user_id`
- `source_id`
- `session_id`
- `project_key`
- `state`
- `occurred_at`
- `rule_phase`
- `final_phase`
- `source_hash`

### evidence_items

保存三层证据。

- `evidence_id`
- `user_id`
- `turn_id`
- `layer`: `core / candidate / pointer / supplemental`
- `source_type`: `user_message / user_answer / assistant_final / tool_result`
- `kind`
- `text_or_summary`
- `raw_pointer_id`
- `message_id`
- `part_id`
- `ordinal`
- `metadata jsonb`

### memories

正式长期记忆。

- `memory_id`
- `user_id`
- `stable_key`
- `type`
- `scope jsonb`
- `title`
- `summary`
- `content jsonb`
- `status`
- `trust_status`
- `verification_status`
- `topic_id`

### memory_revisions

记忆修订链。

- `revision_id`
- `memory_id`
- `user_id`
- `action`
- `before jsonb`
- `after jsonb`
- `reason`
- `writer_plan jsonb`
- `review_decision jsonb`
- `created_at`

### memory_evidence_links

正式记忆和证据的关系。

- `memory_id`
- `evidence_id`
- `quote`
- `role`
- `created_at`

### topics

Markdown 主题归属。

- `topic_id`
- `user_id`
- `name`
- `description`
- `type`
- `file_name`
- `status`

### conflicts

未解决冲突。

- `conflict_id`
- `user_id`
- `stable_key`
- `scope jsonb`
- `status`
- `details jsonb`
- `created_at`
- `resolved_at`

## 记忆类型

当前主要沉淀三类：

1. 用户交互习惯
   - 用户希望 CP 怎么工作；
   - 展示偏好；
   - 确认流程；
   - 不喜欢的工作方式。

2. 编译成功/失败经验
   - 编译错误类型；
   - 出现条件；
   - 根因；
   - 有效修复；
   - 编译是否验证通过。

3. 用户反馈现象后的定位经验
   - 运行现象；
   - 触发条件；
   - 排查路径；
   - 根因；
   - 解决方案；
   - 用户或工具验证结果。

## 分阶段实施

### R0：PostgreSQL 骨架

目标：

- 新增配置文件；
- 建 schema；
- 固定 `user_id = local-default-user`；
- 能插入测试 turn / evidence / memory。

验收：

- PostgreSQL 能看到基础表；
- secrets 不进入仓库。

### R1：CP SQLite → turns

目标：

- 只读扫描 CP SQLite；
- 按一轮任务切 turn；
- 写入 PostgreSQL `turns`。

验收：

- 重复扫描不重复写；
- session / project / turn 关系正确。

### R2：三层证据入库

目标：

- 生成 core / candidate / pointer；
- 写入 `evidence_items`；
- 保留 pointer 回查能力。

验收：

- core 精准；
- candidate 不漏关键候选；
- pointer 覆盖原始 part；
- 普通 read/edit 不进 core。

### R3：LangGraph 阶段路由

目标：

- 用 LangGraph 跑 Phase Router；
- DeepSeek 输出 `final_phase`；
- 写回 PostgreSQL。

验收：

- 真实例子中 `compilation → code_generation` 能被改判；
- 烧录现象能保持 `runtime_symptom_diagnosis`；
- 用户确认能保持 `fix_confirmation`。

### R4：补证据闭环

目标：

- 大模型可请求补证据；
- 程序根据 pointer 回查；
- 最多补充有限轮；
- 重新路由或抽取。

验收：

- 模型不能请求全库；
- 补证据请求类型受限；
- 补充内容可追溯。

### R5：Writer / Reviewer / Commit

目标：

- Writer 生成 JSON 更新计划；
- Reviewer 独立审核；
- 程序校验后写 PostgreSQL。

验收：

- 无证据不写入；
- verified 必须有用户确认或工具验证；
- 同义内容合并；
- 项目例外不污染全局。

### R6：Markdown 投影

目标：

- 从 PostgreSQL 生成 `MEMORY.md` 和主题 Markdown。

验收：

- Markdown 不含完整原始证据；
- `MEMORY.md` 只做索引；
- 主题文件包含当前有效记忆、范围、验证状态和证据数量。

## 当前已完成的实验能力

- CP SQLite 只读增量扫描；
- turn 检测；
- 三层证据池第一版；
- `evidence-pools` CLI；
- DeepSeek 阶段改判；
- 真实 FanGuard/MJ60B 对话验证；
- 边界规则：
  - 普通 `read/edit/write/glob/grep` 不进 core；
  - 需求选项进 candidate；
  - 编译结果进 core；
  - 用户现象和用户确认进 core。

## 当前暂未完成

- PostgreSQL 版事实源；
- LangGraph 工作流；
- 模型补证据回查；
- 证据池 ref 到正式 PostgreSQL evidence link 的闭环；
- 正式 Markdown 从 PostgreSQL 投影。
