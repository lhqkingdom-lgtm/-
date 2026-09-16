# Memory Lab 架构设计文档

本文从头到尾梳理当前 Memory Lab 的架构、数据流、关键设计选择，以及为什么这样设计。

## 1. 项目目标

Memory Lab 的目标是为 CodePulse / CP 对话建立长期记忆系统，让系统能从真实开发长流程中沉淀：

1. 用户交互习惯；
2. 编译失败/成功经验；
3. 用户反馈现象后定位出的原因、解决方法和验证结果。

一句话目标：

```text
需求文档存“当前项目最终要做什么”；
长期记忆存“以后 CP 怎么少问、少错、少返工”。
```

## 2. 当前核心架构

当前链路：

```text
CP SQLite 原始会话
  ↓
turn 切分
  ↓
三层证据池 core / candidate / pointer
  ↓
PostgreSQL 入库
  ↓
ReAct 记忆维护 Agent
  ↓
读工具补证据
  ↓
memory-changes-v1 结构化计划
  ↓
本地校验 / 幂等 / 写库
  ↓
PostgreSQL 当前记忆表
  ↓
Markdown 导出审查
```

## 3. 为什么从 CP SQLite 开始

CP 当前真实会话数据存储在本机 SQLite 中，路径类似：

```text
C:\Users\<user>\.local\share\codepulse\opencode-dev.db
```

选择 SQLite 作为原始数据来源的原因：

1. 它是 CP 的真实运行数据，不是人工构造样例；
2. 能覆盖真实长流程，包括需求对齐、代码生成、编译修复、运行现象反馈；
3. 可回放，适合反复测试记忆系统；
4. 不需要先改 CP 主程序，就能验证记忆链路。

但 SQLite 不适合作为长期记忆主库：

- 原始会话结构不是为长期记忆设计；
- 不适合存用户主键、主题、冲突、证据池；
- 不适合做多轮维护和审查。

所以采用：

```text
SQLite 只作为原始证据来源；
PostgreSQL 作为 Memory Lab 的结构化工作库。
```

## 4. 为什么按 turn 切分

长期记忆抽取的基本单位不是单条 message，而是一轮任务 turn。

一轮 turn 中可能包含：

```text
user 反馈
assistant 诊断
tool 读文件 / 编译 / 修改代码
assistant 最终结论
用户确认
```

如果只看单条 message，容易丢失：

- 问题是否真的修复；
- 诊断是否只是猜测；
- 编译是否通过；
- 用户是否确认；
- 工具输出对应的是哪次修改。

所以当前以 turn 为单位处理：

```text
原始对话 → turn → 一轮证据包 → 一次记忆维护判断
```

## 5. 三层证据池设计

每个 turn 入库前会拆成三层：

### 5.1 core

核心证据，通常包括：

- 用户明确偏好；
- 用户明确反馈；
- 用户确认修复；
- 助手最终稳定结论。

作用：

```text
给模型最关键、最短的信息。
```

### 5.2 candidate

候选证据，通常包括：

- 工具结果摘要；
- 编译结果；
- 诊断过程摘要；
- 可能有长期价值但需要判断的信息。

作用：

```text
让模型看到更多上下文，但不强行认定为长期记忆。
```

### 5.3 pointer

原始指针，保存：

- 原始 message_id；
- part_id；
- source_type；
- 原始片段摘要；
- raw_pointer_id。

作用：

```text
当 core/candidate 不够时，让模型通过 expand_pointer 回查原始 CP 片段。
```

### 5.4 为什么不直接把全部原文给大模型

因为：

1. token 成本高；
2. 编译日志、工具输出很长；
3. 原文里有大量一次性过程信息；
4. 长期记忆只需要“可复用结论”，不是完整聊天记录；
5. 后续模型可通过 pointer 按需补证据。

因此采用：

```text
先给摘要和关键证据；
信息不足再回查 pointer。
```

## 6. PostgreSQL 表设计

当前核心表：

```text
users
sources
turns
evidence_items
memory_topics
memories
memory_conflict_records
memory_agent_runs
```

### 6.1 users

当前 MVP 使用固定用户：

```text
local-default-user
```

设计上预留了 user_id，以后可替换为账号、邮箱或内部用户 ID。

### 6.2 turns

存一轮 CP 对话任务。

关键字段：

```text
turn_id
user_id
source_id
project_key
rule_phase
occurred_at
metadata
```

### 6.3 evidence_items

存三层证据。

关键字段：

```text
evidence_id
turn_id
layer
source_type
kind
text_or_summary
raw_pointer_id
metadata
```

### 6.4 memory_topics

存主题。

例如：

```text
回复语言偏好
Keil C51 编译前自检清单
阈值附近输出自激振荡需时间滤波
```

主题的作用是方便展示、分组、召回。

### 6.5 memories

存正式长期记忆。

关键字段：

```text
memory_id
stable_key
memory_type
scope
summary
content
confidence
metadata
```

### 6.6 memory_conflict_records

存冲突处理过程。

用于记录：

```text
old_claim
new_claim
decision
reason
evidence_refs
```

例如：

```text
旧说法：用户希望默认中文回复
新说法：用户希望默认英文回复
处理：采用更新的明确用户偏好
```

## 7. 为什么用 PostgreSQL

选择 PostgreSQL 的原因：

1. 支持结构化表和 JSONB；
2. 适合存 topic/memory/conflict/evidence；
3. 方便 DBeaver 人工查看；
4. 后续可接入服务端；
5. 比 SQLite 更适合多用户、多任务、多索引场景。

MVP 中暂时没有做复杂版本链。

原因：

```text
先打通“沉淀链路”和“可解释更新”，版本链过早引入会增加维护成本。
```

当前用：

```text
current memory + metadata + conflict_records
```

来满足演示和 MVP 需求。

## 8. ReAct 记忆维护 Agent

当前 Agent 流程：

```text
输入：turn 证据 + memory index
  ↓
模型判断是否信息足够
  ↓
如不足，调用 read_topic_detail / read_memory_detail / expand_pointer
  ↓
模型输出 memory-changes-v1
  ↓
程序校验并写库
```

### 8.1 为什么使用 ReAct

直接一次性让模型输出最终 memory 有几个问题：

- 可能只看摘要就误判；
- 可能不知道已有 memory 正文；
- 可能重复 create；
- 可能无法判断是否冲突。

ReAct 让模型能：

```text
先查，再判断，再提交。
```

### 8.2 为什么模型不能直接写库

因为模型输出不可信，可能：

- 字段多/少；
- ID 编造；
- 引用错误 evidence；
- 误改 topic_key；
- 重复写入；
- 同一 turn 重跑污染数据。

所以采用：

```text
模型只输出计划；
本地代码做 schema 校验、引用校验、幂等校验、落库。
```

## 9. memory-changes-v1 计划格式

写入计划统一为：

```json
{
  "schema_version": "memory-changes-v1",
  "turn_id": "turn_xxx",
  "reason": "整体判断",
  "changes": []
}
```

当前支持 action：

```text
create_memory
update_memory
merge_memory
mark_conflict
update_topic
record_conflict_decision
noop
request_more_evidence
```

这种设计的好处：

1. 一个 turn 可以同时做多个动作；
2. 例如用户偏好反转时可同时：
   ```text
   update_memory
   update_topic
   record_conflict_decision
   ```
3. 本地校验器可以统一处理 schema、幂等和错误返回。

## 10. Markdown 导出

当前导出路径：

```text
reports/p0-memory-review/MEMORY.md
reports/p0-memory-review/*.md
```

作用：

1. 人工审查；
2. 演示长期记忆沉淀结果；
3. 对比 topic 与 memory 是否一致；
4. 作为后续接入 CP 前的可视化检查。

当前已验证：

```text
回复语言偏好 topic 已修正；
memory 内容为英文回复偏好；
不再出现“标题中文、内容英文”的割裂。
```

## 11. 当前已经打通的能力

### 11.1 用户偏好沉淀

例：

```text
用户要求所有对话统一英文回复。
```

系统可：

```text
更新 memory；
修正 topic；
记录旧中文偏好与新英文偏好的冲突决策。
```

### 11.2 编译经验沉淀

例：

```text
Keil C51 C141 / C202 / L250 错误模式。
```

系统可沉淀：

```text
症状 → 原因 → 修复 → 验证
```

### 11.3 用户反馈现象沉淀

例：

```text
温度 36℃ 附近风扇频繁启停。
```

系统可沉淀：

```text
现象：阈值附近输出自激振荡
原因：EMI 耦合回 ADC，噪声方向把读数往回推
解决：连续 N 次采样一致才切换
验证：用户烧录后确认不再频繁启停
```

### 11.4 幂等重跑

已实现：

```text
同 turn 重跑不会重复写 memory；
同冲突重跑不会重复创建 conflict；
topic 无变化返回 skipped_no_change。
```

## 12. 当前架构选择的权衡

### 12.1 为什么先不做 LangGraph

当前没有先上 LangGraph 的原因：

1. 当前最重要是打通 MVP；
2. 核心难点在证据结构、写入校验、幂等和冲突，不在框架；
3. 手写编排更容易观察每一步输出；
4. 之后如果流程稳定，再迁移到 LangGraph 更安全。

后续适合引入 LangGraph 的场景：

```text
多节点状态机稳定；
需要 checkpoint；
需要 human-in-the-loop 审查；
需要失败后续跑；
需要并发任务调度。
```

### 12.2 为什么暂时不做完整版本链

完整版本链会带来：

- revision 表；
- 回滚；
- 冲突分支；
- 历史证据链；
- 更复杂的展示和召回策略。

当前阶段重点是：

```text
能从真实 CP 长流程里稳定沉淀有价值记忆。
```

所以当前采用轻量方案：

```text
当前 memory + metadata + conflict_records
```

### 12.3 为什么保留 Markdown

数据库适合机器读写，Markdown 适合人看。

当前演示需要能回答：

```text
到底沉淀了什么？
主题是什么？
memory 内容是否有用？
是否重复？
是否与用户偏好一致？
```

因此 Markdown 是审查层，不是主存储。

## 13. 当前问题

### 13.1 全量测试仍有 1 个旧测试失败

当前全量测试结果：

```text
301 个测试，300 个通过，1 个失败。
```

失败项：

```text
test_unified_cycle.UnifiedCycleTests.test_long_transcripts_use_longest_model_compatible_prefix_without_starvation
```

表现：

```text
期望 selected_turn_count = 1
实际 selected_turn_count = 3
```

该问题在旧 unified_cycle 长 transcript 批处理策略中，需要单独判断。

### 13.2 召回链路尚未系统验证

目前重点是沉淀。

后续需要验证：

```text
新任务开始时，系统能召回哪些记忆；
召回内容是否真正减少重复问答和返工；
召回是否会误带当前项目参数。
```

### 13.3 主题结构维护还不完整

已有：

```text
update_topic
```

尚缺：

```text
move_memory_to_topic
split_memory
merge_topic
rename_topic_key / 文件名迁移策略
```

### 13.4 conflict 尚未投影到 Markdown

冲突记录已经进 PostgreSQL，但 Markdown 里还没展示。

后续应在主题文件中增加：

```text
## 冲突记录
旧说法
新说法
处理决定
证据来源
```

## 14. 后续路线

建议顺序：

```text
P0：处理唯一全量测试失败，保证全量测试全绿。
P1：Markdown 展示 conflict records。
P2：实现 search_related_memories，开始召回测试。
P3：增加 move_memory_to_topic / split_memory。
P4：设计长期运行压测，验证 20+ turn 连续处理不污染。
P5：评估是否迁移到 LangGraph 编排。
```

## 15. 当前结论

当前架构已经证明：

```text
真实 CP 对话 → 三层证据 → ReAct 判断 → 结构化写库 → 幂等维护 → Markdown 展示
```

这条核心链路可运行。

它已经能支撑：

1. 用户交互偏好沉淀；
2. 编译经验沉淀；
3. 用户反馈现象后的原因/解决/验证沉淀；
4. 主题语义修正；
5. 冲突决策记录；
6. 重跑幂等。

尚未完全完成的是：

```text
召回使用、完整冲突策略、主题结构重构、长期压测、全量测试全绿。
```

