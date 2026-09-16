# 长期记忆测试覆盖文档

本文总结 Memory Lab 长期记忆系统应覆盖的九类测试，以及当前完成情况、验证结果和未完成项的开测条件。

## 1. 测试目标

长期记忆系统不是只要“能生成 memory”就算完成，而是要验证它能长期稳定运行：

```text
能提取、能合并、能跳过、能处理冲突、能回查证据、能展示、能长期重跑不污染、能接入真实链路。
```

因此我们把测试拆成九类。

## 2. 九类长期记忆测试总览

| 编号 | 测试类型 | 当前状态 | 结论 |
|---|---|---|---|
| T1 | 证据提取测试 | 已完成 MVP | 三层证据池可从 CP SQLite 生成并入库 |
| T2 | 记忆新增测试 | 已完成 | 可从真实 turn 新增长期记忆 |
| T3 | 记忆合并/更新测试 | 已完成 | 可 merge/update，且同 turn 重跑跳过 |
| T4 | 幂等测试 | 已完成核心路径 | update/merge/topic/conflict 已有幂等保护 |
| T5 | 冲突测试 | 部分完成 | 可记录冲突决策，但还未做完整冲突策略系统 |
| T6 | 证据回查测试 | 已完成 MVP | pointer 可回查原始 CP 片段 |
| T7 | 召回/使用测试 | 未完成本轮 | 还未系统验证“生成代码前召回哪些记忆” |
| T8 | 展示/审查测试 | 已完成 MVP | Markdown 可导出，topic/memory 语义一致 |
| T9 | 长期运行/全量回归测试 | 部分完成 | 工具链回归通过；全量测试 301 个中 300 个通过 |

## 3. T1：证据提取测试

### 目标

验证系统能从 CP SQLite 原始会话里抽取一轮 turn，并形成三层证据：

```text
core：用户明确表达、关键结论
candidate：工具/助手总结等候选信息
pointer：可回查原始对话或工具输出的指针
```

### 当前完成情况

已完成 MVP。

已实现：

```text
CP SQLite → turn → evidence_items
core / candidate / pointer 分层
raw_pointer_id 可回查
```

真实数据已验证：

- 用户语言偏好 turn；
- Keil C51 编译 turn；
- 运行现象诊断 turn；
- 修复确认 turn。

### 仍需优化

候选摘要的人类可读性仍可继续优化，尤其是工具输出摘要。

## 4. T2：记忆新增测试

### 目标

验证模型能从一轮真实对话中新增长期记忆，而不是只生成需求文档。

### 当前完成情况

已完成。

真实新增过的记忆包括：

```text
Keil C51 现有工程二次开发流程偏好
单片机项目改码后必须实际编译检查
嵌入式硬件输入桩导致假数据
阈值附近输出自激振荡需时间滤波
Keil C51 二次开发常见编译错误模式与修复
```

### 质量评价

新增记忆能覆盖三类核心目标：

1. 用户协作偏好；
2. 编译失败/成功经验；
3. 用户反馈现象后的根因和解决方法。

## 5. T3：记忆合并/更新测试

### 目标

验证新 turn 不一定新建记忆，而是能和已有记忆合并或更新。

### 当前完成情况

已完成核心路径。

已验证：

```text
merge_memory：同类工程经验补充证据。
update_memory：用户偏好从中文更新为英文。
update_topic：memory 语义变化后同步主题标题/概要。
```

真实回归结果：

- 语言偏好已从旧中文偏好更新为英文偏好；
- topic 标题从“默认使用中文回复”修正为“回复语言偏好”；
- 编译 turn 重跑时不重复 merge。

### 仍需优化

尚未实现：

```text
move_memory_to_topic
split_memory
topic 合并
```

这些属于后续结构维护能力。

## 6. T4：幂等测试

### 目标

验证同一 turn / 同一 plan 重跑不会污染数据库。

### 当前完成情况

核心路径已完成。

已实现行为：

```text
update_topic 无变化 → skipped_no_change
record_conflict_decision 已存在 → skipped_existing / updated_existing
update_memory 同 turn 重跑 → skipped_duplicate_turn
merge_memory 同 turn 重跑 → skipped_duplicate_turn
```

真实回归结果：

```text
语言偏好冲突记录最终收敛为 1 条。
编译 turn 重跑返回 skipped_duplicate_turn。
修复确认 turn 重跑返回 skipped_duplicate_turn。
```

### 单元测试

已新增：

```text
tests/test_memory_changes_validation.py
```

覆盖：

- topic 无变化跳过；
- conflict 重复跳过；
- update/merge 同 turn 重跑跳过；
- 错误输入不写库。

## 7. T5：冲突测试

### 目标

验证系统能处理长期记忆中的前后矛盾。

典型例子：

```text
旧记忆：用户希望默认中文回复。
新证据：用户希望以后所有对话英文回复。
```

### 当前完成情况

部分完成。

已实现：

```text
record_conflict_decision
memory_conflict_records 表
old_claim / new_claim / decision / reason / evidence_refs
同一 memory + turn + conflict_type 收敛为 1 条
```

真实验证：

```text
language_conflict_count = 1
topic = 回复语言偏好
memory = 默认英文回复
```

### 未完成

尚未形成完整冲突策略系统：

```text
不同作用域冲突如何保留多版本；
工程经验冲突如何按芯片/工具链/项目条件拆分；
冲突记录如何投影到 Markdown；
冲突如何影响后续召回。
```

### 什么时候开测完整冲突系统

建议在以下能力具备后开测：

1. `search_related_memories` 完成；
2. `move_memory_to_topic` 或 `split_memory` 完成；
3. Markdown 能展示 conflict records；
4. 至少准备 3 组真实/模拟冲突数据：
   - 用户偏好反转；
   - 工程经验同症状不同根因；
   - 不同项目作用域的规则差异。

## 8. T6：证据回查测试

### 目标

验证当摘要不足时，模型能通过 pointer 回查原始对话片段。

### 当前完成情况

已完成 MVP。

已有工具：

```text
expand_pointer
read_topic_detail
read_memory_detail
```

ReAct 提示词要求：

```text
如果 core/candidate 不足以证明症状、根因、修复、验证，应 expand_pointer。
```

真实运行中，模型已能在修改已有 memory 前调用 read 工具读取详情。

### 仍需优化

还缺：

```text
search_related_memories
```

否则模型主要依赖初始 memory_index，如果摘要没命中，可能不知道该查哪个主题。

## 9. T7：召回/使用测试

### 目标

验证长期记忆不只是能存，还能在后续任务里正确召回并影响行为。

例如：

```text
新 Keil C51 项目生成代码前，应召回：
- 不重建工程结构；
- C51 兼容写法；
- 改码后必须实际编译；
- 编译前自检清单。
```

### 当前完成情况

本轮未系统测试。

已有基础：

```text
Markdown 导出
memory index
read_topic_detail
```

但还没有端到端验证：

```text
给新任务 → 系统召回相关记忆 → CP/模型使用这些记忆 → 行为改善。
```

### 什么时候开测

建议在以下条件满足后开测：

1. `search_related_memories` 完成；
2. 有一个“任务开始前召回记忆”的入口；
3. 选择 3 个新任务作为验证：
   - 新 Keil C51 生成代码；
   - 新编译错误修复；
   - 新运行现象诊断。

验收标准：

```text
召回结果相关；
召回内容不超上下文；
模型行为确实减少重复问答/重复错误；
不召回无关项目参数。
```

## 10. T8：展示/审查测试

### 目标

验证记忆可以被人审查，能用于汇报。

### 当前完成情况

已完成 MVP。

导出路径：

```text
reports/p0-memory-review/MEMORY.md
reports/p0-memory-review/*.md
```

已验证：

```text
主文件显示主题列表；
主题文件显示 memory 正文；
topic 标题与 memory 内容一致；
语言偏好不再出现“标题中文、内容英文”的割裂。
```

### 未完成

尚未展示：

```text
conflict records
工具调用流转
每轮 turn 的变化报告
```

建议后续增加：

```text
## 冲突记录
旧说法
新说法
处理决定
证据 turn
```

## 11. T9：长期运行 / 全量回归测试

### 目标

验证系统长期运行不会因为重复 turn、冲突、合并、错误 plan 导致数据库污染。

### 当前完成情况

部分完成。

已跑：

```text
工具相关单元测试：通过
真实多轮回归：通过
Markdown 导出：通过
全量测试：301 个中 300 个通过
```

失败项：

```text
test_unified_cycle.UnifiedCycleTests.test_long_transcripts_use_longest_model_compatible_prefix_without_starvation
```

失败原因表现：

```text
测试期望 selected_turn_count = 1
实际 selected_turn_count = 3
```

初步判断：

该失败在旧的 `unified_cycle` 长 transcript 批处理策略，不是本轮 `memory_changes` 工具层改动直接导致。需要单独判断是：

```text
更新测试期望；
还是调整 unified_cycle 的批处理选择逻辑。
```

### 什么时候认为长期运行测试完成

需要达到：

```text
1. 全量单测全绿；
2. 至少 20 轮真实/模拟 turn 连续跑；
3. memory/topic/conflict 数量增长符合预期；
4. 同一批 turn 重跑后数量不增长；
5. Markdown 导出稳定；
6. 无重复 topic、重复 conflict、重复 memory。
```

## 12. 当前总体结论

当前已经完成的是：

```text
提取 → 写入 → 合并/更新 → 幂等 → 冲突记录 → Markdown 展示
```

当前还未完全完成的是：

```text
召回使用
完整冲突策略
主题结构重构
长期连续运行压测
全量测试全绿
```

下一步优先级建议：

```text
P0：处理唯一全量测试失败。
P1：把 conflict records 投影到 Markdown。
P2：实现 search_related_memories，开始召回测试。
P3：设计 split/move/rename 类结构维护工具。
```

