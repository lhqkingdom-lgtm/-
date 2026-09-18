# ReAct 真实流程验收标准与本轮结果

本文定义 Memory Lab 的 ReAct 记忆维护 Agent 在真实 CP turn 上的验收标准，并记录本轮 dry-run 结果。

## 1. 测试目标

ReAct 测试不是验证“工具能不能写库”，而是验证大模型是否能在真实上下文中自己完成判断：

```text
读取当前 turn 证据
  ↓
查看已有 memory / topic 摘要
  ↓
必要时调用读工具补证据
  ↓
判断新增 / 更新 / 合并 / 冲突记录 / 无操作 / 补证据
  ↓
输出合法 memory-changes-v1
```

本阶段先使用 dry-run，不真实落库，避免测试污染正式记忆。

## 2. 总体验收标准

### 2.1 运行结果标准

每个场景必须满足：

| 检查项 | 验收标准 |
|---|---|
| 运行状态 | `status=complete` |
| 模式 | `mode=dry_run` |
| 结构化输出 | 输出合法 `memory-changes-v1` |
| 提交状态 | `committed=false` |
| 轮数 | 不超过 `max_iterations` |
| 证据引用 | 使用当前 turn 内存在的 `evidence_id` |
| 动作合法性 | 只使用允许的 action |
| 幻觉防护 | 不引用不存在的 memory/topic/evidence |

### 2.2 工具调用标准

模型不应只凭第一眼摘要直接改记忆。理想行为是：

| 情况 | 应有行为 |
|---|---|
| 可能已有相似记忆 | 调用 `search_related_memories` 或读取相关 topic/memory |
| 要修改已有 memory | 先 `read_memory_detail` 或通过自动读工具读取详情 |
| 要判断主题是否已有覆盖 | 调用 `read_topic_detail` |
| 要判断同一主题下是否已有重复 memory | 调用 `list_topic_memories` |
| 要判断已有 memory 是否曾被补充/修正过 | 调用 `read_memory_history` |
| 要处理偏好变化或冲突 | 调用 `list_memory_conflicts` 检查是否已有冲突记录 |
| 当前 turn 证据不够 | 输出 `request_more_evidence` 或调用 `expand_pointer` |
| 需要检查关系链 | 调用 `trace_memory_relations` |

### 2.3 动作选择标准

| 场景 | 合格动作 |
|---|---|
| 用户偏好变化 | `update_memory` / `record_conflict_decision` / `record_memory_relation` |
| 编译失败修复 | `create_memory` / `merge_memory` / `update_memory` |
| 一次编译通过 | 有新增经验才写；否则 `noop` |
| 运行现象定位 | 有新根因则 `create_memory` 或 `merge_memory`；已有覆盖则 `noop` |
| 用户确认修复 | 若是新增验证证据，应 `merge_memory` 提升可信度；若已覆盖则 `noop` |
| 证据不足 | `request_more_evidence` 或 `noop`，不能强行写长期记忆 |
| 同现象不同根因 | 新建独立 memory，并用 relation 关联，不能覆盖旧根因 |

### 2.4 内容质量标准

输出的 memory 内容必须符合：

```text
用户偏好类：记录长期协作习惯，不记录一次性情绪。
编译类：只存“症状 → 原因 → 解决 → 验证”，不存完整日志。
运行现象类：从项目现象抽象为可复用规则，不能只写当前文件名。
冲突类：说明 old_claim / new_claim / decision / reason。
noop：必须说明为什么已有记忆已覆盖，或为什么无长期价值。
```

### 2.5 防污染标准

以下情况必须拒绝或不写：

```text
未验证猜测
当前项目具体参数
完整日志 / 大段代码 diff
能从代码或 Git 稳定恢复的信息
一次性演示安排
不存在的 evidence_id / memory_id / topic_id
```

## 3. 本轮真实 dry-run 测试结果

本轮使用现有 PostgreSQL 中真实 CP turn 运行 `pg-memory-react-run --dry-run`。

### 3.1 用户偏好变化 / 冲突记录

turn：

```text
turn_c8645b85cdfd301c28673037bd5377c44cb3d01e4274f561e0a7d62702a98220
```

结果：

```text
status=complete
iterations=2
observation_count=2
action=record_conflict_decision
```

模型判断：

```text
旧偏好：默认中文回复
新偏好：所有对话统一英文回复
决策：采用新的明确长期偏好，并记录冲突原因
```

结论：通过。模型没有重复创建语言偏好，而是识别为同一语义槽位的偏好变化，并输出冲突决策记录。

### 3.2 一次编译通过

turn：

```text
turn_64bbc76b5ecdea714511940a63629fe45357b2917a67af60000e42a3e853c88a
```

结果：

```text
status=complete
iterations=2
observation_count=1
action=noop
```

模型判断：

```text
已有 memory 已覆盖“Keil C51 生成代码前做兼容性自检可一次编译通过”；
本轮只是重复确认 0 Error / 0 Warning，无新增长期价值。
```

结论：通过。模型没有把一次成功结果重复沉淀成新 memory。

### 3.3 编译修复经验

turn：

```text
turn_f0cc56c0d337e68277a9576a6800e7dc67cc85aa0ebf622a730906db989d92a4
```

结果：

```text
status=complete
iterations=2
observation_count=3
action=merge_memory
```

模型判断：

```text
合并到 Keil C51 编译错误模式；
补充 SFR 寄存器名不一致导致 C202；
补充 Keil fatal exit code 3 属环境级失败，应重试；
保留 L250 为许可限制而非代码错误的结论。
```

结论：通过。模型把新增证据合并到已有编译模式，而不是重复新建。

### 3.4 运行现象定位

turn：

```text
turn_e90cab8f66831ddd470536aae9e6f8b940d2faf01c0db74878705db0db20745e
```

结果：

```text
status=complete
iterations=2
observation_count=3
action=noop
```

模型判断：

```text
已有 memory 已覆盖“adc_sample 固定返回 2048 → 输入恒定 → 下游状态机无响应”；
本轮证据已经被合并，无需重复写入。
```

结论：通过。模型先查已有记忆，再选择不重复沉淀。

### 3.5 用户确认修复

turn：

```text
turn_0e6c9aa67c3a0776783748acfe5838a3e9e0f1a59e1ecc60d982d95ddfb442f6
```

结果：

```text
status=complete
iterations=2
observation_count=1
action=merge_memory
```

模型判断：

```text
本轮是用户烧录实测确认修复有效；
应合并到已有“硬件输入桩固定返回常量导致下游假数据”记忆中；
补充真实验证条件：改变 NTC 输入会影响温度值和占空比，传感器断开进入失效安全。
```

结论：通过。模型能把用户确认作为验证证据，而不是单独新建一条碎片记忆。

## 4. 本轮总体结论

本轮 5 个真实 ReAct dry-run 场景全部通过：

| 场景 | 期望 | 实际 | 结论 |
|---|---|---|---|
| 用户偏好变化 | 冲突记录 / 更新 | `record_conflict_decision` | 通过 |
| 一次编译通过 | 重复则 noop | `noop` | 通过 |
| 编译修复经验 | 合并已有模式 | `merge_memory` | 通过 |
| 运行现象定位 | 已覆盖则 noop | `noop` | 通过 |
| 用户确认修复 | 合并验证证据 | `merge_memory` | 通过 |

说明当前 ReAct 链路已经具备：

```text
查已有记忆
补证据
避免重复沉淀
合并新增证据
记录冲突决策
输出合法 memory-changes-v1
```

## 5. P0 读工具补充验收

本轮新增了三个给 ReAct 使用的确定性读工具：

| 工具 | 验收重点 | 结果 |
|---|---|---|
| `list_topic_memories` | 只返回主题内 memory 摘要；主题不存在时返回 `not_found`；不泄露大段正文 | 通过 |
| `read_memory_history` | 返回 memory 当前信息和版本历史摘要；用于判断补充/修正/重复 | 通过 |
| `list_memory_conflicts` | `memory_id` 和 `topic_id` 必须二选一；可查已有冲突记录 | 通过 |

真实 dry-run 补充结果：

```text
用户偏好冲突 turn：
模型调用 read_memory_detail + read_topic_detail + list_memory_conflicts，
确认“中文回复/英文回复”冲突记录已经存在，最终选择 noop，避免重复写冲突。

编译修复 turn：
模型调用 read_memory_detail 读取两条相关 memory，
再调用 list_topic_memories 查看主题下是否已有重复 memory，
最终判断已有记忆覆盖本轮证据，选择 noop。
```

## 6. 下一步建议

下一阶段可以进入小范围真实 commit 测试。

推荐只选择 1～2 条低风险场景：

1. 用户偏好冲突记录；
2. 用户确认修复的验证证据合并。

真实 commit 后需要检查：

```text
PostgreSQL 写入是否正确
memory_revisions 是否追加
memory_conflict_records 是否新增或幂等跳过
memory 主题 summary 是否仍合理
Markdown 导出是否可读
重复运行是否不会重复写
```
