# U2 正式记忆、合并、更新与冲突验收标准

## 1. 验收原则

先固定标准答案，再生成测试数据。每次只在 CodePulse 发送一条测试消息，等待最终
回复后通知 Memory Lab；完成该步的读取、提取、审核、合并和计数核对后，才能发送
下一条。测试消息不添加 `ML` 标签。

正式层采用：

```text
最新一次成功 admission 决定
        ↓
不可变 proposal
        ↓
用户查看并审批精确、不可变的 proposal_id
        ↓
consolidate 在单一事务内执行
        ↓
memory / revision / evidence / conflict / action 审计
```

`eligible` 不能自动成为 active；未审批、过期审批、范围不明和语义值无法解析的提案
都不得修改正式记忆。Memory Lab 始终只读 CodePulse SQLite。

## 2. Phase 0：现有数据正式化

当前已有 11 个用户事件和 11 个 completed turn。最新规则准入标准答案：

```text
eligible                     5
temporary                    1
covered_by_cp_instructions   2
needs_scope_review           1
not_memory                   2
```

五条 eligible 证据经审批、合并后应形成四条正式记忆：

| Stable Key | Scope | Active 内容 | 证据数 |
|---|---|---|---:|
| `preference.response.language` | global | 默认使用中文回复 | 1 |
| `workflow.code_generation.confirmation` | global | 生成正式代码前需要用户确认 | 2 |
| `workflow.embedded.code_change.compile_check` | domain:embedded | 单片机项目修改代码后进行编译检查 | 1 |
| `preference.response.structure` | global | 先给出结论，再说明处理步骤 | 1 |

业务表标准答案：

```text
memory            4
memory_revision   4（current 4，superseded 0）
evidence          5
conflict          0
```

必须满足：

- A2、B1 合并为同一个 memory 和同一个 current revision，保留两个不同 session/part。
- 两条 Keil C51 项目规则均不进入正式记忆。
- `preference.response.structure` 保持 global；项目 `AGENTS.md` 只能算部分覆盖。
- 单片机编译检查保持 domain:embedded，不能被单个项目规则完全过滤。
- 重复执行 consolidate 后四个业务表计数不变。

## 3. Phase 1：不同 scope 的项目例外

从 `D:\codepulsework` 启动 CodePulse，新建根会话，发送两行：

```text
仅在当前项目中，以后生成正式代码前不需要向我确认，可以直接生成。
这一次只回复“收到”，不要修改任何文件，包括 AGENTS.md。
```

等待最终回复后通知：`U2-1 完成`。

第一行预期：

```text
stable_key = workflow.code_generation.confirmation
scope = project:directory:92c7b4f8593a
semantic_value = not_required
Judge relation = different_scope
正式 proposal relation = create_scope_variant
consolidation action = created_scope_variant
```

第二行必须判为 temporary，不能形成 evidence。审批并 consolidate 后：

```text
memory            5
memory_revision   5（current 5）
evidence          6
conflict          0
```

全局版本仍要求确认，`D:\codepulsework` 项目版本不要求确认；两者并存而不是合并或
冲突。该轮前后的 `AGENTS.md` 哈希应一致。如果 CP 仍修改了文件，本轮标记为受污染，
先分析 CP 行为，不能为了凑计数强行正式化。

## 4. Phase 2：同 scope、没有明确更正的反向要求

在新的根会话发送：

```text
以后在所有对话中，回复不要使用中文，统一使用英文。
这一次只回复“收到”，不要修改任何文件，包括 AGENTS.md。
```

等待最终回复后通知：`U2-2 完成`。

第一行预期：

```text
stable_key = preference.response.language
scope = global
semantic_value = en
Judge relation = conflict
正式 proposal relation = conflict
consolidation action = opened_conflict
```

审批并 consolidate 后：

```text
memory            5
memory_revision   5（current 5，superseded 0）
evidence          7
conflict          1（open 1）
```

原来的“默认中文”revision 继续 current；英文要求只成为可追溯的冲突证据，不创建 revision，
也不允许采用“最新消息优先”覆盖旧记忆。

随后完整重复两次读取和 consolidate。除了 run/observation/no-op 审计允许增加，
memory、revision、evidence、conflict 和已应用 action 的数量都必须不变。

## 5. Phase 3：明确更正并解决冲突

在另一个新的根会话发送：

```text
更正：从现在起，在所有对话中回复不要使用中文，统一使用英文。
这一次只回复“收到”，不要修改任何文件，包括 AGENTS.md。
```

等待最终回复后通知：`U2-3 完成`。

第一行预期：

```text
stable_key = preference.response.language
scope = global
semantic_value = en
Judge relation = correction
正式 proposal relation = revise
consolidation action = revised（同时将旧 conflict 标记 resolved）
```

审批并 consolidate 后最终应为：

```text
memory            5
memory_revision   6（current 5，superseded 1）
evidence          8
conflict          1（open 0，resolved 1）
```

必须满足：

- global language 仍只有一个 memory。
- 中文 revision 1 标记 superseded，不再作为 current。
- 英文 revision 2 为唯一 current。
- 原 conflict 保留历史并标记 resolved，不能删除。
- Phase 1 的全局代码确认规则和项目例外完全不受影响。
- 再次 consolidate 后业务对象数量不变。

## 6. 全程不变量

- 正式身份是 `stable_key + canonical scope_fingerprint`，不是文本相似度。
- 同一 candidate 只有一条 evidence；同一 memory 只有一个 current revision。
- 未明确更正的同 scope 矛盾不能自动覆盖。
- 不同 scope 的相反要求可以并存，未来由更具体范围优先。
- instruction clause 只能用于覆盖判断，不能冒充用户 evidence。
- `toolchain.tg032plus.compiler` 的正式 memory 数始终为 0。
- 原文、session、message、part、项目、时间和处理动作均可追溯。
- 重复执行不增加任何业务对象。

全部通过的含义是：同义不重复、不同范围不串、矛盾不乱覆盖、明确更正可追溯、
CP 自带规则不污染用户记忆。U2 不验收 CP 是否会使用记忆；查询与注入要在后续阶段
单独做 A/B 验收。
