# CodePulse Memory Lab：MVP 与迭代计划

## 1 MVP 要回答的问题

第一版不追求完整的 Claude Code Memory，只回答一个核心问题：

> 能否从同一个用户的多个 CodePulse 会话和项目中，自动找出 CP 指令文件尚未覆盖、
> 值得长期保存的信息，并正确完成原子拆分、合并、更新、范围隔离、冲突和证据管理？

MVP 是本地离线实验，不修改 CodePulse 数据库、不修改 CodePulse 主线、不自动写入
`AGENTS.md`，也不向 CodePulse Agent 自动注入记忆。

## 2 MVP 范围

### 包含

- 读取 CodePulse SQLite 新增用户消息。
- 识别会话和项目目录。
- 扫描当前任务可能加载的 `AGENTS.md`、`CLAUDE.md`、`CONTEXT.md` 和
  `.codepulse/AGENTS.md`。
- 使用硬规则判断临时要求、敏感信息和 CP 已覆盖规则。
- 使用大模型拆分原子候选，生成 Stable Key、scope、activation 和 triggers。
- 使用大模型判断不同会话内容是同义、补充、纠正、冲突还是无关。
- 由确定性程序校验并执行合并，不允许模型直接修改正式记忆。
- 保存候选、证据和审核结果。
- 输出可审核的记忆、关系判断、处理动作、原因和原始 part ID。
- 从 fresh、current、完整的 shadow run 只读生成去重预览目录。
- 按 scope 和确定性词项相关度执行独立的任务预览召回。
- 将候选投影为 `MEMORY.md + 主题 Markdown + pending_conflicts.md`，并只从该投影召回。

### 不包含

- embedding 和向量数据库。
- 工程工具链全过程提取。
- 自动冲突裁决。
- Auto Dream。
- Web UI。
- 自动写入全局或项目 `AGENTS.md`。
- 接入 CodePulse 主 Agent。
- 将召回结果自动注入 CodePulse，以及面向生产的规模化语义检索。

## 3 当前 MVP 输入和标准答案

原始 bootstrap 有六条受控消息，来自三个会话和两个目录。

| Case | 预期处理 | 预期 Stable Key |
|---|---|---|
| ML-S1-A1 默认中文 | 新增候选 | `preference.response.language` |
| ML-S1-A2 代码生成前确认 | 新增候选 | `workflow.code_generation.confirmation` |
| ML-S1-B1 再次确认 | 合并到 A2 | `workflow.code_generation.confirmation` |
| ML-S1-B2 这一次只解释 | 临时要求，拒绝 | 无 |
| ML-S1-P2-01 所有单片机项目修改后编译 | 项目 AGENTS.md 仅部分覆盖；保留跨项目候选 | `workflow.embedded.code_change.compile_check` |
| ML-S1-P2-02 当前项目使用 Keil C51 | 当前项目 AGENTS.md 完整覆盖 | 无新记忆 |

原始 bootstrap 最终应得到三条候选记忆，其中代码生成确认规则有两条独立会话证据。
P2-01 的项目内部分已经存在于 `AGENTS.md`，但“所有单片机项目”的跨项目范围没有被
该项目文件覆盖，因此不能整条丢弃。

随后增加的回归和 U2-1 holdout 使当前 fresh shadow run 包含 13 条原子判断。其只读目录
结果为 6 条正判断合并成 5 条唯一预览记忆、7 条过滤、1 组 scope variant、0 个同 scope
conflict；shadow-only 阶段四张正式表仍均为 0。

## 4 MVP 数据流

```text
opencode-dev.db
        ↓ ingest
标准 user_text 事件
        ↓ turn scan（两次稳定采样）
completed / failed / interrupted / pending
        ↓ prepare
会话、项目、测试编号和纯语义
        ↓ instruction coverage
排除 CP 已通过 Markdown 指令覆盖的规则
        ↓ hard rules
过滤临时、敏感和 CP 已覆盖信息
        ↓ LLM extract
原子候选、Stable Key、scope、triggers
        ↓ LLM relation + deterministic consolidate
模型判断关系，程序新增、合并或拒绝
        ↓ review
人工确认候选
        ↓ Markdown publish
索引、主题文件与冲突审核视图
        ↓ Markdown-only recall
按任务、项目和领域选择可用记忆
        ↓ memory audit report
展示新增、合并、拆分、更新、冲突和证据
```

## 5 MVP 实现检查点

### M0 数据入口

状态：已完成。

已有命令：

```powershell
.\memory-lab.ps1 baseline
.\memory-lab.ps1 ingest
.\memory-lab.ps1 prepare
.\memory-lab.ps1 turns -SettleMilliseconds 1000
```

验收结果：六条消息、三个会话、两个目录；第二次读取新增为零。首次真实回放时已纳入
实验的 9 条用户消息均能与多段 assistant/tool part 合并为 9 个稳定回合；重复扫描
不重复创建 turn，只追加必要的检测审计。

用户配合：已完成三个受控会话。后续每轮等 CP 最终回复出现后，只需通知 Memory Lab
触发扫描；通知不是完成凭据，最终状态由数据库双采样判定。

### M1 CP 指令覆盖检测

实现：

- 根据事件目录发现 CP 会加载的指令文件。
- 解析指令中的规则条目。
- 给事件生成非最终覆盖提示；完成原子拆分后，再给每个候选标记
  `covered_by_cp_instructions`、覆盖范围和具体文件路径。
- 保存不可变指令快照，为“消息前曾见过相同规则”提供时间证据；它不能冒充 CP
  运行时实际加载记录。
- 只读指令文件，不修改它们。

用户配合：核对系统列出的指令文件是否确实对当前项目生效。

通过标准：

- P2-01 识别为 `D:\codepulsework\AGENTS.md` 只覆盖当前项目，不能覆盖“所有单片机项目”。
- P2-02 识别为 `D:\codepulsework\AGENTS.md` 完整覆盖。
- A1、A2、B1 不因另一个项目的 AGENTS.md 被错误过滤。
- 新增消息能够在下一次增量读取中出现，且事前指令快照可以证明规则在消息前已存在。

状态：已通过真实新增会话验收；正确目录、内部 part 过滤、项目规则命中、事前快照和
重复读取均符合预期。

### M2 硬规则准入和大模型原子提取

实现：

- 硬规则处理长期、临时、更正、敏感信息和 CP 指令覆盖信号。
- `LLMExtractor` 把自然语言拆成 `user_contract` 原子候选。
- 模型生成 Stable Key、scope、activation、triggers 和精确原文引用。
- JSON Schema 校验器验证 part ID、原文引用、字段和 scope。
- 模型结果只进入 candidate，不允许直接变为 active。

用户配合：配置独立模型 API 或本地模型服务，并审核六条分类；错误时给出正确答案。

通过标准：

- A1、A2、B1、P2-01 的未覆盖跨项目部分进入候选。
- B2 被识别为临时要求。
- P2-02 被当前项目指令完整覆盖，不进入记忆候选。
- 原文证据引用正确率 100%。

### M3 模型关系判断、确定性合并和本地存储

实现：

- `candidate`、`memory`、`memory_revision`、`evidence`。
- 先按 family、type、scope 和关键词筛出少量可能相关旧记忆。
- 模型只判断 `equivalent`、`extension`、`correction`、`conflict`、
  `different_scope` 或 `unrelated`。
- 程序校验关系结果，并在相同范围、含义一致时合并证据。
- 人工 `accept`、`reject`、`edit`。
- 所有变更保留审计记录。

用户配合：确认合并结果和最终规范表述。

通过标准：

- 最终只有三条候选或已审核记忆。
- A2 和 B1 不生成重复记忆，并保留两个 session/part 来源。
- 模型返回的不存在 memory ID、虚构原文和无依据 scope 必须被拒绝。
- 重复执行不会增加记忆或证据数量。

### M4 记忆管理审计报告

实现：

- 按项目、会话、Stable Key 和状态展示处理结果。
- 显示 `new`、`merge`、`split`、`update`、`conflict`、`ignore`。
- 显示模型判断、程序执行结果、原文证据和审计历史。
- 生成 Demo 的准确率、误存、漏存、重复和冲突统计。

用户配合：审核每条记忆是否值得保存、拆分是否正确、关系判断是否正确。

通过标准：

- 每条输入都有明确处理结果和原因。
- 每条记忆可追溯到项目、会话和 part。
- 已拒绝、已覆盖和临时内容不进入正式记忆。
- 重复、更新和冲突处理结果符合人工答案。

### M5 影子候选目录、Markdown 投影和确定性预览召回

状态：已完成独立只读预览，尚未接入 CodePulse。

实现：

- `catalog` 只消费 fresh、current、`complete` 且完整覆盖 admission 快照的 shadow run；
  正式表状态与 shadow 运行前后不一致时失败关闭；目录与项目解析在同一个 SQLite 只读
  快照内完成，并拒绝读取期间的并发提交。
- 按 `stable_key + canonical scope` 合并同义证据；同 Key 的不同 scope 保留为 variant，
  同 scope 的不兼容值保留为待审核 conflict。
- `recall` 对同一 Key 按 `project > domain > global` 选择最具体适用范围，再按触发词和
  确定性词项重合度排序；单个泛化触发词不能单独命中，当前演示固定最多返回 5 条。
- project、domain、conditions 按“适用 / 不适用 / 未知”三态路由。更具体或同级上下文未知、
  或存在未解决冲突时，会阻断更宽 scope 回退；带 `conditions` 的 scope 在尚无类型化上下文
  时返回待补上下文，而不是被忽略。
- 两个命令都返回 `preview_only: true`，只读 Memory Lab 数据，不写正式表，也不修改或
  注入 CodePulse。
- `markdown-preview` 将通过校验的目录确定性发布为平铺 Markdown：`MEMORY.md` 只做索引，
  三个主题文件保存具体条目，`conflicts/pending_conflicts.md` 是派生审核视图。
- `markdown-recall` 只读取带哈希和 scope 校验的 Markdown，不读取 CP 或 Memory Lab SQLite；
  由此验证 Markdown 能独立承担未来提供给主 Agent 的运行时记忆角色。
- 发布采用临时目录、单文件原子替换和索引最后提交；混合代不会被接受。正式接入前再升级为
  generation + `CURRENT`，以保证发布崩溃期间也持续可读。

真实演示：

```powershell
.\memory-lab.ps1 catalog `
  -ShadowRunId llmshadow_799ce7e3d19446cda48f23d8aa29f7ec

# 当前项目：项目级 not_required 胜过全局 required
.\memory-lab.ps1 recall `
  -Query '请生成正式代码，实现一个新功能。' `
  -ProjectDirectory 'D:\codepulsework' `
  -Limit 5 `
  -ShadowRunId llmshadow_799ce7e3d19446cda48f23d8aa29f7ec

# 另一项目：项目例外不适用，回退到全局 required
.\memory-lab.ps1 recall `
  -Query '请生成正式代码，实现一个新功能。' `
  -ProjectDirectory 'D:\codepulse\Codepulse-TUI1.0.77\codepulse-window-1.0.77' `
  -Limit 5 `
  -ShadowRunId llmshadow_799ce7e3d19446cda48f23d8aa29f7ec

# 把当前 U2-2 影子结果发布为 Claude 风格 Markdown 预览
.\memory-lab.ps1 markdown-preview `
  -ShadowRunId llmshadow_ed67d2dea51648e098e222c82ae4d591

# 仅从 Markdown 查询，不读取任何 SQLite
.\memory-lab.ps1 markdown-recall `
  -Query '请在当前项目生成正式代码，实现一个新功能。' `
  -ProjectDirectory 'D:\codepulsework' `
  -Limit 5
```

当前 U2-1 结果：`positive_judgments = 6`、`unique_memory_count = 5`、
`filtered_count = 7`、`scope_variant_group_count = 1`、`same_scope_conflict_count = 0`；
`memory / memory_revision / evidence / conflict = 0 / 0 / 0 / 0`。
当前全量自动化测试为 143 / 143，其中 Episode 对抗测试 10 条，Markdown 发布、校验和
独立召回边界测试 13 条。

U2-2 同范围冲突盲测结果：冻结集 15 / 15 正确，`positive_judgments = 7`、
`unique_memory_count = 4`、`filtered_count = 8`、`same_scope_conflict_count = 1`。
`preference.response.language / global` 的 `zh` 与 `en` 均未被激活，召回返回待解决冲突；
正式四表仍为 0。

Markdown U2-2 投影结果：3 个主题、4 条可用候选、1 个待解决冲突、5 个受管文件；重复发布
零写入。项目例外、全局规则、回复结构和语言冲突从 Markdown 独立召回的结果与 SQLite
预览召回一致。

查询没有提供项目目录或目录未知时，`project_context_known = false`；只要同 Key 存在项目
变体，就不会直接采用全局值。此行为已与项目冲突、条件冲突和无关查询一起纳入回归测试。

## 6 MVP 完成定义

下面条件同时满足才算 MVP 跑通：

```text
CP 数据库零写入
增量读取幂等
指令覆盖隔离正确
临时要求误存为零
大模型正确识别同义关系
确定性程序正确执行合并
原始 bootstrap 三条规范候选符合人工答案
U2-1 预览目录为 6 positive → 5 unique，7 filtered
项目、领域、全局 scope 选择符合 more-specific-scope-wins
预览召回不写正式表、不注入 CP
Markdown 投影可重复生成且哈希校验通过
Markdown-only 召回不依赖 CP 或 Lab SQLite
同主题内可用条目不被另一条目的冲突整体阻断
每条记忆和处理结果都有来源与原因
```

## 7 MVP 之后的迭代

### V0.2 正式层更新、冲突和项目例外

候选预览层已经能识别 unresolved conflict 和全局/项目例外；本阶段把这些能力迁移到审批后的
正式 revision 状态，并增加：

- explicit correction。
- revision/superseded。
- unresolved conflict 的正式落库与解决记录。
- 全局默认与项目例外的正式发布。

用户测试：分三轮发送相反要求、明确纠正和项目局部例外；每轮单独验收。

### V0.3 工程事件和证据包

状态：E1 v3 自动预验收已通过，等待隔离 Keil 工程的真实 CP 会话验收；E2 语义提炼尚未开始。

增加：

- assistant 可见文本仅在源库只读处理时使用，不复制到 E1 数据库。
- tool completed/error 的白名单结构状态。
- 编译、测试、文件变化和退出状态。
- task episode 和工程证据包。

用户测试：在 CP 正常完成一个真实小型任务，不需要添加 ML 标记，也不需要故意制造失败。

### V0.4 工程经验记忆

增加：

- symptom、cause、attempt、solution、verification。
- observed/hypothesis/failed/verified。
- 项目内经验和跨项目提升。

用户测试：审核一次真实任务提取结果；以后在另一个项目出现相似问题时再次验证。

### V0.5 模型优化和对比

增加：

- 对比不同模型、提示词和低成本模型。
- 比较硬规则、模型和人工标准答案。
- 增加批处理、缓存、重试和成本统计。
- 决定哪些简单情况可由规则处理，哪些必须调用模型。

用户测试：审核一批自然会话结果，比较准确率、成本和延迟。

### V0.6 预览召回后的规模化检索和巩固

增加：

- 已有确定性小规模预览召回；只有其通过价值评审后，才扩展为生产查询链路。
- 增加 scope/状态过滤、BM25、可选 embedding 和任务解析。
- 可选模型 Top 20→Top 5 重排。
- 去重、陈旧、范围修正和整理建议。

用户测试：提供应命中、不应命中和无关查询，审核巩固建议。

### V0.7 A/B 价值验证

增加：

- 无记忆与手工提供记忆摘要的任务对比。
- 重复沟通、失败步骤、工具调用、完成时间和正确性统计。

用户测试：选择风险可控的真实任务，分别完成 A/B 两组。

### V1.0 接入评审

只有 V0.1 至 V0.7 都证明有效后，才讨论是否接入 CodePulse，以及使用全局
`AGENTS.md` 投影、插件、动态上下文或其他方式。

## 8 立即执行顺序

```text
M1 指令覆盖检测
→ 配置独立模型 API 或本地模型
→ M2 硬规则准入和模型原子提取
→ 让用户审核六条结果
→ M3 模型关系判断和确定性合并
→ M4 记忆管理审计报告
→ fresh/current shadow catalog
→ project > domain > global 的独立预览召回
→ 发布 Claude 风格 Markdown 预览
→ 脱离 SQLite 执行同结果召回
→ 输出 MVP 总结报告
```

当前已有独立 `preview_only` 召回；MVP 总结通过价值评审之前，不把结果注入 CodePulse，
也不扩展为规模化语义检索链路。
