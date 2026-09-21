# CodePulse Memory Lab

> 当前最新路线：以 PostgreSQL 为长期记忆主库，围绕真实 CP turn 构建“证据池 → ReAct 记忆维护 Agent → 结构化写库 → 审查报告”的轻量长期记忆系统。
>
> 当前主链路已经从早期 SQLite/Markdown 实验，推进到：
>
> ```text
> CP SQLite / 外部回放 DB
>   → turn 三层证据池 core / candidate / pointer
>   → PostgreSQL 证据与长期记忆表
>   → ReAct 记忆维护 Agent
>   → read tools 补证据 / 查旧记忆
>   → memory-changes-v1 结构化计划
>   → 本地校验、幂等、写库或 dry-run
>   → 报告 / Markdown 审查
> ```
>
> 目前已经完成：三层证据池、PostgreSQL memory/topic/conflict/revision 基础表、ReAct 读工具、结构化写入动作、主题维护工具、冲突记录、真实 turn dry-run 回归和 P0 记忆质量优化。

CodePulse Memory Lab 是一个独立的长期记忆沉淀实验。它目前只把 CodePulse 的 SQLite 会话记录
作为输入，不修改 CP 源码、不扫描项目文件，也不向 CP 主 Agent 注入记忆。

当前目标只有一个：验证多个会话、多个项目里真正有长期价值的信息，能否被稳定地提取、审核、
合并、更新、隔离冲突，并投影成便于人和未来 Agent 阅读的 Markdown。

## 当前状态

- PostgreSQL 长期记忆主链已经打通，核心入口为 `pg-memory-react-run`；
- 已支持从真实 CP turn 生成三层证据池，并交给 ReAct Agent 判断 `create / merge / update / noop / conflict / topic` 等动作；
- 已支持只读补证据工具：搜索相关记忆、读取 memory/topic 详情、读取主题下记忆列表、读取历史、查看冲突、回查 pointer；
- 已支持结构化写入动作：新增、更新、合并证据、合并已有记忆、退役、标记冲突、记录冲突决策、更新主题、移动记忆主题、合并主题、记录关系；
- 已完成 19 条真实 CP turn dry-run 回归，验证 P0 合并粒度优化；
- 当前核心定向回归为 **100 项全部通过**；
- 自动轮询、召回注入 CP 主流程、多设备并发隔离仍属于下一阶段。

完整的两个真实项目测试步骤见 [REAL_PROJECT_TEST_PLAN.md](REAL_PROJECT_TEST_PLAN.md)。

可立即执行的 Keil C51 单项目演示步骤见
[KEIL_FANGUARD_TEST_PLAN.md](KEIL_FANGUARD_TEST_PLAN.md)。

## 当前通用链路

```text
CP SQLite（只读）
  → turn 切分
  → 三层证据池 core / candidate / pointer
  → PostgreSQL 证据表
  → ReAct 记忆维护 Agent
  → 只读工具补充旧记忆、主题、冲突、原始证据
  → 输出 memory-changes-v1
  → 本地程序校验 ID、字段、权限、幂等和事务
  → PostgreSQL memory_topics / memories / revisions / conflict records
  → dry-run 日志、Markdown/报告审查
```

新链路不要求模型直接写数据库。模型只负责判断本轮 turn 是否需要沉淀、该修改哪条记忆、是否需要先补证据；正式动作始终由本地程序执行和校验。

当前 P0 质量策略：

- 只有新证据能用目标 memory 的同一句核心结论解释时，才允许 `merge_memory`；
- 新信息形成独立规则时，应 `create_memory`，不要强塞进宽泛旧 memory；
- 用户协作习惯、工程流程、编译失败经验、运行现象经验尽量保持原子化；
- 如果理想动作是修改旧 memory，但目标 memory 未读取、没有修改权限，必须先请求读工具，不能用 `noop` 掩盖。

## 输入边界

当前唯一输入源为：

```text
%USERPROFILE%\.local\share\codepulse\opencode-dev.db
```

Memory Lab 对该文件只读。证据包包含会话中可观察到的内容：

- 用户消息；
- Assistant 面向用户的最终文本；
- 会话里记录的所有工具事件及其状态、受限输入、受限输出和错误；
- 会话、根会话、项目别名和真实来源 ID。

以下内容不进入模型：

- Assistant 的 Thinking / Reasoning；
- API Key、Token、密码、私钥等敏感值；
- 本机绝对路径；
- 超过本地总预算的大段工具载荷。

原始 CP 会话不会复制到 Markdown。正式证据保存的是脱敏投影以及可回查的
turn/session/message/part/source 标识。

## 如何判断和更新记忆

每条候选必须只有一个中心事实，并引用会话里的精确证据片段。当前正式支持三种范围：

- `global`：用户原话明确表示跨项目、默认或始终适用；
- `domain`：证据明确限定一个领域；
- `project`：只在来源项目中适用。

当前动作规则：

- 相同 Stable Key、范围和语义：不新建记忆，只增加全部证据；
- 相同 Stable Key、不同范围或条件：建立范围变体，互不污染；
- 同范围用户要求出现明确新版本：按真实消息/part/文本位置采用最后要求，旧版本保留为
  `superseded` revision；
- 工程或技术结论相反：不按时间覆盖，登记未解决冲突；
- Reviewer 有效拒绝：正式记忆零写入，但该批回合记为已评估，避免永久重复处理；
- 模型超时、连续两次非法结构、伪造证据、CAS 失效：正式记忆零写入，也不推进成功游标。

通用会话链路中的工具结果最多标记为 `observed`；`verified` 保留给未来理解特定编译器、测试器
或校验工具语义的本地确定性验证器，不能仅凭工具状态为 completed 或输出中出现 0 errors 授予。

`cross_project` 的自动晋级暂不开放。现阶段两个项目验证出的经验先保存为两个隔离的 project
变体；下一阶段再加入多项目证据门槛、跨项目 Markdown 表达和召回规则。

## 三层产物

### 1. 正式 SQLite

实验数据库位于 `data\*.db`，它是事实源和审计账本，保存：

- 哪些完成回合已经处理；
- Writer 计划、Reviewer 结果和失败状态；
- `memory / memory_revision / evidence / conflict`；
- Stable Key、scope、条件、来源顺序、CAS 和回滚链。

### 2. 正式 Markdown

```text
memory/
├── CURRENT
└── generations/
    └── <generation-id>/
        ├── MANIFEST.json
        ├── MEMORY.md
        ├── <动态主题>.md
        └── conflicts/
            ├── pending_conflicts.md
            └── resolution_history.md
```

`MEMORY.md` 只保存主题名、语义概述、类型、更新时间和文件指针；主题文件保存当前有效条目、
Stable Key、范围、条件、revision、验证状态和证据数量。发布时先生成完整代际并校验哈希，最后
原子切换 `CURRENT`；相同正式状态不会重复创建代际。

### 3. 中文变更报告

每次运行会在报告目录生成：

```text
latest-memory-change.md
latest-memory-change.html
latest-memory-change.json
runs/<run-id>.*
```

报告展示本轮涉及的项目、完成回合、工具事件、候选数量、Reviewer 结果、正式动作以及写入前后
的记忆/修订/证据/冲突数量。报告不是事实源，但最适合检查和演示。

## 从零开始一次真实实验

先进入 Memory Lab：

```powershell
Set-Location '<Codepulse-Memory-Lab 仓库目录>'
```

需要 Python 3.11 或更高版本。`memory-lab.ps1` 会依次使用：

1. 环境变量 `MEMORY_LAB_PYTHON` 指向的 `python.exe`；
2. 当前用户目录下的 Codex bundled Python；
3. `PATH` 中的 `python`。

运行测试：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest tests.test_memory_react_prompt tests.test_memory_changes_validation tests.test_real_flow_memory_plans tests.test_pg_memory_search tests.test_pg_memory_versioning
```

当前核心定向回归期望：

```text
100 tests OK
```

PostgreSQL ReAct dry-run 示例：

```powershell
.\memory-lab.ps1 pg-memory-react-run `
  -PgConfigPath .\pg-memory-lab.local.json `
  -TurnId '<turn_id>' `
  -MemoryLimit 30 `
  -TopicLimit 5 `
  -MaxIterations 3 `
  -DryRun `
  -AgentLogDirectory .\reports\react-dry-run
```

在开始新的 CP 会话之前建立基线。基线只表示“当前已有历史不参加这次实验”，不会修改 CP：

```powershell
.\memory-lab.ps1 baseline `
  -LabDatabase .\data\real_projects_20260911.db
```

然后在一个或多个真实项目中正常使用 CP。等 Assistant 已给出最终回复后，运行：

```powershell
.\memory-lab.ps1 session-memory `
  -LabDatabase .\data\real_projects_20260911.db `
  -ConfigPath .\memory-lab.config.json `
  -FormalMemoryRoot .\real-project-memory-20260911 `
  -ReportDirectory .\reports\real-projects-20260911
```

该命令会处理最早的一批未处理完成回合。重复运行是安全的：已经成功处理的回合不会重复沉淀，
同一个 reviewed batch 在异常恢复时也不会重复写正式证据。

演示时主要查看：

```text
real-project-memory-20260911\CURRENT
real-project-memory-20260911\generations\<CURRENT 指向的代际>\MEMORY.md
reports\real-projects-20260911\latest-memory-change.html
```

本轮真实实验建议使用 2 个项目、至少 6 个根会话和 16–24 个完成回合，覆盖同义合并、项目范围
隔离、明确更正、工程失败—修复—验证和负样本。可直接使用的自然语言交互脚本、验收表和明天的
展示顺序都在 [两项目真实测试方案](REAL_PROJECT_TEST_PLAN.md) 中。

## DeepSeek 配置

普通配置位于 `memory-lab.config.json`。API Key 只放在本机忽略文件
`memory-lab.secrets.local.json`：

```json
{
  "schema_version": "memory-lab-secrets-v1",
  "deepseek_api_key": "在本机填写已轮换的新 Key"
}
```

不要在演示、报告或提交中打开该文件。任何曾粘贴到聊天或日志中的 Key 都应撤销并更换。

## GitHub 中不会包含的本地数据

`.gitignore` 明确排除了：

- DeepSeek 本地凭据与 `.env`；
- `data/` 下的真实会话实验数据库及 SQLite sidecar；
- `memory/`、`reports/` 和其他生成的记忆代际；
- Python 缓存、虚拟环境和构建产物；
- 从本地语料复制、授权状态不明确的 `test-projects/`。

因此 GitHub 仓库保存的是可复现实验程序、自动化测试、合成 Gold 数据和文档，不是用户的真实
会话内容。若需要公开示例，应单独生成完全合成、脱敏的 `examples/`，不要提交本机运行产物。

## 当前不在本阶段的能力

- 不向 CP 主 Agent 自动注入记忆；
- 不从项目文件、Git、配置文件或独立编译日志补充证据；
- 不做正式查询/召回效果验收；
- 不自动把两个项目的相似经验提升成 `cross_project`；
- 尚未完成 14–28 天长期质量评测和 Recall A/B。

旧的 `cycle / watch / shadow / catalog / markdown-preview` 命令仍保留用于历史回归，不是当前
通用沉淀主链。当前主入口是 `session-memory`。
