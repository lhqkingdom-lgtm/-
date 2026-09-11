# 三窗口综合实验演示手册

这份手册直接复用已经完成的三窗口 CP 实验，不需要重新向 CP 发送消息。演示使用冻结数据库
`data\full_suite_20260911.db`，展示“跨会话提取结果如何变成正式记忆，以及新任务如何按项目
范围找到正确记忆”。

## 演示目标

用一组数据同时展示：

1. 多会话、多项目的用户要求已经被增量抓取并原子拆分；
2. 同义要求合并证据，不重复建记忆；
3. 项目例外和全局要求同时保存，召回时不跨项目污染；
4. 同范围的新长期要求修订旧值，旧 revision 仍可审计；
5. 临时要求和 CP 已经覆盖的规则不进入正式记忆；
6. 正式 SQLite 能发布为 `MEMORY.md + 主题 Markdown`；
7. 相同数据重跑不新增 memory/revision/evidence，也不创建新 Markdown generation。

## 已冻结的实验结果

| 项目 | 当前结果 |
|---|---:|
| CP 项目数 | 3 |
| 用户消息 / 稳定回合 | 10 / 10 |
| 原子候选 | 19 |
| Writer/Reviewer 接受 / 拒绝 | 9 / 10 |
| 正式 memory / revision / evidence / conflict | 5 / 6 / 9 / 0 |
| Markdown 主题 | 3 |

首次正式运行：

```text
automemory_69ce13ae338e4109ab94aa2019d7ff87
```

正式 generation：

```text
gen-00ce409c9e49fef57e8e0f9714f6f0e2a266f01223212fe5d83c0f8a5d4d01e8
```

正式动作是：创建 4、创建项目 scope 变体 1、合并同义证据 3、修订 1。10 个拒绝项来自
本地硬过滤，包括临时要求和 CP 已有指令覆盖，不是“模型写失败”。

## 0. 演示前准备

在 PowerShell 中执行：

```powershell
Set-Location 'D:\codepulse\Codepulse-Memory-Lab'

$DemoDb = '.\data\full_suite_20260911.db'
$DemoMemoryRoot = '.\full-suite-memory'
$DemoReportDir = '.\reports\full-suite-20260911'
$DemoRunId = 'automemory_69ce13ae338e4109ab94aa2019d7ff87'
```

确认四个输入存在：

```powershell
Test-Path $DemoDb
Test-Path "$DemoMemoryRoot\CURRENT"
Test-Path '.\memory-lab.config.json'
Test-Path '.\memory-lab.secrets.local.json'
```

预期都是 `True`。不要在投屏时打开 `memory-lab.secrets.local.json`。

## 1. 先展示本轮变更报告

从已经完成的正式运行重建报告，不重新请求 CP，也不重新调用 Writer：

```powershell
.\memory-lab.ps1 change-report `
  -LabDatabase $DemoDb `
  -RunId $DemoRunId `
  -ReportDirectory $DemoReportDir `
  -FormalMemoryRoot $DemoMemoryRoot
```

打开可视化报告：

```powershell
Invoke-Item "$DemoReportDir\latest-memory-change.html"
```

重点讲四块：

- Formal state：`0 -> 5 memory`、`0 -> 6 revision`、`0 -> 9 evidence`；
- Writer / Reviewer：两次独立模型判断，而不是模型直接改文件；
- Applied actions：create、scope variant、merge、revision；
- Markdown publication：正式 generation、3 个主题、5 条 active memory。

如果报告命令失败，不要改用旧 `reports\demo-*` 报告。旧报告属于其他实验数据库，不能证明
这次三窗口结果。

## 2. 展示正式 Markdown 长什么样

解析 `CURRENT`，再打开该代际的索引：

```powershell
$DemoGeneration = (Get-Content -Raw "$DemoMemoryRoot\CURRENT" | ConvertFrom-Json).generation_id
$DemoGeneration
Get-Content -Raw "$DemoMemoryRoot\generations\$DemoGeneration\MEMORY.md"
```

预期索引有三个主题：

- `user_response_preferences.md`：回复语言和“先结论后步骤”；
- `workflow_code_generation.md`：全局生成代码前确认 + A 项目例外；
- `workflow_embedded_compile.md`：单片机改代码后的编译检查。

继续查看两个最直观的主题：

```powershell
Get-Content -Raw "$DemoMemoryRoot\generations\$DemoGeneration\user_response_preferences.md"
Get-Content -Raw "$DemoMemoryRoot\generations\$DemoGeneration\workflow_code_generation.md"
```

应看到：

- 回复结构证据数为 2，说明两个窗口的同义表达被合并；
- 当前回复语言为英文、revision 为 2，说明后出现的同范围长期要求更新了旧值；
- 代码确认规则有两个 scope：global=`required`，A 项目=`not_required`；
- Markdown 只展示当前有效内容和证据数量，不复制用户完整会话或工具输出。

## 3. 用同一任务验证项目隔离

### 3.1 A 项目：项目例外生效

```powershell
.\memory-lab.ps1 formal-recall `
  -LabDatabase $DemoDb `
  -FormalMemoryRoot $DemoMemoryRoot `
  -Query '请在当前项目生成正式代码，实现串口功能。' `
  -ProjectDirectory 'D:\codepulse\Codepulse-Memory-Lab\test-projects\e1-keil-mj60b\cpu\MJ60B' `
  -Domain embedded `
  -Limit 5 `
  -TopicLimit 5
```

检查输出：

- `workflow.code_generation.confirmation` 选中 project=`not_required`；
- 同 Key 的 global=`required` 出现在 `suppressed_by_scope`，原因是
  `more_specific_scope_wins`；
- 同时召回当前回复语言、先结论后步骤、嵌入式编译检查；
- `unresolved_context` 和 `unresolved_conflicts` 均为空。

### 3.2 B 项目：不能继承 A 的例外

```powershell
.\memory-lab.ps1 formal-recall `
  -LabDatabase $DemoDb `
  -FormalMemoryRoot $DemoMemoryRoot `
  -Query '请在当前项目生成正式代码，实现串口功能。' `
  -ProjectDirectory 'D:\codepulse\Codepulse-Memory-Lab\test-projects\e1-keil-mj60b-b\cpu\MJ60B' `
  -Domain embedded `
  -Limit 5 `
  -TopicLimit 5
```

检查输出：

- 选中 global=`required`；
- A 项目=`not_required` 出现在 `excluded_by_context`，原因是 `different_project`；
- 回复偏好和嵌入式编译检查仍然正确召回。

这两条命令使用默认确定性主题路由，结果稳定、无需 API。若要展示 DeepSeek 只根据主题概述
选择最多 5 个相关主题，在同一命令末尾加：

```powershell
-ModelRouter -ConfigPath .\memory-lab.config.json
```

无论使用哪一种主题路由，项目 scope、revision、验证状态和冲突过滤都由同一确定性召回器
执行，模型不能绕过它们。

## 4. 验证幂等：相同数据不会重复沉淀

先保存当前指针：

```powershell
$DemoCurrentBefore = Get-Content -Raw "$DemoMemoryRoot\CURRENT"
```

重放已经处理过的准入快照：

```powershell
.\memory-lab.ps1 auto-memory `
  -LabDatabase $DemoDb `
  -ConfigPath .\memory-lab.config.json
```

预期：仍会新增一条“本次检查”的审计运行记录，但 `actions` 为空，正式状态仍是 5 memory、
6 revision、9 evidence、0 conflict。也就是说“正式业务对象零新增”，不是整个实验库绝对零写入。

再次发布：

```powershell
.\memory-lab.ps1 formal-publish `
  -LabDatabase $DemoDb `
  -FormalMemoryRoot $DemoMemoryRoot
```

预期关键字段：

```text
created_generation: false
current_changed: false
generation_id: gen-00ce409c9e49fef57e8e0f9714f6f0e2a266f01223212fe5d83c0f8a5d4d01e8
memory_count: 5
topic_count: 3
```

最后直接比较指针：

```powershell
$DemoCurrentAfter = Get-Content -Raw "$DemoMemoryRoot\CURRENT"
$DemoCurrentBefore -eq $DemoCurrentAfter
```

预期为 `True`。

## 5. 如被问到工程成功/失败经验

可以展示 E1 已经采到什么，但必须明确它还没有进入正式记忆：

```powershell
$Engineering = .\memory-lab.ps1 engineering-candidates -LabDatabase $DemoDb |
  Out-String |
  ConvertFrom-Json

$Engineering |
  Select-Object bundle_count, verification_counts, formal_memory_written, semantic_boundary |
  Format-List
```

当前预期：

```text
bundle_count: 10
verification_counts: unverified = 10
formal_memory_written: False
```

准确讲法是：“已经能从内部工具过程采集脱敏、可追溯的 E1 工程证据；本批没有结构化编译/
测试成功信号，因此系统没有冒充成功经验。E2 的症状、原因、方案语义提炼和正式审核尚未实现。”

不要说“工程成功/失败经验已经自动沉淀并召回”。

## 建议的 5 分钟讲解顺序

1. **30 秒**：说明 CP 数据库只读、CP 核心零修改，当前是独立价值验证。
2. **60 秒**：打开 HTML 报告，讲 19 个原子候选如何变成 5 条正式记忆。
3. **60 秒**：打开 `MEMORY.md` 和两个主题文件，讲索引/正文/SQLite 各自作用。
4. **90 秒**：对 A、B 两个项目运行同一句查询，演示项目例外不污染别的项目。
5. **45 秒**：重跑发布，展示 generation 和 `CURRENT` 不变。
6. **15 秒**：如实说明 E1、E2 和 CP 注入边界。

## 本次演示通过标准

- 报告对应运行 ID `automemory_69ce...`，不是旧 demo；
- `MEMORY.md` 为 3 个主题、正式 active memory 为 5；
- A 项目选择 `not_required`，B 项目选择 `required`；
- 两个查询都召回嵌入式编译检查，且无未解决上下文或冲突；
- 幂等运行不增加正式业务对象，重复发布不创建 generation；
- 工程候选明确显示 `formal_memory_written: false`；
- 全程不修改 CP SQLite，也不宣称已经注入 CP Agent。

## 演示后继续做新 CP 数据时

不要继续向冻结的 `full_suite_20260911.db` 混入新实验。为下一批测试创建单独的 Lab DB，先在
发送 CP 消息前建立基线，再运行完整链路：

```powershell
.\memory-lab.ps1 baseline -LabDatabase .\data\next_experiment.db

# 用户在多个 CP 窗口完成新的自然交互并等到最终回复后：
.\memory-lab.ps1 cycle `
  -LabDatabase .\data\next_experiment.db `
  -ConfigPath .\memory-lab.config.json `
  -FormalMemoryRoot .\next-experiment-memory `
  -ReportDirectory .\reports\next-experiment
```

当前尚未接入 `session.idle`，所以“CP 这一轮完成”仍由实验人员观察最终回复后触发 `cycle`，
或提前启动 `watch` 轮询数据库结构稳定状态。
