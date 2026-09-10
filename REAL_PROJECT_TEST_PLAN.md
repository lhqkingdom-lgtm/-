# 两个真实项目的长期记忆沉淀测试方案

## 1. 这次测试要证明什么

这轮只验证“沉淀链路”，暂不验证向 CP 主 Agent 自动召回和注入。

```text
两个真实项目、多窗口、多根会话
  → CP SQLite 中新增的已结束回合
  → DeepSeek Writer 提取候选
  → 独立 Reviewer 审核
  → 程序处理新建、合并、范围变体、修订和冲突
  → 正式 SQLite
  → MEMORY.md、主题 Markdown、中文变更报告
```

建议样本规模：

- 2 个不同目录的真实项目；
- 每个项目至少 3 个根会话窗口；
- 总计 16–24 个已结束回合，不超过单批上限 32；
- 至少 10 次真实工具事件，包括读取、修改、编译或测试；
- 至少各出现一次用户契约、工程失败—修复—验证、同义复述、项目例外、明确更正和临时要求。

这里追求的是信息质量，不是强行凑足记忆条数。普通进度、闲聊和一次性要求应该被拒绝。

## 2. 项目选择

推荐使用安全副本，不要直接拿生产目录做故障实验。

### 项目 A：SensorGuard

推荐从空目录用 CP 构建一个只依赖 Python 标准库的传感器 CSV 质检 CLI：

- 输入至少包含 `timestamp`、`device_id`、`temperature`；
- 输出合法行数量和逐行错误；
- 使用 `python -m unittest discover -s tests -v` 验证；
- 安排一次 UTF-8 BOM 导致首列表头识别失败的真实测试—修复过程。

### 项目 B：ConfigShift

推荐从另一个空目录用 CP 构建一个只依赖 Node.js 标准库的 JSON 配置迁移 CLI：

- 把 `schema_version: 1` 转换成版本 2；
- 默认生成同目录的 `.migrated.json`，不覆盖输入文件；
- 使用 `node --test` 验证；
- 安排一次 UTF-8 BOM 导致 `JSON.parse` 失败的真实测试—修复过程。

两个项目都处理 BOM，但语言、运行时、失败位置和修复方式不同，适合检查 Memory Lab 是否会因
表面症状相似而错误地跨项目合并。若本机没有 Node.js，可把项目 B 换成另一个具备自动测试命令的
真实项目，但仍需保证技术栈和目录与项目 A 不同。

两个项目必须位于不同的绝对目录。CP 从哪个目录启动，Memory Lab 就把该目录识别成哪个
project scope。

```cmd
mkdir D:\MemoryLabRealTest\20260911\sensor-guard
cd /d D:\MemoryLabRealTest\20260911\sensor-guard
codepulse
```

另开一个窗口：

```cmd
mkdir D:\MemoryLabRealTest\20260911\config-shift
cd /d D:\MemoryLabRealTest\20260911\config-shift
codepulse
```

不要在对话中粘贴真实 API Key、密码、Token、客户数据或私有证书。

## 3. 已建立的实验基线

当前实验库已经在真实 CP 历史之后建立基线：

```text
data\real_projects_20260911.db
```

基线前的会话不会进入本轮实验。从现在开始产生的新根会话才是测试样本。若以后重新开一轮
独立实验，应在开始新对话之前换一个新的 Lab DB 执行：

```powershell
.\memory-lab.ps1 baseline -LabDatabase .\data\another_real_test.db
```

## 4. 建议的交互顺序

以下句子可以直接自然地发给 CP，但不要添加 `A1`、`TEST` 等人工标签。每条消息都等待 CP 给出
最终回复后再发送下一条；CP 弹出选择题时按真实情况回答。

### 第一阶段：项目 A，建立首批记忆

在项目 A 的第一个根会话中发送：

> 以后无论在哪个项目，回复我时先用一句话给出结论，再列出关键处理步骤。

> 以后在所有项目中，生成或修改正式代码前都先给我方案，并等待我明确确认后再修改；这是一条跨项目长期规则。

> 请在当前空目录规划 SensorGuard：一个只使用 Python 标准库的传感器 CSV 质检命令行工具。输入至少包含 timestamp、device_id、temperature 三列，输出合法行数量和错误信息。当前项目的长期产品规则是批处理遇到第一条无效行就立即失败。先给出目录结构、实现方案和验收测试，不要开始写文件。

> 我确认这个方案，请建立源码、单元测试和 README，并运行 `python -m unittest discover -s tests -v`。

在项目 A 的第二个根会话中发送：

> 先检查当前 SensorGuard 项目。以后仅在这个项目中，只要修改 CSV 解析或校验规则，都必须运行 `python -m unittest discover -s tests -v`，最终回复中说明测试数量和结果。现在只检查基线，不做无意义修改。

> 真实设备导出的 CSV 可能在首列表头前带 UTF-8 BOM。请先给测试优先的修改方案：先添加能复现问题的最小回归测试并确认原行为，再做最小修复，最后运行完整测试集；现在不要修改文件。

> 我确认这个方案，请现在执行。不能删除测试或降低断言，也不要顺便扩大支持格式。最终回复要说明症状、根因、失败尝试、有效修复和真实测试结果。如果新增测试第一次就通过，必须如实说明，不能伪造失败。

完成项目 A 后，等待 CP 的最终回复和所有工具调用结束，停 3–5 秒，然后执行第一次沉淀：

```powershell
Set-Location '<Memory Lab 目录>'
.\memory-lab.ps1 session-memory `
  -LabDatabase .\data\real_projects_20260911.db `
  -ConfigPath .\memory-lab.config.json `
  -FormalMemoryRoot .\real-project-memory-20260911 `
  -ReportDirectory .\reports\real-projects-20260911
```

预期：新建全局回复偏好、全局代码确认契约、SensorGuard 产品规则、项目测试工作流，以及至少一条
带真实工具证据的 Python 工程记忆。

### 第二阶段：项目 B，验证跨会话合并和范围隔离

在项目 B 的第一个根会话中，同义复述全局偏好：

> 我再次确认一下：所有项目的回复都先给一句简短结论，然后再说明步骤。

这应当给已有全局偏好增加证据，而不是建立第二条重复记忆。

再让 CP 规划并构建项目 B：

> 请在当前空目录规划 ConfigShift：一个只使用 Node.js 标准库的 JSON 配置迁移命令行工具，把 schema_version 1 转成 2。当前项目长期规则是绝不覆盖输入文件，默认输出同目录的 `.migrated.json`。先给目录结构、实现方案和验收测试，不要写文件。

> 我确认这个方案，请建立源码、测试和 README，并使用 `node --test` 运行全部测试。

> 以后仅在这个项目中，只要修改配置迁移规则，都必须运行 `node --test`，最终回复中说明通过和失败数量。现在检查现有实现；如果测试通过，不要做无意义修改。

在项目 B 的第二个根会话中发送：

> 真实配置文件可能以 UTF-8 BOM 开头。请先给测试优先方案：用最小回归测试确认直接 `JSON.parse` 的当前行为，只修复开头恰好一个 BOM 的情况，不能顺便接受尾逗号；现在不要修改文件。

> 我确认方案，请执行并运行完整 `node --test`。最终说明症状、失败位置、有效修复、为什么没有放宽尾逗号规则和真实测试结果。如果原测试没有失败，必须如实说明。

> 请只复盘刚才 ConfigShift 的 BOM 问题，不修改文件。说明触发条件、失败方案、有效方案、验证命令和可复用范围。

完成项目 B 后执行第二次 `session-memory`。使用和第一阶段完全相同的 Lab DB、Memory Root 和
Report Directory。

预期：

- 全局回复偏好只增加证据；
- 项目 B 工作流不会覆盖项目 A；
- 项目 B 的失败—修复经验形成独立 project scope 工程记忆；
- Python 和 Node 的 BOM 问题不会因为症状相似而错误合并。

### 第三阶段：更正、项目例外和负样本

在项目 A 的第三个根会话中明确更正产品规则：

> 明确更正 SensorGuard 的长期产品规则：默认批处理不再遇错立即终止，而是跳过无效行并汇总全部错误；只有显式使用 `--strict` 时才在第一条无效行停止。这条规则替代之前的默认遇错立即失败。现在只给修改方案，不改文件。

> 我确认刚才的方案，请实施，更新 CLI、README 和回归测试，并运行完整单元测试。最终说明旧行为、新行为和测试结果。

预期：同一 project scope 的最新明确产品要求形成新 revision，旧 revision 保留但变为 superseded，
不能同时作为两条 active 记忆存在。

在项目 B 的第三个根会话中制造项目例外：

> 仅在当前 ConfigShift 项目中，普通迁移代码的小范围修复可以说明计划后直接实施，不需要等待我再次确认；删除数据、改变公开 schema 或破坏兼容性时仍要先确认。这个例外不要应用到其他项目。

这应成为全局“改代码前确认”在 ConfigShift 项目中的范围变体，不应污染 SensorGuard。

继续测试 ConfigShift 的明确修订：

> 明确更正 ConfigShift 的长期输出规则：默认仍输出 `.migrated.json`，但允许用户显式使用 `--in-place`；使用该参数时必须先在同目录生成 `.bak`，再原子替换。这条规则替代“任何情况下都绝不覆盖输入文件”。现在只给方案，不修改文件。

> 我确认方案，请实施 `--in-place`、备份和原子替换，补充失败场景测试与 README，然后运行完整 `node --test`。

再更正一条全局偏好：

> 更正我的跨项目回复习惯：以后仍使用中文，但顺序改为先一句结论，再单独列风险，最后列步骤。这条新规则替代之前结论后直接列步骤的规则。

最后加入两个负样本：

> 仅限这一次，不运行测试，也不要修改文件。请只解释 Node 中原子替换通常如何实现；下一次真正修改迁移代码时仍然必须执行 `node --test`。

> 今天天气不错，我们先停一下。

预期：一次性要求和闲聊均为零正式写入。

完成后执行第三次 `session-memory`。

## 5. 如何判断一个回合已经结束

不需要发送“任务完成”标签。Memory Lab 根据数据库中的可观察状态判断：

- Assistant 已产生终止型最终回复；
- 本轮所有工具调用均已结束；
- 回合内容连续两次读取保持稳定；
- 或者回合明确失败、被后续用户消息中断。

为了避免正在落库的最后一段文本被延后处理，建议看到最终回复后等待 3–5 秒再运行沉淀命令。
尚未稳定的回合不会丢失，只会留到下一次扫描。

## 6. 检查生成结果

读取当前代际：

```powershell
$Pointer = Get-Content .\real-project-memory-20260911\CURRENT -Raw | ConvertFrom-Json
$Generation = Join-Path .\real-project-memory-20260911\generations $Pointer.generation_id
Get-Content (Join-Path $Generation 'MEMORY.md')
```

打开中文报告：

```powershell
Start-Process .\reports\real-projects-20260911\latest-memory-change.html
```

报告历史位于：

```text
reports\real-projects-20260911\runs\<run-id>.html
```

每一个阶段都应当产生新的不可变 generation；旧代际仍可用于对照和回滚。

## 7. 本轮验收标准

### 数据量门槛

- 正好 2 个目标 project key，没有第三个测试目录混入；
- 至少 6 个根会话；
- 16–24 个已结束回合；
- 至少 10 个工具事件；
- 建议得到 6–12 条高质量 active 记忆。条数不是硬凑指标，漏掉重要内容或误存噪声才算失败。

### 行为门槛

| 场景 | 预期结果 |
|---|---|
| 第一次长期要求 | 新建记忆 |
| 跨项目同义复述全局偏好 | 合并证据，不重复新建 |
| 相同概念、不同项目流程 | 保存 project scope 变体 |
| 同项目明确更正 | 新 revision 生效，旧 revision superseded |
| 工程失败—修复—工具验证 | 保存工程经验及全部引用证据 |
| 不同环境下不同工程结果 | 拆分条件或范围，不按时间覆盖 |
| 无法解释的同条件技术矛盾 | 打开冲突并阻断正常使用 |
| 一次性请求、闲聊 | 零正式写入 |

### 安全与一致性门槛

- CP SQLite 写入次数为 0；
- 正式记忆的证据可追溯率为 100%；
- 报告中“与本轮最终 SQLite 状态一致”和“与当前正式 SQLite 投影一致”均为“是”；
- 本机绝对路径、真实凭据和完整工具载荷不能进入 Markdown；
- 项目 A 的规则不能污染项目 B，项目 B 的规则不能污染项目 A；
- 工具结果在本阶段最多标记为 `observed`，不能冒充确定性 `verified`。

## 8. 幂等性终检

所有会话处理完成后，在没有新增 CP 消息的情况下再次运行同一条 `session-memory` 命令。

必须满足：

- `status` 为 `noop`；
- `processed_turn_count` 为 0；
- 正式 `memory / memory_revision / evidence / conflict` 的数量和哈希不变；
- 不创建重复业务对象。

## 9. 明天的展示顺序

1. 展示两个项目目录和多个 CP 根会话，说明输入只来自 CP SQLite。
2. 打开第一、第二、第三次中文历史报告，展示记忆数量逐步变化。
3. 打开当前 `MEMORY.md`，展示主题索引而不是整段会话。
4. 打开一条用户偏好主题，展示跨项目同义证据已经合并。
5. 打开项目 A、B 的工作流主题，展示 project scope 隔离。
6. 展示项目 A 的修订链：旧规则 superseded，新规则 active。
7. 展示至少一条工程经验，包括失败条件、解决方法、验证状态和证据数量。
8. 最后重复扫描，展示 `noop`，证明不会重复沉淀。

GitHub 仓库只保存程序、测试、合成样例和本方案。真实实验数据库、记忆代际、报告、密钥以及
复制的工程目录均由 `.gitignore` 排除，演示时只在本机打开。
