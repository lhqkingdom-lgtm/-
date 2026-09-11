# E1 工程证据采集验收方案

## 本阶段只验什么

E1 只验证 Memory Lab 能否从 CodePulse 已结束的任务中，增量得到安全、可追溯的工程
Episode。它不调用大模型，不生成长期经验 Markdown，也不改变 CodePulse。

本阶段输出的是“真实发生了什么”，下一阶段 E2 才让 DeepSeek 判断“这些事实说明了什么、
是否值得沉淀”。

## 数据链路

```text
CodePulse SQLite（只读）
        ↓
已完成/失败且结构稳定的 conversation turn
        ↓
按 user_message_id → assistant.parentID 重建工具顺序
        ↓
两次白名单结构采样 + 脱敏 reducer
        ↓
trace_episode + trace_episode_event（仅候选/审计）
```

## 自动预验收

在要求用户创建新会话前，先用现有数据库检查：

1. 能扫描已有稳定回合。
2. `tool completed` 不被误当作工程成功。
3. 能识别 Keil、测试/构建命令、文件修改和工具错误的结构化信号。
4. 重复执行不会重复新增 Episode 或 Evidence。
5. CodePulse 源数据库保持只读。
6. Memory Lab 数据库中不出现 reasoning、完整命令、完整工具输出或认证信息。
7. Keil 结果必须同时满足 completed、合法 start/end、时间顺序正确、metadata 自洽且无
   error payload；否则只能是 ambiguous 或 unverified。
8. 每次运行都记录 created / existing / unstable / skipped，能解释为什么写入或跳过。

自动预验收状态（2026-09-10）：已通过。最终提取器为 `cp-trace-episode-v3`；专项测试
10 / 10、全量测试 143 / 143。真实旧实验库首次写入 14 条 v3 Episode，再次运行为
0 条新增、14 条复用；所有版本的 payload hash/精确长度非空行均为 0。详见
`reports/E1_PREVALIDATION.md`。

自动预验收未通过时，不要求用户新增 CP 会话。

## 用户验收任务

自动预验收通过后，由 Memory Lab 给出适合当时测试项目的准确任务句。原则上覆盖以下两类，
用户不需要添加 `ML` 测试编号：

### E1-A 真实验证结果

在一个安全的测试项目中，让 CP 实际执行已有的编译或测试，而不是只阅读代码后判断结果。

预期：Episode 中出现真实验证器、结果、计数或退出码；Assistant 说“完成”本身不算验证。

### E1-B 修改后验证

在安全测试项目中完成一个很小且可撤销的修改，然后实际编译或测试。若自然发生失败，可以让
CP 继续修复并再次执行同一种验证；若一开始成功，不故意制造故障。

预期：Episode 能按顺序显示“修改 → 验证”。若出现失败后再次验证，E1 只保留各次验证
信号；在缺少同一 target/config 身份前，不宣称它们构成“恢复成功”。

## 通过标准

- 每个稳定完成任务最多形成一个 E1 trace Episode；未结束回合不提取。
- 工具顺序、工具状态以及编译/测试结果与 CP 会话一致。
- `verified_success` 必须有结构化成功证据。
- 同一 Episode 同时出现成功和失败信号时标为 `ambiguous`；普通工具报错不能自动成为长期
  失败经验，也不推断 `recovered`。
- 最后一次决定性验证后若又发生非明确只读工具调用，结果降为 `unverified`。
- Evidence 都能通过 `source_id/session_id/message_id/part_id` 回查。
- 重复运行新增 Episode 和 Evidence 均为零。
- reasoning、密钥、完整日志、完整命令和完整代码落盘数量为零。
- E1 记录保持 `candidate/audit only`，不得进入正式 Markdown 或提供给主 Agent。

任何一项不满足，E1 不通过，先修正后重测，不进入 E2。

## 后续节点

E1 通过后进入 E2：把脱敏 Episode 提供给 DeepSeek，提取 `symptom / conditions /
attempts / cause / solution / verification`，并由程序回查 Evidence 后决定 `verified` 或
`pending`。
