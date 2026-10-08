# 变更记录

## 1.0.0-candidate.4 — 2026-10-08

- 增加 `chain_operations[]` 多链决策协议，支持同一 turn 更新既有故障链并创建另一条独立新链。
- 每个链操作显式声明 `update_existing` / `create_new`，正式技术记忆通过 `linked_change_indexes` 精确归属到闭环链。
- PostgreSQL 新链 ID 加入 operation creation key；同类型、同 turn 的不同故障不会再因重叠匹配而串链。
- 增加多链协议、落库、部分失败、真实评测契约和边界测试；全量 502 项测试通过。
- 使用场景 10 的真实 turn 完成 PostgreSQL 集成验证：旧链关闭、新链独立开放。但真实 LLM 全量回放被 DeepSeek HTTP 402 阻断，因此仍保持 9 个发布门禁、1 个 known gap，不提前宣称场景 10 通过。

## 1.0.0-candidate.3 — 2026-10-08

- compilation 故障链新增 `fix_rejected` 状态，支持“修复已实施→原问题仍存在→重新定位/修复”循环。
- 强化编译闭环边界：验证必须对应原始失败步骤；烧录/下载错误不得仅凭上游编译 0 Error 或 HEX 生成提前闭环。
- 故障链 metadata 保留 `transition_history` 与 `rejected_attempt_count`，失败尝试不进入正式记忆，但可通过关联 turn 证据追溯。
- 真实 9 场景回放 26/26 turn 通过；编译修复被否定场景两轮均 noop，同一 `chain_id` 从 `fix_applied` 转为 `fix_rejected`，正式记忆数为 0。
- 将该场景从已知缺口提升为第 9 个发布门禁；当前仍保留 1 个已知缺口：同一 turn 同时关闭旧链与新建另一条链。

## 1.0.0-candidate.2 — 2026-09-28

- 人工回查第三个偏好 turn 的原始证据，确认其中同时存在 Keil 工具 Fatal Exit、同机直接命令构建成功和云构建成功。
- 保持“最终只有一条用户偏好、不得新增正式编译记忆”的金标准，但补全真实理由：失败根因尚未定位，不能把错误码通用释义当作许可证/环境根因。
- 允许该 turn 记录 `compilation/failure_observed` 未闭环链；证据包与 SHA-256 未改变。

## 1.0.0-candidate.1 — 2026-09-28

- 新建独立多轮评测集，不修改 `memory_eval_v1` 历史基线。
- 固定三类正式记忆及编译/上板闭环条件。
- 首批纳入 8 个发布门禁场景和 2 个已知缺口观察场景。
- 所有场景来自真实 CP 数据；未使用合成对话。
- 引入初始状态、逐 turn 状态、必读历史证据、禁止动作和证据包哈希。

当前状态为 `candidate`。完成静态契约校验、证据哈希校验和人工确认后才能标记为 `frozen`。
