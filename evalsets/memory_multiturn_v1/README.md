# memory_multiturn_v1

这是面向当前多轮故障链能力的新评测集。它与旧的 `memory_eval_v1` 独立，旧集继续作为历史基线，不再用旧的 candidate 规则约束当前链路。

## 内容

- 10 个真实连续场景；
- 29 个唯一真实 turn；
- 8 个当前发布门禁场景；
- 2 个已知缺口观察场景；
- 0 个合成对话场景。

详细规则见 `SPEC.md`，人工可读场景见 `场景目录.md`，机器判定见 `scenarios.json` 和 `cases.jsonl`。

## 静态与证据校验

```powershell
powershell -ExecutionPolicy Bypass -File .\memory-lab.ps1 eval-memory `
  -LabDatabase .\data\eval\memory_eval_index.db `
  -EvalsetPath .\evalsets\memory_multiturn_v1 `
  -DryRunPlan
```

该命令会同时检查：

- 用例字段和场景引用；
- 29 个 turn 是否均存在；
- 原始数据库是否可访问；
- 初始状态、步骤顺序和历史 turn 引用是否合法；
- 未闭环步骤是否禁止正式记忆写入；
- 证据包 SHA-256 是否与冻结值一致。

当前版本仍是 `candidate`。人工确认场景语义后，将 manifest 状态改为 `frozen`，后续新增场景进入新版本，不在模型跑分后反改本版本金标准。
