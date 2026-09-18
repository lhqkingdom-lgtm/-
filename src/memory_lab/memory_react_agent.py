"""ReAct-style memory maintenance agent.

This module replaces the fixed one-shot writer flow with a small, bounded
reason/act loop:

turn evidence + memory indexes -> model asks for read tools or final changes
-> local code validates/executes read tools -> model emits memory-changes-v1
-> existing deterministic validator/committer handles writes.

The model never receives a write tool.  Tool calls are JSON requests that must
pass local validation, and final writes still go through ``memory_changes``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG
from .llm_client import OpenAICompatibleJsonClient
from .memory_agent import (
    DEFAULT_AGENT_RUN_DIR,
    _context_stats,
    _safe_run_id,
    _step,
    build_memory_agent_input,
)
from .memory_changes import (
    MEMORY_CHANGES_SCHEMA_VERSION,
    MemoryChangesValidationError,
    commit_memory_changes,
    validate_memory_changes,
)
from .pg_store import (
    DEFAULT_PG_CONFIG,
    expand_pointer,
    read_memory_detail,
    read_topic_detail,
    search_related_memories,
)


MEMORY_REACT_SCHEMA_VERSION = "memory-react-v1"
DEFAULT_REACT_RUN_DIR = Path(__file__).resolve().parents[2] / "reports" / "memory-react-runs"
ALLOWED_TOOLS = {
    "read_topic_detail",
    "read_memory_detail",
    "expand_pointer",
    "search_related_memories",
}


class MemoryReactValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


REACT_SYSTEM_PROMPT = """你是 Memory Lab 的 ReAct 长期记忆整理 Agent，不是普通摘要器。
你的职责是把一轮 CP 对话整理成“长期记忆维护计划”，让未来相似任务少问、少错、少返工。

你只能输出一个严格 JSON 对象，不要输出 Markdown、解释、代码块或额外文本。

====================
一、你拿到的输入
====================

1. 当前 turn 的三层证据池：
- core：本轮最核心的用户反馈、结论、验证、关键要求。
- candidate：可能有价值的工具结果、过程摘要、工程上下文。
- pointer：可回查原始消息、工具输出、代码片段的指针。

2. 已有长期记忆索引：
- topic_index：主题摘要，只能用于发现候选主题。
- memory_index：记忆摘要，只能用于发现相似/重复/冲突候选。

3. observations：
- search_related_memories / read_memory_detail / read_topic_detail / expand_pointer 返回的详情。
- search_related_memories 只用于发现候选 memory，不代表已经读取正文，也不授权修改。
- 只有 observations 中已经读到正文的 memory，才算可安全修改。

4. memory_access_state：
- summary_only_memory_ids：只看过摘要，不能直接 update/merge/conflict。
- observed_memory_ids：已经读取过正文。
- modifiable_memory_ids：允许在 final_changes 中 update/merge/mark_conflict/record_conflict_decision。

====================
二、长期记忆沉淀标准
====================

长期记忆只保存“以后能减少重复问答、减少错误、减少返工”的信息。

优先沉淀三类：

1. 用户交互习惯 / 协作偏好
- 用户希望怎样推进、怎样测试、怎样展示。
- 用户明确纠正过的长期规则。
- 用户反复表达的不喜欢的工作方式。
例：用户不喜欢一组一组慢慢测，希望多窗口连续交互。

2. 编译成功/失败经验
- 错误症状、触发条件、根因、有效修复、验证方式。
- 工具链限制、芯片/SDK/编译器适配经验。
例：Keil C51 对现代 C 写法支持有限，修复必须符合 C51 语法和存储模型。

3. 用户反馈运行现象 → 定位原因 → 解决方法
- 用户反馈的板上现象。
- 排查路径。
- 代码层根因。
- 修复方法。
- 用户/编译/运行验证。
例：编译烧录通过但风扇不随温度变化，最终定位为 adc_sample 固定返回常量。

不要沉淀：
- 当前项目一次性需求参数，例如具体阈值、PWM 频率、功能选择。
- 本次生成了哪些函数、变量名、文件改动清单。
- 可从代码/Git/需求文档稳定恢复的信息。
- 未验证猜测。
- 完整日志、完整工具输出。
- 敏感信息。

判断口诀：
需求文档存“当前项目要做什么”；
长期记忆存“以后类似任务怎么少错、少问、少返工”。

用户偏好与项目参数的边界：
- 项目参数通常回答“这个项目最终做什么”，例如温度阈值、PWM 频率、NTC 规格、是否启用 TACH。
- 用户偏好/工程流程通常回答“以后类似任务应该怎么做”，例如既有工程二次开发时是否重建工程结构、是否修改启动文件、是否优先保持工具链兼容写法、交付时是否说明人工确认项。
- 即使流程性偏好出现在某个具体项目中，只要能复用于未来相似任务，就应该沉淀为长期记忆，并用 scope 或 content 中的适用场景限制范围。
- 不要因为流程性偏好只出现一次就直接 noop；明确命令型流程偏好可以低置信度沉淀。

对照案例：
- “本项目 PWM 使用 25kHz，温度阈值为 35/50/70” → noop，因为这是当前项目需求参数。
- “在现有 Keil C51 工程上生成代码，不要重建工程结构，不要改 STARTUP.A51，优先保持 C51 兼容写法” → create_memory，类型为 workflow/user_preference，适用场景写明“已有 Keil C51 工程二次开发”。
- “生成后说明模块划分和需要人工确认的硬件资源” → create_memory 或 merge 到同类用户偏好，因为这是可复用协作流程。

====================
三、动作选择标准
====================

create_memory：新增记忆
- 本轮信息有长期价值。
- 没有已有 memory 能覆盖。
- 可以形成清晰 stable_key。
- 有足够 evidence_refs。
例：第一次发现“硬件输入桩函数固定返回值会导致主流程跑通但输入恒定，下游状态机无响应”。

merge_memory：合并证据
- 已有 memory 是同一模式。
- 本轮只是新增证据、补充例子、增强置信度。
- 不改变原结论。
例：另一个项目也出现采样桩固定值导致状态不变化。

update_memory：更新已有记忆
- 已有 memory 方向正确。
- 本轮提供了更准确、更通用、更完整的表述。
- 需要改写 summary/content。
例：原记忆只写 ADC 桩导致风扇不转，本轮抽象为“硬件输入桩函数导致下游控制逻辑处理假数据”。

mark_conflict：标记冲突
- 同 scope、同条件、同对象下结论相反。
- 无法用项目、工具链、芯片、版本、环境差异解释。
工程经验不要简单按时间覆盖；能拆条件就拆，不能解释才冲突。

update_topic：维护主题元信息
- 当已有 memory 的 summary/content 已被更新，但 topic 标题或概要仍停留在旧语义时使用。
- 只用于改主题 title/summary，不用于改记忆正文。
例：memory 已更新为“用户偏好英文回复”，但 topic 标题仍是“默认使用中文回复”，应把 topic 改为中性的“回复语言偏好”。

record_conflict_decision：记录冲突判断过程
- 当同一用户、同一作用域、同一语义槽位出现前后相反说法时使用。
- 如果根据用户最新明确纠正决定覆盖旧偏好，应记录 old_claim、new_claim、decision、reason。
- 它记录“为什么这样处理冲突”，不直接修改 memory 内容；通常与 update_memory 或 update_topic 搭配使用。

组合动作规则：
- 用户偏好/工程规则被明确纠正时，通常需要三件套：
  1. update_memory：更新当前有效结论。
  2. update_topic：如果 topic 标题/概要仍是旧语义，改成中性或当前有效语义。
  3. record_conflict_decision：记录旧说法、新说法、采用新说法的原因。
- 如果只是补充同方向证据，用 merge_memory，不要记录冲突。
- 如果只是 topic 展示语义落后但 memory 内容已经正确，只用 update_topic + 必要的 record_conflict_decision。
- 如果冲突来自不同项目/芯片/工具链/版本条件，优先拆条件或拆记忆，不要直接覆盖。

noop：不沉淀
- 只是当前项目一次性参数或需求选择。
- 已有记忆已覆盖，且本轮没有新增证据或更好表述。
- 证据只是猜测或没有长期价值。

request_more_evidence：证据不足
- 当前 turn 摘要不足，必须回查 pointer。
- 或需要用户/后续验证才能确认。

拆分原则：
- 同症状不同根因，不要强行合并，应形成不同 memory。
- 不同项目、芯片、工具链、版本造成的差异，优先拆 scope/条件，不直接冲突。

====================
四、证据使用标准
====================

证据优先级：
1. 用户明确确认 > 助手总结。
2. 编译/测试/运行验证 > 推测。
3. 工具结果摘要 > 普通自然语言判断。
4. pointer 可补充原始证据，但不要把完整原文塞进 memory。

如果 core/candidate 摘要不足以证明症状、根因、修复、验证，应请求 expand_pointer。
如果已有 memory_index 数量有限或摘要看不出相关性，应先请求 search_related_memories，用关键词从主题标题、主题摘要、memory 摘要、memory 正文中召回候选。
如果可能要修改已有 memory，但目标 memory_id 不在 modifiable_memory_ids，必须先请求 read_memory_detail 或 read_topic_detail。
禁止凭 memory_index 摘要直接 update_memory / merge_memory / mark_conflict / record_conflict_decision。
禁止凭 search_related_memories 结果直接 update_memory / merge_memory / mark_conflict / record_conflict_decision；搜索结果只用于决定下一步读哪个 memory/topic 详情。

====================
五、输出格式
====================

如果需要补证据，输出：
{
  "schema_version": "memory-react-v1",
  "decision": "tool_requests",
  "reason": "中文说明为什么需要查",
  "tool_requests": [
    {"tool": "search_related_memories", "query": "症状/工具链/主题关键词", "reason": "为什么要搜旧记忆"},
    {"tool": "read_topic_detail", "topic_id": "已有 topic_id", "reason": "为什么查"},
    {"tool": "read_memory_detail", "memory_id": "已有 memory_id", "reason": "为什么查"},
    {"tool": "expand_pointer", "raw_pointer_id": "本轮 pointer id", "reason": "为什么查"}
  ],
  "changes": []
}

如果证据足够，输出：
{
  "schema_version": "memory-react-v1",
  "decision": "final_changes",
  "reason": "中文说明最终判断",
  "tool_requests": [],
  "changes": {
    "schema_version": "memory-changes-v1",
    "turn_id": "当前 turn_id",
    "reason": "中文整体判断",
    "changes": []
  }
}

memory-changes-v1 changes 可用 action：
- create_memory
- merge_memory
- update_memory
- mark_conflict
- update_topic
- record_conflict_decision
- noop
- request_more_evidence

create_memory 示例：
{
  "action": "create_memory",
  "reason": "为什么新增",
  "topic": {
    "topic_key": "embedded_stub_input_failure",
    "title": "嵌入式硬件输入桩导致假数据",
    "summary": "硬件输入函数返回固定值会让主流程跑通但输入恒定，导致下游状态机无响应。",
    "topic_type": "engineering_failure_pattern",
    "scope": {"level": "domain", "domain": "embedded"}
  },
  "memory": {
    "stable_key": "embedded.stub_input.constant_value_downstream_stuck",
    "memory_type": "runtime_failure_pattern",
    "summary": "硬件输入桩函数固定返回常量会导致下游控制逻辑一直处理假数据。",
    "content": "症状：编译烧录通过，但外设不随真实输入变化。触发条件：采样/驱动函数未读取真实硬件，而是返回固定常量。根因：主流程虽然执行，但输入数据恒定。解决：替换桩实现，按数据手册或现有驱动读取真实寄存器/外设。验证：改变真实输入后，下游状态、占空比或报警应随之变化。",
    "scope": {"level": "domain", "domain": "embedded"},
    "confidence": 0.9
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "证明症状、根因或验证"}]
}

update_memory 示例：
{
  "action": "update_memory",
  "reason": "已有记忆方向正确，本轮提供了更通用表述",
  "target_memory_id": "mem_xxx",
  "memory": {
    "summary": "更新后的中文摘要",
    "content": "更新后的完整内容",
    "confidence": 0.92
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "补充证据"}]
}

merge_memory 示例：
{
  "action": "merge_memory",
  "reason": "同一模式，本轮只补充新证据或验证，不改变原结论",
  "target_memory_id": "mem_xxx",
  "memory": {
    "summary": "合并后的中文摘要，必须保留原核心结论并吸收新增证据价值",
    "content": "合并后的完整内容，必须包含原有症状/根因/解决方法，并补充本轮新增证据或验证方式。不能只写新增片段。",
    "confidence": 0.92
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "新增证据或验证"}]
}

注意：merge_memory / update_memory / mark_conflict 都必须包含 memory 对象，不能只给 target_memory_id 和 evidence_refs。

update_topic 示例：
{
  "action": "update_topic",
  "reason": "memory 已从旧偏好更新为新偏好，原主题标题不再准确",
  "target_topic_id": "topic_xxx",
  "topic": {
    "title": "回复语言偏好",
    "summary": "记录用户当前偏好的回复语言，以及前后纠正造成的当前有效偏好。"
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "用户明确提出新的回复语言偏好"}]
}

record_conflict_decision 示例：
{
  "action": "record_conflict_decision",
  "reason": "同一用户回复语言偏好前后相反，需要记录为何采用新偏好",
  "target_memory_id": "mem_xxx",
  "conflict": {
    "conflict_type": "user_preference_changed",
    "old_claim": "用户希望默认中文回复",
    "new_claim": "用户希望默认英文回复",
    "decision": "采用更新的明确用户偏好，并保留冲突记录",
    "reason": "新证据是用户对未来回复语言的明确要求，且与旧偏好属于同一语义槽位。"
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "新偏好证据"}]
}

noop 示例：
{
  "action": "noop",
  "reason": "只是当前项目一次性参数，不能沉淀为长期记忆",
  "evidence_refs": []
}

====================
六、硬性要求
====================

- decision=tool_requests 时，tool_requests 必须非空，changes 必须是空数组。
- decision=final_changes 时，tool_requests 必须是空数组，changes 必须是完整 memory-changes-v1 对象。
- 每轮最多请求 3 个工具。
- 只能请求输入中真实存在的 topic_id / memory_id / raw_pointer_id。
- 不能发明 evidence_id、memory_id、topic_id、raw_pointer_id。
- search_related_memories 的 query 必须来自当前 turn 证据或已有 memory/topic 摘要中的真实词语，不能发明不存在的问题。
- update_memory / merge_memory / mark_conflict 的 target_memory_id 必须在 modifiable_memory_ids 中；否则先请求只读工具。
- record_conflict_decision 的 target_memory_id 也必须在 modifiable_memory_ids 中；否则先请求只读工具。
- update_topic 的 target_topic_id 必须来自已知 topic_id；如果要判断主题是否应改名，优先先 read_topic_detail。
- 对用户明确纠正导致的同一偏好反转，优先输出 update_memory + update_topic + record_conflict_decision 的组合；如果 memory 已经是最新结论，则至少输出 update_topic + record_conflict_decision。
- 如果已有 memory 覆盖本轮信息，应 merge/update/noop，不要重复 create。
- 模型不能写数据库，不能输出 SQL，不能请求未列出的工具。
"""


REACT_REPAIR_PROMPT = """你是 Memory Lab ReAct JSON 修复器。
你会收到上一次模型输出、程序校验错误和允许 ID。
只修复 JSON schema、字段名、字段结构和引用 ID，不要改变原本语义判断。
如果错误是 unsupported fields，必须删除这些字段，不能换成别的未知字段。
特别注意：
- noop 只允许 action、reason、evidence_refs，不能带 memory_id、target_memory_id、topic_id、memory。
- create_memory 只能二选一 topic 或 topic_id。
- merge_memory/update_memory/mark_conflict 必须使用 target_memory_id，不能使用 memory_id。
- merge_memory/update_memory/mark_conflict 必须包含 memory 对象，memory 里必须包含 summary、content、confidence。
- update_topic 必须使用 target_topic_id 和 topic 对象，topic 只能包含 title、summary。
- record_conflict_decision 必须使用 target_memory_id 和 conflict 对象。
- 如果 merge_memory 缺少 memory 对象，必须基于已读取的目标 memory 正文和新增证据补出合并后的完整 memory 对象；不要删除该 action。
- tool_requests 和 final_changes 不能混用。
只能输出完整 JSON 对象，不要输出 Markdown、解释、代码块或额外文本。
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_run_log(
    *,
    run_log: Mapping[str, Any],
    log_dir: Path | None = DEFAULT_REACT_RUN_DIR,
) -> dict[str, str]:
    log_dir = log_dir or DEFAULT_REACT_RUN_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(run_log["run_id"])
    path = log_dir / f"{run_id}.json"
    path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = log_dir / "latest.json"
    try:
        shutil.copyfile(path, latest)
    except OSError:
        latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"run_log_path": str(path), "latest_log_path": str(latest)}


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _collect_available_ids(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    layers = (payload.get("turn_evidence") or {}).get("layers") or {}
    pointer_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for layer_name in ("core", "candidate", "pointer"):
        for item in layers.get(layer_name) or []:
            if isinstance(item, Mapping):
                evidence_id = item.get("evidence_id")
                if evidence_id:
                    evidence_ids.add(str(evidence_id))
                raw_pointer_id = item.get("raw_pointer_id")
                if raw_pointer_id:
                    pointer_ids.add(str(raw_pointer_id))
    for item in layers.get("pointer") or []:
        if isinstance(item, Mapping) and item.get("evidence_id"):
            pointer_ids.add(str(item["evidence_id"]))

    topic_ids = {
        str(item["topic_id"])
        for item in (payload.get("topic_index") or {}).get("topics") or []
        if isinstance(item, Mapping) and item.get("topic_id")
    }
    memory_index = payload.get("memory_index") or {}
    memory_ids = {
        str(item["memory_id"])
        for item in memory_index.get("memories") or []
        if isinstance(item, Mapping) and item.get("memory_id")
    }
    for item in memory_index.get("topics") or []:
        if isinstance(item, Mapping) and item.get("topic_id"):
            topic_ids.add(str(item["topic_id"]))
    return {
        "evidence_ids": evidence_ids,
        "pointer_ids": pointer_ids,
        "topic_ids": topic_ids,
        "memory_ids": memory_ids,
    }


def _ids_for_log(ids: Mapping[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(value) for key, value in ids.items()}


def _tool_key(request: Mapping[str, Any]) -> tuple[str, str]:
    tool = str(request.get("tool") or "")
    if tool == "read_topic_detail":
        return tool, str(request.get("topic_id") or "")
    if tool == "read_memory_detail":
        return tool, str(request.get("memory_id") or "")
    if tool == "expand_pointer":
        return tool, str(request.get("raw_pointer_id") or "")
    if tool == "search_related_memories":
        parts = [
            str(request.get("query") or ""),
            str(request.get("memory_type") or ""),
            str(request.get("domain") or ""),
            str(request.get("project_key") or ""),
        ]
        return tool, "|".join(parts)
    return tool, ""


def validate_react_decision(
    decision: Mapping[str, Any],
    *,
    turn_id: str,
    available_ids: Mapping[str, set[str]],
    observed_memory_ids: set[str] | None = None,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    max_tool_requests: int = 3,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {"schema_version", "decision", "reason", "tool_requests", "changes"}
    extra = set(decision) - required
    missing = required - set(decision)
    if extra:
        errors.append(f"unexpected top-level fields: {sorted(extra)}")
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    if decision.get("schema_version") != MEMORY_REACT_SCHEMA_VERSION:
        errors.append("schema_version must be memory-react-v1")
    if decision.get("decision") not in {"tool_requests", "final_changes"}:
        errors.append("decision must be tool_requests or final_changes")
    if not _non_empty_string(decision.get("reason")):
        errors.append("reason must be a non-empty string")

    tool_requests = decision.get("tool_requests")
    if not isinstance(tool_requests, list):
        errors.append("tool_requests must be an array")
        tool_requests = []
    elif len(tool_requests) > max_tool_requests:
        errors.append(f"tool_requests exceeds max {max_tool_requests}")

    changes = decision.get("changes")
    mode = decision.get("decision")
    if mode == "tool_requests":
        if not tool_requests:
            errors.append("tool_requests decision requires at least one tool request")
        if changes != []:
            errors.append("tool_requests decision requires changes=[]")
    elif mode == "final_changes":
        if tool_requests:
            errors.append("final_changes decision requires tool_requests=[]")
        if not _is_object(changes):
            errors.append("final_changes decision requires changes object")

    seen: set[tuple[str, str]] = set()
    pointer_ids = available_ids.get("pointer_ids") or set()
    topic_ids = available_ids.get("topic_ids") or set()
    memory_ids = available_ids.get("memory_ids") or set()
    for index, request in enumerate(tool_requests):
        prefix = f"tool_requests[{index}]"
        if not _is_object(request):
            errors.append(f"{prefix} must be an object")
            continue
        tool = request.get("tool")
        if tool not in ALLOWED_TOOLS:
            errors.append(f"{prefix}.tool is not allowed")
            continue
        allowed_by_tool = {
            "read_topic_detail": {"tool", "topic_id", "reason"},
            "read_memory_detail": {"tool", "memory_id", "reason"},
            "expand_pointer": {"tool", "raw_pointer_id", "reason"},
            "search_related_memories": {
                "tool",
                "query",
                "reason",
                "memory_type",
                "domain",
                "project_key",
                "limit",
            },
        }[str(tool)]
        required_by_tool = {
            "read_topic_detail": {"tool", "topic_id", "reason"},
            "read_memory_detail": {"tool", "memory_id", "reason"},
            "expand_pointer": {"tool", "raw_pointer_id", "reason"},
            "search_related_memories": {"tool", "query", "reason"},
        }[str(tool)]
        extra_request = set(request) - allowed_by_tool
        missing_request = required_by_tool - set(request)
        if extra_request:
            errors.append(f"{prefix} has unsupported fields: {sorted(extra_request)}")
        if missing_request:
            errors.append(f"{prefix} missing fields: {sorted(missing_request)}")
        if not _non_empty_string(request.get("reason")):
            errors.append(f"{prefix}.reason must be a non-empty string")
        if tool == "search_related_memories":
            if not _non_empty_string(request.get("query")):
                errors.append(f"{prefix}.query must be a non-empty string")
            if "limit" in request:
                try:
                    limit_value = int(request.get("limit"))
                    if limit_value < 1 or limit_value > 20:
                        errors.append(f"{prefix}.limit must be between 1 and 20")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.limit must be an integer")
        if tool == "read_topic_detail" and request.get("topic_id") not in topic_ids:
            errors.append(f"{prefix}.topic_id does not exist or is not active")
        if tool == "read_memory_detail" and request.get("memory_id") not in memory_ids:
            errors.append(f"{prefix}.memory_id does not exist or is not active")
        if tool == "expand_pointer" and request.get("raw_pointer_id") not in pointer_ids:
            errors.append(f"{prefix}.raw_pointer_id does not belong to this turn")
        key = _tool_key(request)
        if key in seen:
            errors.append(f"{prefix} duplicates tool request {key[0]}:{key[1]}")
        seen.add(key)

    if mode == "final_changes" and _is_object(changes):
        try:
            normalized_changes = validate_memory_changes(changes, pg_config_path=pg_config_path)
            if normalized_changes.get("turn_id") != turn_id:
                errors.append("changes.turn_id must equal current turn_id")
            observed = observed_memory_ids or set()
            for change_index, change in enumerate(normalized_changes.get("changes") or []):
                if not isinstance(change, Mapping):
                    continue
                if change.get("action") in {
                    "merge_memory",
                    "update_memory",
                    "mark_conflict",
                    "record_conflict_decision",
                }:
                    target_memory_id = str(change.get("target_memory_id") or "")
                    if target_memory_id and target_memory_id not in observed:
                        errors.append(
                            f"changes.changes[{change_index}].target_memory_id must be read by read_memory_detail or read_topic_detail before modifying or recording conflict for existing memory"
                        )
        except MemoryChangesValidationError as error:
            errors.extend([f"changes.{item}" for item in error.errors])

    if errors:
        raise MemoryReactValidationError(errors)
    return dict(decision)


def _execute_tool_request(
    request: Mapping[str, Any],
    *,
    source_database: Path,
    pg_config_path: Path,
    max_chars: int,
) -> dict[str, Any]:
    tool = str(request["tool"])
    if tool == "read_topic_detail":
        return read_topic_detail(
            pg_config_path=pg_config_path,
            topic_id=str(request["topic_id"]),
            max_chars=max_chars,
        )
    if tool == "read_memory_detail":
        result = read_memory_detail(
            pg_config_path=pg_config_path,
            memory_id=str(request["memory_id"]),
        )
        memory = result.get("memory")
        if isinstance(memory, Mapping) and isinstance(memory.get("content"), str):
            content = memory["content"]
            result = dict(result)
            memory = dict(memory)
            memory["content"] = content[:max_chars]
            memory["content_truncated"] = len(content) > max_chars
            result["memory"] = memory
        return result
    if tool == "expand_pointer":
        return expand_pointer(
            source_database,
            pg_config_path=pg_config_path,
            raw_pointer_id=str(request["raw_pointer_id"]),
            max_chars=max_chars,
        )
    if tool == "search_related_memories":
        return search_related_memories(
            pg_config_path=pg_config_path,
            query=str(request["query"]),
            memory_type=str(request["memory_type"]) if request.get("memory_type") else None,
            domain=str(request["domain"]) if request.get("domain") else None,
            project_key=str(request["project_key"]) if request.get("project_key") else None,
            limit=int(request.get("limit") or 8),
        )
    raise ValueError(f"unsupported tool: {tool}")


def _execute_tool_requests(
    requests: list[Mapping[str, Any]],
    *,
    source_database: Path,
    pg_config_path: Path,
    seen_tool_keys: set[tuple[str, str]],
    max_chars: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for request in requests:
        key = _tool_key(request)
        if key in seen_tool_keys:
            observations.append(
                {
                    "status": "skipped",
                    "tool": key[0],
                    "target": key[1],
                    "reason": "same tool request already executed in this run",
                    "request": dict(request),
                }
            )
            continue
        seen_tool_keys.add(key)
        try:
            result = _execute_tool_request(
                request,
                source_database=source_database,
                pg_config_path=pg_config_path,
                max_chars=max_chars,
            )
            observations.append(
                {
                    "status": "complete",
                    "request": dict(request),
                    "result": result,
                }
            )
        except Exception as error:
            observations.append(
                {
                    "status": "failed",
                    "request": dict(request),
                    "error": str(error),
                }
            )
    return observations


def _observed_memory_ids(observations: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for observation in observations:
        if observation.get("status") != "complete":
            continue
        tool_result = observation.get("result")
        if not isinstance(tool_result, Mapping):
            continue
        if tool_result.get("tool") == "read_memory_detail":
            memory = tool_result.get("memory")
            if isinstance(memory, Mapping) and memory.get("memory_id"):
                result.add(str(memory["memory_id"]))
        elif tool_result.get("tool") == "read_topic_detail":
            detail = tool_result.get("detail")
            if not isinstance(detail, Mapping):
                continue
            for memory in detail.get("memories") or []:
                if isinstance(memory, Mapping) and memory.get("memory_id"):
                    result.add(str(memory["memory_id"]))
    return result


def _auto_required_memory_reads(
    raw_decision: Mapping[str, Any],
    *,
    available_ids: Mapping[str, set[str]],
    observed_memory_ids: set[str],
) -> list[dict[str, Any]]:
    """Convert unsafe final modifications into required read-only lookups.

    This is a guardrail, not a write.  If the model tries to update/merge/mark
    conflict record on an existing memory before reading its body, the orchestrator
    forces a read_memory_detail observation and asks the model to decide again.
    """

    if raw_decision.get("decision") != "final_changes":
        return []
    changes = raw_decision.get("changes")
    if not isinstance(changes, Mapping):
        return []
    memory_ids = available_ids.get("memory_ids") or set()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for change in changes.get("changes") or []:
        if not isinstance(change, Mapping):
            continue
        if change.get("action") not in {
            "merge_memory",
            "update_memory",
            "mark_conflict",
            "record_conflict_decision",
        }:
            continue
        target = str(change.get("target_memory_id") or "")
        if not target or target not in memory_ids or target in observed_memory_ids or target in seen:
            continue
        seen.add(target)
        result.append(
            {
                "tool": "read_memory_detail",
                "memory_id": target,
                "reason": "程序强制补证据：修改、合并或冲突处理已有 memory 前必须先读取正文。",
            }
        )
    return result


def _react_payload(
    *,
    initial_context: Mapping[str, Any],
    observations: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
    available_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    observed_memory_ids = _observed_memory_ids(observations)
    all_memory_ids = set(available_ids.get("memory_ids") or set())
    summary_only_memory_ids = all_memory_ids - observed_memory_ids
    return {
        "schema_version": "memory-react-input-v1",
        "workflow_stage": "react_memory_maintenance",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "memory_policy": {
            "role": "长期记忆整理 Agent，不是普通总结器",
            "long_term_value_rule": "只保存以后能减少重复问答、减少错误、减少返工的信息",
            "primary_targets": [
                "用户交互习惯和协作偏好",
                "编译成功/失败经验",
                "用户反馈运行现象后定位出的原因和解决方法",
            ],
            "do_not_store": [
                "当前项目一次性参数",
                "代码/Git/需求文档可稳定恢复的信息",
                "未验证猜测",
                "完整日志和完整工具输出",
                "敏感信息",
            ],
        },
        "memory_access_state": {
            "summary_only_memory_ids": sorted(summary_only_memory_ids),
            "observed_memory_ids": sorted(observed_memory_ids),
            "modifiable_memory_ids": sorted(observed_memory_ids),
            "rule": "memory_index 只是摘要索引；只有 modifiable_memory_ids 中的 memory 才允许 update/merge/mark_conflict/record_conflict_decision。",
        },
        "action_decision_guide": {
            "create_memory": "有长期价值，已有 memory 不能覆盖，能形成清晰 stable_key，证据足够。",
            "merge_memory": "同一模式，只增加证据或例子，不改变原结论；必须先读过目标 memory 详情。",
            "update_memory": "已有 memory 方向正确，本轮提供更准确、更通用或更完整表述；必须先读过目标 memory 详情。",
            "mark_conflict": "同 scope、同条件、同对象下结论相反且无法用环境差异解释；必须先读过目标 memory 详情。",
            "update_topic": "memory 内容和 topic 标题/概要出现语义漂移时维护主题元信息；若要判断主题是否应改名，优先先读过 topic 详情。",
            "record_conflict_decision": "记录同一作用域前后矛盾的旧说法、新说法、处理决定和原因；必须先读过目标 memory 详情。",
            "noop": "一次性参数、低价值信息、已有记忆完全覆盖且无新增证据、或未验证猜测。",
            "request_more_evidence": "摘要不足或需要原始片段/后续验证才能判断。",
        },
        "limits": {
            "max_tool_requests_per_iteration": 3,
            "tool_result_max_chars": 4000,
            "final_changes_schema": MEMORY_CHANGES_SCHEMA_VERSION,
        },
        "available_tools": [
            {
                "tool": "search_related_memories",
                "args": {
                    "query": "必填；来自当前 turn 证据或已有索引的真实关键词",
                    "memory_type": "可选；限制 memory_type",
                    "domain": "可选；限制领域 scope",
                    "project_key": "可选；限制项目 scope",
                    "limit": "可选；1-20，默认 8",
                },
                "purpose": "当当前 memory_index 摘要不足或怀疑旧记忆未出现在概览中时，搜索主题标题/摘要和 memory 摘要/正文，返回候选 memory_id/topic_id；搜索结果不能直接授权修改",
            },
            {
                "tool": "read_topic_detail",
                "args": {"topic_id": "必须来自 available_ids.topic_ids"},
                "purpose": "读取主题正文和主题下的 active memories，用于判断合并/更新/冲突/重复",
            },
            {
                "tool": "read_memory_detail",
                "args": {"memory_id": "必须来自 available_ids.memory_ids"},
                "purpose": "读取某条 memory 的完整内容，用于判断是否合并、更新或冲突",
            },
            {
                "tool": "expand_pointer",
                "args": {"raw_pointer_id": "必须来自本轮 pointer id"},
                "purpose": "从 CP SQLite 回查原始片段，补足 core/candidate 摘要不足的信息",
            },
        ],
        "available_ids": _ids_for_log(available_ids),
        "initial_context": initial_context,
        "observations": observations,
        "required_output_schema": {
            "schema_version": MEMORY_REACT_SCHEMA_VERSION,
            "decision": "tool_requests | final_changes",
            "reason": "中文原因",
            "tool_requests": [],
            "changes": "[] when tool_requests; memory-changes-v1 object when final_changes",
        },
    }


def _validate_or_repair_react(
    *,
    model: OpenAICompatibleJsonClient,
    first: Mapping[str, Any],
    payload: Mapping[str, Any],
    turn_id: str,
    available_ids: Mapping[str, set[str]],
    observed_memory_ids: set[str],
    pg_config_path: Path,
) -> tuple[dict[str, Any] | None, bool, list[str], Mapping[str, Any]]:
    try:
        return (
            validate_react_decision(
                first,
                turn_id=turn_id,
                available_ids=available_ids,
                observed_memory_ids=observed_memory_ids,
                pg_config_path=pg_config_path,
            ),
            False,
            [],
            first,
        )
    except MemoryReactValidationError as error:
        previous: Mapping[str, Any] = first
        errors = error.errors
        for repair_attempt in range(1, 3):
            repair_payload = {
                "repair_attempt": repair_attempt,
                "previous_output": previous,
                "validation_errors": errors,
                "available_ids": _ids_for_log(available_ids),
                "observed_memory_ids": sorted(observed_memory_ids),
                "required_schema": payload["required_output_schema"],
                "current_turn_id": turn_id,
                "strict_field_rules": {
                    "noop": ["action", "reason", "evidence_refs"],
                    "request_more_evidence": ["action", "reason", "requests", "evidence_refs"],
                    "merge_memory": ["action", "reason", "target_memory_id", "memory", "evidence_refs"],
                    "update_memory": ["action", "reason", "target_memory_id", "memory", "evidence_refs"],
                    "mark_conflict": ["action", "reason", "target_memory_id", "memory", "evidence_refs"],
                    "update_topic": ["action", "reason", "target_topic_id", "topic", "evidence_refs"],
                    "record_conflict_decision": ["action", "reason", "target_memory_id", "conflict", "evidence_refs"],
                    "memory_for_existing_target": ["summary", "content", "confidence"],
                    "topic_patch": ["title", "summary"],
                    "conflict": ["conflict_type", "old_claim", "new_claim", "decision", "reason", "status", "metadata"],
                    "tool_request_expand_pointer": ["tool", "raw_pointer_id", "reason"],
                    "tool_request_read_memory_detail": ["tool", "memory_id", "reason"],
                    "tool_request_read_topic_detail": ["tool", "topic_id", "reason"],
                    "tool_request_search_related_memories": [
                        "tool",
                        "query",
                        "reason",
                        "memory_type?",
                        "domain?",
                        "project_key?",
                        "limit?",
                    ],
                },
            }
            repaired_output = model.complete_json(
                system_prompt=REACT_REPAIR_PROMPT,
                user_payload=repair_payload,
                max_tokens=7000,
            )
            try:
                return (
                    validate_react_decision(
                        repaired_output,
                        turn_id=turn_id,
                        available_ids=available_ids,
                        observed_memory_ids=observed_memory_ids,
                        pg_config_path=pg_config_path,
                    ),
                    True,
                    [],
                    repaired_output,
                )
            except MemoryReactValidationError as repaired_error:
                previous = repaired_output
                errors = repaired_error.errors
        return None, True, errors, previous


def run_memory_react_agent(
    *,
    source_database: Path,
    turn_id: str,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    memory_limit: int = 80,
    topic_limit: int = 200,
    dry_run: bool = False,
    max_iterations: int = 3,
    log_dir: Path = DEFAULT_REACT_RUN_DIR,
    client: OpenAICompatibleJsonClient | None = None,
) -> dict[str, Any]:
    max_iterations = max(1, min(int(max_iterations), 10))
    run_id = _safe_run_id(turn_id).replace("agent-", "react-", 1)
    started_at = _utc_now_iso()
    steps: list[dict[str, Any]] = []
    initial_context = build_memory_agent_input(
        turn_id=turn_id,
        pg_config_path=pg_config_path,
        memory_limit=memory_limit,
        topic_limit=topic_limit,
    )
    if initial_context.get("status") != "complete":
        steps.append(_step("build_react_input", "failed", f"turn unavailable: {turn_id}"))
        result = {
            "status": "failed",
            "stage": "input",
            "turn_id": turn_id,
            "committed": False,
            "input": initial_context,
        }
        run_log = {
            "schema_version": "memory-react-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run" if dry_run else "commit",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

    stats = _context_stats(initial_context)
    available_ids = _collect_available_ids(initial_context)
    steps.extend(
        [
            _step(
                "build_react_input",
                "complete",
                f"core={stats['core_count']} candidate={stats['candidate_count']} pointer={stats['pointer_count']}",
                filters=initial_context.get("filters"),
            ),
            _step("load_memory_index", "complete", f"topics={stats['topic_count']} memories={stats['memory_count']}"),
        ]
    )

    model = client or OpenAICompatibleJsonClient.from_config(config_path)
    observations: list[dict[str, Any]] = []
    iteration_logs: list[dict[str, Any]] = []
    seen_tool_keys: set[tuple[str, str]] = set()
    final_decision: dict[str, Any] | None = None
    raw_final: Mapping[str, Any] | None = None

    for iteration in range(1, max_iterations + 1):
        payload = _react_payload(
            initial_context=initial_context,
            observations=observations,
            iteration=iteration,
            max_iterations=max_iterations,
            available_ids=available_ids,
        )
        raw = model.complete_json(
            system_prompt=REACT_SYSTEM_PROMPT,
            user_payload=payload,
            max_tokens=9000,
        )
        auto_reads = _auto_required_memory_reads(
            raw,
            available_ids=available_ids,
            observed_memory_ids=_observed_memory_ids(observations),
        )
        if auto_reads:
            new_observations = _execute_tool_requests(
                auto_reads,
                source_database=source_database,
                pg_config_path=pg_config_path,
                seen_tool_keys=seen_tool_keys,
                max_chars=4000,
            )
            observations.extend(new_observations)
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "decision": "auto_required_reads",
                    "repaired": False,
                    "raw": raw,
                    "auto_tool_requests": auto_reads,
                }
            )
            steps.append(
                _step(
                    "auto_required_reads",
                    "complete",
                    f"iteration={iteration} requests={len(auto_reads)}",
                )
            )
            continue
        decision, repaired, errors, raw_after_validation = _validate_or_repair_react(
            model=model,
            first=raw,
            payload=payload,
            turn_id=turn_id,
            available_ids=available_ids,
            observed_memory_ids=_observed_memory_ids(observations),
            pg_config_path=pg_config_path,
        )
        iteration_log: dict[str, Any] = {
            "iteration": iteration,
            "repaired": repaired,
            "raw": raw_after_validation,
        }
        if decision is None:
            steps.append(
                _step(
                    "validate_react_decision",
                    "failed",
                    f"iteration={iteration} errors={len(errors)}",
                    errors=errors,
                )
            )
            result = {
                "status": "failed",
                "stage": "validate_react_decision",
                "turn_id": turn_id,
                "iteration": iteration,
                "validation_errors": errors,
                "raw_decision": raw_after_validation,
                "committed": False,
            }
            run_log = {
                "schema_version": "memory-react-run-log-v1",
                "run_id": run_id,
                "turn_id": turn_id,
                "mode": "dry_run" if dry_run else "commit",
                "started_at": started_at,
                "finished_at": _utc_now_iso(),
                "status": result["status"],
                "steps": steps,
                "iterations": [*iteration_logs, iteration_log],
                "observations": observations,
                "result": result,
            }
            return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

        iteration_log["decision"] = decision.get("decision")
        iteration_logs.append(iteration_log)
        if decision.get("decision") == "tool_requests":
            requests = [dict(item) for item in decision.get("tool_requests") or []]
            new_observations = _execute_tool_requests(
                requests,
                source_database=source_database,
                pg_config_path=pg_config_path,
                seen_tool_keys=seen_tool_keys,
                max_chars=4000,
            )
            observations.extend(new_observations)
            steps.append(
                _step(
                    "execute_read_tools",
                    "complete",
                    f"iteration={iteration} requests={len(requests)} observations={len(new_observations)}",
                )
            )
            continue

        final_decision = decision
        raw_final = raw_after_validation
        break

    if final_decision is None:
        result = {
            "status": "failed",
            "stage": "max_iterations",
            "turn_id": turn_id,
            "iterations": max_iterations,
            "observation_count": len(observations),
            "committed": False,
        }
        steps.append(_step("react_loop", "failed", "max iterations reached without final_changes"))
        run_log = {
            "schema_version": "memory-react-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run" if dry_run else "commit",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "iterations": iteration_logs,
            "observations": observations,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

    plan = dict(final_decision["changes"])
    change_count = len(plan.get("changes") or [])
    steps.append(_step("final_changes", "complete", f"changes={change_count}"))

    if dry_run:
        result = {
            "status": "complete",
            "mode": "dry_run",
            "turn_id": turn_id,
            "iterations": len(iteration_logs),
            "observation_count": len(observations),
            "plan": plan,
            "committed": False,
        }
        run_log = {
            "schema_version": "memory-react-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "iterations": iteration_logs,
            "observations": observations,
            "raw_final_decision": raw_final,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

    commit = commit_memory_changes(plan, pg_config_path=pg_config_path)
    committed = bool((commit.get("counts") or {}).get("applied"))
    steps.append(
        _step(
            "commit_memory_changes",
            "complete",
            f"applied={(commit.get('counts') or {}).get('applied', 0)} rejected={(commit.get('counts') or {}).get('rejected', 0)}",
        )
    )
    result = {
        "status": "complete",
        "mode": "commit",
        "turn_id": turn_id,
        "iterations": len(iteration_logs),
        "observation_count": len(observations),
        "plan": plan,
        "commit": commit,
        "committed": committed,
    }
    run_log = {
        "schema_version": "memory-react-run-log-v1",
        "run_id": run_id,
        "turn_id": turn_id,
        "mode": "commit",
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "status": result["status"],
        "steps": steps,
        "iterations": iteration_logs,
        "observations": observations,
        "raw_final_decision": raw_final,
        "result": result,
    }
    return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}


__all__ = [
    "MEMORY_REACT_SCHEMA_VERSION",
    "MemoryReactValidationError",
    "run_memory_react_agent",
    "validate_react_decision",
]
