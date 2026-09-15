"""Dry-run Writer for lightweight PostgreSQL memory plans.

This module deliberately does not commit memory rows.  It only tests whether a
single LLM Writer can produce a strict executable JSON plan from the fixed
Writer context.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG
from .llm_client import OpenAICompatibleJsonClient
from .pg_store import DEFAULT_PG_CONFIG, build_writer_context


WRITER_PLAN_SCHEMA_VERSION = "memory-writer-plan-v1"
ALLOWED_ACTIONS = {"create", "merge", "update", "conflict", "noop", "need_more_evidence"}
ALLOWED_REQUEST_TOOLS = {"expand_pointer", "read_memory_detail"}
KEY_RE = re.compile(r"^[a-z0-9_]+(?:[._][a-z0-9_]+)*$")


WRITER_SYSTEM_PROMPT = """你是 Memory Lab 的长期记忆 Writer。
你的任务是根据当前 CP turn 的三层证据池和已有 memory 摘要，判断是否需要生成长期记忆计划。

你只能输出一个严格 JSON 对象，不要输出 Markdown、解释、代码块或额外文本。

允许 action:
- create: 新建一条长期记忆
- merge: 与已有 memory 语义一致，合并/丰富当前有效内容
- update: 用户明确更正或旧记忆需要改写
- conflict: 与已有 memory 冲突但无法判断谁对
- noop: 没有长期记忆价值
- need_more_evidence: 当前摘要不够，需要工具补证据

判断原则:
- 长期记忆存“以后怎么少问、少错、少返工”，不是存当前项目完整需求文档。
- 用户协作偏好、可复用流程、工程失败模式、原因、有效修复、验证结论可沉淀。
- 当前项目具体参数如果不能复用，一般 noop。
- 如果 memory_index 为空且证据足够，可 create。
- 如果已有 memory 摘要语义相近，不要重复 create，应 merge/update/conflict/noop。
- 如只缺少原始工具输出或已有 memory 正文，可 need_more_evidence。
- 本阶段不要编造证据，不要引用未提供内容。

输出 JSON 顶层格式必须是:
{
  "schema_version": "memory-writer-plan-v1",
  "action": "create|merge|update|conflict|noop|need_more_evidence",
  "reason": "中文原因",
  "requests": [],
  "topic": null,
  "memory": null,
  "target_memory_id": null
}

create 时:
- topic 必须为对象，包含 topic_key/title/summary/topic_type/scope
- memory 必须为对象，包含 stable_key/memory_type/summary/content/scope/confidence
- target_memory_id 必须为 null
- scope 必须是对象:
  - 全局: {"level":"global"}
  - 领域: {"level":"domain","domain":"embedded"}
  - 项目: {"level":"project","project_key":"directory:..."}

merge/update/conflict 时:
- target_memory_id 必须是已有 memory_id
- memory 必须为对象，至少包含 summary/content/confidence
- topic 通常为 null

noop 时:
- topic=null, memory=null, target_memory_id=null, requests=[]

need_more_evidence 时:
- requests 至少一条
- tool 只能是 expand_pointer 或 read_memory_detail
- expand_pointer 必须带 raw_pointer_id
- read_memory_detail 必须带 memory_id
- topic=null, memory=null, target_memory_id=null
"""


REPAIR_SYSTEM_PROMPT = """你是 JSON 修复器。
你会收到上一次 Writer 输出、程序校验错误和目标 schema。
只修复 JSON 格式和字段结构，不要改变原语义决定。
只能输出一个完整 JSON 对象，不要输出 Markdown、解释、代码块或额外文本。
"""


class WriterPlanValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_key(value: Any) -> bool:
    return isinstance(value, str) and bool(KEY_RE.fullmatch(value))


def _valid_confidence(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= float(value) <= 1


def _valid_scope(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    level = value.get("level")
    if level == "global":
        return True
    if level == "domain":
        return _non_empty_string(value.get("domain"))
    if level == "project":
        return _non_empty_string(value.get("project_key"))
    return False


def validate_writer_plan(plan: Mapping[str, Any], memory_index: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    required = {"schema_version", "action", "reason", "requests", "topic", "memory", "target_memory_id"}
    extra = set(plan) - required
    missing = required - set(plan)
    if extra:
        errors.append(f"unexpected top-level fields: {sorted(extra)}")
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    if plan.get("schema_version") != WRITER_PLAN_SCHEMA_VERSION:
        errors.append("schema_version must be memory-writer-plan-v1")
    action = plan.get("action")
    if action not in ALLOWED_ACTIONS:
        errors.append("action is not allowed")
    if not _non_empty_string(plan.get("reason")):
        errors.append("reason must be a non-empty string")
    if not isinstance(plan.get("requests"), list):
        errors.append("requests must be an array")

    memories = memory_index.get("memories") if isinstance(memory_index, Mapping) else []
    existing_memory_ids = {
        str(item.get("memory_id"))
        for item in memories
        if isinstance(item, Mapping) and item.get("memory_id")
    }

    def check_topic(topic: Any) -> None:
        if not _is_object(topic):
            errors.append("topic must be an object")
            return
        for field in ("topic_key", "title", "summary", "topic_type", "scope"):
            if field not in topic:
                errors.append(f"topic.{field} is required")
        if not _valid_key(topic.get("topic_key")):
            errors.append("topic.topic_key must use lowercase letters/digits/underscore/dot")
        for field in ("title", "summary", "topic_type"):
            if not _non_empty_string(topic.get(field)):
                errors.append(f"topic.{field} must be a non-empty string")
        if not _valid_scope(topic.get("scope")):
            errors.append("topic.scope must be {'level':'global'} or {'level':'domain','domain':'...'} or {'level':'project','project_key':'...'}")

    def check_memory(memory: Any, *, create: bool) -> None:
        if not _is_object(memory):
            errors.append("memory must be an object")
            return
        required_fields = ("summary", "content", "confidence")
        if create:
            required_fields = ("stable_key", "memory_type", "summary", "content", "scope", "confidence")
        for field in required_fields:
            if field not in memory:
                errors.append(f"memory.{field} is required")
        if create and not _valid_key(memory.get("stable_key")):
            errors.append("memory.stable_key must use lowercase letters/digits/underscore/dot")
        if create and not _non_empty_string(memory.get("memory_type")):
            errors.append("memory.memory_type must be a non-empty string")
        for field in ("summary", "content"):
            if not _non_empty_string(memory.get(field)):
                errors.append(f"memory.{field} must be a non-empty string")
        if create and not _valid_scope(memory.get("scope")):
            errors.append("memory.scope must be {'level':'global'} or {'level':'domain','domain':'...'} or {'level':'project','project_key':'...'}")
        if not _valid_confidence(memory.get("confidence")):
            errors.append("memory.confidence must be between 0 and 1")

    if action == "create":
        check_topic(plan.get("topic"))
        check_memory(plan.get("memory"), create=True)
        if plan.get("target_memory_id") is not None:
            errors.append("create.target_memory_id must be null")
        if plan.get("requests") != []:
            errors.append("create.requests must be empty")
    elif action in {"merge", "update", "conflict"}:
        target = plan.get("target_memory_id")
        if not _non_empty_string(target):
            errors.append(f"{action}.target_memory_id must be non-empty")
        elif target not in existing_memory_ids:
            errors.append(f"{action}.target_memory_id does not exist in memory_index")
        if plan.get("topic") is not None:
            errors.append(f"{action}.topic must be null")
        check_memory(plan.get("memory"), create=False)
        if plan.get("requests") != []:
            errors.append(f"{action}.requests must be empty")
    elif action == "noop":
        if plan.get("topic") is not None or plan.get("memory") is not None or plan.get("target_memory_id") is not None:
            errors.append("noop must have topic=null, memory=null, target_memory_id=null")
        if plan.get("requests") != []:
            errors.append("noop.requests must be empty")
    elif action == "need_more_evidence":
        requests = plan.get("requests")
        if not isinstance(requests, list) or not requests:
            errors.append("need_more_evidence.requests must contain at least one request")
        else:
            for index, request in enumerate(requests[:10]):
                if not _is_object(request):
                    errors.append(f"requests[{index}] must be an object")
                    continue
                tool = request.get("tool")
                if tool not in ALLOWED_REQUEST_TOOLS:
                    errors.append(f"requests[{index}].tool is not allowed")
                if tool == "expand_pointer" and not _non_empty_string(request.get("raw_pointer_id")):
                    errors.append(f"requests[{index}].raw_pointer_id is required")
                if tool == "read_memory_detail" and not _non_empty_string(request.get("memory_id")):
                    errors.append(f"requests[{index}].memory_id is required")
                if not _non_empty_string(request.get("reason")):
                    errors.append(f"requests[{index}].reason is required")
        if plan.get("topic") is not None or plan.get("memory") is not None or plan.get("target_memory_id") is not None:
            errors.append("need_more_evidence must have topic=null, memory=null, target_memory_id=null")

    if errors:
        raise WriterPlanValidationError(errors)
    return dict(plan)


def run_writer_dry_run(
    *,
    turn_id: str,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    memory_limit: int = 80,
    client: OpenAICompatibleJsonClient | None = None,
) -> dict[str, Any]:
    context_report = build_writer_context(
        pg_config_path=pg_config_path,
        turn_id=turn_id,
        memory_limit=memory_limit,
    )
    context = context_report["context"]
    model = client or OpenAICompatibleJsonClient.from_config(config_path)
    payload = {
        "writer_context": context,
        "required_schema": {
            "schema_version": WRITER_PLAN_SCHEMA_VERSION,
            "action": "create|merge|update|conflict|noop|need_more_evidence",
            "reason": "中文原因",
            "requests": [],
            "topic": None,
            "memory": None,
            "target_memory_id": None,
        },
        "instruction": "只输出符合 schema 的 JSON 计划；本次 dry-run 不会写库，也不执行补证据。",
    }
    first = model.complete_json(
        system_prompt=WRITER_SYSTEM_PROMPT,
        user_payload=payload,
        max_tokens=2400,
    )
    repaired = False
    try:
        plan = validate_writer_plan(first, context.get("memory_index") or {})
    except WriterPlanValidationError as error:
        repair_payload = {
            "previous_output": first,
            "validation_errors": error.errors,
            "required_schema": payload["required_schema"],
        }
        second = model.complete_json(
            system_prompt=REPAIR_SYSTEM_PROMPT,
            user_payload=repair_payload,
            max_tokens=2400,
        )
        repaired = True
        try:
            plan = validate_writer_plan(second, context.get("memory_index") or {})
        except WriterPlanValidationError as second_error:
            return {
                "status": "failed",
                "turn_id": turn_id,
                "validation_errors": second_error.errors,
                "raw_plan": second,
                "repaired": repaired,
            }
    return {
        "status": "complete",
        "turn_id": turn_id,
        "repaired": repaired,
        "plan": plan,
        "context_stats": {
            "core_count": len((context.get("turn_evidence") or {}).get("core") or []),
            "candidate_count": len((context.get("turn_evidence") or {}).get("candidate") or []),
            "memory_count": len((context.get("memory_index") or {}).get("memories") or []),
            "topic_count": len((context.get("memory_index") or {}).get("topics") or []),
        },
    }


__all__ = [
    "WRITER_PLAN_SCHEMA_VERSION",
    "WriterPlanValidationError",
    "run_writer_dry_run",
    "validate_writer_plan",
]
