"""Three-tier evidence pools and semantic phase routing.

The pool is a review/debug boundary before long-term memory extraction.  It
keeps raw CP data local while showing the model only high-value evidence and
summaries of expandable candidates.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from .llm_client import OpenAICompatibleJsonClient
from .phase import PHASES, classify_turn_phase
from .source import source_id
from .store import open_store
from .turn_evidence import build_turn_evidence
from .turns import run_turn_scan


EVIDENCE_POOL_SCHEMA_VERSION = "memory-lab-evidence-pool-v1"
PHASE_ROUTER_SCHEMA_VERSION = "memory-lab-phase-router-v1"

SUPPORTED_EVIDENCE_REQUEST_TYPES = {
    "expand_candidate",
    "neighbor_messages",
    "build_log_excerpt",
    "tool_result_excerpt",
    "existing_memory_topic",
    "file_change_summary",
}

TERMINAL_STATES = {"completed", "failed", "interrupted"}

CORE_KEYWORDS = (
    "以后",
    "先给结论",
    "不要",
    "仅在当前",
    "需求",
    "确认",
    "编译",
    "报错",
    "error",
    "warning",
    "0 error",
    "0 warning",
    "烧录",
    "现象",
    "不转",
    "没反应",
    "无变化",
    "定位",
    "根因",
    "原因",
    "解决",
    "修好了",
    "正常了",
    "验证通过",
    "adc_sample",
    "return 2048",
    "桩",
)

BUILD_KEYWORDS = (
    "build",
    "compile",
    "compil",
    "keil",
    "c51",
    "error",
    "warning",
    "0 error",
    "0 warning",
    "编译",
    "报错",
)

TOOL_EXCERPT_KEYWORDS = BUILD_KEYWORDS + (
    "adc_sample",
    "return 2048",
    "烧录",
    "pwm",
    "ntc",
    "main.c",
    "adc_hw.c",
)


class PhaseRouterError(ValueError):
    """The semantic phase router produced an invalid decision."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.casefold()
    return any(keyword.casefold() in lower for keyword in keywords)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "…"


def _text_part_items(bundle: Mapping[str, Any], actor: str) -> list[dict[str, Any]]:
    container = bundle.get(actor)
    if not isinstance(container, Mapping):
        return []
    message_id = str(container.get("message_id") or f"unknown-{actor}-message")
    result: list[dict[str, Any]] = []
    parts = container.get("text_parts")
    if not isinstance(parts, list):
        return result
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
            continue
        source_part_id = str(part.get("source_part_id") or f"{actor}-part-{index}")
        result.append(
            {
                "source": "user" if actor == "user" else "assistant",
                "source_type": "user_message" if actor == "user" else "assistant_final",
                "message_id": message_id,
                "part_id": source_part_id,
                "ordinal": index,
                "text": part["text"],
                "truncated": bool(part.get("truncated")),
            }
        )
    return result


def _user_answer_items(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    answers = bundle.get("user_answers")
    if not isinstance(answers, list):
        return []
    result: list[dict[str, Any]] = []
    for index, answer in enumerate(answers):
        if not isinstance(answer, Mapping) or not isinstance(answer.get("text"), str):
            continue
        result.append(
            {
                "source": "user",
                "source_type": "user_answer",
                "message_id": str(answer.get("source_message_id") or "unknown-question-message"),
                "part_id": str(answer.get("source_part_id") or f"user-answer-{index}"),
                "ordinal": index,
                "text": answer["text"],
                "truncated": bool(answer.get("truncated")),
            }
        )
    return result


def _tool_text(tool: Mapping[str, Any]) -> str:
    pieces = [str(tool.get("name") or ""), str(tool.get("status") or "")]
    for field in ("input", "output", "error"):
        channel = tool.get(field)
        if isinstance(channel, Mapping) and isinstance(channel.get("text"), str):
            pieces.append(channel["text"])
    return "\n".join(piece for piece in pieces if piece)


def _tool_is_core_signal(item: Mapping[str, Any], text: str) -> bool:
    name = str(item.get("tool_name") or "").casefold()
    lower = text.casefold()
    non_core_tools = {"read", "edit", "write", "glob", "grep", "todowrite", "skill"}
    if name in non_core_tools:
        return False
    if any(token in name for token in ("build", "compile", "keil")):
        return True
    normalized_start = " ".join(lower.split()[:2])
    if normalized_start.startswith(("read ", "edit ", "write ", "glob ", "grep ", "todowrite ", "skill ")):
        return False
    return any(
        phrase in lower
        for phrase in (
            "## build result",
            "build result:",
            "0 error",
            "0 errors",
            "error c",
            "warning c",
            "编译结果",
            "编译失败",
            "编译通过",
        )
    )


def _tool_items(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    tools = bundle.get("tools")
    if not isinstance(tools, list):
        return []
    result: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            continue
        text = _tool_text(tool)
        if not text.strip():
            continue
        result.append(
            {
                "source": "tool",
                "source_type": "tool_result",
                "message_id": str(tool.get("source_message_id") or "unknown-tool-message"),
                "part_id": str(tool.get("source_part_id") or f"tool-part-{index}"),
                "ordinal": int(tool.get("ordinal") or index),
                "tool_name": str(tool.get("name") or "unknown")[:120],
                "tool_status": str(tool.get("status") or "unknown")[:80],
                "text": text,
                "truncated": any(
                    isinstance(tool.get(field), Mapping) and bool(tool[field].get("truncated"))
                    for field in ("input", "output", "error")
                ),
            }
        )
    return result


def _kind_for_text(source_type: str, text: str, rule_phase: str) -> str:
    lower = text.casefold()
    if source_type in {"user_message", "user_answer"}:
        if any(word in lower for word in ("修好了", "正常了", "验证通过", "已解决", "重新烧录")):
            return "user_confirmation"
        if any(word in lower for word in ("烧录", "不转", "没反应", "现象", "无变化")):
            return "runtime_symptom"
        if any(word in lower for word in ("以后", "我希望", "不要", "别", "先给结论")):
            return "user_contract"
        if rule_phase == "requirement_alignment":
            return "requirement_signal"
        return "user_statement"
    if source_type == "assistant_final":
        if any(word in lower for word in ("根因", "原因", "定位", "导致", "解决", "修复")):
            return "diagnosis_or_solution"
        if any(word in lower for word in ("编译", "error", "warning", "0 error")):
            return "compile_conclusion"
        return "assistant_conclusion"
    if _contains_any(text, BUILD_KEYWORDS):
        return "build_or_tool_result"
    return "tool_context"


def _ref(prefix: str, turn_id: str, source_type: str, part_id: str, ordinal: int) -> str:
    digest = _sha256(f"{turn_id}|{source_type}|{part_id}|{ordinal}")
    return f"{prefix}_{digest[:16]}"


def build_evidence_pool(
    bundle: Mapping[str, Any],
    *,
    phase_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ``core/candidate/pointer`` evidence tiers for one turn."""

    phase = dict(phase_result or classify_turn_phase(bundle))
    turn = bundle.get("turn") if isinstance(bundle.get("turn"), Mapping) else {}
    workspace = bundle.get("workspace") if isinstance(bundle.get("workspace"), Mapping) else {}
    session = bundle.get("session") if isinstance(bundle.get("session"), Mapping) else {}
    turn_id = str(turn.get("turn_id") or phase.get("turn_id") or "unknown-turn")
    rule_phase = str(phase.get("primary_phase") or "other")

    raw_items = [
        *_text_part_items(bundle, "user"),
        *_user_answer_items(bundle),
        *_text_part_items(bundle, "assistant"),
        *_tool_items(bundle),
    ]
    raw_items.sort(key=lambda item: (item["source_type"], item["ordinal"], item["part_id"]))

    core: list[dict[str, Any]] = []
    candidate: list[dict[str, Any]] = []
    pointers: list[dict[str, Any]] = []

    for item in raw_items:
        source_type = item["source_type"]
        text = str(item["text"])
        kind = _kind_for_text(source_type, text, rule_phase)
        base_ref = _ref("ev", turn_id, source_type, item["part_id"], int(item["ordinal"]))
        pointer_ref = _ref("ptr", turn_id, source_type, item["part_id"], int(item["ordinal"]))
        pointers.append(
            {
                "ref": pointer_ref,
                "source_type": source_type,
                "kind": kind,
                "message_id": item["message_id"],
                "part_id": item["part_id"],
                "ordinal": item["ordinal"],
                "summary": _clip(text, 180),
                "expandable": True,
            }
        )

        is_user = source_type in {"user_message", "user_answer"}
        is_assistant_final = source_type == "assistant_final"
        is_tool = source_type == "tool_result"
        text_is_signal = _contains_any(text, CORE_KEYWORDS)
        tool_is_signal = is_tool and _tool_is_core_signal(item, text)
        requirement_answer = source_type == "user_answer" and kind == "requirement_signal"
        include_core = (
            (is_user and not requirement_answer)
            or kind in {"diagnosis_or_solution", "compile_conclusion"}
            or tool_is_signal
            or (is_assistant_final and text_is_signal)
        )
        if include_core:
            core.append(
                {
                    "ref": base_ref,
                    "source": item["source"],
                    "source_type": source_type,
                    "kind": kind,
                    "text": _clip(text, 1800 if is_tool else 2200),
                    "raw_pointer_ref": pointer_ref,
                    "truncated": bool(item.get("truncated")) or len(text) > (1800 if is_tool else 2200),
                }
            )
        else:
            candidate.append(
                {
                    "ref": base_ref,
                    "source": item["source"],
                    "source_type": source_type,
                    "kind": kind,
                    "summary": _clip(text, 260),
                    "raw_pointer_ref": pointer_ref,
                    "expandable": True,
                    "why_candidate": "可能相关但未进入核心证据；保留摘要供模型请求补证据。",
                }
            )

    pool = {
        "schema_version": EVIDENCE_POOL_SCHEMA_VERSION,
        "turn": {
            "turn_id": turn_id,
            "state": turn.get("state"),
            "occurred_at_ms": turn.get("occurred_at_ms"),
        },
        "workspace": {
            "project_key": workspace.get("project_key"),
            "display_name": workspace.get("display_name"),
            "path_alias": workspace.get("path_alias"),
        },
        "session": {
            "session_id": session.get("session_id"),
            "branch_depth": session.get("branch_depth"),
            "has_parent": bool(session.get("parent_session_id")),
        },
        "rule_phase": {
            "primary_phase": rule_phase,
            "primary_label": phase.get("primary_label"),
            "phases": phase.get("phases", []),
            "recommended_memory_targets": phase.get("recommended_memory_targets", []),
            "recommended_evidence_policy": phase.get("recommended_evidence_policy"),
        },
        "core_evidence": core,
        "candidate_evidence": candidate,
        "raw_pointers": pointers,
        "stats": {
            "core_count": len(core),
            "candidate_count": len(candidate),
            "pointer_count": len(pointers),
            "core_chars": sum(len(str(item.get("text") or "")) for item in core),
            "candidate_summary_chars": sum(len(str(item.get("summary") or "")) for item in candidate),
        },
    }
    return pool


PHASE_ROUTER_SYSTEM_PROMPT = """你是 Memory Lab 的阶段路由器。
你的任务是根据证据池判断这一轮 CP 对话的真实阶段，规则初判只是参考，可以改判。

允许阶段：
- user_interaction_preference：用户希望 CP/助手如何工作
- requirement_alignment：需求对齐、选型、澄清，不以修复问题为主
- code_generation：生成或修改代码
- compilation：编译、构建、编译错误或编译通过
- runtime_symptom_diagnosis：烧录/运行现象反馈、排查、定位根因
- fix_confirmation：用户或工具确认修复有效
- other：无长期记忆价值或无法判断

只输出 JSON，格式严格如下：
{
  "schema_version": "memory-lab-phase-router-v1",
  "final_phase": "...",
  "phase_changed": true,
  "changed_from": "规则阶段或 null",
  "confidence": 0.0,
  "reason": "中文理由，说明为什么保留或改判",
  "extract_targets": ["..."],
  "need_more_evidence": false,
  "evidence_requests": []
}

规则：
- 用户原话优先于助手描述，助手结论优先于工具长日志。
- 编译通过如果只是背景，而用户核心是烧录现象，应判 runtime_symptom_diagnosis。
- 用户说“修好了/重新烧录后正常/验证通过”，优先判 fix_confirmation。
- 需求选型问题不要误判成工程经验，除非包含明确失败、根因、修复或验证。
- 不要创造证据；只能基于 core_evidence 和 candidate_evidence 摘要判断。
- 如果证据不够，可设置 need_more_evidence=true，并只使用支持的请求类型。"""


def _validate_router_decision(value: Mapping[str, Any], rule_phase: str) -> dict[str, Any]:
    if value.get("schema_version") != PHASE_ROUTER_SCHEMA_VERSION:
        raise PhaseRouterError("router schema_version is unsupported")
    final_phase = value.get("final_phase")
    if final_phase not in PHASES:
        raise PhaseRouterError("router final_phase is unsupported")
    phase_changed = value.get("phase_changed")
    if not isinstance(phase_changed, bool):
        raise PhaseRouterError("router phase_changed must be boolean")
    changed_from = value.get("changed_from")
    if phase_changed:
        if changed_from != rule_phase:
            raise PhaseRouterError("router changed_from must equal rule_phase")
        if final_phase == rule_phase:
            raise PhaseRouterError("router phase_changed contradicts final_phase")
    else:
        changed_from = None
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise PhaseRouterError("router confidence must be between 0 and 1")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise PhaseRouterError("router reason must be a non-empty string")
    extract_targets = value.get("extract_targets")
    if not isinstance(extract_targets, list) or not all(isinstance(item, str) and item for item in extract_targets):
        raise PhaseRouterError("router extract_targets must be a string array")
    need_more_evidence = value.get("need_more_evidence")
    if not isinstance(need_more_evidence, bool):
        raise PhaseRouterError("router need_more_evidence must be boolean")
    requests = value.get("evidence_requests")
    if not isinstance(requests, list):
        raise PhaseRouterError("router evidence_requests must be an array")
    normalized_requests: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise PhaseRouterError("router evidence request must be an object")
        request_type = request.get("type")
        if request_type not in SUPPORTED_EVIDENCE_REQUEST_TYPES:
            raise PhaseRouterError("router evidence request type is unsupported")
        normalized_requests.append(
            {
                "request_id": str(request.get("request_id") or f"req{index + 1}")[:80],
                "type": request_type,
                "ref": str(request.get("ref") or "")[:120],
                "purpose": str(request.get("purpose") or "")[:240],
            }
        )
    return {
        "schema_version": PHASE_ROUTER_SCHEMA_VERSION,
        "final_phase": final_phase,
        "final_label": PHASES[final_phase]["label"],
        "phase_changed": phase_changed,
        "changed_from": changed_from,
        "confidence": float(confidence),
        "reason": reason.strip()[:500],
        "extract_targets": extract_targets[:12],
        "need_more_evidence": need_more_evidence,
        "evidence_requests": normalized_requests,
    }


class DeepSeekPhaseRouter:
    """Use a stateless JSON model call to correct the deterministic phase."""

    def __init__(self, client: OpenAICompatibleJsonClient):
        self.client = client

    def route(self, evidence_pool: Mapping[str, Any]) -> dict[str, Any]:
        rule_phase = str(
            ((evidence_pool.get("rule_phase") or {}).get("primary_phase") if isinstance(evidence_pool.get("rule_phase"), Mapping) else None)
            or "other"
        )
        payload = {
            "schema_version": EVIDENCE_POOL_SCHEMA_VERSION,
            "allowed_phases": list(PHASES),
            "supported_evidence_request_types": sorted(SUPPORTED_EVIDENCE_REQUEST_TYPES),
            "turn": evidence_pool.get("turn"),
            "workspace": evidence_pool.get("workspace"),
            "rule_phase": evidence_pool.get("rule_phase"),
            "core_evidence": evidence_pool.get("core_evidence", []),
            "candidate_evidence": evidence_pool.get("candidate_evidence", []),
            "instruction": "判断真实阶段；如规则初判错误请改判。不要输出 Markdown。",
        }
        raw = self.client.complete_json(
            system_prompt=PHASE_ROUTER_SYSTEM_PROMPT,
            user_payload=payload,
            max_tokens=1200,
        )
        return _validate_router_decision(raw, rule_phase)


def _terminal_turns(lab_database: Path, source_identifier: str, limit: int) -> list[dict[str, Any]]:
    with closing(open_store(lab_database)) as connection:
        rows = connection.execute(
            """
            SELECT turn_id, session_id, current_state, user_timestamp
            FROM conversation_turn
            WHERE source_id = ?
              AND current_state IN ('completed', 'failed', 'interrupted')
            ORDER BY user_timestamp DESC, turn_id DESC
            LIMIT ?
            """,
            (source_identifier, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def run_evidence_pool_report(
    source_database: Path,
    lab_database: Path,
    *,
    settle_ms: int = 0,
    limit: int = 10,
    layer: str = "all",
    router: DeepSeekPhaseRouter | None = None,
) -> dict[str, Any]:
    if isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if layer not in {"all", "core", "candidate", "pointer", "router"}:
        raise ValueError("layer must be all, core, candidate, pointer, or router")
    run_turn_scan(source_database, lab_database, settle_ms=settle_ms)
    source_identifier = source_id(source_database)
    turns = _terminal_turns(lab_database, source_identifier, limit)
    items: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    changed_count = 0
    for turn in reversed(turns):
        bundle = build_turn_evidence(source_database, lab_database, turn["turn_id"])
        phase = classify_turn_phase(bundle)
        pool = build_evidence_pool(bundle, phase_result=phase)
        decision: dict[str, Any] | None = None
        router_error: str | None = None
        if router is not None:
            try:
                decision = router.route(pool)
                if decision["phase_changed"]:
                    changed_count += 1
                counts[decision["final_phase"]] += 1
            except Exception as error:
                router_error = f"{type(error).__name__}: {error}"
                counts[pool["rule_phase"]["primary_phase"]] += 1
        else:
            counts[pool["rule_phase"]["primary_phase"]] += 1
        item: dict[str, Any] = {
            "turn_id": turn["turn_id"],
            "session_id": turn["session_id"],
            "state": turn["current_state"],
            "user_timestamp": turn["user_timestamp"],
            "project": pool["workspace"],
            "rule_phase": pool["rule_phase"],
            "stats": pool["stats"],
        }
        if layer in {"all", "core"}:
            item["core_evidence"] = pool["core_evidence"]
        if layer in {"all", "candidate"}:
            item["candidate_evidence"] = pool["candidate_evidence"]
        if layer in {"all", "pointer"}:
            item["raw_pointers"] = pool["raw_pointers"]
        if layer in {"all", "router"}:
            item["router_decision"] = decision
            item["router_error"] = router_error
        items.append(item)
    return {
        "status": "complete",
        "schema_version": EVIDENCE_POOL_SCHEMA_VERSION,
        "phase_router_schema_version": PHASE_ROUTER_SCHEMA_VERSION,
        "source_id": source_identifier,
        "turn_count": len(items),
        "layer": layer,
        "model_router_enabled": router is not None,
        "router_changed_count": changed_count,
        "counts": dict(sorted(counts.items())),
        "turns": items,
    }


__all__ = [
    "DeepSeekPhaseRouter",
    "EVIDENCE_POOL_SCHEMA_VERSION",
    "PHASE_ROUTER_SCHEMA_VERSION",
    "PhaseRouterError",
    "build_evidence_pool",
    "run_evidence_pool_report",
]
