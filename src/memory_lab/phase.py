"""Lightweight CP turn phase classification for staged memory extraction.

This module is intentionally deterministic.  It does not decide whether to
write long-term memory; it only labels a completed turn so later extractors can
build smaller, phase-specific evidence packs.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from .source import source_id
from .store import open_store
from .turn_evidence import build_turn_evidence
from .turns import run_turn_scan


PHASE_VERSION = "memory-lab-turn-phase-rules-v1"

TERMINAL_STATES = {"completed", "failed", "interrupted"}


PHASES: dict[str, dict[str, Any]] = {
    "user_interaction_preference": {
        "label": "用户交互习惯",
        "memory_targets": ["user", "workflow"],
        "evidence_policy": "以用户消息为主，少量参考助手是否执行；通常不需要工具输出。",
    },
    "requirement_alignment": {
        "label": "需求对齐",
        "memory_targets": ["workflow", "project"],
        "evidence_policy": "保留需求澄清流程、隐性约束和可复用模板；不保存普通参数选择。",
    },
    "code_generation": {
        "label": "代码生成",
        "memory_targets": ["workflow", "engineering"],
        "evidence_policy": "保留生成前后检查规则、工具链约束；不保存完整代码和变量名。",
    },
    "compilation": {
        "label": "编译成功或失败",
        "memory_targets": ["engineering", "workflow"],
        "evidence_policy": "需要关键报错、修复方案、助手结论和编译通过/失败结果。",
    },
    "runtime_symptom_diagnosis": {
        "label": "运行现象反馈与定位",
        "memory_targets": ["engineering", "project"],
        "evidence_policy": "需要用户现象反馈、可见定位过程、关键代码/配置事实、根因和解决方向。",
    },
    "fix_confirmation": {
        "label": "修复确认",
        "memory_targets": ["engineering", "project"],
        "evidence_policy": "需要用户确认或编译/运行验证，用于更新 verified 状态或关闭旧问题。",
    },
    "other": {
        "label": "其他",
        "memory_targets": [],
        "evidence_policy": "默认不做阶段专项提取，只交给通用长期价值判断。",
    },
}


KEYWORDS: dict[str, tuple[str, ...]] = {
    "user_interaction_preference": (
        "以后",
        "无论",
        "先给结论",
        "先说结论",
        "不要",
        "别",
        "我希望",
        "我想要",
        "每阶段",
        "不要重复实现",
        "不喜欢",
        "习惯",
        "偏好",
        "纠正",
    ),
    "requirement_alignment": (
        "需求",
        "确认",
        "方案",
        "传感器",
        "pwm",
        "滞回",
        "失效安全",
        "adc 来源",
        "定时器",
        "采样周期",
        "看门狗",
        "阈值",
        "分辨率",
    ),
    "code_generation": (
        "生成代码",
        "写代码",
        "修改代码",
        "新增模块",
        "main.c",
        ".c",
        ".h",
        "startup.a51",
        "函数",
        "驱动",
        "实现",
    ),
    "compilation": (
        "编译",
        "build",
        "keil",
        "c51",
        "error",
        "warning",
        "0 error",
        "0 warning",
        "报错",
        "语法",
        "链接",
        "通过",
    ),
    "runtime_symptom_diagnosis": (
        "烧录",
        "板子",
        "运行后",
        "现象",
        "不转",
        "没反应",
        "无变化",
        "温度不变",
        "报警",
        "占空比",
        "ntc",
        "定位",
        "排查",
        "根因",
        "为什么",
        "adc_sample",
        "return 2048",
        "桩",
    ),
    "fix_confirmation": (
        "修好了",
        "解决了",
        "正常了",
        "确认",
        "验证通过",
        "重新烧录",
        "恢复正常",
        "可以变化",
        "能正常",
        "问题已经解决",
    ),
}


def _text_parts(container: Mapping[str, Any]) -> list[str]:
    parts = container.get("text_parts")
    if not isinstance(parts, list):
        return []
    result: list[str] = []
    for part in parts:
        if isinstance(part, Mapping) and isinstance(part.get("text"), str):
            result.append(part["text"])
    return result


def _tool_texts(bundle: Mapping[str, Any]) -> list[str]:
    tools = bundle.get("tools")
    if not isinstance(tools, list):
        return []
    result: list[str] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        values = [str(tool.get("name") or ""), str(tool.get("status") or "")]
        for field in ("input", "output", "error"):
            channel = tool.get(field)
            if isinstance(channel, Mapping) and isinstance(channel.get("text"), str):
                values.append(channel["text"])
        text = "\n".join(item for item in values if item)
        if text:
            result.append(text)
    return result


def _user_answer_texts(bundle: Mapping[str, Any]) -> list[str]:
    answers = bundle.get("user_answers")
    if not isinstance(answers, list):
        return []
    return [
        item["text"]
        for item in answers
        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
    ]


def _contains_any(text: str, keywords: tuple[str, ...]) -> list[str]:
    lower = text.casefold()
    return [keyword for keyword in keywords if keyword.casefold() in lower]


def _has_user_fix_confirmation(user_text: str) -> bool:
    lower = user_text.casefold()
    strong = any(
        phrase in lower
        for phrase in (
            "已解决",
            "问题解决",
            "验证通过",
            "实测",
            "修好了",
            "正常了",
            "确认问题",
        )
    )
    if strong:
        return True
    has_validation_action = any(phrase in lower for phrase in ("重新烧录", "重新编译", "烧录验证"))
    has_positive_outcome = any(
        phrase in lower
        for phrase in ("能正常", "恢复正常", "会变化", "能按", "可以变化", "正常随")
    )
    return has_validation_action and has_positive_outcome


def _has_requirement_alignment_directive(user_text: str) -> bool:
    lower = user_text.casefold()
    return (
        any(
            phrase in lower
            for phrase in (
                "需求澄清",
                "整理需求",
                "需求整理",
                "模块划分",
                "验收条件",
                "验收方案",
            )
        )
        and any(
            phrase in lower
            for phrase in (
                "不修改",
                "不生成代码",
                "不改文件",
                "只完成",
                "只整理",
            )
        )
    )


def classify_turn_phase(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a single evidence bundle into one or more extraction phases."""

    user_text = "\n".join(_text_parts(bundle.get("user", {}) if isinstance(bundle.get("user"), Mapping) else {}))
    user_text += "\n" + "\n".join(_user_answer_texts(bundle))
    assistant_text = "\n".join(
        _text_parts(bundle.get("assistant", {}) if isinstance(bundle.get("assistant"), Mapping) else {})
    )
    tool_text = "\n".join(_tool_texts(bundle))
    all_text = "\n".join([user_text, assistant_text, tool_text])

    scores: Counter[str] = Counter()
    matched: dict[str, list[str]] = {}
    for phase, keywords in KEYWORDS.items():
        hits = _contains_any(all_text, keywords)
        if not hits:
            continue
        matched[phase] = hits[:10]
        scores[phase] += len(hits)

    # Source weighting: user instructions are especially useful for preference
    # and symptom phases; tool output is especially useful for compile/runtime.
    user_pref_hits = _contains_any(user_text, KEYWORDS["user_interaction_preference"])
    if user_pref_hits:
        scores["user_interaction_preference"] += len(user_pref_hits) * 2
    compile_tool_hits = _contains_any(tool_text, KEYWORDS["compilation"])
    if compile_tool_hits:
        scores["compilation"] += len(compile_tool_hits)
    runtime_user_hits = _contains_any(user_text, KEYWORDS["runtime_symptom_diagnosis"])
    if runtime_user_hits:
        scores["runtime_symptom_diagnosis"] += len(runtime_user_hits) * 2
    fix_user_hits = _contains_any(user_text, KEYWORDS["fix_confirmation"])
    if fix_user_hits:
        scores["fix_confirmation"] += len(fix_user_hits) * 3

    phases = [
        {
            "phase": phase,
            "label": PHASES[phase]["label"],
            "score": score,
            "matched_keywords": matched.get(phase, [])[:10],
            "memory_targets": PHASES[phase]["memory_targets"],
            "evidence_policy": PHASES[phase]["evidence_policy"],
        }
        for phase, score in scores.most_common()
        if score > 0
    ]
    if not phases:
        phases = [
            {
                "phase": "other",
                "label": PHASES["other"]["label"],
                "score": 0,
                "matched_keywords": [],
                "memory_targets": [],
                "evidence_policy": PHASES["other"]["evidence_policy"],
            }
        ]

    # Strong user-level directives should decide the primary phase even when
    # assistant/tool text contains many incidental compile/runtime words.
    forced_primary: str | None = None
    if _has_user_fix_confirmation(user_text):
        forced_primary = "fix_confirmation"
    elif _has_requirement_alignment_directive(user_text):
        forced_primary = "requirement_alignment"
    if forced_primary and phases[0]["phase"] != forced_primary:
        forced_item = next(
            (item for item in phases if item["phase"] == forced_primary),
            {
                "phase": forced_primary,
                "label": PHASES[forced_primary]["label"],
                "score": 1,
                "matched_keywords": [],
                "memory_targets": PHASES[forced_primary]["memory_targets"],
                "evidence_policy": PHASES[forced_primary]["evidence_policy"],
            },
        )
        phases = [forced_item, *[item for item in phases if item["phase"] != forced_primary]]

    primary = phases[0]["phase"]
    return {
        "phase_version": PHASE_VERSION,
        "turn_id": (bundle.get("turn") or {}).get("turn_id"),
        "primary_phase": primary,
        "primary_label": PHASES[primary]["label"],
        "phases": phases[:4],
        "recommended_memory_targets": phases[0]["memory_targets"],
        "recommended_evidence_policy": phases[0]["evidence_policy"],
        "signals": {
            "user_chars": len(user_text),
            "assistant_chars": len(assistant_text),
            "tool_event_count": len(bundle.get("tools") or []),
            "tool_chars": len(tool_text),
        },
    }


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


def run_turn_phase_report(
    source_database: Path,
    lab_database: Path,
    *,
    settle_ms: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    if isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    run_turn_scan(source_database, lab_database, settle_ms=settle_ms)
    source_identifier = source_id(source_database)
    turns = _terminal_turns(lab_database, source_identifier, limit)
    classified: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for turn in reversed(turns):
        bundle = build_turn_evidence(source_database, lab_database, turn["turn_id"])
        result = classify_turn_phase(bundle)
        result.update(
            {
                "session_id": turn["session_id"],
                "state": turn["current_state"],
                "user_timestamp": turn["user_timestamp"],
                "project": bundle.get("workspace", {}),
                "user_preview": "\n".join(_text_parts(bundle.get("user", {})))[:240],
                "assistant_preview": "\n".join(_text_parts(bundle.get("assistant", {})))[:240],
            }
        )
        counts[result["primary_phase"]] += 1
        classified.append(result)
    return {
        "status": "complete",
        "phase_version": PHASE_VERSION,
        "source_id": source_identifier,
        "turn_count": len(classified),
        "counts": dict(sorted(counts.items())),
        "turns": classified,
    }


__all__ = ["PHASE_VERSION", "classify_turn_phase", "run_turn_phase_report"]
