"""Offline evaluation-set planning utilities for memory extraction tests."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG
from .memory_react_agent import run_memory_react_agent
from .multiturn_eval_contract import validate_multiturn_evalset
from .pg_store import DEFAULT_PG_CONFIG, ingest_turn_to_postgres, load_pg_config


DEFAULT_EVALSET = Path(__file__).resolve().parents[2] / "evalsets" / "memory_eval_v1"
DEFAULT_EVAL_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eval"
EXPECTED_MEMORY_TYPES = {
    "user_preference",
    "compile_failure_experience",
    "runtime_failure_experience",
    "none",
}
EXPECTED_STATUSES = {"candidate", "verified", "rejected", "none"}
EXPECTED_SCOPES = {"user", "domain", "project", "none", "user_or_domain", "domain_or_project"}
KNOWN_MEMORY_ACTIONS = {
    "create_memory",
    "update_memory",
    "merge_memory",
    "update_topic",
    "noop",
}
REQUIRED_EVAL_CASE_FIELDS = {
    "case_id",
    "turn_id",
    "case_type",
    "allowed_actions",
    "forbidden_actions",
    "expected_keywords",
    "expected_process",
    "expected_memory_type",
    "expected_scope",
    "expected_relation",
    "semantic_expectation",
    "should_extract",
    "should_not_extract",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {line_no} is invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path.name} line {line_no} must be a JSON object")
        case_id = str(item.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"{path.name} line {line_no} missing non-empty case_id")
        if case_id in seen_ids:
            raise ValueError(f"{path.name} line {line_no} duplicates case_id {case_id}")
        seen_ids.add(case_id)
        items.append(item)
    return items


def _validate_eval_case_parameters(cases: list[dict[str, Any]]) -> dict[str, Any]:
    missing_by_case: dict[str, list[str]] = {}
    invalid_by_case: dict[str, list[str]] = {}
    type_counts: Counter[str] = Counter()
    allowed_action_counts: Counter[str] = Counter()
    for case in cases:
        case_id = str(case.get("case_id") or "")
        missing = sorted(field for field in REQUIRED_EVAL_CASE_FIELDS if field not in case)
        if missing:
            missing_by_case[case_id] = missing
        invalid: list[str] = []
        allowed_actions = _string_list(case.get("allowed_actions"))
        forbidden_actions = _string_list(case.get("forbidden_actions"))
        if not isinstance(case.get("allowed_actions"), list) or not allowed_actions:
            invalid.append("allowed_actions 必须是非空数组")
        unknown_actions = sorted(
            set(allowed_actions + forbidden_actions) - KNOWN_MEMORY_ACTIONS
        )
        if unknown_actions:
            invalid.append("未知动作：" + ",".join(unknown_actions))
        memory_type = str(case.get("expected_memory_type") or "")
        type_counts[memory_type or "missing"] += 1
        if memory_type and memory_type not in EXPECTED_MEMORY_TYPES:
            invalid.append(f"expected_memory_type 不在收敛类型内：{memory_type}")
        status = str(case.get("expected_status") or "")
        if status and status not in EXPECTED_STATUSES:
            invalid.append(f"expected_status 不合法：{status}")
        scope = str(case.get("expected_scope") or "")
        if scope and scope not in EXPECTED_SCOPES:
            invalid.append(f"expected_scope 不在建议集合内：{scope}")
        expected_process = case.get("expected_process")
        if not isinstance(expected_process, Mapping):
            invalid.append("expected_process 必须是对象")
        elif "load_memory_index" not in expected_process:
            invalid.append("expected_process 缺少 load_memory_index")
        if not str(case.get("semantic_expectation") or "").strip():
            invalid.append("semantic_expectation 不能为空")
        if memory_type == "none" and "noop" not in allowed_actions:
            invalid.append("expected_memory_type=none 时 allowed_actions 应包含 noop")
        if memory_type != "none" and allowed_actions == ["noop"]:
            invalid.append("非 none 类型不应只允许 noop")
        for action in allowed_actions:
            allowed_action_counts[action] += 1
        if invalid:
            invalid_by_case[case_id] = invalid
    return {
        "case_count": len(cases),
        "missing_required_field_case_count": len(missing_by_case),
        "invalid_case_count": len(invalid_by_case),
        "missing_required_fields": missing_by_case,
        "invalid_cases": invalid_by_case,
        "expected_memory_type_counts": dict(type_counts),
        "allowed_action_counts": dict(allowed_action_counts),
        "status": "pass" if not missing_by_case and not invalid_by_case else "attention_needed",
    }


def _require_string(item: dict[str, Any], key: str, *, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} missing non-empty {key}")
    return value.strip()


def _read_scenarios(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f"{path.name} missing scenarios list")
    seen_ids: set[str] = set()
    for index, scenario in enumerate(scenarios, 1):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario #{index} must be an object")
        scenario_id = _require_string(
            scenario, "scenario_id", context=f"scenario #{index}"
        )
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario_id {scenario_id}")
        seen_ids.add(scenario_id)
        turn_ids = scenario.get("turn_ids")
        if not isinstance(turn_ids, list) or not turn_ids:
            raise ValueError(f"scenario {scenario_id} must have non-empty turn_ids")
        for turn_index, turn_id in enumerate(turn_ids, 1):
            if not isinstance(turn_id, str) or not turn_id.strip():
                raise ValueError(
                    f"scenario {scenario_id} turn #{turn_index} must be a turn_id string"
                )
    return data


def _turn_rows(lab_database: Path, turn_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not turn_ids:
        return {}
    placeholders = ",".join("?" for _ in turn_ids)
    query = f"""
        SELECT
            t.turn_id,
            t.session_id,
            t.user_message_id,
            t.current_state,
            t.assistant_count,
            t.part_count,
            t.terminal_error,
            e.session_title,
            e.directory,
            e.content
        FROM conversation_turn AS t
        LEFT JOIN turn_event AS te ON te.turn_id = t.turn_id
        LEFT JOIN event AS e ON e.event_id = te.event_id
        WHERE t.turn_id IN ({placeholders})
        ORDER BY t.user_timestamp
    """
    with closing(sqlite3.connect(lab_database)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, sorted(turn_ids)).fetchall()
    return {str(row["turn_id"]): dict(row) for row in rows}


def run_eval_memory_plan(
    *,
    evalset_path: Path = DEFAULT_EVALSET,
    lab_database: Path,
    dry_run_plan: bool = False,
) -> dict[str, Any]:
    """Validate an evaluation set and return a model-independent plan report."""

    if not dry_run_plan:
        raise ValueError("eval-memory currently supports only --dry-run-plan")

    evalset_path = Path(evalset_path)
    cases_path = evalset_path / "cases.jsonl"
    scenarios_path = evalset_path / "scenarios.json"
    if not evalset_path.is_dir():
        raise ValueError(f"eval set directory not found: {evalset_path}")
    if not cases_path.is_file():
        raise ValueError(f"cases.jsonl not found: {cases_path}")
    if not scenarios_path.is_file():
        raise ValueError(f"scenarios.json not found: {scenarios_path}")

    cases = _load_jsonl(cases_path)
    case_parameter_quality = _validate_eval_case_parameters(cases)
    scenarios_doc = _read_scenarios(scenarios_path)
    scenarios = scenarios_doc["scenarios"]
    multiturn_contract = None
    if scenarios_doc.get("schema_version") == "memory-multiturn-scenarios-v1":
        multiturn_contract = validate_multiturn_evalset(
            evalset_path,
            lab_database=lab_database,
            verify_evidence_hashes=True,
        )

    case_turn_ids: list[str] = []
    source_paths: set[str] = set()
    case_type_counts: Counter[str] = Counter()
    expected_action_counts: Counter[str] = Counter()
    for case in cases:
        context = f"case {case.get('case_id')}"
        case_turn_ids.append(_require_string(case, "turn_id", context=context))
        case_type_counts[_require_string(case, "case_type", context=context)] += 1
        source_db = _require_string(case, "source_db", context=context)
        source_paths.add(source_db)
        actions = case.get("expected_action")
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, str) and action.strip():
                    expected_action_counts[action.strip()] += 1

    scenario_turn_ids = [
        str(turn_id).strip()
        for scenario in scenarios
        for turn_id in scenario["turn_ids"]
        if str(turn_id).strip()
    ]
    all_turn_ids = set(case_turn_ids) | set(scenario_turn_ids)
    found_rows = _turn_rows(lab_database, all_turn_ids)
    missing_turn_ids = sorted(all_turn_ids - set(found_rows))
    scenario_only_turn_ids = sorted(set(scenario_turn_ids) - set(case_turn_ids))

    missing_sources = [
        source_path
        for source_path in sorted(source_paths)
        if not Path(source_path).is_file()
    ]

    state_counts: Counter[str] = Counter()
    terminal_error_count = 0
    for turn_id in all_turn_ids:
        row = found_rows.get(turn_id)
        if not row:
            continue
        state_counts[str(row.get("current_state") or "unknown")] += 1
        if row.get("terminal_error"):
            terminal_error_count += 1

    scenario_summaries = []
    for scenario in scenarios:
        turn_ids = [str(turn_id).strip() for turn_id in scenario["turn_ids"]]
        scenario_summaries.append(
            {
                "scenario_id": scenario["scenario_id"],
                "title": scenario.get("title", ""),
                "turn_count": len(turn_ids),
                "missing_turn_ids": [
                    turn_id for turn_id in turn_ids if turn_id not in found_rows
                ],
                "states": dict(
                    Counter(
                        str(found_rows[turn_id].get("current_state") or "unknown")
                        for turn_id in turn_ids
                        if turn_id in found_rows
                    )
                ),
            }
        )

    status = "complete" if not missing_turn_ids and not missing_sources else "failed"
    result = {
        "status": status,
        "tool": "eval_memory_plan",
        "evalset_path": str(evalset_path),
        "lab_database": str(lab_database),
        "dry_run_plan": True,
        "case_count": len(cases),
        "scenario_count": len(scenarios),
        "case_parameter_quality": case_parameter_quality,
        "case_type_counts": dict(case_type_counts),
        "expected_action_counts": dict(expected_action_counts),
        "case_turn_count": len(case_turn_ids),
        "scenario_turn_ref_count": len(scenario_turn_ids),
        "unique_turn_ref_count": len(all_turn_ids),
        "scenario_only_turn_ids": scenario_only_turn_ids,
        "found_turn_count": len(found_rows),
        "missing_turn_ids": missing_turn_ids,
        "source_database_count": len(source_paths),
        "missing_source_databases": missing_sources,
        "turn_state_counts": dict(state_counts),
        "terminal_error_count": terminal_error_count,
        "scenario_summaries": scenario_summaries,
        "next_step": (
            "实现单 case 回放：为本次 eval run 创建独立 user_id，"
            "按 cases.jsonl 调用 pg-memory-react-run 并生成事实报告。"
        ),
    }
    if multiturn_contract is not None:
        result["multiturn_contract"] = multiturn_contract
    return result


def _utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:80] or "run"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_eval_pg_config(
    *,
    base_pg_config_path: Path,
    output_dir: Path,
    user_id: str,
) -> Path:
    config = load_pg_config(base_pg_config_path)
    config["user"] = dict(config["user"])
    config["user"]["user_id"] = user_id
    config["user"]["email"] = ""
    path = output_dir / "pg-config.eval.json"
    _write_json(path, {"schema_version": "memory-lab-postgres-config-v1", **config})
    return path


def _action_counts_from_result(result: Mapping[str, Any]) -> Counter[str]:
    plan = result.get("plan") if isinstance(result.get("plan"), Mapping) else {}
    changes = plan.get("changes") if isinstance(plan, Mapping) else []
    counts: Counter[str] = Counter()
    if isinstance(changes, list):
        for change in changes:
            if isinstance(change, Mapping):
                action = change.get("action")
                if isinstance(action, str) and action.strip():
                    counts[action.strip()] += 1
    return counts


def _changes_from_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = result.get("plan") if isinstance(result.get("plan"), Mapping) else {}
    changes = plan.get("changes") if isinstance(plan, Mapping) else []
    summaries: list[dict[str, Any]] = []
    if not isinstance(changes, list):
        return summaries
    for change in changes:
        if not isinstance(change, Mapping):
            continue
        memory = change.get("memory") if isinstance(change.get("memory"), Mapping) else {}
        topic = change.get("topic") if isinstance(change.get("topic"), Mapping) else {}
        evidence_refs = change.get("evidence_refs")
        summaries.append(
            {
                "action": change.get("action"),
                "reason": change.get("reason"),
                "stable_key": memory.get("stable_key") or change.get("stable_key"),
                "memory_id": change.get("memory_id") or change.get("target_memory_id"),
                "memory_type": memory.get("memory_type") or change.get("memory_type"),
                "memory_summary": memory.get("summary") or change.get("summary"),
                "memory_content": memory.get("content") or change.get("content"),
                "topic_key": topic.get("topic_key") or change.get("topic_key"),
                "topic_title": topic.get("title") or change.get("topic_title"),
                "topic_summary": topic.get("summary"),
                "evidence_count": len(evidence_refs) if isinstance(evidence_refs, list) else 0,
            }
        )
    return summaries


def _expected_actions(case: Mapping[str, Any]) -> set[str]:
    raw = case.get("expected_action")
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _case_matches_expected(case: Mapping[str, Any], action_counts: Mapping[str, int]) -> bool:
    expected = _expected_actions(case)
    if not expected:
        return True
    actual = {action for action, count in action_counts.items() if count}
    return bool(expected & actual)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _contains_loose(text: str, phrase: str) -> bool:
    phrase = " ".join(str(phrase or "").split()).strip()
    if not phrase:
        return False
    if phrase in text:
        return True
    tokens = [
        token.strip()
        for token in phrase.replace("/", " ").replace("、", " ").replace("，", " ").split()
        if len(token.strip()) >= 2
    ]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in text)
    return hits >= max(1, min(len(tokens), 2))


def _change_text(changes: list[dict[str, Any]]) -> str:
    return "\n".join(_quality_text(change) for change in changes)


def _read_react_log(path_value: Any) -> dict[str, Any] | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _react_step_names(item: Mapping[str, Any]) -> set[str]:
    log = _read_react_log(item.get("react_log_path"))
    if not log:
        return set()
    steps = log.get("steps")
    if not isinstance(steps, list):
        return set()
    names: set[str] = set()
    for step in steps:
        if isinstance(step, Mapping) and isinstance(step.get("step"), str):
            names.add(step["step"])
    return names


def _expected_process_flags(item: Mapping[str, Any], changes: list[dict[str, Any]]) -> dict[str, bool]:
    case = _evaluation_case(item)
    expected = case.get("expected_process")
    explicit = expected if isinstance(expected, Mapping) else {}
    actions = {str(change.get("action") or "") for change in changes}
    case_type = str(case.get("case_type") or "")
    need_memory_lookup = bool(explicit.get("load_memory_index")) or bool(
        actions & {"create_memory", "update_memory", "merge_memory", "update_topic", "noop"}
    )
    need_final_reason = bool(explicit.get("final_reason_required", True))
    if "evidence_required" in explicit:
        need_evidence = bool(explicit.get("evidence_required"))
    else:
        need_evidence = case_type not in {"low_value_continue", "low_value_location_question", "artifact_conversion"}
    return {
        "need_memory_lookup": need_memory_lookup,
        "need_final_reason": need_final_reason,
        "need_evidence": need_evidence,
    }


def _type_required_markers(memory_type: str, text: str) -> dict[str, bool]:
    text = text or ""
    if memory_type in {"runtime_failure_pattern", "compile_failure_pattern"}:
        return {
            "symptom": any(token in text for token in ("症状", "现象", "表现", "错误")),
            "cause": any(token in text for token in ("根因", "原因", "导致")),
            "solution": any(token in text for token in ("修复", "解决", "处理", "改为")),
            "validation": any(token in text for token in ("验证", "确认", "编译通过", "0 Error", "用户测试", "烧录")),
            "scope": any(token in text for token in ("适用", "触发条件", "场景", "范围")),
        }
    if memory_type == "user_preference":
        return {
            "scenario": any(token in text for token in ("适用场景", "场景", "当用户", "任务")),
            "preference": any(token in text for token in ("用户偏好", "用户要求", "期望", "希望")),
            "behavior": any(token in text for token in ("助手应", "协作要求", "避免", "不要", "需要")),
            "value": any(token in text for token in ("价值", "减少", "避免", "复用")),
        }
    if memory_type in {"workflow", "build_experience", "engineering_failure_pattern"}:
        return {
            "trigger": any(token in text for token in ("触发", "适用", "场景", "当")),
            "process": any(token in text for token in ("流程", "先", "再", "步骤", "应")),
            "boundary": any(token in text for token in ("避免", "不要", "边界", "除非")),
        }
    return {}


def _warning_category(warning: str) -> str:
    if "出现禁止动作" in warning or "过程缺少" in warning or "决策缺少" in warning:
        return "hard_error"
    if "动作不在期望集合" in warning:
        return "action_mismatch"
    if "疑似沉淀禁止信息" in warning or "疑似混入一次性项目细节" in warning:
        return "quality_boundary"
    if "单轮新增" in warning or "长期复用价值表达不明显" in warning or "类型结构缺少" in warning:
        return "quality_review"
    if "期望证据/应提取信息命中" in warning:
        return "semantic_review"
    return "other"


def _warning_category_label(category: str) -> str:
    return {
        "hard_error": "硬错误",
        "action_mismatch": "动作期望不匹配",
        "quality_boundary": "质量边界问题",
        "quality_review": "质量复核项",
        "semantic_review": "语义复核项",
        "other": "其他",
    }.get(category, category)


def _markdown_summary(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['evalset_id']} 回放报告",
        "",
        "## 总览",
        "",
        f"- run_id：`{report['run_id']}`",
        f"- user_id：`{report['user_id']}`",
        f"- 模式：`{report['mode']}`",
        f"- dry_run：`{report['dry_run']}`",
        f"- case 数：{report['case_count']}",
        f"- 成功：{report['success_count']}",
        f"- 失败：{report['failure_count']}",
        f"- 期望动作匹配：{report['expected_match_count']}",
        f"- 期望动作不匹配：{report['expected_mismatch_count']}",
        "",
        "## 动作统计",
        "",
    ]
    action_counts = report.get("action_counts") or {}
    if action_counts:
        for action, count in sorted(action_counts.items()):
            lines.append(f"- {action}: {count}")
    else:
        lines.append("- 无动作")
    lines.extend(["", "## Case 结果", ""])
    for item in report.get("cases") or []:
        lines.append(
            f"- `{item['case_id']}` `{item['case_type']}` "
            f"状态={item['status']} 动作={item.get('action_counts') or {}} "
            f"匹配期望={item['matches_expected']} 日志=`{item.get('react_log_path') or ''}`"
        )
    lines.append("")
    return "\n".join(lines)


def _short_text(value: Any, limit: int = 420) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _quality_text(change: Mapping[str, Any]) -> str:
    return " ".join(
        str(change.get(key) or "")
        for key in (
            "memory_summary",
            "memory_content",
            "reason",
            "topic_title",
            "topic_summary",
        )
    )


def _quality_checks_for_change(change: Mapping[str, Any]) -> dict[str, Any]:
    action = str(change.get("action") or "")
    memory_type = str(change.get("memory_type") or "")
    text = _quality_text(change)
    if action == "noop":
        warnings = [] if change.get("reason") else ["noop 缺少原因"]
        return {
            "action": action,
            "stable_key": change.get("stable_key"),
            "memory_id": change.get("memory_id"),
            "memory_type": memory_type or None,
            "has_symptom": False,
            "has_cause": False,
            "has_solution": False,
            "has_validation": False,
            "has_reusable_marker": False,
            "concrete_marker_count": 0,
            "warnings": warnings,
            "summary": change.get("reason") or "",
        }
    has_symptom = any(token in text for token in ("症状", "现象", "表现"))
    has_cause = any(token in text for token in ("根因", "原因", "导致"))
    has_solution = any(token in text for token in ("修复", "解决", "处理", "改为"))
    has_validation = any(token in text for token in ("验证", "确认", "编译通过", "0 Error", "用户测试"))
    has_reusable_marker = any(
        token in text
        for token in (
            "可复用",
            "适用场景",
            "适用范围",
            "未来",
            "类似",
            "用户偏好",
            "经验",
            "通用",
            "处理原则",
        )
    )
    concrete_markers = sum(
        1
        for token in (
            "本次",
            "当前项目",
            ".c",
            ".h",
            ".hex",
            "函数",
            "变量",
            "引脚",
            "P",
            "GPIO",
            "MJ",
            "EL",
            "FL",
        )
        if token in text
    )
    should_have_chain = action in {
        "create_memory",
        "update_memory",
        "merge_memory",
    } and memory_type in {"runtime_failure_pattern", "compile_failure_pattern"}
    missing_chain = []
    if should_have_chain:
        if not has_symptom:
            missing_chain.append("症状/现象")
        if not has_cause:
            missing_chain.append("原因/根因")
        if not has_solution:
            missing_chain.append("解决/修复")
        if not has_validation:
            missing_chain.append("验证")
    warnings: list[str] = []
    if should_have_chain and missing_chain:
        warnings.append("缺少" + "、".join(missing_chain))
    type_markers = _type_required_markers(memory_type, text)
    missing_type_markers = [
        name for name, present in type_markers.items() if not present
    ]
    if missing_type_markers:
        marker_labels = {
            "symptom": "现象",
            "cause": "根因",
            "solution": "解决方案",
            "validation": "验证",
            "scope": "适用范围",
            "scenario": "适用场景",
            "preference": "用户偏好",
            "behavior": "协作要求",
            "value": "价值说明",
            "trigger": "触发场景",
            "process": "推荐流程",
            "boundary": "边界/避免事项",
        }
        warnings.append(
            "类型结构缺少"
            + "、".join(marker_labels.get(name, name) for name in missing_type_markers)
        )
    if concrete_markers >= 3 and not has_reusable_marker:
        warnings.append("疑似混入一次性项目细节")
    if action in {"create_memory", "update_memory", "merge_memory"} and not (
        has_reusable_marker or memory_type in {"user_preference", "workflow", "build_experience"}
    ):
        warnings.append("长期复用价值表达不明显")
    return {
        "action": action,
        "stable_key": change.get("stable_key"),
        "memory_id": change.get("memory_id"),
        "memory_type": memory_type or None,
        "has_symptom": has_symptom,
        "has_cause": has_cause,
        "has_solution": has_solution,
        "has_validation": has_validation,
        "has_reusable_marker": has_reusable_marker,
        "concrete_marker_count": concrete_markers,
        "type_required_markers": type_markers,
        "type_required_coverage": _percent(
            sum(1 for present in type_markers.values() if present), len(type_markers)
        ),
        "warnings": warnings,
        "summary": change.get("memory_summary") or change.get("reason") or "",
    }


def _evaluation_case(item: Mapping[str, Any]) -> Mapping[str, Any]:
    case = item.get("case")
    return case if isinstance(case, Mapping) else item


def _item_metric_checks(item: Mapping[str, Any], changes: list[dict[str, Any]]) -> dict[str, Any]:
    case = _evaluation_case(item)
    action_counts = item.get("action_counts") if isinstance(item.get("action_counts"), Mapping) else {}
    actual_actions = {str(action) for action, count in action_counts.items() if count}
    if not actual_actions:
        actual_actions = {
            str(change.get("action") or "")
            for change in changes
            if str(change.get("action") or "").strip()
        }

    allowed_actions = set(_string_list(case.get("allowed_actions"))) | _expected_actions(case)
    forbidden_actions = set(_string_list(case.get("forbidden_actions")))
    action_allowed = None if not allowed_actions else bool(actual_actions & allowed_actions)
    forbidden_hits = sorted(actual_actions & forbidden_actions)

    text = _change_text(changes)
    expected_keywords = _string_list(case.get("expected_keywords"))
    expected_evidence = _string_list(case.get("expected_evidence"))
    should_extract = _string_list(case.get("should_extract"))
    should_not_extract = _string_list(case.get("should_not_extract"))
    evidence_expectations = expected_keywords or expected_evidence or should_extract
    evidence_hits = [
        phrase for phrase in evidence_expectations if _contains_loose(text, phrase)
    ]
    extract_hits = [phrase for phrase in should_extract if _contains_loose(text, phrase)]
    leak_hits = [phrase for phrase in should_not_extract if _contains_loose(text, phrase)]

    ingest = item.get("ingest") if isinstance(item.get("ingest"), Mapping) else {}
    evidence_counts = ingest.get("evidence_counts") if isinstance(ingest.get("evidence_counts"), Mapping) else {}
    evidence_pool_count = sum(
        int(evidence_counts.get(name) or 0) for name in ("core", "candidate", "pointer")
    )
    evidence_reference_count = sum(int(change.get("evidence_count") or 0) for change in changes)
    step_names = _react_step_names(item)
    process_flags = _expected_process_flags(item, changes)
    reason_present = all(str(change.get("reason") or "").strip() for change in changes) if changes else True
    final_changes_present = "final_changes" in step_names if step_names else None
    memory_index_loaded = "load_memory_index" in step_names if step_names else None
    evidence_package_present = evidence_pool_count > 0 if evidence_counts else None

    process = {
        "evidence_package_present": evidence_package_present,
        "evidence_pool_count": evidence_pool_count,
        "evidence_reference_count": evidence_reference_count,
        "memory_index_loaded": memory_index_loaded,
        "final_changes_present": final_changes_present,
        "reason_present": reason_present,
        "need_memory_lookup": process_flags["need_memory_lookup"],
        "need_evidence": process_flags["need_evidence"],
        "action_allowed": action_allowed,
        "forbidden_action_hits": forbidden_hits,
    }
    result = {
        "expected_evidence_total": len(evidence_expectations),
        "expected_evidence_hit_count": len(evidence_hits),
        "expected_evidence_missed": [
            phrase for phrase in evidence_expectations if phrase not in evidence_hits
        ],
        "should_extract_total": len(should_extract),
        "should_extract_hit_count": len(extract_hits),
        "should_extract_missed": [
            phrase for phrase in should_extract if phrase not in extract_hits
        ],
        "should_not_extract_total": len(should_not_extract),
        "should_not_extract_leak_count": len(leak_hits),
        "should_not_extract_leaks": leak_hits,
    }
    warnings: list[str] = []
    if action_allowed is False:
        warnings.append(
            "动作不在期望集合内：实际="
            + ",".join(sorted(actual_actions or {"无"}))
            + "；期望="
            + ",".join(sorted(allowed_actions))
        )
    if forbidden_hits:
        warnings.append("出现禁止动作：" + ",".join(forbidden_hits))
    if process_flags["need_memory_lookup"] and memory_index_loaded is False:
        warnings.append("过程缺少已有记忆索引加载")
    if process_flags["need_evidence"] and evidence_package_present is False:
        warnings.append("过程缺少证据包")
    if not reason_present:
        warnings.append("决策缺少明确原因")
    if evidence_expectations and len(evidence_hits) < len(evidence_expectations):
        warnings.append(
            f"期望证据/应提取信息命中 {len(evidence_hits)}/{len(evidence_expectations)}，需语义复核"
        )
    if leak_hits:
        warnings.append("疑似沉淀禁止信息：" + "；".join(leak_hits[:3]))

    return {
        "process": process,
        "result": result,
        "warnings": warnings,
    }


def _build_quality_report(report: Mapping[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    duplicate_keys: Counter[str] = Counter()
    case_by_turn: dict[str, dict[str, Any]] = {}
    evalset_path_value = report.get("evalset_path")
    if isinstance(evalset_path_value, str) and evalset_path_value.strip():
        cases_path = Path(evalset_path_value) / "cases.jsonl"
        if cases_path.is_file():
            try:
                case_by_turn = {
                    str(case.get("turn_id") or ""): case
                    for case in _load_jsonl(cases_path)
                    if str(case.get("turn_id") or "")
                }
            except ValueError:
                case_by_turn = {}

    def add_item(
        *,
        scope: str,
        item_id: str,
        title: str,
        changes: list[dict[str, Any]],
        source_item: Mapping[str, Any],
    ) -> None:
        create_count = sum(1 for change in changes if change.get("action") == "create_memory")
        item_warnings: list[str] = []
        if create_count >= 4:
            item_warnings.append(f"单轮新增 {create_count} 条记忆，疑似过度拆分，需要人工复核")
        change_checks = [_quality_checks_for_change(change) for change in changes]
        metric_checks = _item_metric_checks(source_item, changes)
        item_warnings.extend(metric_checks.get("warnings") or [])
        for check in change_checks:
            key = check.get("stable_key")
            if isinstance(key, str) and key:
                duplicate_keys[key] += 1
            item_warnings.extend(
                f"{check.get('stable_key') or check.get('memory_id') or check.get('action')}: {warning}"
                for warning in check.get("warnings", [])
            )
        items.append(
            {
                "scope": scope,
                "item_id": item_id,
                "title": title,
                "change_count": len(changes),
                "create_count": create_count,
                "warnings": item_warnings,
                "metrics": metric_checks,
                "checks": change_checks,
            }
        )

    for case in report.get("cases") or []:
        add_item(
            scope="case",
            item_id=str(case.get("case_id") or ""),
            title=str(case.get("case_type") or ""),
            changes=list(case.get("changes") or []),
            source_item=case,
        )
    for scenario in report.get("scenarios") or []:
        for turn in scenario.get("turns") or []:
            turn_with_case = dict(turn)
            turn_with_case.setdefault("case", case_by_turn.get(str(turn.get("turn_id") or "")))
            case = _evaluation_case(turn_with_case)
            add_item(
                scope=f"scenario:{scenario.get('scenario_id')}",
                item_id=str(turn.get("turn_id") or ""),
                title=f"turn {turn.get('turn_index')} / {case.get('case_type') or ''}",
                changes=list(turn.get("changes") or []),
                source_item=turn_with_case,
            )

    duplicate_stable_keys = {
        key: count for key, count in duplicate_keys.items() if count > 1
    }
    warning_count = sum(len(item["warnings"]) for item in items)
    warning_categories: Counter[str] = Counter()
    for item in items:
        for warning in item.get("warnings") or []:
            warning_categories[_warning_category(str(warning))] += 1

    def count_bool(path: tuple[str, str], expected: bool = True, *, only_when: tuple[str, str] | None = None) -> tuple[int, int]:
        numerator = 0
        denominator = 0
        for item in items:
            metrics = item.get("metrics") or {}
            target = metrics.get(path[0], {}).get(path[1])
            if only_when is not None:
                gate = metrics.get(only_when[0], {}).get(only_when[1])
                if not gate:
                    continue
            if target is None:
                continue
            denominator += 1
            if target is expected:
                numerator += 1
        return numerator, denominator

    memory_lookup_ok, memory_lookup_total = count_bool(
        ("process", "memory_index_loaded"), True, only_when=("process", "need_memory_lookup")
    )
    evidence_package_ok, evidence_package_total = count_bool(
        ("process", "evidence_package_present"), True, only_when=("process", "need_evidence")
    )
    reason_ok, reason_total = count_bool(("process", "reason_present"), True)
    action_ok, action_total = count_bool(("process", "action_allowed"), True)

    expected_evidence_total = sum(
        int((item.get("metrics") or {}).get("result", {}).get("expected_evidence_total") or 0)
        for item in items
    )
    expected_evidence_hits = sum(
        int((item.get("metrics") or {}).get("result", {}).get("expected_evidence_hit_count") or 0)
        for item in items
    )
    should_extract_total = sum(
        int((item.get("metrics") or {}).get("result", {}).get("should_extract_total") or 0)
        for item in items
    )
    should_extract_hits = sum(
        int((item.get("metrics") or {}).get("result", {}).get("should_extract_hit_count") or 0)
        for item in items
    )
    should_not_extract_total = sum(
        int((item.get("metrics") or {}).get("result", {}).get("should_not_extract_total") or 0)
        for item in items
    )
    should_not_extract_leaks = sum(
        int((item.get("metrics") or {}).get("result", {}).get("should_not_extract_leak_count") or 0)
        for item in items
    )
    process_metrics = {
        "memory_index_loaded": {
            "ok": memory_lookup_ok,
            "total": memory_lookup_total,
            "rate": _percent(memory_lookup_ok, memory_lookup_total),
        },
        "evidence_package_present": {
            "ok": evidence_package_ok,
            "total": evidence_package_total,
            "rate": _percent(evidence_package_ok, evidence_package_total),
        },
        "action_allowed": {
            "ok": action_ok,
            "total": action_total,
            "rate": _percent(action_ok, action_total),
        },
        "reason_present": {
            "ok": reason_ok,
            "total": reason_total,
            "rate": _percent(reason_ok, reason_total),
        },
    }
    result_metrics = {
        "expected_evidence_keyword_recall": {
            "hits": expected_evidence_hits,
            "total": expected_evidence_total,
            "rate": _percent(expected_evidence_hits, expected_evidence_total),
            "note": "关键词近似指标，只用于初筛；语义完整性需要大模型评审",
        },
        "should_extract_keyword_recall": {
            "hits": should_extract_hits,
            "total": should_extract_total,
            "rate": _percent(should_extract_hits, should_extract_total),
            "note": "关键词近似指标，只用于初筛；同义表达可能被低估",
        },
        "should_not_extract_leakage": {
            "leaks": should_not_extract_leaks,
            "total": should_not_extract_total,
            "rate": _percent(should_not_extract_leaks, should_not_extract_total),
            "note": "命中表示疑似泄漏，需要区分作为证据保留还是污染长期记忆主体",
        },
        "duplicate_stable_key_count": len(duplicate_stable_keys),
        "over_split_item_count": sum(
            1 for item in items if int(item.get("create_count") or 0) >= 4
        ),
    }
    return {
        "status": "attention_needed" if warning_count or duplicate_stable_keys else "pass",
        "item_count": len(items),
        "warning_count": warning_count,
        "process_metrics": process_metrics,
        "result_metrics": result_metrics,
        "warning_categories": dict(warning_categories),
        "duplicate_stable_keys": duplicate_stable_keys,
        "items": items,
    }


def _markdown_quality_report(report: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['evalset_id']} 记忆质量评测报告",
        "",
        "这份报告是规则型初筛，用于快速定位需要人工复核的沉淀质量问题；它不是最终裁判。",
        "",
        "## 总览",
        "",
        f"- run_id：`{report['run_id']}`",
        f"- 模式：`{report['mode']}`",
        f"- dry_run：`{report['dry_run']}`",
        f"- 状态：`{quality['status']}`",
        f"- 检查对象数：{quality['item_count']}",
        f"- 风险提示数：{quality['warning_count']}",
        "",
    ]
    process_metrics = quality.get("process_metrics") or {}
    result_metrics = quality.get("result_metrics") or {}
    lines.append("## 过程指标")
    lines.append("")
    process_labels = {
        "evidence_package_present": "证据包存在率",
        "memory_index_loaded": "已有记忆检查率",
        "action_allowed": "动作命中率",
        "reason_present": "决策理由存在率",
    }
    for key, label in process_labels.items():
        metric = process_metrics.get(key) or {}
        rate = metric.get("rate")
        rate_text = "不可计算" if rate is None else f"{rate * 100:.1f}%"
        lines.append(
            f"- {label}：{rate_text}（{metric.get('ok', 0)}/{metric.get('total', 0)}）"
        )
    lines.append("")
    lines.append("## 结果硬指标")
    lines.append("")
    result_labels = {
        "expected_evidence_keyword_recall": "期望证据/应提取关键词命中率",
        "should_extract_keyword_recall": "应提取信息关键词命中率",
        "should_not_extract_leakage": "禁止信息疑似泄漏率",
    }
    for key, label in result_labels.items():
        metric = result_metrics.get(key) or {}
        if key == "should_not_extract_leakage":
            numerator = metric.get("leaks", 0)
        else:
            numerator = metric.get("hits", 0)
        denominator = metric.get("total", 0)
        rate = metric.get("rate")
        rate_text = "不可计算" if rate is None else f"{rate * 100:.1f}%"
        lines.append(f"- {label}：{rate_text}（{numerator}/{denominator}）")
    lines.append(
        f"- 重复 stable_key 数：{result_metrics.get('duplicate_stable_key_count', 0)}"
    )
    lines.append(
        f"- 单轮新增过多项数：{result_metrics.get('over_split_item_count', 0)}"
    )
    lines.append("")
    lines.append("> 说明：关键词命中率只是脚本初筛，不能替代语义评审；范围是否合适、是否真有长期价值、是否该合并/拆分仍需要大模型/人工判断。")
    lines.append("")
    lines.append("## 问题分类汇总")
    lines.append("")
    warning_categories = quality.get("warning_categories") or {}
    if warning_categories:
        for category, count in sorted(warning_categories.items()):
            lines.append(f"- {_warning_category_label(str(category))}：{count}")
    else:
        lines.append("- 暂无风险分类")
    lines.append("")
    lines.append("分类说明：")
    lines.append("")
    lines.append("- 硬错误：禁止动作、缺少必要过程或缺少决策理由，通常应优先修。")
    lines.append("- 动作期望不匹配：可能是模型动作错，也可能是评测期望需要复核。")
    lines.append("- 质量边界问题：疑似把一次性项目细节、禁止信息写进长期记忆。")
    lines.append("- 质量复核项：过度拆分、复用价值表达不明显、结构字段不完整。")
    lines.append("- 语义复核项：关键词未命中或命中不足，需要人工/大模型判断是否真的漏信息。")
    lines.append("")
    duplicate_stable_keys = quality.get("duplicate_stable_keys") or {}
    lines.append("## 重复 stable_key 检查")
    lines.append("")
    if duplicate_stable_keys:
        for key, count in sorted(duplicate_stable_keys.items()):
            lines.append(f"- `{key}` 出现 {count} 次")
    else:
        lines.append("- 未发现同一次回放内重复 stable_key")
    lines.append("")

    lines.append("## 风险项")
    lines.append("")
    risky_items = [item for item in quality.get("items") or [] if item.get("warnings")]
    if not risky_items:
        lines.append("- 未发现需要优先复核的风险项")
    else:
        for item in risky_items:
            lines.append(f"### `{item['item_id']}`")
            lines.append("")
            lines.append(f"- 范围：`{item['scope']}`")
            lines.append(f"- 标题：{item.get('title') or ''}")
            lines.append(f"- 变化数：{item['change_count']}")
            lines.append(f"- 新增记忆数：{item['create_count']}")
            metrics = item.get("metrics") or {}
            process = metrics.get("process") or {}
            result = metrics.get("result") or {}
            lines.append(
                "- 过程："
                f"证据包={process.get('evidence_package_present')}，"
                f"查已有记忆={process.get('memory_index_loaded')}，"
                f"动作命中={process.get('action_allowed')}，"
                f"理由={process.get('reason_present')}"
            )
            if result.get("should_extract_total") or result.get("should_not_extract_total"):
                lines.append(
                    "- 结果："
                    f"应提取命中={result.get('should_extract_hit_count', 0)}/{result.get('should_extract_total', 0)}，"
                    f"禁止泄漏={result.get('should_not_extract_leak_count', 0)}/{result.get('should_not_extract_total', 0)}"
                )
            lines.append("- 风险：")
            for warning in item.get("warnings") or []:
                lines.append(f"  - {warning}")
            lines.append("")

    lines.append("## 明细")
    lines.append("")
    for item in quality.get("items") or []:
        lines.append(f"### `{item['item_id']}`")
        lines.append("")
        lines.append(f"- 范围：`{item['scope']}`")
        lines.append(f"- 变化数：{item['change_count']}")
        lines.append(f"- 新增记忆数：{item['create_count']}")
        metrics = item.get("metrics") or {}
        process = metrics.get("process") or {}
        result = metrics.get("result") or {}
        lines.append(
            "- 过程指标："
            f"证据包={process.get('evidence_package_present')}，"
            f"证据引用数={process.get('evidence_reference_count')}，"
            f"查已有记忆={process.get('memory_index_loaded')}，"
            f"动作命中={process.get('action_allowed')}，"
            f"理由={process.get('reason_present')}"
        )
        if result.get("expected_evidence_total") or result.get("should_not_extract_total"):
            lines.append(
                "- 结果指标："
                f"期望证据命中={result.get('expected_evidence_hit_count', 0)}/{result.get('expected_evidence_total', 0)}，"
                f"应提取命中={result.get('should_extract_hit_count', 0)}/{result.get('should_extract_total', 0)}，"
                f"禁止泄漏={result.get('should_not_extract_leak_count', 0)}/{result.get('should_not_extract_total', 0)}"
            )
        checks = item.get("checks") or []
        if not checks:
            lines.extend(["- 无变化", ""])
            continue
        for index, check in enumerate(checks, 1):
            label = check.get("stable_key") or check.get("memory_id") or check.get("action")
            lines.append(f"- {index}. `{check.get('action')}` `{label}`")
            summary = _short_text(check.get("summary"), 220)
            if summary:
                lines.append(f"  - 摘要：{summary}")
            markers = []
            if check.get("has_symptom"):
                markers.append("症状")
            if check.get("has_cause"):
                markers.append("原因")
            if check.get("has_solution"):
                markers.append("解决")
            if check.get("has_validation"):
                markers.append("验证")
            if check.get("has_reusable_marker"):
                markers.append("复用表达")
            lines.append(f"  - 结构标签：{', '.join(markers) if markers else '无'}")
            if check.get("warnings"):
                lines.append(f"  - 风险：{'; '.join(check['warnings'])}")
        lines.append("")
    return "\n".join(lines)


def _markdown_generated_memories(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['evalset_id']} 生成记忆明细",
        "",
        "这份文件用于人工快速判断：每条 turn 到底沉淀了什么，或者为什么没有沉淀。",
        "",
        "## 总览",
        "",
        f"- run_id：`{report['run_id']}`",
        f"- 模式：`{report['mode']}`",
        f"- dry_run：`{report['dry_run']}`",
        f"- case 数：{report['case_count']}",
        "",
    ]
    for item in report.get("cases") or []:
        lines.extend(
            [
                f"## {item['case_index']:03d}. `{item['case_id']}`",
                "",
                f"- 类型：`{item.get('case_type')}`",
                f"- turn_id：`{item.get('turn_id')}`",
                f"- user_id：`{item.get('user_id')}`",
                f"- 状态：`{item.get('status')}`",
                f"- 期望动作：`{', '.join(item.get('expected_action') or [])}`",
                f"- 实际动作：`{item.get('action_counts') or {}}`",
                f"- 匹配期望：`{item.get('matches_expected')}`",
                f"- ReAct 日志：`{item.get('react_log_path') or ''}`",
                "",
            ]
        )
        changes = item.get("changes") or []
        if not changes:
            lines.extend(["无记忆变化。", ""])
            continue
        for change_index, change in enumerate(changes, 1):
            action = change.get("action") or "unknown"
            lines.append(f"### 变化 {change_index}: `{action}`")
            lines.append("")
            if change.get("topic_title") or change.get("topic_key"):
                lines.append(
                    f"- 主题：{change.get('topic_title') or ''} "
                    f"`{change.get('topic_key') or ''}`"
                )
            if change.get("stable_key") or change.get("memory_id"):
                lines.append(
                    f"- 记忆：`{change.get('stable_key') or change.get('memory_id') or ''}`"
                )
            if change.get("memory_type"):
                lines.append(f"- 类型：`{change.get('memory_type')}`")
            if change.get("memory_summary"):
                lines.append(f"- 摘要：{_short_text(change.get('memory_summary'), 320)}")
            if change.get("memory_content"):
                lines.append(f"- 内容：{_short_text(change.get('memory_content'), 620)}")
            if change.get("reason"):
                lines.append(f"- 原因：{_short_text(change.get('reason'), 420)}")
            lines.append(f"- 证据数：{change.get('evidence_count') or 0}")
            lines.append("")
    return "\n".join(lines)


def _scenario_source_db_for_turn(
    *,
    turn_id: str,
    source_by_turn: Mapping[str, str],
    fallback_source_db: str | None,
) -> Path:
    source_db = source_by_turn.get(turn_id) or fallback_source_db
    if not source_db:
        raise ValueError(
            f"scenario turn {turn_id} has no source_db in cases.jsonl and no unique fallback source"
        )
    return Path(source_db)


def _refresh_report_cases_from_evalset(
    report: Mapping[str, Any], *, evalset_path: Path
) -> dict[str, Any]:
    refreshed = json.loads(json.dumps(report, ensure_ascii=False))
    cases_path = Path(evalset_path) / "cases.jsonl"
    case_by_turn: dict[str, dict[str, Any]] = {}
    if cases_path.is_file():
        case_by_turn = {
            str(case.get("turn_id") or ""): case
            for case in _load_jsonl(cases_path)
            if str(case.get("turn_id") or "")
        }
    for case in refreshed.get("cases") or []:
        turn_id = str(case.get("turn_id") or "")
        latest = case_by_turn.get(turn_id)
        if latest:
            case["case"] = latest
    for scenario in refreshed.get("scenarios") or []:
        for turn in scenario.get("turns") or []:
            turn_id = str(turn.get("turn_id") or "")
            latest = case_by_turn.get(turn_id)
            if latest:
                turn["case"] = latest
    refreshed["evalset_path"] = str(evalset_path)
    return refreshed


def run_eval_memory_review_existing(
    *,
    evalset_path: Path = DEFAULT_EVALSET,
    existing_report_directory: Path,
) -> dict[str, Any]:
    """Recalculate quality reports for an existing eval run without rerunning ReAct."""

    existing_report_directory = Path(existing_report_directory)
    detail_path = existing_report_directory / "eval-detail.json"
    if not detail_path.is_file():
        raise ValueError(f"eval-detail.json not found: {detail_path}")
    try:
        report = json.loads(detail_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid eval-detail.json: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("eval-detail.json must contain a JSON object")
    refreshed = _refresh_report_cases_from_evalset(
        report, evalset_path=Path(evalset_path)
    )
    quality = _build_quality_report(refreshed)
    refreshed["quality"] = quality
    rejudged_detail_path = existing_report_directory / "eval-detail.rejudged.json"
    quality_json_path = existing_report_directory / "quality-rejudged.json"
    quality_md_path = existing_report_directory / "quality-report-rejudged.md"
    _write_json(rejudged_detail_path, refreshed)
    _write_json(quality_json_path, quality)
    quality_md_path.write_text(
        _markdown_quality_report(refreshed, quality), encoding="utf-8"
    )
    return {
        "status": "complete",
        "tool": "eval_memory_review_existing",
        "evalset_path": str(evalset_path),
        "existing_report_directory": str(existing_report_directory),
        "run_id": refreshed.get("run_id"),
        "quality": {
            "status": quality.get("status"),
            "warning_count": quality.get("warning_count"),
            "process_metrics": quality.get("process_metrics"),
            "result_metrics": quality.get("result_metrics"),
            "warning_categories": quality.get("warning_categories"),
        },
        "outputs": {
            "eval_detail": str(rejudged_detail_path),
            "quality_json": str(quality_json_path),
            "quality_report": str(quality_md_path),
        },
    }


def _markdown_scenario_generated_memories(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['evalset_id']} 场景顺序回放记忆明细",
        "",
        "这份文件用于人工判断：同一个场景的多个 turn 连续处理时，记忆如何逐步新增、更新或过滤。",
        "",
        "## 总览",
        "",
        f"- run_id：`{report['run_id']}`",
        f"- 模式：`{report['mode']}`",
        f"- dry_run：`{report['dry_run']}`",
        f"- 场景数：{report['scenario_count']}",
        f"- turn 数：{report['turn_count']}",
        f"- 成功 turn：{report['success_count']}",
        f"- 失败 turn：{report['failure_count']}",
        "",
    ]
    action_counts = report.get("action_counts") or {}
    lines.append("## 动作统计")
    lines.append("")
    if action_counts:
        for action, count in sorted(action_counts.items()):
            lines.append(f"- {action}: {count}")
    else:
        lines.append("- 无动作")
    lines.append("")

    for scenario in report.get("scenarios") or []:
        lines.extend(
            [
                f"## 场景 {scenario['scenario_index']:03d}. `{scenario['scenario_id']}`",
                "",
                f"- 标题：{scenario.get('title') or ''}",
                f"- user_id：`{scenario.get('user_id')}`",
                f"- 状态：`{scenario.get('status')}`",
                f"- turn 数：{scenario.get('turn_count')}",
                f"- 实际动作：`{scenario.get('action_counts') or {}}`",
                "",
            ]
        )
        expected = scenario.get("expected") if isinstance(scenario.get("expected"), Mapping) else {}
        if expected:
            lines.append("### 场景期望")
            lines.append("")
            for key, value in expected.items():
                lines.append(f"- {key}：`{value}`")
            lines.append("")

        for turn in scenario.get("turns") or []:
            lines.extend(
                [
                    f"### Turn {turn['turn_index']:03d}. `{turn.get('turn_id')}`",
                    "",
                    f"- 状态：`{turn.get('status')}`",
                    f"- 实际动作：`{turn.get('action_counts') or {}}`",
                    f"- ReAct 日志：`{turn.get('react_log_path') or ''}`",
                    "",
                ]
            )
            changes = turn.get("changes") or []
            if not changes:
                lines.extend(["无记忆变化。", ""])
                continue
            for change_index, change in enumerate(changes, 1):
                action = change.get("action") or "unknown"
                lines.append(f"#### 变化 {change_index}: `{action}`")
                lines.append("")
                if change.get("topic_title") or change.get("topic_key"):
                    lines.append(
                        f"- 主题：{change.get('topic_title') or ''} "
                        f"`{change.get('topic_key') or ''}`"
                    )
                if change.get("stable_key") or change.get("memory_id"):
                    lines.append(
                        f"- 记忆：`{change.get('stable_key') or change.get('memory_id') or ''}`"
                    )
                if change.get("memory_type"):
                    lines.append(f"- 类型：`{change.get('memory_type')}`")
                if change.get("memory_summary"):
                    lines.append(f"- 摘要：{_short_text(change.get('memory_summary'), 320)}")
                if change.get("memory_content"):
                    lines.append(f"- 内容：{_short_text(change.get('memory_content'), 620)}")
                if change.get("reason"):
                    lines.append(f"- 原因：{_short_text(change.get('reason'), 420)}")
                lines.append(f"- 证据数：{change.get('evidence_count') or 0}")
                lines.append("")
    return "\n".join(lines)


def run_eval_memory_replay(
    *,
    evalset_path: Path = DEFAULT_EVALSET,
    lab_database: Path,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    report_directory: Path = DEFAULT_EVAL_REPORT_DIR,
    run_name: str | None = None,
    mode: str = "single-batch",
    limit: int = 3,
    dry_run: bool = True,
    max_iterations: int = 3,
    memory_limit: int = 80,
    topic_limit: int = 200,
) -> dict[str, Any]:
    """Replay a small batch of eval cases through the ReAct memory agent."""

    if mode not in {"single-batch", "single-isolated", "scenario-sequential"}:
        raise ValueError(f"unsupported eval replay mode for this step: {mode}")

    evalset_path = Path(evalset_path)
    cases = _load_jsonl(evalset_path / "cases.jsonl")
    if mode == "scenario-sequential":
        scenarios_doc = _read_scenarios(evalset_path / "scenarios.json")
        scenarios = scenarios_doc["scenarios"]
        selected_scenarios = scenarios[: max(1, int(limit))]
        evalset_id = evalset_path.name
        stamp = _utc_run_stamp()
        label = _safe_name(run_name or mode)
        run_id = f"{evalset_id}-{label}-{stamp}"
        batch_user_id = f"eval-{_safe_name(evalset_id)}-{_safe_name(label)}-{stamp.lower()}"
        output_dir = Path(report_directory) / evalset_id / run_id
        react_log_dir = output_dir / "react-logs"
        source_by_turn = {
            _require_string(case, "turn_id", context=f"case {case.get('case_id')}"): _require_string(
                case, "source_db", context=f"case {case.get('case_id')}"
            )
            for case in cases
        }
        case_by_turn = {
            _require_string(case, "turn_id", context=f"case {case.get('case_id')}"): case
            for case in cases
        }
        unique_sources = sorted(set(source_by_turn.values()))
        fallback_source_db = unique_sources[0] if len(unique_sources) == 1 else None

        scenario_results: list[dict[str, Any]] = []
        total_actions: Counter[str] = Counter()
        success_count = 0
        failure_count = 0
        turn_count = 0

        for scenario_index, scenario in enumerate(selected_scenarios, 1):
            scenario_id = _require_string(
                scenario, "scenario_id", context=f"scenario #{scenario_index}"
            )
            scenario_user_id = (
                f"{batch_user_id}-scenario-{scenario_index:03d}-{_safe_name(scenario_id).lower()}"
            )
            scenario_dir = output_dir / "runs" / f"{scenario_index:03d}-{_safe_name(scenario_id)}"
            scenario_pg_config = _write_eval_pg_config(
                base_pg_config_path=pg_config_path,
                output_dir=scenario_dir,
                user_id=scenario_user_id,
            )
            scenario_action_counts: Counter[str] = Counter()
            turn_results: list[dict[str, Any]] = []

            for turn_index, raw_turn_id in enumerate(scenario["turn_ids"], 1):
                turn_id = str(raw_turn_id).strip()
                turn_count += 1
                scenario_log_name = f"{scenario_index:03d}-{_safe_name(scenario_id)[:32]}"
                turn_log_name = f"{turn_index:03d}-{_safe_name(turn_id[-12:])}"
                case_log_dir = (
                    react_log_dir
                    / scenario_log_name
                    / turn_log_name
                )
                source_db = _scenario_source_db_for_turn(
                    turn_id=turn_id,
                    source_by_turn=source_by_turn,
                    fallback_source_db=fallback_source_db,
                )
                result: dict[str, Any]
                ingest_report: dict[str, Any] | None = None
                try:
                    ingest_report = ingest_turn_to_postgres(
                        source_db,
                        lab_database,
                        pg_config_path=scenario_pg_config,
                        turn_id=turn_id,
                        settle_ms=0,
                    )
                    result = run_memory_react_agent(
                        source_database=source_db,
                        turn_id=turn_id,
                        pg_config_path=scenario_pg_config,
                        config_path=config_path,
                        memory_limit=memory_limit,
                        topic_limit=topic_limit,
                        max_iterations=max_iterations,
                        dry_run=dry_run,
                        log_dir=case_log_dir,
                    )
                except Exception as exc:  # pragma: no cover - exercised by integration runs
                    result = {
                        "status": "failed",
                        "stage": "exception",
                        "turn_id": turn_id,
                        "error": str(exc),
                        "committed": False,
                    }

                action_counts = _action_counts_from_result(result)
                changes = _changes_from_result(result)
                scenario_action_counts.update(action_counts)
                total_actions.update(action_counts)
                if result.get("status") == "complete":
                    success_count += 1
                else:
                    failure_count += 1
                react_log_path = (
                    (result.get("log") or {}).get("run_log_path")
                    if isinstance(result.get("log"), Mapping)
                    else None
                )
                turn_result = {
                    "turn_index": turn_index,
                    "turn_id": turn_id,
                    "case": case_by_turn.get(turn_id),
                    "source_db": str(source_db),
                    "status": result.get("status"),
                    "stage": result.get("stage"),
                    "dry_run": dry_run,
                    "run_id": result.get("run_id"),
                    "react_log_path": react_log_path,
                    "ingest": ingest_report,
                    "action_counts": dict(action_counts),
                    "changes": changes,
                    "committed": bool(result.get("committed")),
                    "error": result.get("error"),
                }
                _write_json(
                    scenario_dir / f"turn-{turn_index:03d}.json",
                    {"scenario": scenario, "result": result, "summary": turn_result},
                )
                turn_results.append(turn_result)

            scenario_results.append(
                {
                    "scenario_index": scenario_index,
                    "scenario_id": scenario_id,
                    "title": scenario.get("title", ""),
                    "user_id": scenario_user_id,
                    "eval_pg_config_path": str(scenario_pg_config),
                    "status": "complete"
                    if all(turn["status"] == "complete" for turn in turn_results)
                    else "complete_with_errors",
                    "turn_count": len(turn_results),
                    "action_counts": dict(scenario_action_counts),
                    "expected": scenario.get("expected"),
                    "turns": turn_results,
                }
            )

        report: dict[str, Any] = {
            "status": "complete" if failure_count == 0 else "complete_with_errors",
            "tool": "eval_memory_replay",
            "evalset_id": evalset_id,
            "evalset_path": str(evalset_path),
            "run_id": run_id,
            "user_id": None,
            "user_ids": [scenario["user_id"] for scenario in scenario_results],
            "mode": mode,
            "dry_run": dry_run,
            "scenario_count": len(selected_scenarios),
            "turn_count": turn_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "action_counts": dict(total_actions),
            "report_directory": str(output_dir),
            "eval_pg_config_path": None,
            "scenarios": scenario_results,
        }
        quality = _build_quality_report(report)
        report["quality"] = quality
        _write_json(output_dir / "eval-detail.json", report)
        (output_dir / "generated-memories.md").write_text(
            _markdown_scenario_generated_memories(report), encoding="utf-8"
        )
        (output_dir / "quality-report.md").write_text(
            _markdown_quality_report(report, quality), encoding="utf-8"
        )
        (output_dir / "eval-summary.md").write_text(
            _markdown_scenario_generated_memories(report), encoding="utf-8"
        )
        bad_scenarios = [
            scenario for scenario in scenario_results if scenario["status"] != "complete"
        ]
        (output_dir / "bad-cases.md").write_text(
            _markdown_scenario_generated_memories({**report, "scenarios": bad_scenarios}),
            encoding="utf-8",
        )
        return report

    selected_cases = cases[: max(1, int(limit))]
    evalset_id = evalset_path.name
    stamp = _utc_run_stamp()
    label = _safe_name(run_name or mode)
    run_id = f"{evalset_id}-{label}-{stamp}"
    batch_user_id = f"eval-{_safe_name(evalset_id)}-{_safe_name(label)}-{stamp.lower()}"
    output_dir = Path(report_directory) / evalset_id / run_id
    react_log_dir = output_dir / "react-logs"
    batch_pg_config = (
        _write_eval_pg_config(
            base_pg_config_path=pg_config_path,
            output_dir=output_dir,
            user_id=batch_user_id,
        )
        if mode == "single-batch"
        else None
    )

    case_results: list[dict[str, Any]] = []
    total_actions: Counter[str] = Counter()
    success_count = 0
    failure_count = 0
    expected_match_count = 0
    expected_mismatch_count = 0

    for index, case in enumerate(selected_cases, 1):
        case_id = _require_string(case, "case_id", context=f"case #{index}")
        source_db = Path(_require_string(case, "source_db", context=case_id))
        turn_id = _require_string(case, "turn_id", context=case_id)
        case_dir = output_dir / "runs" / f"{index:03d}-{_safe_name(case_id)}"
        case_log_dir = react_log_dir / f"{index:03d}-{_safe_name(case_id)}"
        case_user_id = (
            batch_user_id
            if mode == "single-batch"
            else f"{batch_user_id}-case-{index:03d}-{_safe_name(case_id).lower()}"
        )
        eval_pg_config = (
            batch_pg_config
            if batch_pg_config is not None
            else _write_eval_pg_config(
                base_pg_config_path=pg_config_path,
                output_dir=case_dir,
                user_id=case_user_id,
            )
        )
        result: dict[str, Any]
        ingest_report: dict[str, Any] | None = None
        try:
            ingest_report = ingest_turn_to_postgres(
                source_db,
                lab_database,
                pg_config_path=eval_pg_config,
                turn_id=turn_id,
                settle_ms=0,
            )
            result = run_memory_react_agent(
                source_database=source_db,
                turn_id=turn_id,
                pg_config_path=eval_pg_config,
                config_path=config_path,
                memory_limit=memory_limit,
                topic_limit=topic_limit,
                max_iterations=max_iterations,
                dry_run=dry_run,
                log_dir=case_log_dir,
            )
        except Exception as exc:  # pragma: no cover - exercised by integration runs
            result = {
                "status": "failed",
                "stage": "exception",
                "turn_id": turn_id,
                "error": str(exc),
                "committed": False,
            }

        action_counts = _action_counts_from_result(result)
        changes = _changes_from_result(result)
        total_actions.update(action_counts)
        matches_expected = _case_matches_expected(case, action_counts)
        if matches_expected:
            expected_match_count += 1
        else:
            expected_mismatch_count += 1
        if result.get("status") == "complete":
            success_count += 1
        else:
            failure_count += 1

        react_log_path = ((result.get("log") or {}).get("run_log_path") if isinstance(result.get("log"), Mapping) else None)
        case_result = {
            "case_index": index,
            "case_id": case_id,
            "case_type": case.get("case_type"),
            "turn_id": turn_id,
            "source_db": str(source_db),
            "user_id": case_user_id,
            "eval_pg_config_path": str(eval_pg_config),
            "status": result.get("status"),
            "stage": result.get("stage"),
            "dry_run": dry_run,
            "run_id": result.get("run_id"),
            "react_log_path": react_log_path,
            "ingest": ingest_report,
            "action_counts": dict(action_counts),
            "changes": changes,
            "expected_action": sorted(_expected_actions(case)),
            "matches_expected": matches_expected,
            "committed": bool(result.get("committed")),
            "error": result.get("error"),
        }
        _write_json(case_dir / "result.json", {"case": case, "result": result, "summary": case_result})
        case_results.append(case_result)

    report: dict[str, Any] = {
        "status": "complete" if failure_count == 0 else "complete_with_errors",
        "tool": "eval_memory_replay",
        "evalset_id": evalset_id,
        "evalset_path": str(evalset_path),
        "run_id": run_id,
        "user_id": batch_user_id if mode == "single-batch" else None,
        "user_ids": [case["user_id"] for case in case_results],
        "mode": mode,
        "dry_run": dry_run,
        "case_count": len(selected_cases),
        "success_count": success_count,
        "failure_count": failure_count,
        "expected_match_count": expected_match_count,
        "expected_mismatch_count": expected_mismatch_count,
        "action_counts": dict(total_actions),
        "report_directory": str(output_dir),
        "eval_pg_config_path": str(batch_pg_config) if batch_pg_config is not None else None,
        "cases": case_results,
    }
    quality = _build_quality_report(report)
    report["quality"] = quality
    _write_json(output_dir / "eval-detail.json", report)
    (output_dir / "eval-summary.md").write_text(
        _markdown_summary(report), encoding="utf-8"
    )
    (output_dir / "generated-memories.md").write_text(
        _markdown_generated_memories(report), encoding="utf-8"
    )
    (output_dir / "quality-report.md").write_text(
        _markdown_quality_report(report, quality), encoding="utf-8"
    )
    bad_cases = [case for case in case_results if case["status"] != "complete" or not case["matches_expected"]]
    (output_dir / "bad-cases.md").write_text(
        _markdown_summary({**report, "cases": bad_cases}), encoding="utf-8"
    )
    return report
