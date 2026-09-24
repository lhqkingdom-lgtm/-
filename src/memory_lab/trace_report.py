"""Human-readable trace report for CP turn processing.

The report is intentionally read-only.  It stitches together:

* local CP SQLite messages;
* Memory Lab turn metadata;
* PostgreSQL processing state and evidence counts;
* ReAct JSON run logs;
* committed memory revisions, when running in commit mode.

This gives humans and later evaluators one auditable view of what entered the
pipeline, how it was split, which read tools the model requested, and what
memory changes were finally proposed or committed.
"""

from __future__ import annotations

import html
import json
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .pg_store import (
    DEFAULT_PG_CONFIG,
    DEFAULT_USER_ID,
    _json_from_psql_array,
    _json_literal,
    _run_psql,
    _sql_literal,
    load_pg_config,
)
from .process_new_turns import ensure_turn_processing_state
from .store import open_store


TRACE_REPORT_SCHEMA_VERSION = "cp-memory-trace-report-v1"
TRACE_ANNOTATION_TEMPLATE_SCHEMA_VERSION = "cp-memory-trace-annotations-template-v1"
DEFAULT_TRACE_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "cp-memory-trace"
DEFAULT_REACT_LOG_DIR = Path(__file__).resolve().parents[2] / "reports" / "memory-react-runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _truncate(text: Any, limit: int = 600) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _compact_value(value: Any, *, max_chars: int = 260) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate(value, max_chars)
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        payload = str(value)
    return _truncate(payload, max_chars)


def _time_from_ms(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000).isoformat(sep=" ", timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return None


def _read_message_parts(
    connection: sqlite3.Connection,
    message_id: str,
    *,
    text_limit: int = 1200,
) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    tools: list[dict[str, Any]] = []
    rows = connection.execute(
        """
        SELECT id, data, time_created, time_updated
        FROM part
        WHERE message_id = ?
        ORDER BY time_created, id
        """,
        (message_id,),
    ).fetchall()
    for row in rows:
        data = _safe_json(row["data"])
        part_type = data.get("type")
        if part_type == "text":
            texts.append(_truncate(data.get("text") or "", text_limit))
        elif part_type == "tool":
            state = data.get("state") if isinstance(data.get("state"), Mapping) else {}
            tool_input = state.get("input") if isinstance(state, Mapping) else None
            tools.append(
                {
                    "tool": data.get("tool"),
                    "status": state.get("status") if isinstance(state, Mapping) else None,
                    "input_summary": _compact_value(tool_input),
                }
            )
    return texts, tools


def _read_cp_turn_context(
    *,
    source_database: Path,
    lab_database: Path,
    turn_id: str,
    max_chars: int,
) -> dict[str, Any]:
    """Return bounded raw CP context for one turn."""

    try:
        with closing(open_store(lab_database)) as lab:
            turn = lab.execute(
                """
                SELECT turn_id, session_id, user_message_id, terminal_assistant_id,
                       user_timestamp, current_state, assistant_count, part_count
                FROM conversation_turn
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchone()
    except sqlite3.Error as error:
        return {"status": "unavailable", "reason": f"lab sqlite error: {error}"}
    if turn is None:
        return {"status": "not_found", "reason": "turn_id not found in local lab sqlite"}
    if not source_database.is_file():
        return {"status": "unavailable", "reason": f"source sqlite not found: {source_database}"}

    try:
        source = sqlite3.connect(source_database)
        source.row_factory = sqlite3.Row
        with closing(source) as connection:
            user_row = connection.execute(
                "SELECT id, data, time_created, time_updated FROM message WHERE id = ?",
                (str(turn["user_message_id"]),),
            ).fetchone()
            assistant_rows = connection.execute(
                """
                SELECT id, data, time_created, time_updated
                FROM message
                WHERE session_id = ?
                  AND json_extract(data, '$.role') = 'assistant'
                  AND json_extract(data, '$.parentID') = ?
                ORDER BY time_created, id
                """,
                (str(turn["session_id"]), str(turn["user_message_id"])),
            ).fetchall()
            user_texts: list[str] = []
            if user_row is not None:
                user_texts, _ = _read_message_parts(
                    connection,
                    str(user_row["id"]),
                    text_limit=max_chars,
                )
            assistant_texts: list[str] = []
            assistant_tools: list[dict[str, Any]] = []
            for assistant in assistant_rows:
                texts, tools = _read_message_parts(
                    connection,
                    str(assistant["id"]),
                    text_limit=max_chars,
                )
                assistant_texts.extend(texts)
                assistant_tools.extend(tools)
    except sqlite3.Error as error:
        return {"status": "unavailable", "reason": f"source sqlite error: {error}"}

    final_text = assistant_texts[-1] if assistant_texts else ""
    return {
        "status": "complete",
        "session_id": str(turn["session_id"]),
        "user_message_id": str(turn["user_message_id"]),
        "terminal_assistant_id": turn["terminal_assistant_id"],
        "state": str(turn["current_state"]),
        "occurred_at": _time_from_ms(turn["user_timestamp"]),
        "assistant_count": int(turn["assistant_count"] or 0),
        "part_count": int(turn["part_count"] or 0),
        "user_text": _truncate("\n".join(user_texts), max_chars),
        "assistant_final_text": _truncate(final_text, max_chars),
        "cp_tool_calls": [
            {
                "tool": item.get("tool"),
                "status": item.get("status"),
                "input_summary": item.get("input_summary"),
            }
            for item in assistant_tools[:40]
        ],
        "cp_tool_call_count": len(assistant_tools),
    }


def _processing_rows(
    *,
    pg_config_path: Path,
    limit: int,
) -> list[dict[str, Any]]:
    ensure_turn_processing_state(pg_config_path=pg_config_path)
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH latest_processing AS (
  SELECT DISTINCT ON (s.turn_id)
    s.*
  FROM turn_processing_state s
  ORDER BY s.turn_id, COALESCE(s.finished_at, s.started_at, s.created_at) DESC, s.state_id DESC
),
evidence_counts AS (
  SELECT
    turn_id,
    count(*) FILTER (WHERE layer = 'core') AS core_count,
    count(*) FILTER (WHERE layer = 'candidate') AS candidate_count,
    count(*) FILTER (WHERE layer = 'pointer') AS pointer_count,
    count(*) AS total_count
  FROM evidence_items
  WHERE user_id = {_sql_literal(user_id)}
  GROUP BY turn_id
),
revision_rows AS (
  SELECT
    r.source_turn_id AS turn_id,
    jsonb_agg(jsonb_build_object(
      'memory_id', m.memory_id,
      'stable_key', m.stable_key,
      'memory_type', m.memory_type,
      'status', m.status,
      'summary', m.summary,
      'topic_id', mt.topic_id,
      'topic_title', mt.title,
      'revision_id', r.revision_id,
      'revision_number', r.revision_number,
      'change_type', r.change_type,
      'reason', r.reason,
      'created_at', r.created_at
    ) ORDER BY r.created_at, r.revision_number) AS revisions
  FROM memory_revisions r
  JOIN memories m ON m.user_id = r.user_id AND m.memory_id = r.memory_id
  JOIN memory_topics mt ON mt.user_id = m.user_id AND mt.topic_id = m.topic_id
  WHERE r.user_id = {_sql_literal(user_id)}
    AND r.source_turn_id IS NOT NULL
  GROUP BY r.source_turn_id
)
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'turn_id', p.turn_id,
  'session_id', p.session_id,
  'input_hash', p.input_hash,
  'dry_run', p.dry_run,
  'processing_status', p.status,
  'agent_run_id', p.agent_run_id,
  'last_error', p.last_error,
  'created_at', p.created_at,
  'started_at', p.started_at,
  'finished_at', p.finished_at,
  'processing_metadata', p.metadata,
  'turn', CASE WHEN t.turn_id IS NULL THEN NULL ELSE jsonb_build_object(
    'state', t.state,
    'rule_phase', t.rule_phase,
    'final_phase', t.final_phase,
    'project_key', t.project_key,
    'occurred_at', t.occurred_at,
    'metadata', t.metadata
  ) END,
  'evidence_counts', jsonb_build_object(
    'core', COALESCE(e.core_count, 0),
    'candidate', COALESCE(e.candidate_count, 0),
    'pointer', COALESCE(e.pointer_count, 0),
    'total', COALESCE(e.total_count, 0)
  ),
  'committed_revisions', COALESCE(r.revisions, '[]'::jsonb)
) ORDER BY COALESCE(p.finished_at, p.started_at, p.created_at) DESC), '[]'::jsonb)::text
FROM (
  SELECT *
  FROM latest_processing
  ORDER BY COALESCE(finished_at, started_at, created_at) DESC, state_id DESC
  LIMIT {int(limit)}
) p
LEFT JOIN turns t
  ON t.turn_id = p.turn_id
 AND t.user_id = {_sql_literal(user_id)}
LEFT JOIN evidence_counts e
  ON e.turn_id = p.turn_id
LEFT JOIN revision_rows r
  ON r.turn_id = p.turn_id;
"""
    return _json_from_psql_array(config, sql)


def _read_react_log(agent_run_id: str | None, log_dir: Path | None) -> dict[str, Any] | None:
    if not agent_run_id:
        return None
    candidates: list[Path] = []
    if log_dir is not None:
        candidates.append(log_dir / f"{agent_run_id}.json")
    candidates.append(DEFAULT_REACT_LOG_DIR / f"{agent_run_id}.json")
    for path in candidates:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"status": "invalid_json", "path": str(path)}
            if isinstance(value, Mapping):
                return {**dict(value), "_path": str(path)}
    return {"status": "not_found", "agent_run_id": agent_run_id}


def _requested_tools_from_iteration(iteration: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = iteration.get("raw") if isinstance(iteration.get("raw"), Mapping) else {}
    requests = raw.get("tool_requests") if isinstance(raw, Mapping) else None
    if not isinstance(requests, list):
        requests = iteration.get("auto_tool_requests")
    if not isinstance(requests, list):
        return []
    result: list[dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, Mapping):
            continue
        result.append(
            {
                "tool": item.get("tool"),
                "memory_id": item.get("memory_id"),
                "topic_id": item.get("topic_id"),
                "raw_pointer_id": item.get("raw_pointer_id"),
                "query": item.get("query"),
                "reason": item.get("reason"),
            }
        )
    return result


def _tool_request_parameters(request: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        return {}
    keys = (
        "memory_id",
        "topic_id",
        "raw_pointer_id",
        "query",
        "limit",
        "scope",
        "include_inactive",
    )
    return {key: request.get(key) for key in keys if request.get(key) not in (None, "")}


def _summarize_tool_result(result: Mapping[str, Any] | None) -> str:
    if not isinstance(result, Mapping):
        return ""
    status = result.get("status")
    if result.get("memory") and isinstance(result.get("memory"), Mapping):
        memory = result["memory"]
        topic = memory.get("topic") if isinstance(memory.get("topic"), Mapping) else {}
        return _compact_value(
            {
                "status": status,
                "memory_id": memory.get("memory_id"),
                "stable_key": memory.get("stable_key"),
                "memory_type": memory.get("memory_type"),
                "memory_status": memory.get("status"),
                "topic_title": topic.get("title"),
                "summary": memory.get("summary"),
                "content_excerpt": _truncate(memory.get("content"), 220),
            },
            max_chars=520,
        )
    if result.get("topic") and isinstance(result.get("topic"), Mapping):
        topic = result["topic"]
        return _compact_value(
            {
                "status": status,
                "topic_id": topic.get("topic_id"),
                "topic_key": topic.get("topic_key"),
                "topic_type": topic.get("topic_type"),
                "topic_status": topic.get("status"),
                "title": topic.get("title"),
                "summary": topic.get("summary"),
            },
            max_chars=520,
        )
    for list_key in ("memories", "conflicts", "revisions", "evidence", "items", "rows"):
        value = result.get(list_key)
        if isinstance(value, list):
            samples: list[Any] = []
            for item in value[:3]:
                if isinstance(item, Mapping):
                    samples.append(
                        {
                            key: item.get(key)
                            for key in (
                                "memory_id",
                                "topic_id",
                                "conflict_id",
                                "revision_id",
                                "title",
                                "summary",
                                "status",
                                "change_type",
                                "reason",
                            )
                            if item.get(key) not in (None, "")
                        }
                    )
                else:
                    samples.append(item)
            return _compact_value(
                {
                    "status": status,
                    list_key + "_count": len(value),
                    "sample": samples,
                },
                max_chars=520,
            )
    return _compact_value(
        {key: value for key, value in result.items() if key not in {"content", "raw_text"}},
        max_chars=520,
    )


def _summarize_observation(item: Mapping[str, Any]) -> dict[str, Any]:
    request = item.get("request") if isinstance(item.get("request"), Mapping) else {}
    result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
    return {
        "status": item.get("status"),
        "tool": request.get("tool"),
        "request": {
            "tool": request.get("tool"),
            "parameters": _tool_request_parameters(request),
            "reason": request.get("reason"),
        },
        "result_status": result.get("status"),
        "result_summary": _summarize_tool_result(result),
    }


def _summarize_react_log(log: Mapping[str, Any] | None) -> dict[str, Any]:
    if not log:
        return {"status": "missing"}
    iterations = log.get("iterations") if isinstance(log.get("iterations"), list) else []
    observations = log.get("observations") if isinstance(log.get("observations"), list) else []
    result = log.get("result") if isinstance(log.get("result"), Mapping) else {}
    plan = result.get("plan") if isinstance(result.get("plan"), Mapping) else {}
    changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []
    observed_memories: dict[str, Mapping[str, Any]] = {}
    for item in observations:
        if not isinstance(item, Mapping):
            continue
        result_obj = item.get("result") if isinstance(item.get("result"), Mapping) else {}
        memory = result_obj.get("memory") if isinstance(result_obj.get("memory"), Mapping) else None
        if memory and memory.get("memory_id"):
            observed_memories[str(memory["memory_id"])] = memory
    normalized_changes = [
        {
            "action": item.get("action"),
            "reason": item.get("reason"),
            "target_memory_id": item.get("target_memory_id"),
            "target_topic_id": item.get("target_topic_id"),
            "memory_summary": ((item.get("memory") or {}).get("summary") if isinstance(item.get("memory"), Mapping) else None),
            "topic_title": ((item.get("topic") or {}).get("title") if isinstance(item.get("topic"), Mapping) else None),
            "evidence_refs": item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else [],
        }
        for item in changes
        if isinstance(item, Mapping)
    ]
    memory_diffs: list[dict[str, Any]] = []
    for change in normalized_changes:
        action = str(change.get("action") or "")
        if action == "noop":
            continue
        target_memory_id = change.get("target_memory_id")
        before = observed_memories.get(str(target_memory_id)) if target_memory_id else None
        after_summary = change.get("memory_summary") or change.get("topic_title")
        diff_kind = "memory" if change.get("memory_summary") else ("topic" if change.get("topic_title") else "operation")
        memory_diffs.append(
            {
                "mode": log.get("mode"),
                "status": "planned" if log.get("mode") == "dry_run" else "committed_or_planned",
                "kind": diff_kind,
                "action": action,
                "target_memory_id": target_memory_id,
                "target_topic_id": change.get("target_topic_id"),
                "before_summary": before.get("summary") if before else None,
                "after_summary": after_summary,
                "before_status": before.get("status") if before else None,
                "reason": change.get("reason"),
                "before_found": bool(before),
            }
        )
    return {
        "status": log.get("status"),
        "mode": log.get("mode"),
        "run_id": log.get("run_id"),
        "path": log.get("_path"),
        "steps": log.get("steps") if isinstance(log.get("steps"), list) else [],
        "iterations": [
            {
                "iteration": item.get("iteration"),
                "decision": item.get("decision"),
                "repaired": bool(item.get("repaired")),
                "requested_tools": _requested_tools_from_iteration(item),
                "reason": _truncate(
                    ((item.get("raw") or {}).get("reason") if isinstance(item.get("raw"), Mapping) else ""),
                    500,
                ),
            }
            for item in iterations
            if isinstance(item, Mapping)
        ],
        "observations": [
            _summarize_observation(item)
            for item in observations
            if isinstance(item, Mapping)
        ],
        "plan_reason": plan.get("reason"),
        "changes": normalized_changes,
        "memory_diffs": memory_diffs,
    }


def _turn_actions(turn: Mapping[str, Any]) -> list[str]:
    react = turn.get("react") if isinstance(turn.get("react"), Mapping) else {}
    changes = react.get("changes") if isinstance(react.get("changes"), list) else []
    return [
        str(change.get("action"))
        for change in changes
        if isinstance(change, Mapping) and change.get("action")
    ]


def _turn_requested_tools(turn: Mapping[str, Any]) -> list[str]:
    react = turn.get("react") if isinstance(turn.get("react"), Mapping) else {}
    iterations = react.get("iterations") if isinstance(react.get("iterations"), list) else []
    tools: list[str] = []
    for item in iterations:
        if not isinstance(item, Mapping):
            continue
        requests = item.get("requested_tools") if isinstance(item.get("requested_tools"), list) else []
        for request in requests:
            if isinstance(request, Mapping) and request.get("tool"):
                tools.append(str(request["tool"]))
    observations = react.get("observations") if isinstance(react.get("observations"), list) else []
    for item in observations:
        if isinstance(item, Mapping) and item.get("tool"):
            tools.append(str(item["tool"]))
    return tools


def _quality_flags_for_turn(turn: Mapping[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    cp = turn.get("cp_context") if isinstance(turn.get("cp_context"), Mapping) else {}
    react = turn.get("react") if isinstance(turn.get("react"), Mapping) else {}
    changes = react.get("changes") if isinstance(react.get("changes"), list) else []
    actions = _turn_actions(turn)
    requested_tools = set(_turn_requested_tools(turn))
    user_text = str(cp.get("user_text") or "")
    final_reason = str(react.get("plan_reason") or " ".join(
        str(change.get("reason") or "")
        for change in changes
        if isinstance(change, Mapping)
    ))

    if turn.get("processing_status") == "failed":
        flags.append(
            {
                "level": "error",
                "label": "❌ 处理失败",
                "reason": "turn_processing_state 标记为 failed，需要先看 last_error。",
            }
        )
    if react.get("status") in {None, "missing", "not_found", "invalid_json"}:
        flags.append(
            {
                "level": "warn",
                "label": "⚠️ ReAct 日志缺失",
                "reason": "没有找到可用于解释模型调用过程的 ReAct run log。",
            }
        )

    durable_keywords = (
        "以后",
        "长期",
        "记住",
        "偏好",
        "不要",
        "别",
        "必须",
        "确认",
        "已解决",
        "解决了",
        "编译通过",
        "烧录",
        "验证",
    )
    safe_noop_reasons = (
        "项目一次性",
        "当前项目",
        "已覆盖",
        "不应沉淀",
        "无新增",
        "非可复用",
        "可从",
    )
    if actions and set(actions) == {"noop"}:
        if any(token in user_text for token in durable_keywords) and not any(
            token in final_reason for token in safe_noop_reasons
        ):
            flags.append(
                {
                    "level": "warn",
                    "label": "⚠️ 可能漏沉淀",
                    "reason": "用户输入含长期/确认/验证/必须等信号，但最终 noop，建议人工复核。",
                }
            )

    write_actions = {
        "create_memory",
        "merge_memory",
        "merge_existing_memories",
        "update_memory",
        "retire_memory",
        "move_memory_topic",
        "merge_topics",
        "mark_conflict",
        "update_topic",
        "record_conflict_decision",
        "record_memory_relation",
    }
    existing_memory_write_actions = {
        "merge_memory",
        "merge_existing_memories",
        "update_memory",
        "retire_memory",
        "move_memory_topic",
        "mark_conflict",
        "record_conflict_decision",
        "record_memory_relation",
    }
    for change in changes:
        if not isinstance(change, Mapping):
            continue
        action = str(change.get("action") or "")
        if action in write_actions and not change.get("evidence_refs"):
            flags.append(
                {
                    "level": "warn",
                    "label": "⚠️ 证据不足",
                    "reason": f"{action} 没有 evidence_refs，沉淀依据不足。",
                }
            )
        if action in existing_memory_write_actions and not (
            {"read_memory_detail", "read_topic_detail", "list_topic_memories"} & requested_tools
        ):
            flags.append(
                {
                    "level": "warn",
                    "label": "⚠️ 工具不足",
                    "reason": f"{action} 修改/合并已有记忆，但未看到读取旧记忆详情的工具调用。",
                }
            )
        summary = str(change.get("memory_summary") or "")
        if action == "create_memory" and any(
            token in summary
            for token in ("25kHz", "35℃", "36", "50℃", "70℃", "2048", "具体阈值")
        ):
            flags.append(
                {
                    "level": "warn",
                    "label": "⚠️ 可能误沉淀项目参数",
                    "reason": "新增记忆摘要疑似包含具体项目参数，需确认是否应放入需求文档而不是长期记忆。",
                }
            )

    if not flags:
        if actions and set(actions) == {"noop"}:
            flags.append(
                {
                    "level": "ok",
                    "label": "✅ 合理跳过",
                    "reason": "最终动作为 noop，且未命中漏沉淀/证据不足等复核规则。",
                }
            )
        elif any(action in write_actions for action in actions):
            flags.append(
                {
                    "level": "ok",
                    "label": "✅ 合理沉淀",
                    "reason": "存在 create/merge/update 等记忆动作，且未命中明显风险规则。",
                }
            )
        else:
            flags.append(
                {
                    "level": "warn",
                    "label": "⚠️ 无最终动作",
                    "reason": "没有检测到最终 memory action，建议确认 ReAct 是否完整结束。",
                }
            )
    return flags


def _report_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for turn in turns:
        status = str(turn.get("processing_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        for action in _turn_actions(turn):
            action_counts[action] = action_counts.get(action, 0) + 1
        for flag in turn.get("quality_flags") or []:
            if isinstance(flag, Mapping):
                label = str(flag.get("label") or "unknown")
                quality_counts[label] = quality_counts.get(label, 0) + 1
    return {
        "turn_count": len(turns),
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "suspicious_turn_count": sum(
            1
            for turn in turns
            if any(
                isinstance(flag, Mapping) and flag.get("level") in {"warn", "error"}
                for flag in turn.get("quality_flags") or []
            )
        ),
    }


def build_cp_memory_trace_report(
    *,
    source_database: Path,
    lab_database: Path,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    log_dir: Path | None = None,
    limit: int = 20,
    max_chars: int = 900,
) -> dict[str, Any]:
    rows = _processing_rows(pg_config_path=pg_config_path, limit=limit)
    turns: list[dict[str, Any]] = []
    for row in rows:
        turn_id = str(row.get("turn_id") or "")
        react_log = _read_react_log(
            row.get("agent_run_id") if isinstance(row.get("agent_run_id"), str) else None,
            log_dir,
        )
        turn = {
                **row,
                "cp_context": _read_cp_turn_context(
                    source_database=source_database,
                    lab_database=lab_database,
                    turn_id=turn_id,
                    max_chars=max_chars,
                ),
                "react": _summarize_react_log(react_log),
        }
        turn["quality_flags"] = _quality_flags_for_turn(turn)
        turns.append(turn)
    summary = _report_summary(turns)
    return {
        "schema_version": TRACE_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "generated_at": _utc_now_iso(),
        "source_database": str(source_database),
        "lab_database": str(lab_database),
        "pg_config": str(pg_config_path),
        "log_dir": str(log_dir) if log_dir else str(DEFAULT_REACT_LOG_DIR),
        "limit": int(limit),
        "turn_count": len(turns),
        "summary": summary,
        "turns": turns,
    }


def _md_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


def _md_inline(value: Any, *, limit: int = 120) -> str:
    text = _truncate(value, limit)
    text = " ".join(str(text).split())
    return _md_escape(text)


def _md_json(value: Any) -> str:
    try:
        return _md_escape(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return _md_escape(value)


def render_trace_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    lines = [
        "# CP 交互到长期记忆链路报告",
        "",
        f"- 生成时间：`{report.get('generated_at')}`",
        f"- CP 原始库：`{report.get('source_database')}`",
        f"- Lab SQLite：`{report.get('lab_database')}`",
        f"- ReAct 日志目录：`{report.get('log_dir')}`",
        f"- turn 数量：{report.get('turn_count')}",
        "",
        "## 总览",
        "",
        f"- 可疑 turn 数：{summary.get('suspicious_turn_count', 0)}",
        f"- 处理状态统计：{_md_json(summary.get('status_counts') or {})}",
        f"- 记忆动作统计：{_md_json(summary.get('action_counts') or {})}",
        f"- 质量标签统计：{_md_json(summary.get('quality_counts') or {})}",
        "",
    ]
    turns = report.get("turns") if isinstance(report.get("turns"), list) else []
    if turns:
        lines.extend(
            [
                "## Turn 总览表",
                "",
                "| # | turn | 用户输入摘要 | 证据池 | ReAct | 动作 | 质量标签 |",
                "|---:|---|---|---|---|---|---|",
            ]
        )
        for index, turn in enumerate(turns, 1):
            if not isinstance(turn, Mapping):
                continue
            cp = turn.get("cp_context") if isinstance(turn.get("cp_context"), Mapping) else {}
            react = turn.get("react") if isinstance(turn.get("react"), Mapping) else {}
            counts = turn.get("evidence_counts") if isinstance(turn.get("evidence_counts"), Mapping) else {}
            flags = turn.get("quality_flags") if isinstance(turn.get("quality_flags"), list) else []
            actions = ", ".join(_turn_actions(turn)) or "无"
            labels = ", ".join(
                str(flag.get("label"))
                for flag in flags
                if isinstance(flag, Mapping) and flag.get("label")
            ) or "无"
            lines.append(
                "| "
                + " | ".join(
                        [
                            str(index),
                            f"`{_md_inline(turn.get('turn_id'), limit=18)}`",
                            _md_inline(cp.get("user_text"), limit=90),
                            f"{counts.get('core', 0)}/{counts.get('candidate', 0)}/{counts.get('pointer', 0)}",
                            _md_inline(f"{react.get('status')}/{react.get('mode')}", limit=80),
                            _md_inline(actions, limit=120),
                            _md_inline(labels, limit=160),
                        ]
                    )
                    + " |"
                )
        lines.append("")
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, Mapping):
            continue
        cp = turn.get("cp_context") if isinstance(turn.get("cp_context"), Mapping) else {}
        react = turn.get("react") if isinstance(turn.get("react"), Mapping) else {}
        counts = turn.get("evidence_counts") if isinstance(turn.get("evidence_counts"), Mapping) else {}
        flags = turn.get("quality_flags") if isinstance(turn.get("quality_flags"), list) else []
        flag_summary = "；".join(
            f"{flag.get('label')}：{flag.get('reason')}"
            for flag in flags
            if isinstance(flag, Mapping)
        ) or "无"
        lines.extend(
            [
                f"## {index}. turn `{turn.get('turn_id')}`",
                "",
                f"- 处理状态：`{turn.get('processing_status')}`；模式：`{'dry_run' if turn.get('dry_run') else 'commit'}`",
                f"- session：`{turn.get('session_id')}`；agent_run_id：`{turn.get('agent_run_id') or '无'}`",
                f"- 证据池：core={counts.get('core', 0)}，candidate={counts.get('candidate', 0)}，pointer={counts.get('pointer', 0)}",
                f"- ReAct：status=`{react.get('status')}`，mode=`{react.get('mode')}`",
                f"- 质量标签：{_md_escape(flag_summary)}",
                "",
                "### CP 原始交互",
                "",
                f"- 用户输入：{_md_escape(_truncate(cp.get('user_text'), 500)) or '（无）'}",
                f"- CP 原始工具事件数：{cp.get('cp_tool_call_count', 0)}（CP 主流程读文件/改文件/编译等事件，不等于记忆 Agent 工具调用）",
                f"- 最终回答摘要：{_md_escape(_truncate(cp.get('assistant_final_text'), 700)) or '（无）'}",
                "",
                "### ReAct 调用流转",
                "",
            ]
        )
        iterations = react.get("iterations") if isinstance(react.get("iterations"), list) else []
        if iterations:
            lines.extend(["| 轮次 | 决策 | 修复过 JSON | 请求工具 | 原因摘要 |", "|---:|---|---|---|---|"])
            for item in iterations:
                if not isinstance(item, Mapping):
                    continue
                tools = item.get("requested_tools") if isinstance(item.get("requested_tools"), list) else []
                tool_names = ", ".join(
                    str(tool.get("tool"))
                    for tool in tools
                    if isinstance(tool, Mapping) and tool.get("tool")
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_escape(item.get("iteration")),
                            _md_escape(item.get("decision")),
                            "是" if item.get("repaired") else "否",
                            _md_escape(tool_names or "无"),
                            _md_escape(_truncate(item.get("reason"), 260)),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("无 ReAct 迭代日志。")
        observations = react.get("observations") if isinstance(react.get("observations"), list) else []
        if observations:
            lines.extend(["", "### 工具调用明细", ""])
            lines.extend(["| # | 工具 | 状态 | 参数 | 请求原因 | 返回摘要 |", "|---:|---|---|---|---|---|"])
            for obs_index, obs in enumerate(observations, 1):
                if not isinstance(obs, Mapping):
                    continue
                request = obs.get("request") if isinstance(obs.get("request"), Mapping) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(obs_index),
                            _md_escape(obs.get("tool")),
                            _md_escape(obs.get("result_status") or obs.get("status")),
                            _md_inline(_md_json(request.get("parameters") or {}), limit=180),
                            _md_inline(request.get("reason"), limit=220),
                            _md_inline(obs.get("result_summary"), limit=260),
                        ]
                    )
                    + " |"
                )
        lines.extend(["", "### 最终沉淀/计划", ""])
        changes = react.get("changes") if isinstance(react.get("changes"), list) else []
        if changes:
            for change in changes:
                if not isinstance(change, Mapping):
                    continue
                target = change.get("target_memory_id") or change.get("target_topic_id") or change.get("topic_title") or ""
                summary = change.get("memory_summary") or change.get("reason")
                lines.append(
                    f"- `{change.get('action')}` {('→ `' + str(target) + '`') if target else ''}："
                    f"{_md_escape(_truncate(summary, 360))}"
                )
        else:
            lines.append("- 无最终 changes。")
        memory_diffs = react.get("memory_diffs") if isinstance(react.get("memory_diffs"), list) else []
        if memory_diffs:
            lines.extend(["", "### Memory before / after", ""])
            for diff in memory_diffs:
                if not isinstance(diff, Mapping):
                    continue
                mode_label = "计划变更，未写入" if diff.get("mode") == "dry_run" else "可能已提交"
                target = diff.get("target_memory_id") or diff.get("target_topic_id") or "新建"
                lines.extend(
                    [
                        f"- `{diff.get('action')}` → `{target}`（{mode_label}）",
                        f"  - before：{_md_escape(_truncate(diff.get('before_summary') or '无；可能是新建或未读取旧对象', 280))}",
                        f"  - after：{_md_escape(_truncate(diff.get('after_summary') or '无 after 摘要', 280))}",
                    ]
                )
        committed = turn.get("committed_revisions") if isinstance(turn.get("committed_revisions"), list) else []
        if committed:
            lines.extend(["", "### 已提交到正式记忆的 revision", ""])
            for item in committed:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    f"- `{item.get('change_type')}` `{item.get('memory_id')}`："
                    f"{_md_escape(_truncate(item.get('summary'), 260))}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_trace_html(report: Mapping[str, Any]) -> str:
    markdown = render_trace_markdown(report)
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    turns = report.get("turns") if isinstance(report.get("turns"), list) else []

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value), quote=True)

    def compact_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)

    def badge_class(label: Any) -> str:
        text = str(label or "")
        if text.startswith("✅"):
            return "badge ok"
        if text.startswith("❌"):
            return "badge error"
        if text.startswith("⚠️"):
            return "badge warn"
        return "badge"

    stat_cards = [
        ("turn 数量", report.get("turn_count", 0), "本报告覆盖的最近 turn 数"),
        ("可疑 turn", summary.get("suspicious_turn_count", 0), "命中 warn/error 质量标签的 turn"),
        ("记忆动作", compact_json(summary.get("action_counts") or {}), "create / merge / noop 等动作统计"),
        ("质量标签", compact_json(summary.get("quality_counts") or {}), "自动初筛标签统计"),
    ]
    stat_html = "\n".join(
        f"""
        <article class="stat-card">
          <div class="stat-title">{esc(title)}</div>
          <div class="stat-value">{esc(value)}</div>
          <div class="stat-hint">{esc(hint)}</div>
        </article>
        """
        for title, value, hint in stat_cards
    )

    overview_rows: list[str] = []
    detail_cards: list[str] = []
    for index, raw_turn in enumerate(turns, 1):
        if not isinstance(raw_turn, Mapping):
            continue
        turn = raw_turn
        cp = turn.get("cp_context") if isinstance(turn.get("cp_context"), Mapping) else {}
        react = turn.get("react") if isinstance(turn.get("react"), Mapping) else {}
        counts = turn.get("evidence_counts") if isinstance(turn.get("evidence_counts"), Mapping) else {}
        flags = turn.get("quality_flags") if isinstance(turn.get("quality_flags"), list) else []
        actions = _turn_actions(turn)
        action_text = ", ".join(actions) or "无"
        label_html = " ".join(
            f'<span class="{badge_class(flag.get("label"))}">{esc(flag.get("label"))}</span>'
            for flag in flags
            if isinstance(flag, Mapping) and flag.get("label")
        ) or '<span class="badge muted">无</span>'
        turn_anchor = f"turn-{index}"
        overview_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><a href=\"#{turn_anchor}\">{esc(_truncate(turn.get('turn_id'), 18))}</a></td>"
            f"<td>{esc(_truncate(cp.get('user_text'), 100))}</td>"
            f"<td><span class=\"pill\">core {esc(counts.get('core', 0))}</span> "
            f"<span class=\"pill\">candidate {esc(counts.get('candidate', 0))}</span> "
            f"<span class=\"pill\">pointer {esc(counts.get('pointer', 0))}</span></td>"
            f"<td>{esc(action_text)}</td>"
            f"<td>{label_html}</td>"
            "</tr>"
        )

        iterations = react.get("iterations") if isinstance(react.get("iterations"), list) else []
        iteration_rows: list[str] = []
        for item in iterations:
            if not isinstance(item, Mapping):
                continue
            tools = item.get("requested_tools") if isinstance(item.get("requested_tools"), list) else []
            tool_names = ", ".join(
                str(tool.get("tool"))
                for tool in tools
                if isinstance(tool, Mapping) and tool.get("tool")
            ) or "无"
            iteration_rows.append(
                "<tr>"
                f"<td>{esc(item.get('iteration'))}</td>"
                f"<td>{esc(item.get('decision'))}</td>"
                f"<td>{'是' if item.get('repaired') else '否'}</td>"
                f"<td>{esc(tool_names)}</td>"
                f"<td>{esc(_truncate(item.get('reason'), 260))}</td>"
                "</tr>"
            )
        if iteration_rows:
            iterations_html = (
                '<table class="inner-table"><thead><tr>'
                "<th>轮次</th><th>决策</th><th>JSON 修复</th><th>请求工具</th><th>原因</th>"
                "</tr></thead><tbody>"
                + "\n".join(iteration_rows)
                + "</tbody></table>"
            )
        else:
            iterations_html = '<p class="muted-text">无 ReAct 迭代日志。</p>'

        observations = react.get("observations") if isinstance(react.get("observations"), list) else []
        observation_rows: list[str] = []
        for obs_index, obs in enumerate(observations, 1):
            if not isinstance(obs, Mapping):
                continue
            request = obs.get("request") if isinstance(obs.get("request"), Mapping) else {}
            observation_rows.append(
                "<tr>"
                f"<td>{obs_index}</td>"
                f"<td><code>{esc(obs.get('tool'))}</code></td>"
                f"<td>{esc(obs.get('result_status') or obs.get('status'))}</td>"
                f"<td>{esc(_compact_value(request.get('parameters') or {}, max_chars=180))}</td>"
                f"<td>{esc(_truncate(request.get('reason'), 240))}</td>"
                f"<td>{esc(_truncate(obs.get('result_summary'), 360))}</td>"
                "</tr>"
            )
        if observation_rows:
            observations_html = (
                '<table class="inner-table tool-table"><thead><tr>'
                "<th>#</th><th>工具</th><th>状态</th><th>参数</th><th>请求原因</th><th>返回摘要</th>"
                "</tr></thead><tbody>"
                + "\n".join(observation_rows)
                + "</tbody></table>"
            )
        else:
            observations_html = '<p class="muted-text">无记忆 Agent 工具调用明细。</p>'

        changes = react.get("changes") if isinstance(react.get("changes"), list) else []
        change_items: list[str] = []
        for change in changes:
            if not isinstance(change, Mapping):
                continue
            target = change.get("target_memory_id") or change.get("target_topic_id") or change.get("topic_title") or ""
            change_items.append(
                "<li>"
                f"<code>{esc(change.get('action'))}</code> "
                f"{('→ <code>' + esc(target) + '</code>') if target else ''}"
                f"<div>{esc(_truncate(change.get('memory_summary') or change.get('reason'), 360))}</div>"
                "</li>"
            )
        changes_html = "<ul>" + "\n".join(change_items) + "</ul>" if change_items else '<p class="muted-text">无最终 changes。</p>'

        memory_diffs = react.get("memory_diffs") if isinstance(react.get("memory_diffs"), list) else []
        diff_items: list[str] = []
        for diff in memory_diffs:
            if not isinstance(diff, Mapping):
                continue
            mode_label = "计划变更，未写入" if diff.get("mode") == "dry_run" else "可能已提交"
            target = diff.get("target_memory_id") or diff.get("target_topic_id") or "新建"
            diff_items.append(
                f"""
                <div class="diff-card">
                  <div><code>{esc(diff.get('action'))}</code> → <code>{esc(target)}</code> <span class="badge muted">{esc(mode_label)}</span></div>
                  <div class="diff-grid">
                    <div><div class="diff-title">before</div><div>{esc(_truncate(diff.get('before_summary') or '无；可能是新建或未读取旧对象', 320))}</div></div>
                    <div><div class="diff-title">after</div><div>{esc(_truncate(diff.get('after_summary') or '无 after 摘要', 320))}</div></div>
                  </div>
                </div>
                """
            )
        diffs_html = "\n".join(diff_items) if diff_items else '<p class="muted-text">无 memory before / after。</p>'

        flag_reason_items = [
            f"<li><span class=\"{badge_class(flag.get('label'))}\">{esc(flag.get('label'))}</span> {esc(flag.get('reason'))}</li>"
            for flag in flags
            if isinstance(flag, Mapping)
        ]
        flag_reasons_html = "<ul>" + "\n".join(flag_reason_items) + "</ul>" if flag_reason_items else '<p class="muted-text">无质量标签。</p>'
        focus_reason = react.get("plan_reason") or (
            changes[0].get("reason") if changes and isinstance(changes[0], Mapping) else ""
        )

        detail_cards.append(
            f"""
            <section id="{turn_anchor}" class="turn-card">
              <div class="turn-head">
                <div>
                  <div class="turn-index">Turn {index}</div>
                  <h2>{esc(_truncate(turn.get('turn_id'), 28))}</h2>
                </div>
                <div class="labels">{label_html}</div>
              </div>
              <div class="meta-grid">
                <div><b>处理状态</b><span>{esc(turn.get('processing_status'))}</span></div>
                <div><b>模式</b><span>{'dry-run' if turn.get('dry_run') else 'commit'}</span></div>
                <div><b>证据池</b><span>core={esc(counts.get('core', 0))} / candidate={esc(counts.get('candidate', 0))} / pointer={esc(counts.get('pointer', 0))}</span></div>
                <div><b>动作</b><span>{esc(action_text)}</span></div>
              </div>
              <section class="focus-box">
                <h3>本轮重点判断</h3>
                <div class="focus-grid">
                  <div><b>用户输入摘要</b><p>{esc(_truncate(cp.get('user_text'), 280)) or '（无）'}</p></div>
                  <div><b>模型判断理由</b><p>{esc(_truncate(focus_reason, 420)) or '（无）'}</p></div>
                  <div><b>最终动作</b><p>{esc(action_text)}</p></div>
                  <div><b>质量风险</b><p>{label_html}</p></div>
                </div>
              </section>
              <details>
                <summary>CP 原始交互摘要</summary>
                <p><b>用户：</b>{esc(_truncate(cp.get('user_text'), 700)) or '（无）'}</p>
                <p><b>最终回答：</b>{esc(_truncate(cp.get('assistant_final_text'), 900)) or '（无）'}</p>
                <p><b>CP 原始工具事件数：</b>{esc(cp.get('cp_tool_call_count', 0))} <span class="muted-text">（CP 主流程读文件/改文件/编译等事件，不等于记忆 Agent 工具调用）</span></p>
              </details>
              <details>
                <summary>质量标签依据</summary>
                {flag_reasons_html}
              </details>
              <details>
                <summary>ReAct 调用流转</summary>
                {iterations_html}
              </details>
              <details>
                <summary>记忆 Agent 工具调用明细</summary>
                {observations_html}
              </details>
              <details open>
                <summary>最终沉淀/计划</summary>
                {changes_html}
              </details>
              <details open>
                <summary>Memory before / after</summary>
                {diffs_html}
              </details>
            </section>
            """
        )

    overview_table = (
        '<table class="overview-table"><thead><tr>'
        "<th>#</th><th>turn</th><th>用户输入摘要</th><th>证据池</th><th>动作</th><th>质量标签</th>"
        "</tr></thead><tbody>"
        + "\n".join(overview_rows)
        + "</tbody></table>"
        if overview_rows
        else '<p class="muted-text">没有 turn。</p>'
    )
    raw_markdown = html.escape(markdown)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>CP 交互到长期记忆链路报告</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fb; --panel:#fff; --line:#e6e8ef; --text:#1f2937; --muted:#6b7280; --accent:#2563eb; --ok:#0f766e; --warn:#b45309; --error:#b91c1c; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: ui-sans-serif, system-ui, "Microsoft YaHei", sans-serif; margin: 0; background:var(--bg); color:var(--text); }}
    header {{ padding: 28px 36px 18px; background: linear-gradient(135deg, #111827, #1e3a8a); color: white; }}
    header h1 {{ margin: 0 0 10px; font-size: 24px; }}
    header p {{ margin: 4px 0; color: #dbeafe; }}
    main {{ padding: 24px 36px 48px; max-width: 1480px; margin: 0 auto; }}
    .dashboard {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 20px; }}
    .stat-card, .turn-card, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .stat-card {{ padding:16px; min-height:118px; }}
    .stat-title {{ color:var(--muted); font-size:13px; }}
    .stat-value {{ margin-top:8px; font-size:22px; font-weight:700; word-break:break-word; }}
    .stat-hint {{ margin-top:8px; color:var(--muted); font-size:12px; }}
    .panel {{ padding:18px; margin-bottom:20px; overflow:auto; }}
    .overview-table, .inner-table {{ width:100%; border-collapse: collapse; }}
    th, td {{ border-bottom:1px solid var(--line); padding:10px 9px; text-align:left; vertical-align:top; font-size:13px; }}
    th {{ color:#374151; background:#f8fafc; position:sticky; top:0; }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .pill {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; margin:2px; font-size:12px; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#f3f4f6; color:#374151; margin:2px; font-size:12px; white-space:nowrap; }}
    .badge.ok {{ background:#ccfbf1; color:var(--ok); }}
    .badge.warn {{ background:#fef3c7; color:var(--warn); }}
    .badge.error {{ background:#fee2e2; color:var(--error); }}
    .badge.muted {{ background:#f3f4f6; color:var(--muted); }}
    .turn-card {{ padding:20px; margin:18px 0; }}
    .turn-head {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
    .turn-head h2 {{ margin:2px 0 12px; font-size:18px; }}
    .turn-index {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    .meta-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:12px 0 16px; }}
    .meta-grid div {{ border:1px solid var(--line); background:#f9fafb; border-radius:10px; padding:10px; }}
    .meta-grid b {{ display:block; font-size:12px; color:var(--muted); margin-bottom:5px; }}
    .focus-box {{ border:1px solid #bfdbfe; background:#eff6ff; border-radius:14px; padding:14px; margin:12px 0 16px; }}
    .focus-box h3 {{ margin:0 0 10px; font-size:16px; color:#1e3a8a; }}
    .focus-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
    .focus-grid > div {{ background:rgba(255,255,255,.72); border:1px solid #dbeafe; border-radius:10px; padding:10px; }}
    .focus-grid b {{ display:block; color:#1e40af; font-size:12px; margin-bottom:6px; }}
    .focus-grid p {{ margin:0; }}
    details {{ border-top:1px solid var(--line); padding:12px 0; }}
    summary {{ cursor:pointer; font-weight:700; }}
    p {{ line-height:1.65; }}
    code {{ background:#f3f4f6; border-radius:5px; padding:1px 5px; }}
    .diff-card {{ border:1px solid var(--line); border-radius:12px; padding:12px; margin:10px 0; background:#fbfdff; }}
    .diff-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; }}
    .diff-grid > div {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:white; line-height:1.55; }}
    .diff-title {{ font-size:12px; color:var(--muted); font-weight:700; margin-bottom:6px; }}
    .muted-text {{ color:var(--muted); }}
    .tool-table td:nth-child(4), .tool-table td:nth-child(5), .tool-table td:nth-child(6) {{ min-width:220px; }}
    pre {{ white-space: pre-wrap; background:#0b1020; color:#d1d5db; border-radius:14px; padding:18px; overflow:auto; line-height:1.55; }}
    @media (max-width: 980px) {{ .dashboard, .meta-grid, .focus-grid, .diff-grid {{ grid-template-columns:1fr; }} main, header {{ padding-left:18px; padding-right:18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>CP 交互到长期记忆链路报告</h1>
    <p>生成时间：<code>{esc(report.get('generated_at'))}</code></p>
    <p>CP 原始库：<code>{esc(report.get('source_database'))}</code></p>
    <p>ReAct 日志目录：<code>{esc(report.get('log_dir'))}</code></p>
  </header>
  <main>
    <section class="dashboard trace-dashboard">
      {stat_html}
    </section>
    <section class="panel">
      <h2>Turn 总览</h2>
      {overview_table}
    </section>
    <section>
      {"".join(detail_cards)}
    </section>
    <section class="panel">
      <details>
        <summary>原始 Markdown 报告</summary>
        <pre>{raw_markdown}</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""


def build_annotation_template(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build a human-fillable evaluation template from a trace report.

    The template is intentionally separate from the report payload: the report is
    generated evidence, while this file is for human review labels.  Keeping them
    separate makes it safe to regenerate reports without overwriting future
    filled-in annotation files.
    """

    turns = report.get("turns") if isinstance(report.get("turns"), list) else []
    template_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, Mapping):
            continue
        cp = turn.get("cp_context") if isinstance(turn.get("cp_context"), Mapping) else {}
        flags = turn.get("quality_flags") if isinstance(turn.get("quality_flags"), list) else []
        template_turns.append(
            {
                "index": index,
                "turn_id": turn.get("turn_id"),
                "session_id": turn.get("session_id"),
                "processing_status": turn.get("processing_status"),
                "dry_run": bool(turn.get("dry_run")),
                "user_summary": _truncate(cp.get("user_text"), 240),
                "assistant_summary": _truncate(cp.get("assistant_final_text"), 240),
                "actual_actions": _turn_actions(turn),
                "quality_labels": [
                    str(flag.get("label"))
                    for flag in flags
                    if isinstance(flag, Mapping) and flag.get("label")
                ],
                "expected_action": "",
                "human_label": "",
                "reviewer": "",
                "needs_followup": False,
                "note": "",
            }
        )
    return {
        "schema_version": TRACE_ANNOTATION_TEMPLATE_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "source_report_schema_version": report.get("schema_version"),
        "source_report_generated_at": report.get("generated_at"),
        "source_database": report.get("source_database"),
        "lab_database": report.get("lab_database"),
        "instructions": {
            "purpose": "人工复核每个 turn 的长期记忆沉淀是否合理；填写后可用于后续评测统计。",
            "expected_action_options": [
                "create_memory",
                "update_memory",
                "merge_memory",
                "update_topic",
                "merge_topic",
                "mark_conflict",
                "request_more_evidence",
                "noop",
                "other",
            ],
            "human_label_options": [
                "合理",
                "漏沉淀",
                "误沉淀",
                "重复记忆",
                "证据不足",
                "工具不足",
                "处理失败",
                "其他",
            ],
            "fields": {
                "expected_action": "人工认为这一轮最应该采取的动作；可填多个，用逗号分隔。",
                "human_label": "人工最终评价标签。",
                "needs_followup": "是否需要后续补证据、改提示词或修工具。",
                "note": "简短写明判断依据，最好引用 trace 报告里的 turn 内容。",
            },
        },
        "turn_count": len(template_turns),
        "turns": template_turns,
    }


def write_trace_report(
    *,
    report: Mapping[str, Any],
    output_dir: Path = DEFAULT_TRACE_REPORT_DIR,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    markdown = render_trace_markdown(report)
    html_payload = render_trace_html(report)
    json_payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    annotation_template = build_annotation_template(report)
    annotation_payload = json.dumps(
        annotation_template,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    latest_md = output_dir / "latest-trace.md"
    latest_html = output_dir / "latest-trace.html"
    latest_json = output_dir / "latest-trace.json"
    latest_annotation_template = output_dir / "latest-annotations.template.json"
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_md = history_dir / f"{stamp}.md"
    history_html = history_dir / f"{stamp}.html"
    history_json = history_dir / f"{stamp}.json"
    history_annotation_template = history_dir / f"{stamp}.annotations.template.json"

    for path, payload in (
        (history_md, markdown),
        (history_html, html_payload),
        (history_json, json_payload),
        (history_annotation_template, annotation_payload),
        (latest_md, markdown),
        (latest_html, html_payload),
        (latest_json, json_payload),
        (latest_annotation_template, annotation_payload),
    ):
        path.write_text(payload, encoding="utf-8")
    # Keep a stable copy operation shape for future replacement with atomic writes.
    try:
        shutil.copyfile(history_json, latest_json)
    except OSError:
        latest_json.write_text(json_payload, encoding="utf-8")
    return {
        "markdown": str(latest_md),
        "html": str(latest_html),
        "json": str(latest_json),
        "annotation_template": str(latest_annotation_template),
        "history_markdown": str(history_md),
        "history_html": str(history_html),
        "history_json": str(history_json),
        "history_annotation_template": str(history_annotation_template),
    }


def generate_cp_memory_trace_report(
    *,
    source_database: Path,
    lab_database: Path,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    log_dir: Path | None = None,
    output_dir: Path = DEFAULT_TRACE_REPORT_DIR,
    limit: int = 20,
    max_chars: int = 900,
) -> dict[str, Any]:
    report = build_cp_memory_trace_report(
        source_database=source_database,
        lab_database=lab_database,
        pg_config_path=pg_config_path,
        log_dir=log_dir,
        limit=limit,
        max_chars=max_chars,
    )
    files = write_trace_report(report=report, output_dir=output_dir)
    turn_summaries = []
    for item in report.get("turns", [])[:5]:
        if not isinstance(item, Mapping):
            continue
        react = item.get("react") if isinstance(item.get("react"), Mapping) else {}
        changes = react.get("changes") if isinstance(react.get("changes"), list) else []
        memory_diffs = react.get("memory_diffs") if isinstance(react.get("memory_diffs"), list) else []
        turn_summaries.append(
            {
                "turn_id": item.get("turn_id"),
                "processing_status": item.get("processing_status"),
                "dry_run": item.get("dry_run"),
                "agent_run_id": item.get("agent_run_id"),
                "evidence_counts": item.get("evidence_counts"),
                "quality_flags": item.get("quality_flags"),
                "memory_diff_count": len(memory_diffs),
                "actions": [
                    change.get("action")
                    for change in changes
                    if isinstance(change, Mapping) and change.get("action")
                ],
            }
        )
    return {
        "status": "complete",
        "schema_version": TRACE_REPORT_SCHEMA_VERSION,
        "turn_count": report["turn_count"],
        "summary": report.get("summary"),
        "files": files,
        "turn_summaries": turn_summaries,
    }


__all__ = [
    "TRACE_REPORT_SCHEMA_VERSION",
    "TRACE_ANNOTATION_TEMPLATE_SCHEMA_VERSION",
    "DEFAULT_TRACE_REPORT_DIR",
    "build_annotation_template",
    "build_cp_memory_trace_report",
    "generate_cp_memory_trace_report",
    "render_trace_html",
    "render_trace_markdown",
    "write_trace_report",
]
