"""Commit validated lightweight Writer plans into PostgreSQL.

The LLM never calls this module directly.  It emits a JSON plan; local code
validates the plan again at the write boundary and then applies a small,
deterministic set of database mutations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .pg_store import (
    DEFAULT_PG_CONFIG,
    _json_from_psql_object,
    _json_literal,
    _run_psql,
    _sql_literal,
    load_pg_config,
)
from .writer_dry_run import WriterPlanValidationError, validate_writer_plan


MAX_SUMMARY_CHARS = 500
MAX_CONTENT_CHARS = 8000


class WriterCommitError(ValueError):
    """A Writer plan is not safe to commit."""


def _digest_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WriterCommitError(f"{name} must be an object")
    return dict(value)


def _memory_index(config: Mapping[str, Any], user_id: str, limit: int = 500) -> dict[str, Any]:
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT jsonb_build_object(
  'topics', COALESCE((
    SELECT jsonb_agg(
      jsonb_build_object(
        'topic_id', topic_id,
        'topic_key', topic_key,
        'title', title,
        'summary', summary,
        'topic_type', topic_type,
        'scope', scope,
        'status', status,
        'updated_at', updated_at
      )
      ORDER BY updated_at DESC, topic_key
    )
    FROM memory_topics
    WHERE user_id = {_sql_literal(user_id)}
  ), '[]'::jsonb),
  'memories', COALESCE((
    SELECT jsonb_agg(
      jsonb_build_object(
        'memory_id', memory_id,
        'topic_id', topic_id,
        'stable_key', stable_key,
        'memory_type', memory_type,
        'summary', summary,
        'scope', scope,
        'status', status,
        'confidence', confidence,
        'updated_at', updated_at
      )
      ORDER BY updated_at DESC, stable_key
    )
    FROM (
      SELECT *
      FROM memories
      WHERE user_id = {_sql_literal(user_id)}
      ORDER BY updated_at DESC, stable_key
      LIMIT {int(limit)}
    ) AS limited_memories
  ), '[]'::jsonb)
)::text;
"""
    return _json_from_psql_object(config, sql)


def _strict_commit_checks(plan: Mapping[str, Any], index: Mapping[str, Any]) -> None:
    errors: list[str] = []
    action = plan.get("action")
    memory = plan.get("memory")
    topic = plan.get("topic")
    memories = index.get("memories") if isinstance(index, Mapping) else []
    existing_stable_keys = {
        str(item.get("stable_key"))
        for item in memories
        if isinstance(item, Mapping) and item.get("stable_key")
    }

    def text_length(obj: Mapping[str, Any], field: str, limit: int) -> None:
        value = obj.get(field)
        if isinstance(value, str) and len(value) > limit:
            errors.append(f"{field} exceeds {limit} characters")

    if isinstance(memory, Mapping):
        extra_memory = set(memory) - {
            "stable_key",
            "memory_type",
            "summary",
            "content",
            "scope",
            "confidence",
        }
        if extra_memory:
            errors.append(f"memory has unsupported fields: {sorted(extra_memory)}")
        text_length(memory, "summary", MAX_SUMMARY_CHARS)
        text_length(memory, "content", MAX_CONTENT_CHARS)
    if isinstance(topic, Mapping):
        extra_topic = set(topic) - {"topic_key", "title", "summary", "topic_type", "scope"}
        if extra_topic:
            errors.append(f"topic has unsupported fields: {sorted(extra_topic)}")
        text_length(topic, "summary", MAX_SUMMARY_CHARS)

    if action == "create" and isinstance(memory, Mapping):
        stable_key = str(memory.get("stable_key") or "")
        if stable_key in existing_stable_keys:
            errors.append("create stable_key already exists; use merge/update instead")
    if action == "need_more_evidence":
        # Safe no-write action.  Commit returns not_committed.
        pass
    if errors:
        raise WriterCommitError("; ".join(errors))


def _validate_for_commit(plan: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = validate_writer_plan(plan, index)
    except WriterPlanValidationError as error:
        raise WriterCommitError("; ".join(error.errors)) from error
    _strict_commit_checks(normalized, index)
    return normalized


def _create_topic_sql(user_id: str, topic: Mapping[str, Any]) -> tuple[str, str]:
    topic_key = str(topic["topic_key"])
    topic_id = _digest_id("topic", user_id, topic_key)
    sql = f"""
INSERT INTO memory_topics(topic_id, user_id, topic_key, title, summary, topic_type, scope, status, updated_at)
VALUES (
  {_sql_literal(topic_id)},
  {_sql_literal(user_id)},
  {_sql_literal(topic_key)},
  {_sql_literal(str(topic["title"]))},
  {_sql_literal(str(topic["summary"]))},
  {_sql_literal(str(topic["topic_type"]))},
  {_json_literal(topic["scope"])},
  'active',
  now()
)
ON CONFLICT (user_id, topic_key) DO UPDATE SET
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  topic_type = EXCLUDED.topic_type,
  scope = EXCLUDED.scope,
  status = 'active',
  updated_at = now()
RETURNING topic_id;
"""
    return topic_id, sql


def _create_memory_sql(user_id: str, topic_id: str, memory: Mapping[str, Any]) -> tuple[str, str]:
    stable_key = str(memory["stable_key"])
    memory_id = _digest_id("mem", user_id, stable_key)
    sql = f"""
INSERT INTO memories(
  memory_id, user_id, topic_id, stable_key, memory_type, scope,
  summary, content, status, confidence, metadata, updated_at
)
VALUES (
  {_sql_literal(memory_id)},
  {_sql_literal(user_id)},
  {_sql_literal(topic_id)},
  {_sql_literal(stable_key)},
  {_sql_literal(str(memory["memory_type"]))},
  {_json_literal(memory["scope"])},
  {_sql_literal(str(memory["summary"]))},
  {_sql_literal(str(memory["content"]))},
  'active',
  {float(memory["confidence"]):.3f},
  '{{}}'::jsonb,
  now()
);
"""
    return memory_id, sql


def _update_memory_sql(user_id: str, memory_id: str, memory: Mapping[str, Any], action: str, reason: str) -> str:
    metadata = {"last_action": action, "last_reason": reason}
    return f"""
UPDATE memories
SET summary = {_sql_literal(str(memory["summary"]))},
    content = {_sql_literal(str(memory["content"]))},
    confidence = {float(memory["confidence"]):.3f},
    metadata = metadata || {_json_literal(metadata)},
    updated_at = now()
WHERE memory_id = {_sql_literal(memory_id)}
  AND user_id = {_sql_literal(user_id)};
"""


def _mark_conflict_sql(user_id: str, memory_id: str, memory: Mapping[str, Any], reason: str) -> str:
    note = {
        "conflict_summary": str(memory.get("summary") or ""),
        "conflict_content": str(memory.get("content") or ""),
        "reason": reason,
    }
    return f"""
UPDATE memories
SET status = 'conflict',
    metadata = metadata || {_json_literal({"conflict": note})},
    updated_at = now()
WHERE memory_id = {_sql_literal(memory_id)}
  AND user_id = {_sql_literal(user_id)};
"""


def commit_writer_plan(
    plan: Mapping[str, Any],
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
) -> dict[str, Any]:
    """Validate and commit one Writer plan.

    This is the local write boundary.  It intentionally supports no arbitrary
    SQL and no direct model tool invocation.
    """

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or "local-default-user")
    index = _memory_index(config, user_id)
    normalized = _validate_for_commit(plan, index)
    action = str(normalized["action"])
    reason = str(normalized["reason"])

    if action == "noop":
        return {
            "status": "complete",
            "committed": False,
            "action": "noop",
            "reason": reason,
        }
    if action == "need_more_evidence":
        return {
            "status": "complete",
            "committed": False,
            "action": "need_more_evidence",
            "reason": reason,
            "requests": normalized["requests"],
        }

    if action == "create":
        topic = _object(normalized.get("topic"), "topic")
        memory = _object(normalized.get("memory"), "memory")
        topic_id, topic_sql = _create_topic_sql(user_id, topic)
        memory_id, memory_sql = _create_memory_sql(user_id, topic_id, memory)
        sql = f"BEGIN;\n{topic_sql}\n{memory_sql}\nCOMMIT;"
        _run_psql(config, sql)
        return {
            "status": "complete",
            "committed": True,
            "action": "create",
            "topic_id": topic_id,
            "memory_id": memory_id,
            "stable_key": memory["stable_key"],
        }

    target_memory_id = str(normalized["target_memory_id"])
    memory = _object(normalized.get("memory"), "memory")
    if action in {"merge", "update"}:
        sql = "BEGIN;\n" + _update_memory_sql(user_id, target_memory_id, memory, action, reason) + "\nCOMMIT;"
        _run_psql(config, sql)
        return {
            "status": "complete",
            "committed": True,
            "action": action,
            "memory_id": target_memory_id,
        }
    if action == "conflict":
        sql = "BEGIN;\n" + _mark_conflict_sql(user_id, target_memory_id, memory, reason) + "\nCOMMIT;"
        _run_psql(config, sql)
        return {
            "status": "complete",
            "committed": True,
            "action": "conflict",
            "memory_id": target_memory_id,
        }
    raise WriterCommitError(f"unsupported action at commit boundary: {action}")


def commit_plan_file(
    plan_path: Path,
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
) -> dict[str, Any]:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise WriterCommitError(f"plan file not found: {plan_path}") from error
    except json.JSONDecodeError as error:
        raise WriterCommitError(f"plan file is not valid JSON: {error}") from error
    if not isinstance(plan, Mapping):
        raise WriterCommitError("plan file must contain a JSON object")
    return commit_writer_plan(plan, pg_config_path=pg_config_path)


def run_writer_commit(
    *,
    turn_id: str,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path | None = None,
    memory_limit: int = 80,
) -> dict[str, Any]:
    """Run the Writer once and commit the validated plan if it is writable.

    This is the first end-to-end write path:
    turn evidence + memory index -> LLM JSON plan -> deterministic validation
    -> deterministic commit.  The LLM still never receives a database write
    tool; it only produces data for this function to validate.
    """

    from .config import DEFAULT_CONFIG
    from .writer_dry_run import run_writer_dry_run

    dry_report = run_writer_dry_run(
        turn_id=turn_id,
        pg_config_path=pg_config_path,
        config_path=config_path or DEFAULT_CONFIG,
        memory_limit=memory_limit,
    )
    if dry_report.get("status") != "complete":
        return {
            "status": "failed",
            "turn_id": turn_id,
            "stage": "writer",
            "writer_report": dry_report,
            "committed": False,
        }
    plan = dry_report.get("plan")
    if not isinstance(plan, Mapping):
        return {
            "status": "failed",
            "turn_id": turn_id,
            "stage": "writer",
            "error": "writer did not return a plan object",
            "writer_report": dry_report,
            "committed": False,
        }
    try:
        commit_report = commit_writer_plan(plan, pg_config_path=pg_config_path)
    except WriterCommitError as error:
        return {
            "status": "failed",
            "turn_id": turn_id,
            "stage": "commit",
            "error": str(error),
            "writer_report": dry_report,
            "committed": False,
        }
    return {
        "status": "complete",
        "turn_id": turn_id,
        "writer_repaired": bool(dry_report.get("repaired")),
        "writer_plan": plan,
        "context_stats": dry_report.get("context_stats") or {},
        "commit": commit_report,
        "committed": bool(commit_report.get("committed")),
    }


__all__ = [
    "WriterCommitError",
    "commit_plan_file",
    "commit_writer_plan",
    "run_writer_commit",
]
