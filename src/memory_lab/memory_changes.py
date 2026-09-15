"""Validation boundary for multi-change memory plans.

The Writer/agent may propose several memory changes for one turn, but this
module treats that proposal as untrusted data.  It verifies shape, references,
targets, scope, and duplicate keys before any later commit tool can write.
"""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .pg_store import (
    DEFAULT_PG_CONFIG,
    DEFAULT_USER_ID,
    _json_from_psql_object,
    _json_literal,
    _run_psql,
    _sql_literal,
    load_pg_config,
)


MEMORY_CHANGES_SCHEMA_VERSION = "memory-changes-v1"
ALLOWED_ACTIONS = {
    "create_memory",
    "merge_memory",
    "update_memory",
    "mark_conflict",
    "noop",
    "request_more_evidence",
}
ALLOWED_REQUEST_TOOLS = {"expand_pointer", "read_memory_detail", "read_topic_detail"}
KEY_RE = re.compile(r"^[a-z0-9_]+(?:[._][a-z0-9_]+)*$")
MAX_CHANGES = 10
MAX_SUMMARY_CHARS = 500
MAX_CONTENT_CHARS = 8000


class MemoryChangesValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class MemoryChangesCommitError(ValueError):
    pass


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_key(value: Any) -> bool:
    return isinstance(value, str) and bool(KEY_RE.fullmatch(value))


def _valid_confidence(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= float(value) <= 1
    )


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


def _digest_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _reference_snapshot(config: Mapping[str, Any], user_id: str, turn_id: str) -> dict[str, Any]:
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT jsonb_build_object(
  'turn_exists', EXISTS(
    SELECT 1 FROM turns
    WHERE turn_id = {_sql_literal(turn_id)} AND user_id = {_sql_literal(user_id)}
  ),
  'evidence_ids', COALESCE((
    SELECT jsonb_agg(evidence_id)
    FROM evidence_items
    WHERE turn_id = {_sql_literal(turn_id)} AND user_id = {_sql_literal(user_id)}
  ), '[]'::jsonb),
  'pointer_ids', COALESCE((
    SELECT jsonb_agg(evidence_id)
    FROM evidence_items
    WHERE turn_id = {_sql_literal(turn_id)}
      AND user_id = {_sql_literal(user_id)}
      AND layer = 'pointer'
  ), '[]'::jsonb),
  'memory_ids', COALESCE((
    SELECT jsonb_agg(memory_id)
    FROM memories
    WHERE user_id = {_sql_literal(user_id)} AND status = 'active'
  ), '[]'::jsonb),
  'topic_ids', COALESCE((
    SELECT jsonb_agg(topic_id)
    FROM memory_topics
    WHERE user_id = {_sql_literal(user_id)} AND status = 'active'
  ), '[]'::jsonb),
  'stable_keys', COALESCE((
    SELECT jsonb_agg(stable_key)
    FROM memories
    WHERE user_id = {_sql_literal(user_id)}
  ), '[]'::jsonb)
)::text;
"""
    return _json_from_psql_object(config, sql)


def _check_topic(topic: Any, errors: list[str], prefix: str) -> None:
    if not _is_object(topic):
        errors.append(f"{prefix}.topic must be an object")
        return
    allowed = {"topic_key", "title", "summary", "topic_type", "scope"}
    extra = set(topic) - allowed
    if extra:
        errors.append(f"{prefix}.topic has unsupported fields: {sorted(extra)}")
    for field in allowed:
        if field not in topic:
            errors.append(f"{prefix}.topic.{field} is required")
    if not _valid_key(topic.get("topic_key")):
        errors.append(f"{prefix}.topic.topic_key must use lowercase key format")
    for field in ("title", "summary", "topic_type"):
        if not _non_empty_string(topic.get(field)):
            errors.append(f"{prefix}.topic.{field} must be a non-empty string")
    if isinstance(topic.get("summary"), str) and len(topic["summary"]) > MAX_SUMMARY_CHARS:
        errors.append(f"{prefix}.topic.summary exceeds {MAX_SUMMARY_CHARS} characters")
    if not _valid_scope(topic.get("scope")):
        errors.append(f"{prefix}.topic.scope is invalid")


def _check_memory(memory: Any, errors: list[str], prefix: str, *, create: bool) -> None:
    if not _is_object(memory):
        errors.append(f"{prefix}.memory must be an object")
        return
    required = {"summary", "content", "confidence"}
    allowed = {"summary", "content", "confidence"}
    if create:
        required |= {"stable_key", "memory_type", "scope"}
        allowed |= {"stable_key", "memory_type", "scope"}
    extra = set(memory) - allowed
    if extra:
        errors.append(f"{prefix}.memory has unsupported fields: {sorted(extra)}")
    for field in required:
        if field not in memory:
            errors.append(f"{prefix}.memory.{field} is required")
    if create and not _valid_key(memory.get("stable_key")):
        errors.append(f"{prefix}.memory.stable_key must use lowercase key format")
    if create and not _non_empty_string(memory.get("memory_type")):
        errors.append(f"{prefix}.memory.memory_type must be a non-empty string")
    if create and not _valid_scope(memory.get("scope")):
        errors.append(f"{prefix}.memory.scope is invalid")
    for field, limit in (("summary", MAX_SUMMARY_CHARS), ("content", MAX_CONTENT_CHARS)):
        value = memory.get(field)
        if not _non_empty_string(value):
            errors.append(f"{prefix}.memory.{field} must be a non-empty string")
        elif len(value) > limit:
            errors.append(f"{prefix}.memory.{field} exceeds {limit} characters")
    if not _valid_confidence(memory.get("confidence")):
        errors.append(f"{prefix}.memory.confidence must be between 0 and 1")


def _check_evidence_refs(
    refs: Any,
    errors: list[str],
    prefix: str,
    *,
    evidence_ids: set[str],
    pointer_ids: set[str],
    required: bool,
) -> None:
    if not isinstance(refs, list):
        errors.append(f"{prefix}.evidence_refs must be an array")
        return
    if required and not refs:
        errors.append(f"{prefix}.evidence_refs must not be empty")
    for ref_index, ref in enumerate(refs[:20]):
        ref_prefix = f"{prefix}.evidence_refs[{ref_index}]"
        if not _is_object(ref):
            errors.append(f"{ref_prefix} must be an object")
            continue
        allowed = {"evidence_id", "raw_pointer_id", "reason"}
        extra = set(ref) - allowed
        if extra:
            errors.append(f"{ref_prefix} has unsupported fields: {sorted(extra)}")
        evidence_id = ref.get("evidence_id")
        raw_pointer_id = ref.get("raw_pointer_id")
        if bool(evidence_id) == bool(raw_pointer_id):
            errors.append(f"{ref_prefix} must contain exactly one of evidence_id/raw_pointer_id")
        if evidence_id and evidence_id not in evidence_ids:
            errors.append(f"{ref_prefix}.evidence_id does not belong to this turn")
        if raw_pointer_id and raw_pointer_id not in pointer_ids:
            errors.append(f"{ref_prefix}.raw_pointer_id does not belong to this turn")
        if "reason" in ref and not _non_empty_string(ref.get("reason")):
            errors.append(f"{ref_prefix}.reason must be a non-empty string")


def validate_memory_changes(
    proposal: Mapping[str, Any],
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
) -> dict[str, Any]:
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    errors: list[str] = []
    top_required = {"schema_version", "turn_id", "reason", "changes"}
    extra_top = set(proposal) - top_required
    missing_top = top_required - set(proposal)
    if extra_top:
        errors.append(f"unexpected top-level fields: {sorted(extra_top)}")
    if missing_top:
        errors.append(f"missing top-level fields: {sorted(missing_top)}")
    if proposal.get("schema_version") != MEMORY_CHANGES_SCHEMA_VERSION:
        errors.append("schema_version must be memory-changes-v1")
    if not _non_empty_string(proposal.get("turn_id")):
        errors.append("turn_id must be a non-empty string")
    if not _non_empty_string(proposal.get("reason")):
        errors.append("reason must be a non-empty string")
    changes = proposal.get("changes")
    if not isinstance(changes, list):
        errors.append("changes must be an array")
        changes = []
    elif not changes:
        errors.append("changes must not be empty")
    elif len(changes) > MAX_CHANGES:
        errors.append(f"changes exceeds max {MAX_CHANGES}")

    snapshot = _reference_snapshot(config, user_id, str(proposal.get("turn_id") or ""))
    if not snapshot.get("turn_exists"):
        errors.append("turn_id does not exist for current user")
    evidence_ids = {str(item) for item in snapshot.get("evidence_ids") or []}
    pointer_ids = {str(item) for item in snapshot.get("pointer_ids") or []}
    memory_ids = {str(item) for item in snapshot.get("memory_ids") or []}
    topic_ids = {str(item) for item in snapshot.get("topic_ids") or []}
    existing_stable_keys = {str(item) for item in snapshot.get("stable_keys") or []}
    batch_stable_keys: set[str] = set()
    batch_targets: dict[str, str] = {}

    for index, change in enumerate(changes[:MAX_CHANGES]):
        prefix = f"changes[{index}]"
        if not _is_object(change):
            errors.append(f"{prefix} must be an object")
            continue
        action = change.get("action")
        if action not in ALLOWED_ACTIONS:
            errors.append(f"{prefix}.action is not allowed")
            continue
        if not _non_empty_string(change.get("reason")):
            errors.append(f"{prefix}.reason must be a non-empty string")

        if action == "create_memory":
            allowed = {"action", "reason", "topic", "topic_id", "memory", "evidence_refs"}
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            has_topic = change.get("topic") is not None
            has_topic_id = change.get("topic_id") is not None
            if has_topic == has_topic_id:
                errors.append(f"{prefix} must contain exactly one of topic/topic_id")
            if has_topic:
                _check_topic(change.get("topic"), errors, prefix)
            if has_topic_id and change.get("topic_id") not in topic_ids:
                errors.append(f"{prefix}.topic_id does not exist")
            _check_memory(change.get("memory"), errors, prefix, create=True)
            memory = change.get("memory")
            if _is_object(memory):
                stable_key = str(memory.get("stable_key") or "")
                if stable_key in existing_stable_keys:
                    errors.append(f"{prefix}.memory.stable_key already exists")
                if stable_key in batch_stable_keys:
                    errors.append(f"{prefix}.memory.stable_key is duplicated in this batch")
                if stable_key:
                    batch_stable_keys.add(stable_key)
            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action in {"merge_memory", "update_memory", "mark_conflict"}:
            allowed = {"action", "reason", "target_memory_id", "memory", "evidence_refs"}
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            target = change.get("target_memory_id")
            if not _non_empty_string(target):
                errors.append(f"{prefix}.target_memory_id must be a non-empty string")
            elif target not in memory_ids:
                errors.append(f"{prefix}.target_memory_id does not exist or is not active")
            elif target in batch_targets:
                errors.append(
                    f"{prefix}.target_memory_id already used by {batch_targets[target]} in this batch"
                )
            else:
                batch_targets[str(target)] = action
            _check_memory(change.get("memory"), errors, prefix, create=False)
            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "noop":
            allowed = {"action", "reason", "evidence_refs"}
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            _check_evidence_refs(
                change.get("evidence_refs", []),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=False,
            )

        elif action == "request_more_evidence":
            allowed = {"action", "reason", "requests", "evidence_refs"}
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            requests = change.get("requests")
            if not isinstance(requests, list) or not requests:
                errors.append(f"{prefix}.requests must contain at least one request")
            else:
                for req_index, request in enumerate(requests[:10]):
                    req_prefix = f"{prefix}.requests[{req_index}]"
                    if not _is_object(request):
                        errors.append(f"{req_prefix} must be an object")
                        continue
                    allowed_request = {
                        "tool",
                        "raw_pointer_id",
                        "memory_id",
                        "topic_id",
                        "reason",
                    }
                    extra = set(request) - allowed_request
                    if extra:
                        errors.append(f"{req_prefix} has unsupported fields: {sorted(extra)}")
                    tool = request.get("tool")
                    if tool not in ALLOWED_REQUEST_TOOLS:
                        errors.append(f"{req_prefix}.tool is not allowed")
                    if tool == "expand_pointer" and request.get("raw_pointer_id") not in pointer_ids:
                        errors.append(f"{req_prefix}.raw_pointer_id does not belong to this turn")
                    if tool == "read_memory_detail" and request.get("memory_id") not in memory_ids:
                        errors.append(f"{req_prefix}.memory_id does not exist or is not active")
                    if tool == "read_topic_detail" and request.get("topic_id") not in topic_ids:
                        errors.append(f"{req_prefix}.topic_id does not exist or is not active")
                    if not _non_empty_string(request.get("reason")):
                        errors.append(f"{req_prefix}.reason must be a non-empty string")
            _check_evidence_refs(
                change.get("evidence_refs", []),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=False,
            )

    if errors:
        raise MemoryChangesValidationError(errors)
    return dict(proposal)


def validate_memory_changes_file(
    plan_path: Path,
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
) -> dict[str, Any]:
    try:
        proposal = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MemoryChangesValidationError([f"changes file not found: {plan_path}"]) from error
    except json.JSONDecodeError as error:
        raise MemoryChangesValidationError([f"changes file is not valid JSON: {error}"]) from error
    if not isinstance(proposal, Mapping):
        raise MemoryChangesValidationError(["changes file must contain a JSON object"])
    normalized = validate_memory_changes(proposal, pg_config_path=pg_config_path)
    return {
        "status": "complete",
        "tool": "validate_memory_changes",
        "valid": True,
        "turn_id": normalized["turn_id"],
        "change_count": len(normalized["changes"]),
        "proposal": normalized,
    }


def _load_proposal_file(plan_path: Path) -> dict[str, Any]:
    try:
        proposal = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MemoryChangesValidationError([f"changes file not found: {plan_path}"]) from error
    except json.JSONDecodeError as error:
        raise MemoryChangesValidationError([f"changes file is not valid JSON: {error}"]) from error
    if not isinstance(proposal, Mapping):
        raise MemoryChangesValidationError(["changes file must contain a JSON object"])
    return dict(proposal)


def _single_change_proposal(proposal: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": proposal.get("schema_version"),
        "turn_id": proposal.get("turn_id"),
        "reason": proposal.get("reason"),
        "changes": [dict(change)],
    }


def _evidence_metadata(proposal: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_turn_id": proposal.get("turn_id"),
        "change_reason": change.get("reason"),
        "evidence_refs": change.get("evidence_refs") or [],
        "last_action": change.get("action"),
    }


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
  updated_at = now();
"""
    return topic_id, sql


def _insert_memory_sql(
    *,
    user_id: str,
    topic_id: str,
    memory: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
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
  {_json_literal(metadata)},
  now()
);
"""
    return memory_id, sql


def _update_memory_sql(
    *,
    user_id: str,
    memory_id: str,
    memory: Mapping[str, Any],
    metadata: Mapping[str, Any],
    status: str | None = None,
) -> str:
    status_sql = f"status = {_sql_literal(status)}," if status else ""
    return f"""
UPDATE memories
SET {status_sql}
    summary = {_sql_literal(str(memory["summary"]))},
    content = {_sql_literal(str(memory["content"]))},
    confidence = {float(memory["confidence"]):.3f},
    metadata = metadata || {_json_literal(metadata)},
    updated_at = now()
WHERE memory_id = {_sql_literal(memory_id)}
  AND user_id = {_sql_literal(user_id)}
  AND status = 'active';
"""


def _memory_already_processed_turn(
    *,
    config: Mapping[str, Any],
    user_id: str,
    memory_id: str,
    turn_id: str,
) -> bool:
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT jsonb_build_object(
  'processed', EXISTS(
    SELECT 1
    FROM memories
    WHERE user_id = {_sql_literal(user_id)}
      AND memory_id = {_sql_literal(memory_id)}
      AND metadata ->> 'source_turn_id' = {_sql_literal(turn_id)}
  )
)::text;
"""
    result = _json_from_psql_object(config, sql)
    return bool(result.get("processed"))


def _commit_validated_change(
    *,
    config: Mapping[str, Any],
    user_id: str,
    proposal: Mapping[str, Any],
    change: Mapping[str, Any],
) -> dict[str, Any]:
    action = str(change["action"])
    if action == "noop":
        return {
            "action": action,
            "committed": False,
            "reason": change.get("reason"),
        }
    if action == "request_more_evidence":
        return {
            "action": action,
            "committed": False,
            "reason": change.get("reason"),
            "requests": change.get("requests") or [],
        }

    metadata = _evidence_metadata(proposal, change)
    if action == "create_memory":
        topic_sql = ""
        if change.get("topic") is not None:
            topic_id, topic_sql = _create_topic_sql(user_id, change["topic"])
        else:
            topic_id = str(change["topic_id"])
        memory_id, memory_sql = _insert_memory_sql(
            user_id=user_id,
            topic_id=topic_id,
            memory=change["memory"],
            metadata=metadata,
        )
        _run_psql(config, "BEGIN;\n" + topic_sql + memory_sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "topic_id": topic_id,
            "memory_id": memory_id,
            "stable_key": change["memory"]["stable_key"],
        }

    target_memory_id = str(change["target_memory_id"])
    if _memory_already_processed_turn(
        config=config,
        user_id=user_id,
        memory_id=target_memory_id,
        turn_id=str(proposal.get("turn_id") or ""),
    ):
        raise MemoryChangesCommitError(
            "target memory already processed this source_turn_id; skipping duplicate write"
        )
    status = "conflict" if action == "mark_conflict" else None
    sql = _update_memory_sql(
        user_id=user_id,
        memory_id=target_memory_id,
        memory=change["memory"],
        metadata=metadata,
        status=status,
    )
    _run_psql(config, "BEGIN;\n" + sql + "\nCOMMIT;")
    return {
        "action": action,
        "committed": True,
        "memory_id": target_memory_id,
        "status": status or "active",
    }


def commit_memory_changes(
    proposal: Mapping[str, Any],
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
) -> dict[str, Any]:
    """Validate and commit a multi-change proposal with per-change isolation."""

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    top_errors: list[str] = []
    if proposal.get("schema_version") != MEMORY_CHANGES_SCHEMA_VERSION:
        top_errors.append("schema_version must be memory-changes-v1")
    if not _non_empty_string(proposal.get("turn_id")):
        top_errors.append("turn_id must be a non-empty string")
    if not _non_empty_string(proposal.get("reason")):
        top_errors.append("reason must be a non-empty string")
    changes = proposal.get("changes")
    if not isinstance(changes, list):
        top_errors.append("changes must be an array")
        changes = []
    elif not changes:
        top_errors.append("changes must not be empty")
    elif len(changes) > MAX_CHANGES:
        top_errors.append(f"changes exceeds max {MAX_CHANGES}")
    if top_errors:
        raise MemoryChangesValidationError(top_errors)

    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    noops: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for index, change in enumerate(changes):
        if not isinstance(change, Mapping):
            rejected.append(
                {"index": index, "action": None, "errors": [f"changes[{index}] must be an object"]}
            )
            continue
        single = _single_change_proposal(proposal, change)
        try:
            validate_memory_changes(single, pg_config_path=pg_config_path)
            result = _commit_validated_change(
                config=config,
                user_id=user_id,
                proposal=single,
                change=change,
            )
        except MemoryChangesValidationError as error:
            rejected.append(
                {
                    "index": index,
                    "action": change.get("action"),
                    "errors": error.errors,
                }
            )
            continue
        except Exception as error:
            rejected.append(
                {
                    "index": index,
                    "action": change.get("action"),
                    "errors": [str(error)],
                }
            )
            continue
        row = {"index": index, **result}
        if result["action"] == "noop":
            noops.append(row)
        elif result["action"] == "request_more_evidence":
            requests.append(row)
        else:
            applied.append(row)
    return {
        "status": "complete",
        "tool": "commit_memory_changes",
        "turn_id": proposal.get("turn_id"),
        "applied": applied,
        "rejected": rejected,
        "noop": noops,
        "requests": requests,
        "counts": {
            "applied": len(applied),
            "rejected": len(rejected),
            "noop": len(noops),
            "requests": len(requests),
        },
    }


def commit_memory_changes_file(
    plan_path: Path,
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
) -> dict[str, Any]:
    proposal = _load_proposal_file(plan_path)
    return commit_memory_changes(proposal, pg_config_path=pg_config_path)


__all__ = [
    "MEMORY_CHANGES_SCHEMA_VERSION",
    "MemoryChangesValidationError",
    "MemoryChangesCommitError",
    "commit_memory_changes",
    "commit_memory_changes_file",
    "validate_memory_changes",
    "validate_memory_changes_file",
]
