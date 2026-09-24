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
    "merge_existing_memories",
    "update_memory",
    "retire_memory",
    "mark_conflict",
    "update_topic",
    "move_memory_topic",
    "merge_topics",
    "record_conflict_decision",
    "record_memory_relation",
    "noop",
    "request_more_evidence",
}
ALLOWED_REQUEST_TOOLS = {"expand_pointer", "read_memory_detail", "read_topic_detail"}
ALLOWED_MEMORY_TYPES = {
    "user_preference",
    "compile_failure_experience",
    "runtime_failure_experience",
}
CONFLICT_RECORD_STATUSES = (
    "needs_review",
    "resolved_by_update",
    "resolved_by_merge",
    "discarded_as_stale",
    "carried_forward",
)
ACTIVE_CONFLICT_RECORD_STATUSES = ("needs_review", "carried_forward")
KEY_RE = re.compile(r"^[a-z0-9_]+(?:[._][a-z0-9_]+)*$")
CLIENT_TEMP_ID_RE = re.compile(r"^temp:[a-z0-9_]+(?:[._-][a-z0-9_]+)*$")
MAX_CHANGES = 10
MAX_SUMMARY_CHARS = 500
MAX_CONTENT_CHARS = 8000


class MemoryChangesValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class MemoryChangesCommitError(ValueError):
    pass


def _error_codes_for_messages(errors: list[str]) -> list[str]:
    codes: list[str] = []
    patterns = [
        ("topic.topic_key already exists", "TOPIC_KEY_EXISTS"),
        ("target_memory_id does not exist", "MEMORY_NOT_FOUND"),
        ("memory_id does not exist", "MEMORY_NOT_FOUND"),
        ("from_memory_id does not exist", "MEMORY_NOT_FOUND"),
        ("to_memory_id does not exist", "MEMORY_NOT_FOUND"),
        ("target_topic_id does not exist", "TOPIC_NOT_FOUND"),
        ("topic_id does not exist", "TOPIC_NOT_FOUND"),
        ("evidence_id does not belong to this turn", "EVIDENCE_NOT_IN_TURN"),
        ("raw_pointer_id does not belong to this turn", "POINTER_NOT_IN_TURN"),
        ("unsupported fields", "UNSUPPORTED_FIELD"),
        ("memory.memory_type must be one of", "UNSUPPORTED_MEMORY_TYPE"),
        ("is required", "MISSING_REQUIRED_FIELD"),
        ("must be a non-empty string", "MISSING_REQUIRED_FIELD"),
        ("stable_key already exists", "DUPLICATE_STABLE_KEY"),
        ("action is not allowed", "UNKNOWN_ACTION"),
    ]
    for error in errors:
        for needle, code in patterns:
            if needle in error and code not in codes:
                codes.append(code)
    return codes or ["VALIDATION_ERROR"]


def _suggestions_for_error_codes(codes: list[str]) -> list[str]:
    suggestions: list[str] = []
    mapping = {
        "MEMORY_NOT_FOUND": "Call read_memory_detail or search/list memory index again, then use an active memory_id.",
        "TOPIC_NOT_FOUND": "Call read_topic_detail or list_memory_topics again, then use an active topic_id.",
        "EVIDENCE_NOT_IN_TURN": "Use only evidence_id values from the current turn evidence pool, or expand the relevant pointer first.",
        "POINTER_NOT_IN_TURN": "Use only raw_pointer_id values from the current turn pointer layer.",
        "UNSUPPORTED_FIELD": "Remove unsupported fields and keep only the schema fields allowed for this action.",
        "UNSUPPORTED_MEMORY_TYPE": "Use one of the allowed memory_type values: user_preference, compile_failure_experience, runtime_failure_experience. Use noop when nothing should be stored.",
        "MISSING_REQUIRED_FIELD": "Add the missing required field using the action schema.",
        "DUPLICATE_STABLE_KEY": "Update or merge the existing memory instead of creating a new memory with the same stable_key.",
        "TOPIC_KEY_EXISTS": "Use topic_id to create a memory under the existing topic, or emit a separate update_topic action if the topic title/summary should change.",
        "UNKNOWN_ACTION": "Use one of the allowed memory-changes-v1 actions.",
        "VALIDATION_ERROR": "Fix the reported validation errors and resubmit the corrected change.",
    }
    for code in codes:
        suggestion = mapping.get(code)
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
    return suggestions


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_key(value: Any) -> bool:
    return isinstance(value, str) and bool(KEY_RE.fullmatch(value))


def _valid_client_temp_id(value: Any) -> bool:
    return isinstance(value, str) and bool(CLIENT_TEMP_ID_RE.fullmatch(value))


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
  'memory_topics', COALESCE((
    SELECT jsonb_object_agg(memory_id, topic_id)
    FROM memories
    WHERE user_id = {_sql_literal(user_id)} AND status = 'active'
  ), '{{}}'::jsonb),
  'conflict_memory_ids', COALESCE((
    SELECT jsonb_agg(memory_id)
    FROM memories
    WHERE user_id = {_sql_literal(user_id)} AND status = 'conflict'
  ), '[]'::jsonb),
  'topic_ids', COALESCE((
    SELECT jsonb_agg(topic_id)
    FROM memory_topics
    WHERE user_id = {_sql_literal(user_id)} AND status = 'active'
  ), '[]'::jsonb),
  'topic_keys', COALESCE((
    SELECT jsonb_agg(topic_key)
    FROM memory_topics
    WHERE user_id = {_sql_literal(user_id)}
  ), '[]'::jsonb),
  'stable_keys', COALESCE((
    SELECT jsonb_agg(stable_key)
    FROM memories
    WHERE user_id = {_sql_literal(user_id)}
  ), '[]'::jsonb),
  'conflict_records', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
      'conflict_id', conflict_id,
      'memory_id', memory_id,
      'topic_id', topic_id,
      'status', status
    ))
    FROM memory_conflict_records
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
    elif create and str(memory.get("memory_type")) not in ALLOWED_MEMORY_TYPES:
        errors.append(
            f"{prefix}.memory.memory_type must be one of {', '.join(sorted(ALLOWED_MEMORY_TYPES))}"
        )
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


def _check_memory_relation(relation: Any, errors: list[str], prefix: str) -> None:
    if not _is_object(relation):
        errors.append(f"{prefix}.relation must be an object")
        return
    allowed = {"from_memory_id", "to_memory_id", "relation_type", "reason", "metadata"}
    required = {"from_memory_id", "to_memory_id", "relation_type", "reason"}
    extra = set(relation) - allowed
    missing = required - set(relation)
    if extra:
        errors.append(f"{prefix}.relation has unsupported fields: {sorted(extra)}")
    if missing:
        errors.append(f"{prefix}.relation missing fields: {sorted(missing)}")
    if not _non_empty_string(relation.get("from_memory_id")):
        errors.append(f"{prefix}.relation.from_memory_id must be a non-empty string")
    if not _non_empty_string(relation.get("to_memory_id")):
        errors.append(f"{prefix}.relation.to_memory_id must be a non-empty string")
    if relation.get("from_memory_id") and relation.get("from_memory_id") == relation.get("to_memory_id"):
        errors.append(f"{prefix}.relation from/to memory must be different")
    if relation.get("relation_type") not in {"supersedes", "merges", "conflicts_with"}:
        errors.append(f"{prefix}.relation.relation_type must be supersedes, merges or conflicts_with")
    if not _non_empty_string(relation.get("reason")):
        errors.append(f"{prefix}.relation.reason must be a non-empty string")
    if "metadata" in relation and not _is_object(relation.get("metadata")):
        errors.append(f"{prefix}.relation.metadata must be an object")


def _check_topic_patch(patch: Any, errors: list[str], prefix: str) -> None:
    if not _is_object(patch):
        errors.append(f"{prefix}.topic must be an object")
        return
    allowed = {"title", "summary"}
    extra = set(patch) - allowed
    if extra:
        errors.append(f"{prefix}.topic has unsupported fields: {sorted(extra)}")
    if not any(field in patch for field in allowed):
        errors.append(f"{prefix}.topic must contain at least one of title/summary")
    if "title" in patch and not _non_empty_string(patch.get("title")):
        errors.append(f"{prefix}.topic.title must be a non-empty string")
    if "summary" in patch:
        if not _non_empty_string(patch.get("summary")):
            errors.append(f"{prefix}.topic.summary must be a non-empty string")
        elif len(str(patch["summary"])) > MAX_SUMMARY_CHARS:
            errors.append(f"{prefix}.topic.summary exceeds {MAX_SUMMARY_CHARS} characters")


def _check_conflict_decision(decision: Any, errors: list[str], prefix: str) -> None:
    if not _is_object(decision):
        errors.append(f"{prefix}.conflict must be an object")
        return
    required = {"conflict_type", "old_claim", "new_claim", "decision", "reason"}
    allowed = {*required, "status", "metadata"}
    extra = set(decision) - allowed
    if extra:
        errors.append(f"{prefix}.conflict has unsupported fields: {sorted(extra)}")
    for field in required:
        if not _non_empty_string(decision.get(field)):
            errors.append(f"{prefix}.conflict.{field} must be a non-empty string")
    if "status" in decision and str(decision.get("status")) not in set(CONFLICT_RECORD_STATUSES):
        errors.append(
            f"{prefix}.conflict.status must be one of {', '.join(CONFLICT_RECORD_STATUSES)}"
        )
    if "metadata" in decision and not _is_object(decision.get("metadata")):
        errors.append(f"{prefix}.conflict.metadata must be an object")


def _check_conflict_resolution(resolution: Any, errors: list[str], prefix: str) -> None:
    if not _is_object(resolution):
        errors.append(f"{prefix}.conflict_resolution must be an object")
        return
    allowed = {"conflict_ids", "status", "reason", "metadata"}
    required = {"conflict_ids", "status", "reason"}
    extra = set(resolution) - allowed
    missing = required - set(resolution)
    if extra:
        errors.append(f"{prefix}.conflict_resolution has unsupported fields: {sorted(extra)}")
    if missing:
        errors.append(f"{prefix}.conflict_resolution missing fields: {sorted(missing)}")
    conflict_ids = resolution.get("conflict_ids")
    if not isinstance(conflict_ids, list) or not conflict_ids:
        errors.append(f"{prefix}.conflict_resolution.conflict_ids must be a non-empty array")
    elif len(conflict_ids) > 20:
        errors.append(f"{prefix}.conflict_resolution.conflict_ids exceeds 20")
    else:
        seen: set[str] = set()
        for item_index, conflict_id in enumerate(conflict_ids):
            if not _non_empty_string(conflict_id):
                errors.append(f"{prefix}.conflict_resolution.conflict_ids[{item_index}] must be a non-empty string")
            elif str(conflict_id) in seen:
                errors.append(f"{prefix}.conflict_resolution.conflict_ids[{item_index}] is duplicated")
            else:
                seen.add(str(conflict_id))
    if resolution.get("status") != "resolved_by_update":
        errors.append(f"{prefix}.conflict_resolution.status must be resolved_by_update")
    if not _non_empty_string(resolution.get("reason")):
        errors.append(f"{prefix}.conflict_resolution.reason must be a non-empty string")
    if "metadata" in resolution and not _is_object(resolution.get("metadata")):
        errors.append(f"{prefix}.conflict_resolution.metadata must be an object")


def _check_conflict_disposition(
    disposition: Any,
    errors: list[str],
    prefix: str,
    *,
    allowed_statuses: set[str] | None = None,
) -> None:
    if not _is_object(disposition):
        errors.append(f"{prefix}.conflict_disposition must be an object")
        return
    allowed_statuses = allowed_statuses or {"discarded_as_stale", "carried_forward"}
    allowed = {"conflict_ids", "status", "reason", "metadata"}
    required = {"status", "reason"}
    extra = set(disposition) - allowed
    missing = required - set(disposition)
    if extra:
        errors.append(f"{prefix}.conflict_disposition has unsupported fields: {sorted(extra)}")
    if missing:
        errors.append(f"{prefix}.conflict_disposition missing fields: {sorted(missing)}")
    if disposition.get("status") not in allowed_statuses:
        errors.append(
            f"{prefix}.conflict_disposition.status must be one of {', '.join(sorted(allowed_statuses))}"
        )
    if not _non_empty_string(disposition.get("reason")):
        errors.append(f"{prefix}.conflict_disposition.reason must be a non-empty string")
    conflict_ids = disposition.get("conflict_ids")
    if conflict_ids is not None:
        if not isinstance(conflict_ids, list) or not conflict_ids:
            errors.append(f"{prefix}.conflict_disposition.conflict_ids must be a non-empty array when provided")
        elif len(conflict_ids) > 20:
            errors.append(f"{prefix}.conflict_disposition.conflict_ids exceeds 20")
        else:
            seen: set[str] = set()
            for item_index, conflict_id in enumerate(conflict_ids):
                if not _non_empty_string(conflict_id):
                    errors.append(f"{prefix}.conflict_disposition.conflict_ids[{item_index}] must be a non-empty string")
                elif str(conflict_id) in seen:
                    errors.append(f"{prefix}.conflict_disposition.conflict_ids[{item_index}] is duplicated")
                else:
                    seen.add(str(conflict_id))
    if "metadata" in disposition and not _is_object(disposition.get("metadata")):
        errors.append(f"{prefix}.conflict_disposition.metadata must be an object")


def validate_memory_changes(
    proposal: Mapping[str, Any],
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    extra_memory_ids: Mapping[str, str] | None = None,
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
    memory_topics = {
        str(key): str(value)
        for key, value in (snapshot.get("memory_topics") or {}).items()
        if _non_empty_string(key) and _non_empty_string(value)
    }
    conflict_memory_ids = {str(item) for item in snapshot.get("conflict_memory_ids") or []}
    extra_memory_ids = {
        str(key): str(value)
        for key, value in (extra_memory_ids or {}).items()
        if _non_empty_string(key) and _non_empty_string(value)
    }
    memory_ids.update(extra_memory_ids)
    memory_ids.update(extra_memory_ids.values())
    topic_ids = {str(item) for item in snapshot.get("topic_ids") or []}
    existing_topic_keys = {str(item) for item in snapshot.get("topic_keys") or []}
    existing_stable_keys = {str(item) for item in snapshot.get("stable_keys") or []}
    conflict_records: dict[str, Mapping[str, Any]] = {}
    for item in snapshot.get("conflict_records") or []:
        if isinstance(item, Mapping) and item.get("conflict_id"):
            conflict_records[str(item["conflict_id"])] = item
    batch_stable_keys: set[str] = set()
    batch_targets: dict[str, str] = {}
    batch_temp_memory_ids: dict[str, str] = {}

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
            allowed = {
                "action",
                "reason",
                "client_temp_id",
                "topic",
                "topic_id",
                "memory",
                "evidence_refs",
            }
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            client_temp_id = change.get("client_temp_id")
            if client_temp_id is not None:
                if not _valid_client_temp_id(client_temp_id):
                    errors.append(f"{prefix}.client_temp_id must use format temp:lowercase_id")
                elif client_temp_id in memory_ids or client_temp_id in batch_temp_memory_ids:
                    errors.append(f"{prefix}.client_temp_id is duplicated in this batch")
            has_topic = change.get("topic") is not None
            has_topic_id = change.get("topic_id") is not None
            if has_topic == has_topic_id:
                errors.append(f"{prefix} must contain exactly one of topic/topic_id")
            if has_topic:
                _check_topic(change.get("topic"), errors, prefix)
                if _is_object(change.get("topic")):
                    topic_key = str(change["topic"].get("topic_key") or "")
                    if topic_key in existing_topic_keys:
                        errors.append(
                            f"{prefix}.topic.topic_key already exists; use topic_id for existing topics or update_topic to change title/summary"
                        )
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
                    if _valid_client_temp_id(client_temp_id):
                        memory_id = _digest_id("mem", user_id, stable_key)
                        batch_temp_memory_ids[str(client_temp_id)] = memory_id
                        memory_ids.add(str(client_temp_id))
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
            if action == "update_memory":
                allowed = {*allowed, "conflict_resolution"}
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
            if "conflict_resolution" in change:
                _check_conflict_resolution(change.get("conflict_resolution"), errors, prefix)
                resolution = change.get("conflict_resolution")
                if _is_object(resolution) and isinstance(resolution.get("conflict_ids"), list):
                    for conflict_id in resolution.get("conflict_ids") or []:
                        conflict_id = str(conflict_id)
                        conflict_record = conflict_records.get(conflict_id)
                        if not conflict_record:
                            errors.append(f"{prefix}.conflict_resolution.conflict_ids contains unknown conflict_id: {conflict_id}")
                            continue
                        if str(conflict_record.get("memory_id") or "") != str(target or ""):
                            errors.append(
                                f"{prefix}.conflict_resolution.conflict_id {conflict_id} does not belong to target_memory_id"
                            )
                        if str(conflict_record.get("status") or "") not in set(ACTIVE_CONFLICT_RECORD_STATUSES):
                            errors.append(
                                f"{prefix}.conflict_resolution.conflict_id {conflict_id} is not an active conflict"
                            )
            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "retire_memory":
            allowed = {"action", "reason", "target_memory_id", "evidence_refs", "conflict_disposition"}
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

            active_conflicts_for_target = [
                conflict_id
                for conflict_id, conflict_record in conflict_records.items()
                if str(conflict_record.get("memory_id") or "") == str(target or "")
                and str(conflict_record.get("status") or "") in set(ACTIVE_CONFLICT_RECORD_STATUSES)
            ]
            if "conflict_disposition" in change:
                _check_conflict_disposition(change.get("conflict_disposition"), errors, prefix)
                disposition = change.get("conflict_disposition")
                if _is_object(disposition):
                    provided_conflict_ids = disposition.get("conflict_ids")
                    if isinstance(provided_conflict_ids, list):
                        for conflict_id in provided_conflict_ids:
                            conflict_id = str(conflict_id)
                            conflict_record = conflict_records.get(conflict_id)
                            if not conflict_record:
                                errors.append(f"{prefix}.conflict_disposition.conflict_ids contains unknown conflict_id: {conflict_id}")
                                continue
                            if str(conflict_record.get("memory_id") or "") != str(target or ""):
                                errors.append(
                                    f"{prefix}.conflict_disposition.conflict_id {conflict_id} does not belong to target_memory_id"
                                )
                            if str(conflict_record.get("status") or "") not in set(ACTIVE_CONFLICT_RECORD_STATUSES):
                                errors.append(
                                    f"{prefix}.conflict_disposition.conflict_id {conflict_id} is not an active conflict"
                                )
            elif active_conflicts_for_target:
                errors.append(
                    f"{prefix}.conflict_disposition is required because target memory has active conflicts"
                )

            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "merge_existing_memories":
            allowed = {
                "action",
                "reason",
                "target_memory_id",
                "source_memory_ids",
                "memory",
                "evidence_refs",
                "conflict_disposition",
            }
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

            source_memory_ids = change.get("source_memory_ids")
            normalized_sources: list[str] = []
            if not isinstance(source_memory_ids, list) or not source_memory_ids:
                errors.append(f"{prefix}.source_memory_ids must be a non-empty array")
            elif len(source_memory_ids) > 20:
                errors.append(f"{prefix}.source_memory_ids exceeds 20")
            else:
                seen_sources: set[str] = set()
                for source_index, source_id in enumerate(source_memory_ids):
                    if not _non_empty_string(source_id):
                        errors.append(f"{prefix}.source_memory_ids[{source_index}] must be a non-empty string")
                        continue
                    source_id = str(source_id)
                    if source_id in seen_sources:
                        errors.append(f"{prefix}.source_memory_ids[{source_index}] is duplicated")
                    elif source_id == str(target or ""):
                        errors.append(f"{prefix}.source_memory_ids[{source_index}] must not equal target_memory_id")
                    elif source_id not in memory_ids:
                        errors.append(f"{prefix}.source_memory_ids[{source_index}] does not exist or is not active")
                    elif source_id in batch_targets:
                        errors.append(
                            f"{prefix}.source_memory_ids[{source_index}] already used by {batch_targets[source_id]} in this batch"
                        )
                    else:
                        seen_sources.add(source_id)
                        normalized_sources.append(source_id)
                        batch_targets[source_id] = action

            _check_memory(change.get("memory"), errors, prefix, create=False)

            active_conflicts_for_sources = [
                conflict_id
                for conflict_id, conflict_record in conflict_records.items()
                if str(conflict_record.get("memory_id") or "") in set(normalized_sources)
                and str(conflict_record.get("status") or "") in set(ACTIVE_CONFLICT_RECORD_STATUSES)
            ]
            if "conflict_disposition" in change:
                _check_conflict_disposition(
                    change.get("conflict_disposition"),
                    errors,
                    prefix,
                    allowed_statuses={"resolved_by_merge", "discarded_as_stale", "carried_forward"},
                )
                disposition = change.get("conflict_disposition")
                if _is_object(disposition):
                    provided_conflict_ids = disposition.get("conflict_ids")
                    if isinstance(provided_conflict_ids, list):
                        for conflict_id in provided_conflict_ids:
                            conflict_id = str(conflict_id)
                            conflict_record = conflict_records.get(conflict_id)
                            if not conflict_record:
                                errors.append(f"{prefix}.conflict_disposition.conflict_ids contains unknown conflict_id: {conflict_id}")
                                continue
                            if str(conflict_record.get("memory_id") or "") not in set(normalized_sources):
                                errors.append(
                                    f"{prefix}.conflict_disposition.conflict_id {conflict_id} does not belong to source_memory_ids"
                                )
                            if str(conflict_record.get("status") or "") not in set(ACTIVE_CONFLICT_RECORD_STATUSES):
                                errors.append(
                                    f"{prefix}.conflict_disposition.conflict_id {conflict_id} is not an active conflict"
                                )
            elif active_conflicts_for_sources:
                errors.append(
                    f"{prefix}.conflict_disposition is required because source memories have active conflicts"
                )

            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "update_topic":
            allowed = {"action", "reason", "target_topic_id", "topic", "evidence_refs"}
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            target = change.get("target_topic_id")
            if not _non_empty_string(target):
                errors.append(f"{prefix}.target_topic_id must be a non-empty string")
            elif target not in topic_ids:
                errors.append(f"{prefix}.target_topic_id does not exist or is not active")
            _check_topic_patch(change.get("topic"), errors, prefix)
            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "move_memory_topic":
            allowed = {
                "action",
                "reason",
                "target_memory_id",
                "from_topic_id",
                "to_topic_id",
                "evidence_refs",
            }
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            target = change.get("target_memory_id")
            from_topic_id = change.get("from_topic_id")
            to_topic_id = change.get("to_topic_id")
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
                actual_topic_id = memory_topics.get(str(target))
                if actual_topic_id and _non_empty_string(from_topic_id) and str(from_topic_id) != actual_topic_id:
                    errors.append(
                        f"{prefix}.from_topic_id does not match current memory topic"
                    )
            if not _non_empty_string(from_topic_id):
                errors.append(f"{prefix}.from_topic_id must be a non-empty string")
            elif from_topic_id not in topic_ids:
                errors.append(f"{prefix}.from_topic_id does not exist or is not active")
            if not _non_empty_string(to_topic_id):
                errors.append(f"{prefix}.to_topic_id must be a non-empty string")
            elif to_topic_id not in topic_ids:
                errors.append(f"{prefix}.to_topic_id does not exist or is not active")
            if _non_empty_string(from_topic_id) and _non_empty_string(to_topic_id) and str(from_topic_id) == str(to_topic_id):
                errors.append(f"{prefix}.to_topic_id must be different from from_topic_id")
            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "merge_topics":
            allowed = {
                "action",
                "reason",
                "target_topic_id",
                "source_topic_ids",
                "evidence_refs",
            }
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            target_topic_id = change.get("target_topic_id")
            if not _non_empty_string(target_topic_id):
                errors.append(f"{prefix}.target_topic_id must be a non-empty string")
            elif target_topic_id not in topic_ids:
                errors.append(f"{prefix}.target_topic_id does not exist or is not active")
            source_topic_ids = change.get("source_topic_ids")
            if not isinstance(source_topic_ids, list) or not source_topic_ids:
                errors.append(f"{prefix}.source_topic_ids must be a non-empty array")
            elif len(source_topic_ids) > 20:
                errors.append(f"{prefix}.source_topic_ids exceeds 20")
            else:
                seen_topics: set[str] = set()
                for source_index, source_topic_id in enumerate(source_topic_ids):
                    if not _non_empty_string(source_topic_id):
                        errors.append(f"{prefix}.source_topic_ids[{source_index}] must be a non-empty string")
                        continue
                    source_topic_id = str(source_topic_id)
                    if source_topic_id in seen_topics:
                        errors.append(f"{prefix}.source_topic_ids[{source_index}] is duplicated")
                    elif source_topic_id == str(target_topic_id or ""):
                        errors.append(f"{prefix}.source_topic_ids[{source_index}] must not equal target_topic_id")
                    elif source_topic_id not in topic_ids:
                        errors.append(f"{prefix}.source_topic_ids[{source_index}] does not exist or is not active")
                    else:
                        seen_topics.add(source_topic_id)
            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "record_conflict_decision":
            allowed = {
                "action",
                "reason",
                "target_memory_id",
                "conflict",
                "evidence_refs",
            }
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            target = change.get("target_memory_id")
            if not _non_empty_string(target):
                errors.append(f"{prefix}.target_memory_id must be a non-empty string")
            elif target not in memory_ids and target not in conflict_memory_ids:
                errors.append(f"{prefix}.target_memory_id does not exist or is not active/conflict")
            _check_conflict_decision(change.get("conflict"), errors, prefix)
            _check_evidence_refs(
                change.get("evidence_refs"),
                errors,
                prefix,
                evidence_ids=evidence_ids,
                pointer_ids=pointer_ids,
                required=True,
            )

        elif action == "record_memory_relation":
            allowed = {"action", "reason", "relation", "evidence_refs"}
            extra = set(change) - allowed
            if extra:
                errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
            _check_memory_relation(change.get("relation"), errors, prefix)
            relation = change.get("relation")
            if _is_object(relation):
                from_memory_id = relation.get("from_memory_id")
                to_memory_id = relation.get("to_memory_id")
                if _non_empty_string(from_memory_id) and from_memory_id not in memory_ids:
                    errors.append(f"{prefix}.relation.from_memory_id does not exist or is not active")
                if _non_empty_string(to_memory_id) and to_memory_id not in memory_ids:
                    errors.append(f"{prefix}.relation.to_memory_id does not exist or is not active")
                resolved_from = batch_temp_memory_ids.get(str(from_memory_id), str(from_memory_id))
                resolved_to = batch_temp_memory_ids.get(str(to_memory_id), str(to_memory_id))
                if (
                    _non_empty_string(from_memory_id)
                    and _non_empty_string(to_memory_id)
                    and resolved_from == resolved_to
                ):
                    errors.append(f"{prefix}.relation resolved from/to memory must be different")
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


def _resolve_temp_memory_references(
    change: Mapping[str, Any],
    temp_memory_ids: Mapping[str, str],
) -> dict[str, Any]:
    resolved = json.loads(json.dumps(dict(change), ensure_ascii=False))
    relation = resolved.get("relation")
    if isinstance(relation, dict):
        for field in ("from_memory_id", "to_memory_id"):
            value = relation.get(field)
            if isinstance(value, str) and value in temp_memory_ids:
                relation[field] = temp_memory_ids[value]
    target_memory_id = resolved.get("target_memory_id")
    if isinstance(target_memory_id, str) and target_memory_id in temp_memory_ids:
        resolved["target_memory_id"] = temp_memory_ids[target_memory_id]
    return resolved


def _evidence_metadata(proposal: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    turn_id = proposal.get("turn_id")
    return {
        "source_turn_id": turn_id,
        "source_turn_ids": [turn_id] if turn_id else [],
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
);
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


def _insert_memory_revision_sql(
    *,
    user_id: str,
    memory_id: str,
    turn_id: str,
    change_type: str,
    reason: str,
    metadata: Mapping[str, Any],
    initial: bool = False,
) -> str:
    """Append a lightweight revision for the current state of one memory."""

    if initial:
        revision_number_sql = "1"
        supersede_sql = ""
    else:
        revision_number_sql = """
        COALESCE((
          SELECT max(r.revision_number)
          FROM memory_revisions r
          WHERE r.user_id = m.user_id
            AND r.memory_id = m.memory_id
        ), 0) + 1
        """
        supersede_sql = f"""
UPDATE memory_revisions
SET status = 'superseded'
WHERE user_id = {_sql_literal(user_id)}
  AND memory_id = {_sql_literal(memory_id)}
  AND status = 'current';
"""
    return f"""
{supersede_sql}
INSERT INTO memory_revisions(
  revision_id, user_id, memory_id, revision_number, summary, content,
  confidence, trust_level, status, change_type, reason, source_turn_id, metadata
)
SELECT
  'rev_' || substr(md5(
    m.user_id || '|' || m.memory_id || '|' || {_sql_literal(change_type)} || '|' ||
    COALESCE({_sql_literal(turn_id)}, '') || '|' || m.summary || '|' || m.content
  ), 1, 16),
  m.user_id,
  m.memory_id,
  {revision_number_sql},
  m.summary,
  m.content,
  m.confidence,
  m.trust_level,
  'current',
  {_sql_literal(change_type)},
  {_sql_literal(reason)},
  {_sql_literal(turn_id)}::text,
  {_json_literal({"change_metadata": metadata})}
FROM memories m
WHERE m.user_id = {_sql_literal(user_id)}
  AND m.memory_id = {_sql_literal(memory_id)}
ON CONFLICT (revision_id) DO NOTHING;
"""


def _update_memory_sql(
    *,
    user_id: str,
    memory_id: str,
    memory: Mapping[str, Any],
    metadata: Mapping[str, Any],
    status: str | None = None,
) -> str:
    status_sql = f"status = {_sql_literal(status)}," if status else ""
    metadata_json = _json_literal(metadata)
    return f"""
UPDATE memories
SET {status_sql}
    summary = {_sql_literal(str(memory["summary"]))},
    content = {_sql_literal(str(memory["content"]))},
    confidence = {float(memory["confidence"]):.3f},
    metadata =
      (metadata || {metadata_json})
      || jsonb_build_object(
        'source_turn_ids',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT value), '[]'::jsonb)
          FROM jsonb_array_elements_text(
            COALESCE(metadata -> 'source_turn_ids', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'source_turn_ids', '[]'::jsonb)
            || CASE
                 WHEN metadata ? 'source_turn_id' THEN jsonb_build_array(metadata ->> 'source_turn_id')
                 ELSE '[]'::jsonb
               END
          ) AS merged(value)
        ),
        'evidence_refs',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
          FROM jsonb_array_elements(
            COALESCE(metadata -> 'evidence_refs', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'evidence_refs', '[]'::jsonb)
          ) AS merged(item)
        )
      ),
    updated_at = now()
WHERE memory_id = {_sql_literal(memory_id)}
  AND user_id = {_sql_literal(user_id)}
  AND status = 'active';
"""


def _update_topic_sql(
    *,
    user_id: str,
    topic_id: str,
    topic: Mapping[str, Any],
) -> str:
    assignments: list[str] = []
    if "title" in topic:
        assignments.append(f"title = {_sql_literal(str(topic['title']))}")
    if "summary" in topic:
        assignments.append(f"summary = {_sql_literal(str(topic['summary']))}")
    assignments.append("updated_at = now()")
    return f"""
UPDATE memory_topics
SET {", ".join(assignments)}
WHERE topic_id = {_sql_literal(topic_id)}
  AND user_id = {_sql_literal(user_id)}
  AND status = 'active';
"""


def _move_memory_topic_sql(
    *,
    user_id: str,
    memory_id: str,
    from_topic_id: str,
    to_topic_id: str,
    metadata: Mapping[str, Any],
    reason: str,
) -> str:
    move_metadata = {
        **metadata,
        "move_reason": reason,
        "old_topic_id": from_topic_id,
        "new_topic_id": to_topic_id,
    }
    metadata_json = _json_literal(move_metadata)
    return f"""
UPDATE memories
SET topic_id = {_sql_literal(to_topic_id)},
    metadata =
      (metadata || {metadata_json})
      || jsonb_build_object(
        'source_turn_ids',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT value), '[]'::jsonb)
          FROM jsonb_array_elements_text(
            COALESCE(metadata -> 'source_turn_ids', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'source_turn_ids', '[]'::jsonb)
            || CASE
                 WHEN metadata ? 'source_turn_id' THEN jsonb_build_array(metadata ->> 'source_turn_id')
                 ELSE '[]'::jsonb
               END
          ) AS merged(value)
        ),
        'evidence_refs',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
          FROM jsonb_array_elements(
            COALESCE(metadata -> 'evidence_refs', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'evidence_refs', '[]'::jsonb)
          ) AS merged(item)
        )
      ),
    updated_at = now()
WHERE memory_id = {_sql_literal(memory_id)}
  AND user_id = {_sql_literal(user_id)}
  AND status = 'active'
  AND topic_id = {_sql_literal(from_topic_id)};
"""


def _merge_topics_sql(
    *,
    user_id: str,
    target_topic_id: str,
    source_topic_ids: list[str],
    metadata: Mapping[str, Any],
    reason: str,
) -> str:
    source_values = ", ".join(_sql_literal(item) for item in source_topic_ids)
    merge_metadata = {
        **metadata,
        "merge_topics_reason": reason,
        "target_topic_id": target_topic_id,
        "source_topic_ids": source_topic_ids,
    }
    metadata_json = _json_literal(merge_metadata)
    return f"""
UPDATE memories
SET topic_id = {_sql_literal(target_topic_id)},
    metadata =
      (metadata || {metadata_json})
      || jsonb_build_object(
        'source_turn_ids',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT value), '[]'::jsonb)
          FROM jsonb_array_elements_text(
            COALESCE(metadata -> 'source_turn_ids', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'source_turn_ids', '[]'::jsonb)
            || CASE
                 WHEN metadata ? 'source_turn_id' THEN jsonb_build_array(metadata ->> 'source_turn_id')
                 ELSE '[]'::jsonb
               END
          ) AS merged(value)
        ),
        'evidence_refs',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
          FROM jsonb_array_elements(
            COALESCE(metadata -> 'evidence_refs', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'evidence_refs', '[]'::jsonb)
          ) AS merged(item)
        )
      ),
    updated_at = now()
WHERE user_id = {_sql_literal(user_id)}
  AND status = 'active'
  AND topic_id IN ({source_values});

UPDATE memory_topics
SET status = 'merged',
    metadata = metadata || jsonb_build_object(
      'merged_into', {_sql_literal(target_topic_id)},
      'merge_reason', {_sql_literal(reason)},
      'source_turn_id', {_sql_literal(str(metadata.get("source_turn_id") or ""))},
      'source_topic_ids', {metadata_json} -> 'source_topic_ids'
    ),
    updated_at = now()
WHERE user_id = {_sql_literal(user_id)}
  AND status = 'active'
  AND topic_id IN ({source_values});
"""


def _read_topic_state(
    *,
    config: Mapping[str, Any],
    user_id: str,
    topic_id: str,
) -> dict[str, Any]:
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT COALESCE((
  SELECT jsonb_build_object(
    'found', true,
    'topic_id', topic_id,
    'title', title,
    'summary', summary
  )
  FROM memory_topics
  WHERE topic_id = {_sql_literal(topic_id)}
    AND user_id = {_sql_literal(user_id)}
    AND status = 'active'
), jsonb_build_object('found', false))::text;
"""
    return _json_from_psql_object(config, sql)


def _read_conflict_state(
    *,
    config: Mapping[str, Any],
    user_id: str,
    conflict_id: str,
) -> dict[str, Any]:
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT COALESCE((
  SELECT jsonb_build_object(
    'found', true,
      'conflict_id', conflict_id,
      'memory_id', memory_id,
      'topic_id', topic_id,
      'reason', reason,
      'status', status,
      'evidence_refs', evidence_refs
  )
  FROM memory_conflict_records
  WHERE conflict_id = {_sql_literal(conflict_id)}
    AND user_id = {_sql_literal(user_id)}
), jsonb_build_object('found', false))::text;
"""
    return _json_from_psql_object(config, sql)


def _resolve_conflicts_by_update_sql(
    *,
    user_id: str,
    memory_id: str,
    conflict_resolution: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    conflict_ids = [str(item) for item in conflict_resolution.get("conflict_ids") or []]
    values = ", ".join(_sql_literal(item) for item in conflict_ids)
    resolution_metadata = {
        **(conflict_resolution.get("metadata") if isinstance(conflict_resolution.get("metadata"), Mapping) else {}),
        **metadata,
        "resolution_status": "resolved_by_update",
        "resolution_reason": str(conflict_resolution.get("reason") or ""),
    }
    return f"""
UPDATE memory_conflict_records
SET status = 'resolved_by_update',
    metadata = metadata || {_json_literal(resolution_metadata)}
WHERE user_id = {_sql_literal(user_id)}
  AND memory_id = {_sql_literal(memory_id)}
  AND conflict_id IN ({values})
  AND status IN ('needs_review', 'carried_forward');
"""


def _retire_memory_sql(
    *,
    user_id: str,
    memory_id: str,
    metadata: Mapping[str, Any],
    reason: str,
) -> str:
    retirement_metadata = {
        **metadata,
        "retirement_reason": reason,
    }
    metadata_json = _json_literal(retirement_metadata)
    return f"""
UPDATE memories
SET status = 'retired',
    metadata =
      (metadata || {metadata_json})
      || jsonb_build_object(
        'source_turn_ids',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT value), '[]'::jsonb)
          FROM jsonb_array_elements_text(
            COALESCE(metadata -> 'source_turn_ids', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'source_turn_ids', '[]'::jsonb)
            || CASE
                 WHEN metadata ? 'source_turn_id' THEN jsonb_build_array(metadata ->> 'source_turn_id')
                 ELSE '[]'::jsonb
               END
          ) AS merged(value)
        ),
        'evidence_refs',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
          FROM jsonb_array_elements(
            COALESCE(metadata -> 'evidence_refs', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'evidence_refs', '[]'::jsonb)
          ) AS merged(item)
        )
      ),
    updated_at = now()
WHERE memory_id = {_sql_literal(memory_id)}
  AND user_id = {_sql_literal(user_id)}
  AND status = 'active';
"""


def _dispose_conflicts_for_retire_sql(
    *,
    user_id: str,
    memory_id: str,
    disposition: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    conflict_ids = disposition.get("conflict_ids")
    if isinstance(conflict_ids, list) and conflict_ids:
        conflict_filter = "AND conflict_id IN (" + ", ".join(_sql_literal(str(item)) for item in conflict_ids) + ")"
    else:
        conflict_filter = ""
    status = str(disposition["status"])
    disposition_metadata = {
        **(disposition.get("metadata") if isinstance(disposition.get("metadata"), Mapping) else {}),
        **metadata,
        "retire_disposition_status": status,
        "retire_disposition_reason": str(disposition.get("reason") or ""),
    }
    return f"""
UPDATE memory_conflict_records
SET status = {_sql_literal(status)},
    metadata = metadata || {_json_literal(disposition_metadata)}
WHERE user_id = {_sql_literal(user_id)}
  AND memory_id = {_sql_literal(memory_id)}
  AND status IN ('needs_review', 'carried_forward')
  {conflict_filter};
"""


def _mark_source_memory_merged_sql(
    *,
    user_id: str,
    memory_id: str,
    target_memory_id: str,
    metadata: Mapping[str, Any],
    reason: str,
) -> str:
    merge_metadata = {
        **metadata,
        "merge_target_memory_id": target_memory_id,
        "merge_reason": reason,
    }
    metadata_json = _json_literal(merge_metadata)
    return f"""
UPDATE memories
SET status = 'merged',
    metadata =
      (metadata || {metadata_json})
      || jsonb_build_object(
        'source_turn_ids',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT value), '[]'::jsonb)
          FROM jsonb_array_elements_text(
            COALESCE(metadata -> 'source_turn_ids', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'source_turn_ids', '[]'::jsonb)
            || CASE
                 WHEN metadata ? 'source_turn_id' THEN jsonb_build_array(metadata ->> 'source_turn_id')
                 ELSE '[]'::jsonb
               END
          ) AS merged(value)
        ),
        'evidence_refs',
        (
          SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
          FROM jsonb_array_elements(
            COALESCE(metadata -> 'evidence_refs', '[]'::jsonb)
            || COALESCE({metadata_json} -> 'evidence_refs', '[]'::jsonb)
          ) AS merged(item)
        )
      ),
    updated_at = now()
WHERE memory_id = {_sql_literal(memory_id)}
  AND user_id = {_sql_literal(user_id)}
  AND status = 'active';
"""


def _dispose_conflicts_for_merge_sql(
    *,
    user_id: str,
    source_memory_ids: list[str],
    disposition: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    if not source_memory_ids:
        return ""
    source_filter = ", ".join(_sql_literal(item) for item in source_memory_ids)
    conflict_ids = disposition.get("conflict_ids")
    if isinstance(conflict_ids, list) and conflict_ids:
        conflict_filter = "AND conflict_id IN (" + ", ".join(_sql_literal(str(item)) for item in conflict_ids) + ")"
    else:
        conflict_filter = ""
    status = str(disposition["status"])
    disposition_metadata = {
        **(disposition.get("metadata") if isinstance(disposition.get("metadata"), Mapping) else {}),
        **metadata,
        "merge_disposition_status": status,
        "merge_disposition_reason": str(disposition.get("reason") or ""),
    }
    return f"""
UPDATE memory_conflict_records
SET status = {_sql_literal(status)},
    metadata = metadata || {_json_literal(disposition_metadata)}
WHERE user_id = {_sql_literal(user_id)}
  AND memory_id IN ({source_filter})
  AND status IN ('needs_review', 'carried_forward')
  {conflict_filter};
"""


def _insert_conflict_decision_sql(
    *,
    user_id: str,
    memory_id: str,
    turn_id: str,
    conflict: Mapping[str, Any],
    evidence_refs: list[Any],
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
    conflict_type = str(conflict["conflict_type"])
    old_claim = str(conflict["old_claim"])
    new_claim = str(conflict["new_claim"])
    decision = str(conflict["decision"])
    conflict_id = _digest_id(
        "conflict",
        user_id,
        memory_id,
        turn_id,
        conflict_type,
    )
    status = str(conflict.get("status") or "needs_review")
    conflict_metadata = {
        **(conflict.get("metadata") if isinstance(conflict.get("metadata"), Mapping) else {}),
        **metadata,
    }
    sql = f"""
INSERT INTO memory_conflict_records(
  conflict_id, user_id, memory_id, topic_id, turn_id, conflict_type,
  old_claim, new_claim, decision, reason, evidence_refs, status, metadata
)
SELECT
  {_sql_literal(conflict_id)},
  {_sql_literal(user_id)},
  m.memory_id,
  m.topic_id,
  {_sql_literal(turn_id)},
  {_sql_literal(conflict_type)},
  {_sql_literal(old_claim)},
  {_sql_literal(new_claim)},
  {_sql_literal(decision)},
  {_sql_literal(str(conflict["reason"]))},
  {_json_literal(evidence_refs)},
  {_sql_literal(status)},
  {_json_literal(conflict_metadata)}
FROM memories m
WHERE m.memory_id = {_sql_literal(memory_id)}
  AND m.user_id = {_sql_literal(user_id)}
  AND m.status IN ('active', 'conflict')
ON CONFLICT (conflict_id)
DO UPDATE SET
  old_claim = EXCLUDED.old_claim,
  new_claim = EXCLUDED.new_claim,
  decision = EXCLUDED.decision,
  reason = EXCLUDED.reason,
  evidence_refs = EXCLUDED.evidence_refs,
  status = EXCLUDED.status,
  metadata = memory_conflict_records.metadata || EXCLUDED.metadata;
"""
    return conflict_id, sql


def _read_memory_relation_state(
    *,
    config: Mapping[str, Any],
    user_id: str,
    relation_id: str,
) -> dict[str, Any]:
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT COALESCE((
  SELECT jsonb_build_object(
    'found', true,
    'relation_id', relation_id,
    'reason', reason,
    'evidence_refs', evidence_refs
  )
  FROM memory_relations
  WHERE relation_id = {_sql_literal(relation_id)}
    AND user_id = {_sql_literal(user_id)}
), jsonb_build_object('found', false))::text;
"""
    return _json_from_psql_object(config, sql)


def _insert_memory_relation_sql(
    *,
    user_id: str,
    turn_id: str,
    relation: Mapping[str, Any],
    evidence_refs: list[Any],
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
    from_memory_id = str(relation["from_memory_id"])
    to_memory_id = str(relation["to_memory_id"])
    relation_type = str(relation["relation_type"])
    relation_id = _digest_id(
        "rel",
        user_id,
        from_memory_id,
        to_memory_id,
        relation_type,
    )
    relation_metadata = {
        **(relation.get("metadata") if isinstance(relation.get("metadata"), Mapping) else {}),
        **metadata,
    }
    sql = f"""
INSERT INTO memory_relations(
  relation_id, user_id, from_memory_id, to_memory_id, relation_type,
  reason, source_turn_id, evidence_refs, metadata
)
VALUES (
  {_sql_literal(relation_id)},
  {_sql_literal(user_id)},
  {_sql_literal(from_memory_id)},
  {_sql_literal(to_memory_id)},
  {_sql_literal(relation_type)},
  {_sql_literal(str(relation["reason"]))},
  {_sql_literal(turn_id)},
  {_json_literal(evidence_refs)},
  {_json_literal(relation_metadata)}
)
ON CONFLICT (relation_id)
DO UPDATE SET
  reason = EXCLUDED.reason,
  source_turn_id = EXCLUDED.source_turn_id,
  evidence_refs = EXCLUDED.evidence_refs,
  metadata = memory_relations.metadata || EXCLUDED.metadata;
"""
    return relation_id, sql


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
      AND (
        metadata ->> 'source_turn_id' = {_sql_literal(turn_id)}
        OR COALESCE(metadata -> 'source_turn_ids', '[]'::jsonb) ? {_sql_literal(turn_id)}
        OR EXISTS (
          SELECT 1
          FROM jsonb_array_elements(COALESCE(metadata -> 'evidence_refs', '[]'::jsonb)) AS ref(item)
          JOIN evidence_items e ON e.evidence_id = ref.item ->> 'evidence_id'
          WHERE e.turn_id = {_sql_literal(turn_id)}
            AND e.user_id = {_sql_literal(user_id)}
        )
      )
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
            "result": "noop",
            "target_id": None,
            "reason": change.get("reason"),
            "message": "no durable memory update is needed",
        }
    if action == "request_more_evidence":
        return {
            "action": action,
            "committed": False,
            "result": "request_more_evidence",
            "target_id": None,
            "reason": change.get("reason"),
            "requests": change.get("requests") or [],
            "message": "more evidence is required before committing memory changes",
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
        revision_sql = _insert_memory_revision_sql(
            user_id=user_id,
            memory_id=memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
            change_type="create",
            reason=str(change.get("reason") or "create_memory"),
            metadata=metadata,
            initial=True,
        )
        _run_psql(config, "BEGIN;\n" + topic_sql + memory_sql + revision_sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": "created",
            "target_id": memory_id,
            "topic_id": topic_id,
            "memory_id": memory_id,
            "stable_key": change["memory"]["stable_key"],
            "message": "memory created",
        }

    if action == "update_topic":
        target_topic_id = str(change["target_topic_id"])
        current = _read_topic_state(
            config=config,
            user_id=user_id,
            topic_id=target_topic_id,
        )
        changed_fields = [
            field
            for field in ("title", "summary")
            if field in change["topic"] and str(change["topic"][field]) != str(current.get(field) or "")
        ]
        if current.get("found") and not changed_fields:
            return {
                "action": action,
                "committed": False,
                "result": "skipped_no_change",
                "target_id": target_topic_id,
                "topic_id": target_topic_id,
                "message": "topic title/summary already match the requested values",
            }
        sql = _update_topic_sql(
            user_id=user_id,
            topic_id=target_topic_id,
            topic=change["topic"],
        )
        _run_psql(config, "BEGIN;\n" + sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": "updated",
            "target_id": target_topic_id,
            "topic_id": target_topic_id,
            "title": change["topic"].get("title"),
            "summary": change["topic"].get("summary"),
            "changed_fields": changed_fields,
            "message": "topic title/summary updated",
        }

    if action == "move_memory_topic":
        target_memory_id = str(change["target_memory_id"])
        from_topic_id = str(change["from_topic_id"])
        to_topic_id = str(change["to_topic_id"])
        if _memory_already_processed_turn(
            config=config,
            user_id=user_id,
            memory_id=target_memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
        ):
            return {
                "action": action,
                "committed": False,
                "result": "skipped_duplicate_turn",
                "target_id": target_memory_id,
                "memory_id": target_memory_id,
                "from_topic_id": from_topic_id,
                "to_topic_id": to_topic_id,
                "message": "target memory already processed this source_turn_id; skipping duplicate topic move",
            }
        sql = _move_memory_topic_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            from_topic_id=from_topic_id,
            to_topic_id=to_topic_id,
            metadata=metadata,
            reason=str(change.get("reason") or action),
        )
        revision_sql = _insert_memory_revision_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
            change_type=action,
            reason=str(change.get("reason") or action),
            metadata={
                **metadata,
                "old_topic_id": from_topic_id,
                "new_topic_id": to_topic_id,
            },
            initial=False,
        )
        _run_psql(config, "BEGIN;\n" + sql + revision_sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": "moved",
            "target_id": target_memory_id,
            "memory_id": target_memory_id,
            "from_topic_id": from_topic_id,
            "to_topic_id": to_topic_id,
            "message": "memory moved to target topic",
        }

    if action == "merge_topics":
        target_topic_id = str(change["target_topic_id"])
        source_topic_ids = [str(item) for item in change.get("source_topic_ids") or []]
        sql = _merge_topics_sql(
            user_id=user_id,
            target_topic_id=target_topic_id,
            source_topic_ids=source_topic_ids,
            metadata=metadata,
            reason=str(change.get("reason") or action),
        )
        revision_sql = f"""
UPDATE memory_revisions
SET status = 'superseded'
WHERE user_id = {_sql_literal(user_id)}
  AND status = 'current'
  AND memory_id IN (
    SELECT memory_id
    FROM memories
    WHERE user_id = {_sql_literal(user_id)}
      AND topic_id = {_sql_literal(target_topic_id)}
      AND metadata ->> 'source_turn_id' = {_sql_literal(str(proposal.get("turn_id") or ""))}
  );

INSERT INTO memory_revisions(
  revision_id, user_id, memory_id, revision_number, summary, content,
  confidence, trust_level, status, change_type, reason, source_turn_id, metadata
)
SELECT
  'rev_' || substr(md5(
    m.user_id || '|' || m.memory_id || '|merge_topics|' ||
    COALESCE({_sql_literal(str(proposal.get("turn_id") or ""))}, '') || '|' ||
    m.summary || '|' || m.content
  ), 1, 16),
  m.user_id,
  m.memory_id,
  COALESCE((
    SELECT max(r.revision_number)
    FROM memory_revisions r
    WHERE r.user_id = m.user_id
      AND r.memory_id = m.memory_id
  ), 0) + 1,
  m.summary,
  m.content,
  m.confidence,
  m.trust_level,
  'current',
  'merge_topics',
  {_sql_literal(str(change.get("reason") or action))},
  {_sql_literal(str(proposal.get("turn_id") or ""))}::text,
  {_json_literal({"change_metadata": {**metadata, "target_topic_id": target_topic_id, "source_topic_ids": source_topic_ids}})}
FROM memories m
WHERE m.user_id = {_sql_literal(user_id)}
  AND m.status = 'active'
  AND m.topic_id = {_sql_literal(target_topic_id)}
  AND m.metadata ->> 'source_turn_id' = {_sql_literal(str(proposal.get("turn_id") or ""))}
ON CONFLICT (revision_id) DO NOTHING;
"""
        _run_psql(config, "BEGIN;\n" + sql + revision_sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": "merged",
            "target_id": target_topic_id,
            "target_topic_id": target_topic_id,
            "source_topic_ids": source_topic_ids,
            "message": "source topics merged into target topic",
        }

    if action == "record_conflict_decision":
        target_memory_id = str(change["target_memory_id"])
        conflict_id, sql = _insert_conflict_decision_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
            conflict=change["conflict"],
            evidence_refs=change.get("evidence_refs") or [],
            metadata=metadata,
        )
        current = _read_conflict_state(
            config=config,
            user_id=user_id,
            conflict_id=conflict_id,
        )
        desired_status = str(change["conflict"].get("status") or "needs_review")
        desired_reason = str(change["conflict"]["reason"])
        desired_evidence_refs = change.get("evidence_refs") or []
        if (
            current.get("found")
            and str(current.get("status") or "") == desired_status
            and str(current.get("reason") or "") == desired_reason
            and current.get("evidence_refs") == desired_evidence_refs
        ):
            return {
                "action": action,
                "committed": False,
                "result": "skipped_existing",
                "target_id": conflict_id,
                "memory_id": target_memory_id,
                "conflict_id": conflict_id,
                "status": desired_status,
                "message": "conflict decision already recorded with identical content",
            }
        result = "updated_existing" if current.get("found") else "created"
        _run_psql(config, "BEGIN;\n" + sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": result,
            "target_id": conflict_id,
            "memory_id": target_memory_id,
            "conflict_id": conflict_id,
            "status": str(change["conflict"].get("status") or "needs_review"),
            "message": "conflict decision recorded" if result == "created" else "conflict decision updated",
        }

    if action == "record_memory_relation":
        relation_id, sql = _insert_memory_relation_sql(
            user_id=user_id,
            turn_id=str(proposal.get("turn_id") or ""),
            relation=change["relation"],
            evidence_refs=change.get("evidence_refs") or [],
            metadata=metadata,
        )
        current = _read_memory_relation_state(
            config=config,
            user_id=user_id,
            relation_id=relation_id,
        )
        desired_reason = str(change["relation"]["reason"])
        desired_evidence_refs = change.get("evidence_refs") or []
        if (
            current.get("found")
            and str(current.get("reason") or "") == desired_reason
            and current.get("evidence_refs") == desired_evidence_refs
        ):
            return {
                "action": action,
                "committed": False,
                "result": "skipped_existing",
                "target_id": relation_id,
                "relation_id": relation_id,
                "from_memory_id": change["relation"]["from_memory_id"],
                "to_memory_id": change["relation"]["to_memory_id"],
                "relation_type": change["relation"]["relation_type"],
                "message": "memory relation already recorded with identical content",
            }
        result = "updated_existing" if current.get("found") else "created"
        _run_psql(config, "BEGIN;\n" + sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": result,
            "target_id": relation_id,
            "relation_id": relation_id,
            "from_memory_id": change["relation"]["from_memory_id"],
            "to_memory_id": change["relation"]["to_memory_id"],
            "relation_type": change["relation"]["relation_type"],
            "message": "memory relation recorded" if result == "created" else "memory relation updated",
        }

    if action == "retire_memory":
        target_memory_id = str(change["target_memory_id"])
        if _memory_already_processed_turn(
            config=config,
            user_id=user_id,
            memory_id=target_memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
        ):
            return {
                "action": action,
                "committed": False,
                "result": "skipped_duplicate_turn",
                "target_id": target_memory_id,
                "memory_id": target_memory_id,
                "message": "target memory already processed this source_turn_id; skipping duplicate retire",
            }
        sql = _retire_memory_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            metadata=metadata,
            reason=str(change.get("reason") or action),
        )
        revision_sql = _insert_memory_revision_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
            change_type=action,
            reason=str(change.get("reason") or action),
            metadata=metadata,
            initial=False,
        )
        conflict_disposition_sql = ""
        disposed_conflict_ids: list[str] = []
        if isinstance(change.get("conflict_disposition"), Mapping):
            disposition = change["conflict_disposition"]
            disposed_conflict_ids = [str(item) for item in disposition.get("conflict_ids") or []]
            conflict_disposition_sql = _dispose_conflicts_for_retire_sql(
                user_id=user_id,
                memory_id=target_memory_id,
                disposition=disposition,
                metadata=metadata,
            )
        _run_psql(config, "BEGIN;\n" + sql + revision_sql + conflict_disposition_sql + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": "retired",
            "target_id": target_memory_id,
            "memory_id": target_memory_id,
            "status": "retired",
            "disposed_conflict_ids": disposed_conflict_ids,
            "message": "memory retired",
        }

    if action == "merge_existing_memories":
        target_memory_id = str(change["target_memory_id"])
        source_memory_ids = [str(item) for item in change.get("source_memory_ids") or []]
        all_processed = [_memory_already_processed_turn(
            config=config,
            user_id=user_id,
            memory_id=memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
        ) for memory_id in [target_memory_id, *source_memory_ids]]
        if all(all_processed):
            return {
                "action": action,
                "committed": False,
                "result": "skipped_duplicate_turn",
                "target_id": target_memory_id,
                "memory_id": target_memory_id,
                "source_memory_ids": source_memory_ids,
                "message": "target and source memories already processed this source_turn_id; skipping duplicate merge",
            }
        sql_parts: list[str] = []
        sql_parts.append(_update_memory_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            memory=change["memory"],
            metadata=metadata,
            status=None,
        ))
        sql_parts.append(_insert_memory_revision_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            turn_id=str(proposal.get("turn_id") or ""),
            change_type=action,
            reason=str(change.get("reason") or action),
            metadata=metadata,
            initial=False,
        ))
        for source_memory_id in source_memory_ids:
            sql_parts.append(_mark_source_memory_merged_sql(
                user_id=user_id,
                memory_id=source_memory_id,
                target_memory_id=target_memory_id,
                metadata=metadata,
                reason=str(change.get("reason") or action),
            ))
            sql_parts.append(_insert_memory_revision_sql(
                user_id=user_id,
                memory_id=source_memory_id,
                turn_id=str(proposal.get("turn_id") or ""),
                change_type=action,
                reason=str(change.get("reason") or action),
                metadata={**metadata, "merge_target_memory_id": target_memory_id},
                initial=False,
            ))
            relation = {
                "from_memory_id": source_memory_id,
                "to_memory_id": target_memory_id,
                "relation_type": "merges",
                "reason": str(change.get("reason") or action),
                "metadata": {"created_by": "merge_existing_memories"},
            }
            _, relation_sql = _insert_memory_relation_sql(
                user_id=user_id,
                turn_id=str(proposal.get("turn_id") or ""),
                relation=relation,
                evidence_refs=change.get("evidence_refs") or [],
                metadata=metadata,
            )
            sql_parts.append(relation_sql)
        disposed_conflict_ids: list[str] = []
        if isinstance(change.get("conflict_disposition"), Mapping):
            disposition = change["conflict_disposition"]
            disposed_conflict_ids = [str(item) for item in disposition.get("conflict_ids") or []]
            sql_parts.append(_dispose_conflicts_for_merge_sql(
                user_id=user_id,
                source_memory_ids=source_memory_ids,
                disposition=disposition,
                metadata=metadata,
            ))
        _run_psql(config, "BEGIN;\n" + "\n".join(sql_parts) + "\nCOMMIT;")
        return {
            "action": action,
            "committed": True,
            "result": "merged",
            "target_id": target_memory_id,
            "memory_id": target_memory_id,
            "source_memory_ids": source_memory_ids,
            "status": "active",
            "disposed_conflict_ids": disposed_conflict_ids,
            "message": "existing memories merged",
        }

    target_memory_id = str(change["target_memory_id"])
    if _memory_already_processed_turn(
        config=config,
        user_id=user_id,
        memory_id=target_memory_id,
        turn_id=str(proposal.get("turn_id") or ""),
    ):
        return {
            "action": action,
            "committed": False,
            "result": "skipped_duplicate_turn",
            "target_id": target_memory_id,
            "memory_id": target_memory_id,
            "message": "target memory already processed this source_turn_id; skipping duplicate write",
        }
    status = "conflict" if action == "mark_conflict" else None
    sql = _update_memory_sql(
        user_id=user_id,
        memory_id=target_memory_id,
        memory=change["memory"],
        metadata=metadata,
        status=status,
    )
    revision_sql = _insert_memory_revision_sql(
        user_id=user_id,
        memory_id=target_memory_id,
        turn_id=str(proposal.get("turn_id") or ""),
        change_type=action,
        reason=str(change.get("reason") or action),
        metadata=metadata,
        initial=False,
    )
    conflict_resolution_sql = ""
    if action == "update_memory" and isinstance(change.get("conflict_resolution"), Mapping):
        conflict_resolution_sql = _resolve_conflicts_by_update_sql(
            user_id=user_id,
            memory_id=target_memory_id,
            conflict_resolution=change["conflict_resolution"],
            metadata=metadata,
        )
    _run_psql(config, "BEGIN;\n" + sql + revision_sql + conflict_resolution_sql + "\nCOMMIT;")
    return {
        "action": action,
        "committed": True,
        "result": "marked_conflict" if action == "mark_conflict" else "updated",
        "target_id": target_memory_id,
        "memory_id": target_memory_id,
        "status": status or "active",
        "resolved_conflict_ids": (
            list(change["conflict_resolution"].get("conflict_ids") or [])
            if action == "update_memory" and isinstance(change.get("conflict_resolution"), Mapping)
            else []
        ),
        "message": "memory marked as conflict" if action == "mark_conflict" else "memory updated",
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
    skipped: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    noops: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    temp_memory_ids: dict[str, str] = {}
    for index, change in enumerate(changes):
        if not isinstance(change, Mapping):
            errors = [f"changes[{index}] must be an object"]
            codes = _error_codes_for_messages(errors)
            rejected.append(
                {
                    "index": index,
                    "action": None,
                    "errors": errors,
                    "error_codes": codes,
                    "suggested_next_actions": _suggestions_for_error_codes(codes),
                }
            )
            continue
        resolved_change = _resolve_temp_memory_references(change, temp_memory_ids)
        single = _single_change_proposal(proposal, resolved_change)
        try:
            validate_memory_changes(
                single,
                pg_config_path=pg_config_path,
                extra_memory_ids=temp_memory_ids,
            )
            result = _commit_validated_change(
                config=config,
                user_id=user_id,
                proposal=single,
                change=resolved_change,
            )
        except MemoryChangesValidationError as error:
            codes = _error_codes_for_messages(error.errors)
            rejected.append(
                {
                    "index": index,
                    "action": change.get("action"),
                    "errors": error.errors,
                    "error_codes": codes,
                    "suggested_next_actions": _suggestions_for_error_codes(codes),
                }
            )
            continue
        except Exception as error:
            errors = [str(error)]
            codes = _error_codes_for_messages(errors)
            rejected.append(
                {
                    "index": index,
                    "action": change.get("action"),
                    "errors": errors,
                    "error_codes": codes,
                    "suggested_next_actions": _suggestions_for_error_codes(codes),
                }
            )
            continue
        client_temp_id = change.get("client_temp_id")
        if (
            resolved_change.get("action") == "create_memory"
            and isinstance(client_temp_id, str)
            and result.get("memory_id")
        ):
            temp_memory_ids[client_temp_id] = str(result["memory_id"])
            result = {
                **result,
                "client_temp_id": client_temp_id,
                "resolved_memory_id": str(result["memory_id"]),
            }
        row = {"index": index, **result}
        if result["action"] == "noop":
            noops.append(row)
        elif result["action"] == "request_more_evidence":
            requests.append(row)
        elif result.get("committed") is False:
            skipped.append(row)
        else:
            applied.append(row)
    warnings: list[dict[str, Any]] = []
    suggested_next_actions: list[str] = []
    if rejected:
        rejected_suggestions: list[str] = []
        for item in rejected:
            for suggestion in item.get("suggested_next_actions") or []:
                if isinstance(suggestion, str) and suggestion not in rejected_suggestions:
                    rejected_suggestions.append(suggestion)
        warnings.append(
            {
                "type": "PARTIAL_REJECTION",
                "message": "one or more changes were rejected; inspect rejected[].errors and regenerate only the failed changes",
            }
        )
        suggested_next_actions.extend(
            rejected_suggestions
            or ["Fix rejected changes using the reported field errors, then resubmit a corrected plan."]
        )
    if requests:
        suggested_next_actions.append(
            "Run the requested read tools, then submit a new memory-changes-v1 plan with the additional evidence."
        )
    return {
        "ok": not rejected,
        "status": "complete",
        "tool": "commit_memory_changes",
        "turn_id": proposal.get("turn_id"),
        "applied": applied,
        "skipped": skipped,
        "rejected": rejected,
        "noop": noops,
        "requests": requests,
        "warnings": warnings,
        "suggested_next_actions": suggested_next_actions,
        "counts": {
            "applied": len(applied),
            "skipped": len(skipped),
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
    "ALLOWED_MEMORY_TYPES",
    "MEMORY_CHANGES_SCHEMA_VERSION",
    "MemoryChangesValidationError",
    "MemoryChangesCommitError",
    "commit_memory_changes",
    "commit_memory_changes_file",
    "validate_memory_changes",
    "validate_memory_changes_file",
]
