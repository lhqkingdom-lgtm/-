"""Immutable Markdown projection and scoped recall for formal memories.

The formal SQLite tables remain the source of truth.  A publication is a
content-addressed, immutable generation beneath ``memory/generations`` and the
small ``CURRENT`` file is the only mutable pointer.  Every published file is
covered by SHA-256, while topic Markdown deliberately contains no evidence
quotes or source conversation payloads.

Front matter uses the same deliberately small JSON-scalar subset of YAML as
the preview projection.  This keeps the files readable without accepting YAML
tags, aliases, duplicate keys, or implicit scalar conversions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .judge import WIRE_SECRET_PATTERNS


STATE_SCHEMA = "memory-lab-formal-state-v1"
INDEX_SCHEMA = "memory-lab-formal-index-v1"
TOPIC_SCHEMA = "memory-lab-formal-topic-v1"
PENDING_SCHEMA = "memory-lab-formal-pending-conflicts-v1"
HISTORY_SCHEMA = "memory-lab-formal-resolution-history-v1"
MANIFEST_SCHEMA = "memory-lab-formal-manifest-v1"
CURRENT_SCHEMA = "memory-lab-formal-current-v1"
RECALL_VERSION = "memory-lab-formal-recall-v1"
MANAGED_BY = "codepulse-memory-lab"
ENGINEERING_RECALLABLE_VERIFICATION = frozenset(
    {"verified", "cross_project_verified"}
)

MAX_INDEX_BYTES = 25 * 1024
MAX_TOPIC_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_HEADER_BYTES = 192 * 1024
MAX_TOPICS = 200
MAX_QUERY_CHARS = 4000
_EMPTY_TIME = "1970-01-01T00:00:00.000+00:00"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^gen-[0-9a-f]{64}$")
_HEADER_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


class FormalMarkdownError(ValueError):
    """The formal database or Markdown projection is unsafe/inconsistent."""


_TOPIC_SPECS = (
    (
        "preference.response.",
        {
            "topic_id": "user-response-preferences",
            "path": "user_response_preferences.md",
            "name": "回复偏好",
            "description": "跨项目生效的回复语言与组织方式",
            "type": "user",
        },
    ),
    (
        "workflow.code_generation.",
        {
            "topic_id": "workflow-code-generation",
            "path": "workflow_code_generation.md",
            "name": "代码生成流程",
            "description": "生成正式代码前的确认要求及项目例外",
            "type": "workflow",
        },
    ),
    (
        "workflow.embedded.",
        {
            "topic_id": "workflow-embedded-compile",
            "path": "workflow_embedded_compile.md",
            "name": "嵌入式流程",
            "description": "嵌入式代码修改后的编译和验证要求",
            "type": "workflow",
        },
    ),
)

_ENTRY_KEYS = {
    "memory_id",
    "revision_id",
    "revision_number",
    "stable_key",
    "family",
    "memory_type",
    "semantic_value",
    "statement",
    "scope",
    "state_version",
    "status",
    "verification_status",
    "activation_triggers",
    "evidence_count",
    "updated_at",
}
_TOPIC_SPEC_KEYS = {"topic_id", "path", "name", "description", "type"}
_TOPIC_INDEX_KEYS = _TOPIC_SPEC_KEYS | {
    "status",
    "updated_at",
    "stable_keys",
    "triggers",
    "summary",
    "memory_count",
    "sha256",
}
_PENDING_KEYS = {
    "conflict_id",
    "memory_id",
    "stable_key",
    "scope",
    "status",
    "current_revision_id",
    "current_semantic_value",
    "current_statement",
    "incoming_semantic_value",
    "incoming_statement",
    "activation_triggers",
    "created_at",
}
_HISTORY_KEYS = {
    "conflict_id",
    "memory_id",
    "stable_key",
    "scope",
    "status",
    "previous_revision_id",
    "previous_semantic_value",
    "incoming_semantic_value",
    "resolution_revision_id",
    "resolution_semantic_value",
    "resolution_statement",
    "resolution_reason",
    "created_at",
    "resolved_at",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_object(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _front_matter(fields: Mapping[str, Any]) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if not _HEADER_KEY.fullmatch(key):
            raise FormalMarkdownError(f"invalid front-matter key: {key}")
        lines.append(f"{key}: {_canonical_json(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _parse_front_matter(text: str, *, source: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise FormalMarkdownError(f"{source} has no front matter")
    lines = text.splitlines(keepends=True)
    consumed = len(lines[0].encode("utf-8"))
    result: dict[str, Any] = {}
    closing: int | None = None
    for index, raw in enumerate(lines[1:], start=1):
        consumed += len(raw.encode("utf-8"))
        if consumed > MAX_HEADER_BYTES:
            raise FormalMarkdownError(f"{source} front matter is too large")
        line = raw.rstrip("\r\n")
        if line == "---":
            closing = index
            break
        if not line or ": " not in line:
            raise FormalMarkdownError(f"{source} has invalid front matter")
        key, encoded = line.split(": ", 1)
        if not _HEADER_KEY.fullmatch(key) or key in result:
            raise FormalMarkdownError(f"{source} has invalid or duplicate key")
        try:
            result[key] = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise FormalMarkdownError(
                f"{source} has a non-JSON YAML scalar for {key}"
            ) from error
    if closing is None:
        raise FormalMarkdownError(f"{source} front matter is not closed")
    return result, "".join(lines[closing + 1 :])


def _exact_keys(value: Mapping[str, Any], expected: set[str], source: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FormalMarkdownError(
            f"{source} schema mismatch "
            f"(missing={sorted(expected - actual)}, unknown={sorted(actual - expected)})"
        )


def _clean_text(value: Any, field: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise FormalMarkdownError(f"{field} must be text")
    cleaned = unicodedata.normalize("NFKC", value).strip()
    if not cleaned or len(cleaned) > maximum or "\x00" in cleaned:
        raise FormalMarkdownError(f"{field} is empty, too long, or unsafe")
    return cleaned


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _assert_no_secrets(value: Any) -> None:
    for text in _walk_strings(value):
        if any(pattern.search(text) for pattern in WIRE_SECRET_PATTERNS):
            raise FormalMarkdownError("formal projection contains a probable secret")


def _parse_json_object(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise FormalMarkdownError(f"{field} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise FormalMarkdownError(f"{field} must be an object")
    result = json.loads(_canonical_json(dict(value)))
    if not isinstance(result.get("level"), str):
        raise FormalMarkdownError(f"{field}.level is required")
    if result["level"] not in {"global", "domain", "project"}:
        raise FormalMarkdownError(f"{field}.level is invalid")
    if result["level"] == "domain" and not isinstance(result.get("domain"), str):
        raise FormalMarkdownError(f"{field}.domain is required")
    if result["level"] == "project" and not isinstance(result.get("project_key"), str):
        raise FormalMarkdownError(f"{field}.project_key is required")
    conditions = result.get("conditions")
    if conditions is not None and not isinstance(conditions, Mapping):
        raise FormalMarkdownError(f"{field}.conditions must be an object")
    return result


def _parse_json_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise FormalMarkdownError(f"{field} is not valid JSON") from error
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise FormalMarkdownError(f"{field} must be a string list")
    cleaned = [_clean_text(item, field, 200) for item in value]
    return sorted(set(cleaned), key=lambda item: (item.casefold(), item))


def _slug(value: str, maximum: int = 40) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    result = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return (result or "memory")[:maximum].rstrip("_") or "memory"


def _topic_spec(stable_key: str, family: str, memory_type: str) -> dict[str, str]:
    for prefix, spec in _TOPIC_SPECS:
        if stable_key.startswith(prefix):
            return dict(spec)
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:10]
    pieces = stable_key.split(".")
    is_engineering = (
        stable_key.startswith(("engineering.", "toolchain."))
        or "engineering" in family.casefold()
        or "engineering" in memory_type.casefold()
    )
    if is_engineering:
        stem = _slug("_".join(pieces[:2]))
        return {
            "topic_id": f"engineering-{stem}-{digest}",
            "path": f"engineering_{stem}_{digest}.md",
            "name": f"工程经验 · {'.'.join(pieces[:2])}",
            "description": f"{'.'.join(pieces[:2])} 相关的已沉淀工程经验",
            "type": "engineering",
        }
    stem = _slug("_".join(pieces[:2]))
    return {
        "topic_id": f"topic-{stem}-{digest}",
        "path": f"topic_{stem}_{digest}.md",
        "name": f"记忆主题 · {'.'.join(pieces[:2])}",
        "description": f"{'.'.join(pieces[:2])} 相关的长期记忆",
        "type": memory_type or family or "memory",
    }


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if exists is None:
        raise FormalMarkdownError(f"formal database is missing table: {table}")
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _optional_table_columns(connection: sqlite3.Connection, table: str) -> set[str] | None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if exists is None:
        return None
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _require_columns(
    connection: sqlite3.Connection, table: str, required: set[str]
) -> set[str]:
    columns = _table_columns(connection, table)
    missing = required - columns
    if missing:
        raise FormalMarkdownError(
            f"formal database table {table} is missing columns: {sorted(missing)}"
        )
    return columns


def _readonly_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FormalMarkdownError(f"formal database does not exist: {resolved}")
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _verification_status(
    row: Mapping[str, Any], family: str, memory_type: str
) -> str:
    keys = set(row.keys()) if hasattr(row, "keys") else set(row)
    for key in ("revision_verification_status", "memory_verification_status"):
        if key in keys and row[key] is not None:
            return _clean_text(row[key], key, 100).casefold()
    if "engineering" in family.casefold() or "engineering" in memory_type.casefold():
        signals = row["source_signals_json"] if "source_signals_json" in keys else None
        if isinstance(signals, str):
            try:
                signals = json.loads(signals)
            except json.JSONDecodeError:
                signals = None
        if isinstance(signals, Mapping):
            contested = bool(
                signals.get("contested")
                or signals.get("unresolved_conflict")
                or signals.get("verification_conflict")
            )
            value = (
                signals.get("verification_status")
                or signals.get("domain_verification_outcome")
                or signals.get("verification")
            )
            if value in ENGINEERING_RECALLABLE_VERIFICATION and not contested:
                return str(value)
        return "unverified"
    return "not_applicable"


def _stored_topic_owners(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, str]] | None:
    """Read Curator ownership when the topic subsystem has been initialized.

    Older lab databases do not have these tables and retain deterministic
    prefix routing.  Once either topic table exists, however, publication
    fails closed on an incomplete owner map so a Curator operation can never
    be silently undone by fallback routing.
    """

    topic_columns = _optional_table_columns(connection, "topic")
    membership_columns = _optional_table_columns(connection, "topic_membership")
    if topic_columns is None and membership_columns is None:
        return None
    if topic_columns is None or membership_columns is None:
        raise FormalMarkdownError(
            "topic schema is incomplete; run ensure_topic_assignments"
        )
    expected_topic = {
        "topic_id",
        "slug",
        "name",
        "description",
        "family",
        "status",
        "updated_at",
    }
    expected_membership = {"stable_key", "topic_id"}
    if expected_topic - topic_columns or expected_membership - membership_columns:
        raise FormalMarkdownError(
            "topic schema is incompatible; run ensure_topic_assignments"
        )
    owners: dict[str, dict[str, str]] = {}
    for row in connection.execute(
        """
        SELECT tm.stable_key, t.topic_id, t.slug, t.name, t.description,
               t.family, t.status, t.updated_at
        FROM topic_membership AS tm
        JOIN topic AS t ON t.topic_id = tm.topic_id
        ORDER BY tm.stable_key
        """
    ):
        stable_key = _clean_text(row["stable_key"], "topic_membership.stable_key", 256)
        slug = _clean_text(row["slug"], "topic.slug", 80)
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,79}", slug):
            raise FormalMarkdownError(f"topic has an unsafe slug: {slug}")
        path = f"{slug}.md"
        if path.casefold() == "memory.md" or path.startswith("conflicts/"):
            raise FormalMarkdownError(f"topic uses a reserved path: {path}")
        if row["status"] != "active":
            raise FormalMarkdownError(
                f"stable key {stable_key} belongs to a retired topic; run ensure_topic_assignments"
            )
        family = _clean_text(row["family"], "topic.family", 100)
        if family == "engineering_memory":
            topic_type = "engineering"
        elif slug.startswith("user_response"):
            topic_type = "user"
        elif slug.startswith("workflow_"):
            topic_type = "workflow"
        else:
            topic_type = family
        owners[stable_key] = {
            "topic_id": _clean_text(row["topic_id"], "topic.topic_id", 256),
            "path": path,
            "name": _clean_text(row["name"], "topic.name", 200),
            "description": _clean_text(row["description"], "topic.description", 500),
            "type": topic_type,
            "topic_updated_at": _clean_text(
                row["updated_at"], "topic.updated_at", 100
            ),
        }
    return owners


def _snapshot_from_connection(connection: sqlite3.Connection) -> dict[str, Any]:
    memory_columns = _require_columns(
        connection,
        "memory",
        {
            "memory_id",
            "stable_key",
            "scope_fingerprint",
            "scope_json",
            "memory_type",
            "status",
            "current_revision_id",
            "updated_at",
        },
    )
    revision_columns = _require_columns(
        connection,
        "memory_revision",
        {
            "revision_id",
            "memory_id",
            "revision_number",
            "semantic_value",
            "statement",
            "activation_triggers_json",
            "status",
            "created_at",
        },
    )
    evidence_columns = _require_columns(connection, "evidence", {"memory_id"})
    conflict_columns = _require_columns(
        connection,
        "conflict",
        {
            "conflict_id",
            "memory_id",
            "current_revision_id",
            "incoming_semantic_value",
            "incoming_statement",
            "status",
            "resolution_revision_id",
            "resolution_reason",
            "created_at",
            "resolved_at",
        },
    )
    del evidence_columns, conflict_columns

    family_expression = "m.family" if "family" in memory_columns else "m.memory_type"
    version_expression = "m.state_version" if "state_version" in memory_columns else "0"
    revision_verify = (
        "r.verification_status" if "verification_status" in revision_columns else "NULL"
    )
    memory_verify = (
        "m.verification_status" if "verification_status" in memory_columns else "NULL"
    )
    admission_columns = _optional_table_columns(connection, "admission_decision")
    can_read_signals = (
        admission_columns is not None
        and {"decision_id", "signals_json"} <= admission_columns
        and "source_decision_id" in revision_columns
    )
    signals_expression = "sd.signals_json" if can_read_signals else "NULL"
    signals_join = (
        "LEFT JOIN admission_decision AS sd ON sd.decision_id = r.source_decision_id"
        if can_read_signals
        else ""
    )
    stored_owners = _stored_topic_owners(connection)
    rows = connection.execute(
        f"""
        SELECT m.memory_id, m.stable_key, m.scope_json, m.memory_type,
               m.status AS memory_status, m.current_revision_id, m.updated_at,
               {family_expression} AS family, {version_expression} AS state_version,
               {memory_verify} AS memory_verification_status,
               r.revision_id, r.revision_number, r.semantic_value, r.statement,
               r.activation_triggers_json, r.status AS revision_status,
               r.created_at AS revision_created_at,
               {revision_verify} AS revision_verification_status,
               {signals_expression} AS source_signals_json,
               (SELECT COUNT(*) FROM evidence AS e WHERE e.memory_id = m.memory_id)
                   AS evidence_count
        FROM memory AS m
        LEFT JOIN memory_revision AS r ON r.revision_id = m.current_revision_id
        {signals_join}
        ORDER BY m.stable_key, m.scope_fingerprint, m.memory_id
        """
    ).fetchall()

    entries: list[dict[str, Any]] = []
    memory_by_id: dict[str, dict[str, Any]] = {}
    all_memory_ids: set[str] = set()
    timestamps: list[str] = []
    for row in rows:
        memory_id = _clean_text(row["memory_id"], "memory.memory_id", 256)
        if memory_id in all_memory_ids:
            raise FormalMarkdownError("formal database repeats a memory_id")
        all_memory_ids.add(memory_id)
        if row["current_revision_id"] is None or row["revision_id"] is None:
            raise FormalMarkdownError(f"memory {memory_id} has no current revision")
        if row["revision_status"] != "current":
            raise FormalMarkdownError(f"memory {memory_id} points to a non-current revision")
        stable_key = _clean_text(row["stable_key"], "memory.stable_key", 256)
        family = _clean_text(row["family"] or row["memory_type"], "memory.family", 100)
        memory_type = _clean_text(row["memory_type"], "memory.memory_type", 100)
        scope = _parse_json_object(row["scope_json"], "memory.scope_json")
        updated_at = _clean_text(row["updated_at"], "memory.updated_at", 100)
        entry = {
            "memory_id": memory_id,
            "revision_id": _clean_text(row["revision_id"], "revision.revision_id", 256),
            "revision_number": int(row["revision_number"]),
            "stable_key": stable_key,
            "family": family,
            "memory_type": memory_type,
            "semantic_value": _clean_text(
                row["semantic_value"], "revision.semantic_value", 1000
            ),
            "statement": _clean_text(row["statement"], "revision.statement", 2000),
            "scope": scope,
            "state_version": int(row["state_version"] or 0),
            "status": "active",
            "verification_status": _verification_status(row, family, memory_type),
            "activation_triggers": _parse_json_list(
                row["activation_triggers_json"], "revision.activation_triggers_json"
            ),
            "evidence_count": int(row["evidence_count"]),
            "updated_at": updated_at,
        }
        if entry["revision_number"] < 1 or entry["state_version"] < 0:
            raise FormalMarkdownError("revision/state version must be non-negative")
        if entry["evidence_count"] < 0:
            raise FormalMarkdownError("evidence_count must be non-negative")
        memory_by_id[memory_id] = entry
        timestamps.extend([updated_at, str(row["revision_created_at"] or "")])
        if row["memory_status"] == "active":
            entries.append(entry)
        elif row["memory_status"] != "retired":
            raise FormalMarkdownError(f"memory {memory_id} has an invalid status")

    conflict_rows = connection.execute(
        """
        SELECT c.*, m.stable_key, m.scope_json,
               current.semantic_value AS current_semantic_value,
               current.statement AS current_statement,
               current.activation_triggers_json AS current_triggers,
               resolution.semantic_value AS resolution_semantic_value,
               resolution.statement AS resolution_statement
        FROM conflict AS c
        JOIN memory AS m ON m.memory_id = c.memory_id
        JOIN memory_revision AS current ON current.revision_id = c.current_revision_id
        LEFT JOIN memory_revision AS resolution
          ON resolution.revision_id = c.resolution_revision_id
        ORDER BY c.created_at, c.conflict_id
        """
    ).fetchall()
    pending: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    seen_conflicts: set[str] = set()
    for row in conflict_rows:
        conflict_id = _clean_text(row["conflict_id"], "conflict.conflict_id", 256)
        if conflict_id in seen_conflicts:
            raise FormalMarkdownError("formal database repeats a conflict_id")
        seen_conflicts.add(conflict_id)
        memory_id = _clean_text(row["memory_id"], "conflict.memory_id", 256)
        if memory_id not in all_memory_ids:
            raise FormalMarkdownError("conflict references an unknown memory")
        scope = _parse_json_object(row["scope_json"], "conflict.scope_json")
        stable_key = _clean_text(row["stable_key"], "conflict.stable_key", 256)
        status = row["status"]
        timestamps.extend([str(row["created_at"] or ""), str(row["resolved_at"] or "")])
        if status == "open":
            if row["resolution_revision_id"] is not None or row["resolved_at"] is not None:
                raise FormalMarkdownError("open conflict contains resolution fields")
            pending.append(
                {
                    "conflict_id": conflict_id,
                    "memory_id": memory_id,
                    "stable_key": stable_key,
                    "scope": scope,
                    "status": "open",
                    "current_revision_id": _clean_text(
                        row["current_revision_id"], "conflict.current_revision_id", 256
                    ),
                    "current_semantic_value": _clean_text(
                        row["current_semantic_value"], "conflict.current_semantic_value"
                    ),
                    "current_statement": _clean_text(
                        row["current_statement"], "conflict.current_statement", 2000
                    ),
                    "incoming_semantic_value": _clean_text(
                        row["incoming_semantic_value"],
                        "conflict.incoming_semantic_value",
                    ),
                    "incoming_statement": _clean_text(
                        row["incoming_statement"], "conflict.incoming_statement", 2000
                    ),
                    "activation_triggers": _parse_json_list(
                        row["current_triggers"], "conflict.current_triggers"
                    ),
                    "created_at": _clean_text(row["created_at"], "conflict.created_at", 100),
                }
            )
        elif status == "resolved":
            if row["resolution_revision_id"] is None or row["resolved_at"] is None:
                raise FormalMarkdownError("resolved conflict has no resolution revision/time")
            history.append(
                {
                    "conflict_id": conflict_id,
                    "memory_id": memory_id,
                    "stable_key": stable_key,
                    "scope": scope,
                    "status": "resolved",
                    "previous_revision_id": _clean_text(
                        row["current_revision_id"], "conflict.current_revision_id", 256
                    ),
                    "previous_semantic_value": _clean_text(
                        row["current_semantic_value"], "conflict.current_semantic_value"
                    ),
                    "incoming_semantic_value": _clean_text(
                        row["incoming_semantic_value"],
                        "conflict.incoming_semantic_value",
                    ),
                    "resolution_revision_id": _clean_text(
                        row["resolution_revision_id"],
                        "conflict.resolution_revision_id",
                        256,
                    ),
                    "resolution_semantic_value": _clean_text(
                        row["resolution_semantic_value"],
                        "conflict.resolution_semantic_value",
                    ),
                    "resolution_statement": _clean_text(
                        row["resolution_statement"], "conflict.resolution_statement", 2000
                    ),
                    "resolution_reason": _clean_text(
                        row["resolution_reason"] or "automatically resolved",
                        "conflict.resolution_reason",
                        1000,
                    ),
                    "created_at": _clean_text(row["created_at"], "conflict.created_at", 100),
                    "resolved_at": _clean_text(row["resolved_at"], "conflict.resolved_at", 100),
                }
            )
        else:
            raise FormalMarkdownError(f"conflict {conflict_id} has an invalid status")

    grouped: dict[str, dict[str, Any]] = {}
    owner_by_key: dict[str, str] = {}
    for entry in entries:
        if stored_owners is not None:
            stored = stored_owners.get(entry["stable_key"])
            if stored is None:
                raise FormalMarkdownError(
                    f"stable key {entry['stable_key']} has no active topic owner; "
                    "run ensure_topic_assignments"
                )
            spec = {key: stored[key] for key in _TOPIC_SPEC_KEYS}
        else:
            stored = None
            spec = _topic_spec(entry["stable_key"], entry["family"], entry["memory_type"])
        prior = owner_by_key.setdefault(entry["stable_key"], spec["topic_id"])
        if prior != spec["topic_id"]:
            raise FormalMarkdownError("one stable key is owned by multiple topics")
        record = grouped.setdefault(spec["topic_id"], {"spec": spec, "entries": []})
        if record["spec"] != spec:
            raise FormalMarkdownError("one topic id has inconsistent metadata")
        record["entries"].append(entry)
    paths_by_casefold: dict[str, str] = {}
    for topic_id, record in grouped.items():
        path = record["spec"]["path"]
        prior = paths_by_casefold.setdefault(path.casefold(), topic_id)
        if prior != topic_id:
            raise FormalMarkdownError("topic paths collide on a case-insensitive filesystem")
    topics = []
    for topic_id in sorted(grouped):
        record = grouped[topic_id]
        topic_entries = sorted(
            record["entries"],
            key=lambda item: (
                item["stable_key"],
                _canonical_json(item["scope"]),
                item["memory_id"],
            ),
        )
        owner_times = [
            stored_owners[item["stable_key"]]["topic_updated_at"]
            for item in topic_entries
            if stored_owners is not None
        ]
        topics.append(
            {
                "spec": record["spec"],
                "updated_at": max(
                    [item["updated_at"] for item in topic_entries] + owner_times
                ),
                "entries": topic_entries,
            }
        )
    if len(topics) > MAX_TOPICS:
        raise FormalMarkdownError(f"formal projection exceeds {MAX_TOPICS} topics")

    generated_at = max((item for item in timestamps if item), default=_EMPTY_TIME)
    state = {
        "schema": STATE_SCHEMA,
        "generated_at": generated_at,
        "topics": topics,
        "pending_conflicts": pending,
        "resolution_history": history,
    }
    _assert_no_secrets(state)
    state_hash = _sha256_object(state)
    return {
        **state,
        "state_hash": state_hash,
        "generation_id": f"gen-{state_hash}",
    }


def snapshot_formal_database(lab_database: Path) -> dict[str, Any]:
    """Return one consistent, read-only projection of the formal SQLite state."""

    connection = _readonly_connection(Path(lab_database))
    try:
        connection.execute("BEGIN")
        snapshot = _snapshot_from_connection(connection)
        connection.rollback()
        return snapshot
    finally:
        connection.close()


def _scope_label(scope: Mapping[str, Any]) -> str:
    level = scope["level"]
    if level == "global":
        result = "全局"
    elif level == "domain":
        result = f"领域：{scope['domain']}"
    else:
        result = f"项目：{scope['project_key']}"
    conditions = scope.get("conditions") or {}
    if conditions:
        rendered = ", ".join(f"{key}={conditions[key]}" for key in sorted(conditions))
        result += f"（{rendered}）"
    return result


def _topic_summary(spec: Mapping[str, str], entries: list[Mapping[str, Any]]) -> str:
    scopes = sorted({str(item["scope"]["level"]) for item in entries})
    return f"{spec['description']}；{len(entries)} 条有效记忆；范围：{', '.join(scopes)}"


def _render_topic(header: Mapping[str, Any]) -> str:
    entries = header["entries"]
    lines = [_front_matter(header), f"# {header['name']}", ""]
    for entry in entries:
        lines.extend(
            [
                f"## `{entry['stable_key']}`",
                "",
                f"- 内容：{entry['statement']}",
                f"- 适用范围：{_scope_label(entry['scope'])}",
                f"- 修订：{entry['revision_number']}（`{entry['revision_id']}`）",
                f"- 验证状态：`{entry['verification_status']}`",
                f"- 证据数量：{entry['evidence_count']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_pending(header: Mapping[str, Any]) -> str:
    lines = [_front_matter(header), "# 待解决冲突", ""]
    if not header["conflicts"]:
        lines.extend(["当前没有待解决冲突。", ""])
    for item in header["conflicts"]:
        lines.extend(
            [
                f"## `{item['stable_key']}`",
                "",
                f"- 范围：{_scope_label(item['scope'])}",
                f"- 当前值：{item['current_statement']}",
                f"- 新值：{item['incoming_statement']}",
                "- 状态：阻断正常召回",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_history(header: Mapping[str, Any]) -> str:
    lines = [_front_matter(header), "# 冲突解决历史", ""]
    if not header["resolutions"]:
        lines.extend(["当前没有已解决冲突。", ""])
    for item in header["resolutions"]:
        lines.extend(
            [
                f"## `{item['stable_key']}`",
                "",
                f"- 范围：{_scope_label(item['scope'])}",
                f"- 最终值：{item['resolution_statement']}",
                f"- 原因：{item['resolution_reason']}",
                f"- 解决时间：{item['resolved_at']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_index_body(topics: list[Mapping[str, Any]]) -> str:
    lines = ["# Memory Index", ""]
    if not topics:
        lines.extend(["当前没有有效长期记忆。", ""])
    for topic in topics:
        lines.append(f"- [{topic['name']}]({topic['path']}) — {topic['summary']}")
    lines.extend(
        [
            "",
            "## 冲突视图",
            "",
            "- [待解决冲突](conflicts/pending_conflicts.md)",
            "- [解决历史](conflicts/resolution_history.md)",
            "",
        ]
    )
    return "\n".join(lines)


def render_formal_generation(snapshot: Mapping[str, Any]) -> dict[str, bytes]:
    """Render a database snapshot into a complete immutable generation."""

    required = {
        "schema",
        "generated_at",
        "topics",
        "pending_conflicts",
        "resolution_history",
        "state_hash",
        "generation_id",
    }
    _exact_keys(snapshot, required, "formal snapshot")
    state = {key: snapshot[key] for key in required - {"state_hash", "generation_id"}}
    expected_hash = _sha256_object(state)
    if snapshot["schema"] != STATE_SCHEMA or snapshot["state_hash"] != expected_hash:
        raise FormalMarkdownError("formal snapshot hash is invalid")
    if snapshot["generation_id"] != f"gen-{expected_hash}":
        raise FormalMarkdownError("formal snapshot generation id is invalid")
    _assert_no_secrets(snapshot)

    generation_id = snapshot["generation_id"]
    files: dict[str, bytes] = {}
    topic_index: list[dict[str, Any]] = []
    for topic in snapshot["topics"]:
        spec = dict(topic["spec"])
        _exact_keys(spec, _TOPIC_SPEC_KEYS, "topic spec")
        entries = list(topic["entries"])
        header = {
            "schema": TOPIC_SCHEMA,
            "managed_by": MANAGED_BY,
            "generation_id": generation_id,
            **spec,
            "status": "active",
            "updated_at": topic["updated_at"],
            "stable_keys": sorted({item["stable_key"] for item in entries}),
            "triggers": sorted(
                {
                    trigger
                    for item in entries
                    for trigger in item["activation_triggers"]
                },
                key=lambda item: (item.casefold(), item),
            ),
            "entries": entries,
        }
        content = _render_topic(header).encode("utf-8")
        if len(content) > MAX_TOPIC_BYTES:
            raise FormalMarkdownError(f"topic exceeds size budget: {spec['path']}")
        files[spec["path"]] = content
        topic_index.append(
            {
                **spec,
                "status": "active",
                "updated_at": topic["updated_at"],
                "stable_keys": header["stable_keys"],
                "triggers": header["triggers"],
                "summary": _topic_summary(spec, entries),
                "memory_count": len(entries),
                "sha256": _sha256_bytes(content),
            }
        )

    pending_header = {
        "schema": PENDING_SCHEMA,
        "managed_by": MANAGED_BY,
        "generation_id": generation_id,
        "conflict_count": len(snapshot["pending_conflicts"]),
        "conflicts": snapshot["pending_conflicts"],
    }
    history_header = {
        "schema": HISTORY_SCHEMA,
        "managed_by": MANAGED_BY,
        "generation_id": generation_id,
        "resolution_count": len(snapshot["resolution_history"]),
        "resolutions": snapshot["resolution_history"],
    }
    files["conflicts/pending_conflicts.md"] = _render_pending(pending_header).encode("utf-8")
    files["conflicts/resolution_history.md"] = _render_history(history_header).encode("utf-8")

    index_header = {
        "schema": INDEX_SCHEMA,
        "managed_by": MANAGED_BY,
        "generation_id": generation_id,
        "state_hash": snapshot["state_hash"],
        "generated_at": snapshot["generated_at"],
        "topic_count": len(topic_index),
        "memory_count": sum(item["memory_count"] for item in topic_index),
        "open_conflict_count": len(snapshot["pending_conflicts"]),
        "resolved_conflict_count": len(snapshot["resolution_history"]),
        "topics": topic_index,
        "conflict_views": {
            path: _sha256_bytes(files[path])
            for path in (
                "conflicts/pending_conflicts.md",
                "conflicts/resolution_history.md",
            )
        },
    }
    index = (_front_matter(index_header) + _render_index_body(topic_index)).encode("utf-8")
    if len(index) > MAX_INDEX_BYTES:
        raise FormalMarkdownError("MEMORY.md exceeds the 25 KiB index budget")
    files["MEMORY.md"] = index

    manifest_files = {
        path: {
            "sha256": _sha256_bytes(content),
            "size_bytes": len(content),
            "kind": (
                "index"
                if path == "MEMORY.md"
                else "conflict_view"
                if path.startswith("conflicts/")
                else "topic"
            ),
        }
        for path, content in sorted(files.items())
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "managed_by": MANAGED_BY,
        "generation_id": generation_id,
        "state_hash": snapshot["state_hash"],
        "generated_at": snapshot["generated_at"],
        "files": manifest_files,
    }
    manifest_bytes = _json_bytes(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise FormalMarkdownError("generation manifest exceeds its size budget")
    files["MANIFEST.json"] = manifest_bytes
    return dict(sorted(files.items()))


def _safe_relative_path(raw: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise FormalMarkdownError("manifest contains an unsafe path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FormalMarkdownError("manifest contains an unsafe path")
    return path


def _assert_not_link(path: Path) -> None:
    if path.is_symlink():
        raise FormalMarkdownError(f"managed path must not be a link: {path}")


def _read_file(path: Path, *, maximum: int) -> bytes:
    _assert_not_link(path)
    if not path.is_file():
        raise FormalMarkdownError(f"managed file is missing: {path}")
    size = path.stat().st_size
    if size > maximum:
        raise FormalMarkdownError(f"managed file exceeds size budget: {path}")
    return path.read_bytes()


def _load_current(memory_root: Path) -> tuple[dict[str, Any], Path]:
    root = Path(memory_root).resolve()
    _assert_not_link(root)
    raw = _read_file(root / "CURRENT", maximum=4096)
    try:
        pointer = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalMarkdownError("CURRENT is not valid UTF-8 JSON") from error
    if not isinstance(pointer, Mapping):
        raise FormalMarkdownError("CURRENT must contain one JSON object")
    pointer = dict(pointer)
    _exact_keys(
        pointer,
        {"schema", "generation_id", "manifest_sha256", "state_hash"},
        "CURRENT",
    )
    if pointer["schema"] != CURRENT_SCHEMA:
        raise FormalMarkdownError("CURRENT schema is unsupported")
    if not _GENERATION.fullmatch(str(pointer["generation_id"])):
        raise FormalMarkdownError("CURRENT generation id is unsafe")
    if not _SHA256.fullmatch(str(pointer["manifest_sha256"])):
        raise FormalMarkdownError("CURRENT manifest hash is invalid")
    if not _SHA256.fullmatch(str(pointer["state_hash"])):
        raise FormalMarkdownError("CURRENT state hash is invalid")
    if pointer["generation_id"] != f"gen-{pointer['state_hash']}":
        raise FormalMarkdownError("CURRENT generation and state hash disagree")
    generation = root / "generations" / pointer["generation_id"]
    _assert_not_link(root / "generations")
    _assert_not_link(generation)
    if not generation.is_dir():
        raise FormalMarkdownError("CURRENT generation directory is missing")
    return pointer, generation


def _load_manifest(
    generation: Path, *, expected_id: str, expected_hash: str | None = None
) -> tuple[dict[str, Any], bytes]:
    raw = _read_file(generation / "MANIFEST.json", maximum=MAX_MANIFEST_BYTES)
    if expected_hash is not None and _sha256_bytes(raw) != expected_hash:
        raise FormalMarkdownError("MANIFEST.json hash does not match CURRENT")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalMarkdownError("MANIFEST.json is invalid") from error
    if not isinstance(manifest, Mapping):
        raise FormalMarkdownError("MANIFEST.json must contain one object")
    manifest = dict(manifest)
    _exact_keys(
        manifest,
        {"schema", "managed_by", "generation_id", "state_hash", "generated_at", "files"},
        "MANIFEST.json",
    )
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["managed_by"] != MANAGED_BY:
        raise FormalMarkdownError("MANIFEST.json schema/owner is invalid")
    if manifest["generation_id"] != expected_id:
        raise FormalMarkdownError("MANIFEST.json generation id is invalid")
    if manifest["state_hash"] != expected_id.removeprefix("gen-"):
        raise FormalMarkdownError("MANIFEST.json state hash is invalid")
    if not isinstance(manifest["files"], Mapping):
        raise FormalMarkdownError("MANIFEST.json files must be an object")
    for path, record in manifest["files"].items():
        _safe_relative_path(path)
        if not isinstance(record, Mapping):
            raise FormalMarkdownError("MANIFEST.json contains an invalid file record")
        _exact_keys(record, {"sha256", "size_bytes", "kind"}, f"manifest.{path}")
        if not _SHA256.fullmatch(str(record["sha256"])):
            raise FormalMarkdownError("MANIFEST.json contains an invalid file hash")
        if (
            isinstance(record["size_bytes"], bool)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 0
        ):
            raise FormalMarkdownError("MANIFEST.json contains an invalid file size")
        if record["kind"] not in {"index", "topic", "conflict_view"}:
            raise FormalMarkdownError("MANIFEST.json contains an invalid file kind")
    return manifest, raw


def _verified_manifest_file(
    generation: Path, manifest: Mapping[str, Any], relative: str
) -> bytes:
    record = manifest["files"].get(relative)
    if not isinstance(record, Mapping):
        raise FormalMarkdownError(f"manifest does not cover {relative}")
    maximum = MAX_INDEX_BYTES if relative == "MEMORY.md" else MAX_TOPIC_BYTES
    raw = _read_file(generation.joinpath(*PurePosixPath(relative).parts), maximum=maximum)
    if len(raw) != record["size_bytes"] or _sha256_bytes(raw) != record["sha256"]:
        raise FormalMarkdownError(f"managed file hash/size mismatch: {relative}")
    return raw


def _load_index(memory_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    pointer, generation = _load_current(memory_root)
    manifest, _ = _load_manifest(
        generation,
        expected_id=pointer["generation_id"],
        expected_hash=pointer["manifest_sha256"],
    )
    if manifest["state_hash"] != pointer["state_hash"]:
        raise FormalMarkdownError("CURRENT and manifest state hashes disagree")
    index_raw = _verified_manifest_file(generation, manifest, "MEMORY.md")
    try:
        index, body = _parse_front_matter(index_raw.decode("utf-8"), source="MEMORY.md")
    except UnicodeDecodeError as error:
        raise FormalMarkdownError("MEMORY.md is not UTF-8") from error
    expected_index_keys = {
        "schema",
        "managed_by",
        "generation_id",
        "state_hash",
        "generated_at",
        "topic_count",
        "memory_count",
        "open_conflict_count",
        "resolved_conflict_count",
        "topics",
        "conflict_views",
    }
    _exact_keys(index, expected_index_keys, "MEMORY.md")
    if (
        index["schema"] != INDEX_SCHEMA
        or index["managed_by"] != MANAGED_BY
        or index["generation_id"] != pointer["generation_id"]
        or index["state_hash"] != pointer["state_hash"]
    ):
        raise FormalMarkdownError("MEMORY.md identity is inconsistent")
    if not isinstance(index["topics"], list) or len(index["topics"]) > MAX_TOPICS:
        raise FormalMarkdownError("MEMORY.md topics are malformed or exceed the scan limit")
    if index["topic_count"] != len(index["topics"]):
        raise FormalMarkdownError("MEMORY.md topic count is invalid")
    paths: set[str] = set()
    for position, topic in enumerate(index["topics"]):
        if not isinstance(topic, Mapping):
            raise FormalMarkdownError("MEMORY.md topic entry is not an object")
        _exact_keys(topic, _TOPIC_INDEX_KEYS, f"MEMORY.md.topics[{position}]")
        path = str(topic["path"])
        _safe_relative_path(path)
        if path.startswith("conflicts/") or path in paths:
            raise FormalMarkdownError("MEMORY.md contains an invalid/duplicate topic path")
        paths.add(path)
        record = manifest["files"].get(path)
        if not isinstance(record, Mapping) or record["kind"] != "topic":
            raise FormalMarkdownError("MEMORY.md references an unmanifested topic")
        if topic["sha256"] != record["sha256"]:
            raise FormalMarkdownError("MEMORY.md topic hash disagrees with manifest")
        if not isinstance(topic["stable_keys"], list) or not isinstance(topic["triggers"], list):
            raise FormalMarkdownError("MEMORY.md topic routing fields are malformed")
        forbidden = {"entries", "statement", "semantic_value", "scope", "memory_id"}
        if forbidden & set(topic):
            raise FormalMarkdownError("MEMORY.md contains topic body data")
    if body != _render_index_body(index["topics"]):
        raise FormalMarkdownError("MEMORY.md body is not canonical")
    conflict_views = index["conflict_views"]
    if not isinstance(conflict_views, Mapping) or set(conflict_views) != {
        "conflicts/pending_conflicts.md",
        "conflicts/resolution_history.md",
    }:
        raise FormalMarkdownError("MEMORY.md conflict views are malformed")
    for path, digest in conflict_views.items():
        if manifest["files"].get(path, {}).get("sha256") != digest:
            raise FormalMarkdownError("MEMORY.md conflict hash disagrees with manifest")
    _assert_no_secrets(index)
    return index, manifest, generation


def scan_formal_memory_index(memory_root: Path) -> dict[str, Any]:
    """Verify CURRENT/manifest/index and return at most 200 routing headers."""

    index, manifest, _generation = _load_index(Path(memory_root))
    return {
        "generation_id": index["generation_id"],
        "state_hash": index["state_hash"],
        "generated_at": index["generated_at"],
        "topic_count": index["topic_count"],
        "memory_count": index["memory_count"],
        "open_conflict_count": index["open_conflict_count"],
        "resolved_conflict_count": index["resolved_conflict_count"],
        "topics": index["topics"],
        "manifest_file_count": len(manifest["files"]),
    }


def load_formal_topic_bodies(
    memory_root: Path, selected_topic_ids: Iterable[str]
) -> dict[str, Any]:
    """Load 1-5 router-selected topic bodies after manifest verification.

    This is the update-router boundary: it exposes normalized formal entries
    for comparison, including non-recallable engineering entries that a Writer
    may need to revise, while never exposing raw evidence quotes.
    """

    topic_ids = list(selected_topic_ids)
    if (
        not topic_ids
        or len(topic_ids) > 5
        or any(not isinstance(item, str) or not item for item in topic_ids)
        or len(set(topic_ids)) != len(topic_ids)
    ):
        raise ValueError("selected_topic_ids must contain 1-5 unique topic ids")
    index, manifest, generation = _load_index(Path(memory_root))
    by_id = {str(topic["topic_id"]): topic for topic in index["topics"]}
    unknown = [item for item in topic_ids if item not in by_id]
    if unknown:
        raise FormalMarkdownError(f"router selected unknown topic ids: {sorted(unknown)}")
    topics = [_load_topic(generation, manifest, by_id[item]) for item in topic_ids]
    return {
        "generation_id": index["generation_id"],
        "state_hash": index["state_hash"],
        "selected_topic_ids": topic_ids,
        "topic_bodies_loaded": len(topics),
        "topics": topics,
        "integrity": "current_manifest_index_and_selected_files_verified",
    }


def _validate_entry(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalMarkdownError(f"{source} must be an object")
    entry = dict(value)
    _exact_keys(entry, _ENTRY_KEYS, source)
    for key in (
        "memory_id",
        "revision_id",
        "stable_key",
        "family",
        "memory_type",
        "semantic_value",
        "statement",
        "verification_status",
        "updated_at",
    ):
        entry[key] = _clean_text(entry[key], f"{source}.{key}", 2000)
    entry["scope"] = _parse_json_object(entry["scope"], f"{source}.scope")
    entry["activation_triggers"] = _parse_json_list(
        entry["activation_triggers"], f"{source}.activation_triggers"
    )
    for key in ("revision_number", "state_version", "evidence_count"):
        if isinstance(entry[key], bool) or not isinstance(entry[key], int) or entry[key] < 0:
            raise FormalMarkdownError(f"{source}.{key} must be non-negative")
    if entry["revision_number"] < 1 or entry["status"] != "active":
        raise FormalMarkdownError(f"{source} is not an active current revision")
    _assert_no_secrets(entry)
    return entry


def _load_topic(
    generation: Path, manifest: Mapping[str, Any], topic_index: Mapping[str, Any]
) -> dict[str, Any]:
    path = str(topic_index["path"])
    raw = _verified_manifest_file(generation, manifest, path)
    try:
        header, body = _parse_front_matter(raw.decode("utf-8"), source=path)
    except UnicodeDecodeError as error:
        raise FormalMarkdownError(f"{path} is not UTF-8") from error
    expected = {
        "schema",
        "managed_by",
        "generation_id",
        *_TOPIC_SPEC_KEYS,
        "status",
        "updated_at",
        "stable_keys",
        "triggers",
        "entries",
    }
    _exact_keys(header, expected, path)
    if header["schema"] != TOPIC_SCHEMA or header["managed_by"] != MANAGED_BY:
        raise FormalMarkdownError(f"{path} schema/owner is invalid")
    for key in _TOPIC_SPEC_KEYS | {"status", "updated_at", "stable_keys", "triggers"}:
        if header[key] != topic_index[key]:
            raise FormalMarkdownError(f"{path} disagrees with MEMORY.md: {key}")
    if not isinstance(header["entries"], list) or not header["entries"]:
        raise FormalMarkdownError(f"{path} has no entries")
    header["entries"] = [
        _validate_entry(item, f"{path}.entries[{index}]")
        for index, item in enumerate(header["entries"])
    ]
    ordered = sorted(
        header["entries"],
        key=lambda item: (
            item["stable_key"],
            _canonical_json(item["scope"]),
            item["memory_id"],
        ),
    )
    if header["entries"] != ordered:
        raise FormalMarkdownError(f"{path} entries are not deterministically ordered")
    if sorted({item["stable_key"] for item in ordered}) != header["stable_keys"]:
        raise FormalMarkdownError(f"{path} stable keys are invalid")
    if len(ordered) != topic_index["memory_count"]:
        raise FormalMarkdownError(f"{path} memory count is invalid")
    if body != _render_topic(header).removeprefix(_front_matter(header)):
        raise FormalMarkdownError(f"{path} body is not canonical")
    return header


def _load_conflict_view(
    generation: Path, manifest: Mapping[str, Any], relative: str
) -> dict[str, Any]:
    raw = _verified_manifest_file(generation, manifest, relative)
    try:
        header, body = _parse_front_matter(raw.decode("utf-8"), source=relative)
    except UnicodeDecodeError as error:
        raise FormalMarkdownError(f"{relative} is not UTF-8") from error
    if relative.endswith("pending_conflicts.md"):
        expected = {"schema", "managed_by", "generation_id", "conflict_count", "conflicts"}
        _exact_keys(header, expected, relative)
        if header["schema"] != PENDING_SCHEMA or not isinstance(header["conflicts"], list):
            raise FormalMarkdownError("pending conflict view is malformed")
        if header["conflict_count"] != len(header["conflicts"]):
            raise FormalMarkdownError("pending conflict count is invalid")
        for index, item in enumerate(header["conflicts"]):
            if not isinstance(item, Mapping):
                raise FormalMarkdownError("pending conflict entry is malformed")
            _exact_keys(item, _PENDING_KEYS, f"pending[{index}]")
        if body != _render_pending(header).removeprefix(_front_matter(header)):
            raise FormalMarkdownError("pending conflict body is not canonical")
    else:
        expected = {"schema", "managed_by", "generation_id", "resolution_count", "resolutions"}
        _exact_keys(header, expected, relative)
        if header["schema"] != HISTORY_SCHEMA or not isinstance(header["resolutions"], list):
            raise FormalMarkdownError("resolution history is malformed")
        if header["resolution_count"] != len(header["resolutions"]):
            raise FormalMarkdownError("resolution history count is invalid")
        for index, item in enumerate(header["resolutions"]):
            if not isinstance(item, Mapping):
                raise FormalMarkdownError("resolution history entry is malformed")
            _exact_keys(item, _HISTORY_KEYS, f"history[{index}]")
        if body != _render_history(header).removeprefix(_front_matter(header)):
            raise FormalMarkdownError("resolution history body is not canonical")
    _assert_no_secrets(header)
    return header


def _projection_state_from_loaded(
    index: Mapping[str, Any], topics: list[Mapping[str, Any]], pending: Mapping[str, Any], history: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "generated_at": index["generated_at"],
        "topics": [
            {
                "spec": {key: topic[key] for key in _TOPIC_SPEC_KEYS},
                "updated_at": topic["updated_at"],
                "entries": topic["entries"],
            }
            for topic in topics
        ],
        "pending_conflicts": pending["conflicts"],
        "resolution_history": history["resolutions"],
    }


def validate_formal_memory(
    memory_root: Path, generation_id: str | None = None
) -> dict[str, Any]:
    """Fully validate hashes, schemas, canonical bodies, and semantic state."""

    root = Path(memory_root).resolve()
    if generation_id is None:
        pointer, generation = _load_current(root)
        manifest, _ = _load_manifest(
            generation,
            expected_id=pointer["generation_id"],
            expected_hash=pointer["manifest_sha256"],
        )
        expected_state_hash = pointer["state_hash"]
    else:
        if not _GENERATION.fullmatch(generation_id):
            raise FormalMarkdownError("generation id is unsafe")
        generation = root / "generations" / generation_id
        _assert_not_link(generation)
        if not generation.is_dir():
            raise FormalMarkdownError("generation directory is missing")
        manifest, _ = _load_manifest(generation, expected_id=generation_id)
        expected_state_hash = generation_id.removeprefix("gen-")

    actual_paths: set[str] = set()
    for path in generation.rglob("*"):
        _assert_not_link(path)
        if path.is_file():
            actual_paths.add(path.relative_to(generation).as_posix())
        elif not path.is_dir():
            raise FormalMarkdownError("generation contains a non-file entry")
    expected_paths = set(manifest["files"]) | {"MANIFEST.json"}
    if actual_paths != expected_paths:
        raise FormalMarkdownError("generation file set disagrees with manifest")
    for relative in manifest["files"]:
        _verified_manifest_file(generation, manifest, relative)

    index_raw = _verified_manifest_file(generation, manifest, "MEMORY.md")
    index, _ = _parse_front_matter(index_raw.decode("utf-8"), source="MEMORY.md")
    # Re-use all strict index checks without depending on CURRENT when an old
    # generation is being audited.
    if generation_id is None:
        strict_index, _, _ = _load_index(root)
        index = strict_index
    else:
        if index.get("generation_id") != generation_id or index.get("state_hash") != expected_state_hash:
            raise FormalMarkdownError("old generation index identity is invalid")
        if not isinstance(index.get("topics"), list) or len(index["topics"]) > MAX_TOPICS:
            raise FormalMarkdownError("old generation index topics are malformed")
        if index.get("topic_count") != len(index["topics"]):
            raise FormalMarkdownError("old generation topic count is invalid")
        if _parse_front_matter(index_raw.decode("utf-8"), source="MEMORY.md")[1] != _render_index_body(index["topics"]):
            raise FormalMarkdownError("old generation index body is not canonical")

    topics = [_load_topic(generation, manifest, topic) for topic in index["topics"]]
    pending = _load_conflict_view(
        generation, manifest, "conflicts/pending_conflicts.md"
    )
    history = _load_conflict_view(
        generation, manifest, "conflicts/resolution_history.md"
    )
    if index["memory_count"] != sum(len(topic["entries"]) for topic in topics):
        raise FormalMarkdownError("MEMORY.md memory count is invalid")
    if index["open_conflict_count"] != pending["conflict_count"]:
        raise FormalMarkdownError("MEMORY.md open conflict count is invalid")
    if index["resolved_conflict_count"] != history["resolution_count"]:
        raise FormalMarkdownError("MEMORY.md resolved conflict count is invalid")
    state = _projection_state_from_loaded(index, topics, pending, history)
    if _sha256_object(state) != expected_state_hash or manifest["state_hash"] != expected_state_hash:
        raise FormalMarkdownError("generation semantic state hash is invalid")
    return {
        "status": "valid",
        "generation_id": index["generation_id"],
        "state_hash": expected_state_hash,
        "generated_at": index["generated_at"],
        "topic_count": len(topics),
        "memory_count": sum(len(topic["entries"]) for topic in topics),
        "open_conflict_count": pending["conflict_count"],
        "resolved_conflict_count": history["resolution_count"],
        "verified_file_count": len(expected_paths),
        "topics": topics,
        "pending_conflicts": pending["conflicts"],
        "resolution_history": history["resolutions"],
    }


def _write_new_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_current(root: Path, pointer: Mapping[str, Any]) -> None:
    temporary = root / f".CURRENT-{uuid4().hex}.tmp"
    try:
        _write_new_file(temporary, _json_bytes(pointer))
        os.replace(temporary, root / "CURRENT")
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _publication_lock(root: Path):
    lock = root / ".publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FormalMarkdownError("another formal publication is in progress") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if lock.exists():
            lock.unlink()


def publish_formal_memory(lab_database: Path, memory_root: Path) -> dict[str, Any]:
    """Publish the current formal DB state and atomically advance CURRENT.

    Re-publishing an identical state is a no-op.  Before CURRENT is switched,
    the database is sampled a second time; a changed state fails closed rather
    than exposing a stale generation.
    """

    database = Path(lab_database)
    root = Path(memory_root).resolve()
    if root.exists():
        _assert_not_link(root)
        if not root.is_dir():
            raise FormalMarkdownError("memory root is not a directory")
    else:
        root.mkdir(parents=True)
    generations = root / "generations"
    if generations.exists():
        _assert_not_link(generations)
        if not generations.is_dir():
            raise FormalMarkdownError("generations path is not a directory")
    else:
        generations.mkdir()

    with _publication_lock(root):
        snapshot = snapshot_formal_database(database)
        files = render_formal_generation(snapshot)
        generation_id = snapshot["generation_id"]
        target = generations / generation_id
        created = False
        staging: Path | None = None
        if target.exists():
            validate_formal_memory(root, generation_id=generation_id)
        else:
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations))
            try:
                for relative, content in files.items():
                    safe = _safe_relative_path(relative)
                    _write_new_file(staging.joinpath(*safe.parts), content)
                # Audit the exact bytes before making the directory visible.
                staged_manifest, _ = _load_manifest(
                    staging, expected_id=generation_id
                )
                for relative in staged_manifest["files"]:
                    _verified_manifest_file(staging, staged_manifest, relative)
                os.replace(staging, target)
                staging = None
                created = True
            finally:
                if staging is not None and staging.exists():
                    shutil.rmtree(staging)

        second = snapshot_formal_database(database)
        if second["state_hash"] != snapshot["state_hash"]:
            raise FormalMarkdownError("formal database changed during publication")

        previous: str | None = None
        current_path = root / "CURRENT"
        if current_path.exists():
            pointer, _ = _load_current(root)
            previous = str(pointer["generation_id"])
        manifest_bytes = files["MANIFEST.json"]
        pointer = {
            "schema": CURRENT_SCHEMA,
            "generation_id": generation_id,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "state_hash": snapshot["state_hash"],
        }
        changed = previous != generation_id
        if changed:
            _replace_current(root, pointer)
        validated = validate_formal_memory(root)
        return {
            "status": "published",
            "generation_id": generation_id,
            "previous_generation_id": previous,
            "created_generation": created,
            "current_changed": changed,
            "state_hash": snapshot["state_hash"],
            "topic_count": validated["topic_count"],
            "memory_count": validated["memory_count"],
            "open_conflict_count": validated["open_conflict_count"],
            "resolved_conflict_count": validated["resolved_conflict_count"],
            "verified_file_count": validated["verified_file_count"],
        }


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


def _features(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    latin = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    chinese = {
        run[index : index + 2]
        for run in chinese_runs
        for index in range(max(0, len(run) - 1))
    }
    return latin | chinese


def _topic_relevance(topic: Mapping[str, Any], query: str) -> tuple[float, list[str]]:
    if topic["type"] == "user" and any(
        str(key).startswith("preference.response.") for key in topic["stable_keys"]
    ):
        return 0.35, ["always_global_response_preference"]
    normalized_query = _normalized_text(query)
    matched = [
        trigger
        for trigger in topic["triggers"]
        if len(_normalized_text(str(trigger))) >= 2
        and _normalized_text(str(trigger)) in normalized_query
    ]
    query_features = _features(query)
    document_features = _features(
        " ".join(
            [
                str(topic["name"]),
                str(topic["description"]),
                *[str(item).replace(".", " ") for item in topic["stable_keys"]],
                *[str(item) for item in topic["triggers"]],
            ]
        )
    )
    overlap = len(query_features & document_features)
    lexical = overlap / math.sqrt(max(1, len(query_features) * len(document_features)))
    score = min(1.0, 0.7 * min(1.0, len(matched) / 2) + 0.6 * lexical)
    if not matched and overlap < 2:
        score = 0.0
    reasons = [f"trigger:{item}" for item in matched]
    if overlap:
        reasons.append(f"lexical_overlap:{overlap}")
    return round(score, 6), reasons


def _scope_state(
    scope: Mapping[str, Any],
    *,
    project_key: str | None,
    domain: str | None,
    conditions: Mapping[str, str] | None,
) -> tuple[str, str | None, int]:
    level = scope["level"]
    if level == "global":
        state, reason, rank = "applicable", None, 0
    elif level == "domain":
        rank = 100
        if domain is None:
            state, reason = "unknown", "domain_context_required"
        elif str(scope.get("domain")).casefold() == domain.casefold():
            state, reason = "applicable", None
        else:
            state, reason = "inapplicable", "different_domain"
    else:
        rank = 200
        if project_key is None:
            state, reason = "unknown", "project_context_required"
        elif scope.get("project_key") == project_key:
            state, reason = "applicable", None
        else:
            state, reason = "inapplicable", "different_project"
    required = scope.get("conditions") or {}
    rank += len(required)
    if state == "applicable" and required:
        if conditions is None or any(key not in conditions for key in required):
            return "unknown", "typed_conditions_required", rank
        if any(str(conditions[key]).casefold() != str(value).casefold() for key, value in required.items()):
            return "inapplicable", "condition_mismatch", rank
    return state, reason, rank


def recall_formal_memory(
    memory_root: Path,
    query: str,
    *,
    project_key: str | None = None,
    domain: str | None = None,
    conditions: Mapping[str, str] | None = None,
    topic_limit: int = 5,
    memory_limit: int = 5,
    _selected_topic_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Route over index summaries, then hash-check/read only selected topics."""

    query = _clean_text(query, "query", MAX_QUERY_CHARS)
    if topic_limit < 1 or topic_limit > 5:
        raise ValueError("topic_limit must be between 1 and 5")
    if memory_limit < 1 or memory_limit > 20:
        raise ValueError("memory_limit must be between 1 and 20")
    index, manifest, generation = _load_index(Path(memory_root))
    if _selected_topic_ids is None:
        ranked_topics = []
        for topic in index["topics"]:
            score, reasons = _topic_relevance(topic, query)
            if score > 0:
                ranked_topics.append((score, str(topic["topic_id"]), reasons, topic))
        ranked_topics.sort(key=lambda item: (-item[0], item[1]))
        chosen = ranked_topics[:topic_limit]
    else:
        selected_topic_ids = list(_selected_topic_ids)
        if (
            not selected_topic_ids
            or len(selected_topic_ids) > 5
            or any(not isinstance(item, str) or not item for item in selected_topic_ids)
            or len(set(selected_topic_ids)) != len(selected_topic_ids)
        ):
            raise ValueError("selected_topic_ids must contain 1-5 unique topic ids")
        by_id = {str(topic["topic_id"]): topic for topic in index["topics"]}
        unknown = [item for item in selected_topic_ids if item not in by_id]
        if unknown:
            raise FormalMarkdownError(
                f"router selected unknown topic ids: {sorted(unknown)}"
            )
        chosen = [
            (1.0, topic_id, ["external_router_selection"], by_id[topic_id])
            for topic_id in selected_topic_ids
        ]
    loaded_topics = [_load_topic(generation, manifest, item[3]) for item in chosen]

    pending_header = _load_conflict_view(
        generation, manifest, "conflicts/pending_conflicts.md"
    )
    blocked_memory_ids = {item["memory_id"] for item in pending_header["conflicts"]}
    candidates_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    excluded_by_verification: list[dict[str, Any]] = []
    unresolved_context: list[dict[str, Any]] = []
    for (topic_score, _topic_id, reasons, _), topic in zip(chosen, loaded_topics):
        for entry in topic["entries"]:
            if entry["memory_id"] in blocked_memory_ids:
                continue
            is_engineering = (
                "engineering" in entry["family"].casefold()
                or "engineering" in entry["memory_type"].casefold()
                or entry["stable_key"].startswith("engineering.")
            )
            if (
                is_engineering
                and entry["verification_status"]
                not in ENGINEERING_RECALLABLE_VERIFICATION
            ):
                excluded_by_verification.append(
                    {
                        "memory_id": entry["memory_id"],
                        "revision_id": entry["revision_id"],
                        "stable_key": entry["stable_key"],
                        "verification_status": entry["verification_status"],
                        "reason": "engineering_memory_not_verified",
                    }
                )
                continue
            state, reason, rank = _scope_state(
                entry["scope"],
                project_key=project_key,
                domain=domain,
                conditions=conditions,
            )
            record = {
                "entry": entry,
                "state": state,
                "reason": reason,
                "scope_rank": rank,
                "score": topic_score,
                "match_reasons": reasons,
                "topic_id": topic["topic_id"],
            }
            candidates_by_key[entry["stable_key"]].append(record)
            if state == "inapplicable":
                excluded.append(
                    {
                        "memory_id": entry["memory_id"],
                        "stable_key": entry["stable_key"],
                        "scope": entry["scope"],
                        "reason": reason,
                    }
                )

    selected: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for stable_key, records in sorted(candidates_by_key.items()):
        applicable = [item for item in records if item["state"] == "applicable"]
        unknown = [item for item in records if item["state"] == "unknown"]
        highest_applicable = max((item["scope_rank"] for item in applicable), default=-1)
        blockers = [item for item in unknown if item["scope_rank"] >= highest_applicable]
        if blockers:
            unresolved_context.append(
                {
                    "stable_key": stable_key,
                    "reason": "more_specific_scope_context_is_unknown",
                    "scopes": [item["entry"]["scope"] for item in blockers],
                }
            )
            continue
        if highest_applicable < 0:
            continue
        winners = [item for item in applicable if item["scope_rank"] == highest_applicable]
        for winner in winners:
            entry = winner["entry"]
            selected.append(
                {
                    "memory_id": entry["memory_id"],
                    "revision_id": entry["revision_id"],
                    "stable_key": stable_key,
                    "memory_type": entry["memory_type"],
                    "statement": entry["statement"],
                    "scope": entry["scope"],
                    "verification_status": entry["verification_status"],
                    "evidence_count": entry["evidence_count"],
                    "score": winner["score"],
                    "match_reasons": winner["match_reasons"],
                    "topic_id": winner["topic_id"],
                }
            )
        for item in applicable:
            if item not in winners:
                suppressed.append(
                    {
                        "memory_id": item["entry"]["memory_id"],
                        "stable_key": stable_key,
                        "scope": item["entry"]["scope"],
                        "reason": "more_specific_scope_wins",
                    }
                )
    selected.sort(
        key=lambda item: (
            -float(item["score"]),
            item["stable_key"],
            item["memory_id"],
        )
    )

    relevant_conflicts = []
    relevant_keys = {
        key
        for _, _, _, topic in chosen
        for key in topic["stable_keys"]
    }
    for conflict in pending_header["conflicts"]:
        state, reason, _ = _scope_state(
            conflict["scope"],
            project_key=project_key,
            domain=domain,
            conditions=conditions,
        )
        if conflict["stable_key"] in relevant_keys and state != "inapplicable":
            relevant_conflicts.append(
                {
                    "conflict_id": conflict["conflict_id"],
                    "memory_id": conflict["memory_id"],
                    "stable_key": conflict["stable_key"],
                    "scope": conflict["scope"],
                    "status": "blocked",
                    "context_state": state,
                    "context_reason": reason,
                }
            )
    return {
        "recall_version": RECALL_VERSION,
        "generation_id": index["generation_id"],
        "state_hash": index["state_hash"],
        "query": query,
        "context": {
            "project_key": project_key,
            "domain": domain,
            "conditions": dict(conditions) if conditions is not None else None,
        },
        "topic_headers_scanned": len(index["topics"]),
        "topic_bodies_loaded": len(loaded_topics),
        "selected_topics": [
            {
                "topic_id": item[3]["topic_id"],
                "path": item[3]["path"],
                "score": item[0],
                "match_reasons": item[2],
            }
            for item in chosen
        ],
        "selected_count": min(len(selected), memory_limit),
        "selected": selected[:memory_limit],
        "suppressed_by_scope": suppressed,
        "excluded_by_context": excluded,
        "excluded_by_verification": excluded_by_verification,
        "unresolved_context": unresolved_context,
        "unresolved_conflicts": relevant_conflicts,
        "integrity": "current_manifest_index_and_selected_files_verified",
    }


def recall_selected_formal_topics(
    memory_root: Path,
    query: str,
    selected_topic_ids: Iterable[str],
    *,
    project_key: str | None = None,
    domain: str | None = None,
    conditions: Mapping[str, str] | None = None,
    memory_limit: int = 5,
) -> dict[str, Any]:
    """Recall from 1-5 topic IDs chosen by an external/LLM router.

    Topic IDs are validated against the current hashed index, and all normal
    scope, conflict, verification, and selected-file integrity gates still
    apply.  A hallucinated topic ID therefore fails closed.
    """

    return recall_formal_memory(
        memory_root,
        query,
        project_key=project_key,
        domain=domain,
        conditions=conditions,
        topic_limit=5,
        memory_limit=memory_limit,
        _selected_topic_ids=selected_topic_ids,
    )
