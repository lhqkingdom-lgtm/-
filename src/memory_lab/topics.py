"""Deterministic topic ownership and semantics-preserving maintenance.

Topics are a routing/presentation layer.  They never own the memory value: a
stable key has exactly one active topic owner, while the formal memory,
revision, evidence and conflict tables remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from .store import now_iso


TOPIC_SCHEMA_VERSION = "memory-topic-schema-v1"
TOPIC_PLAN_VERSION = "memory-topic-plan-v1"
_SLUG = re.compile(r"^[a-z][a-z0-9_]{1,79}$")


TOPIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic (
    topic_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    family TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'retired')),
    state_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_membership (
    stable_key TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES topic(topic_id)
);

CREATE TABLE IF NOT EXISTS topic_maintenance_run (
    run_id TEXT PRIMARY KEY,
    plan_hash TEXT NOT NULL UNIQUE,
    baseline_memory_hash TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('complete', 'failed', 'noop')),
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    error TEXT
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any, prefix: str = "") -> str:
    return prefix + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def ensure_topic_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(TOPIC_SCHEMA)


def formal_semantic_hash(connection: sqlite3.Connection) -> str:
    """Hash formal semantics, excluding topic-only presentation state."""

    projection: dict[str, list[dict[str, Any]]] = {}
    for table, order in (
        ("memory", "memory_id"),
        ("memory_revision", "revision_id"),
        ("evidence", "evidence_id"),
        ("conflict", "conflict_id"),
    ):
        projection[table] = [
            dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}")
        ]
    return _hash(projection, "formal_")


def _default_topic(
    stable_key: str, family: str, summary: str | None = None
) -> dict[str, str]:
    summary = (summary or "").strip()[:500]
    if stable_key.startswith("preference.response."):
        return {
            "slug": "user_response_preferences",
            "name": "用户回复偏好",
            "description": "跨会话生效的回复语言、结构和表达偏好",
            "family": "user_contract",
        }
    if stable_key.startswith("workflow.code_generation."):
        return {
            "slug": "workflow_code_generation",
            "name": "代码生成流程",
            "description": "生成或修改正式代码时应遵循的用户流程",
            "family": "user_contract",
        }
    if stable_key.startswith("workflow.embedded."):
        return {
            "slug": "workflow_embedded_compile",
            "name": "嵌入式编译流程",
            "description": "嵌入式代码修改后的编译和验证要求",
            "family": "user_contract",
        }
    if family == "engineering_memory":
        component = re.sub(r"[^a-z0-9]+", "_", stable_key.split(".")[1]).strip("_")
        component = component[:50] or "general"
        return {
            "slug": f"engineering_{component}",
            "name": f"工程经验：{component}",
            "description": summary or "工程故障、解决方案和可复用流程",
            "family": "engineering_memory",
        }
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:10]
    return {
        "slug": f"memory_{digest}",
        "name": stable_key,
        "description": summary or f"与 {stable_key} 相关的长期记忆",
        "family": family or "user_contract",
    }


def _upsert_topic(connection: sqlite3.Connection, spec: Mapping[str, str]) -> str:
    slug = spec["slug"]
    if not _SLUG.fullmatch(slug):
        raise ValueError(f"invalid topic slug: {slug}")
    existing = connection.execute("SELECT topic_id FROM topic WHERE slug = ?", (slug,)).fetchone()
    if existing is not None:
        return str(existing[0])
    topic_id = "topic_" + uuid4().hex
    timestamp = now_iso()
    connection.execute(
        """
        INSERT INTO topic(topic_id, slug, name, description, family, status,
                          state_version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'active', 1, ?, ?)
        """,
        (
            topic_id,
            slug,
            spec["name"].strip(),
            spec["description"].strip(),
            spec["family"].strip(),
            timestamp,
            timestamp,
        ),
    )
    return topic_id


def ensure_topic_assignments(connection: sqlite3.Connection) -> dict[str, int]:
    """Assign every formal stable key once without moving existing owners."""

    ensure_topic_schema(connection)
    created_topics_before = connection.execute("SELECT COUNT(*) FROM topic").fetchone()[0]
    memberships_before = connection.execute("SELECT COUNT(*) FROM topic_membership").fetchone()[0]
    rows = connection.execute(
        """
        SELECT stable_key, MIN(COALESCE(family, 'user_contract')) AS family
        FROM memory GROUP BY stable_key ORDER BY stable_key
        """
    ).fetchall()
    timestamp = now_iso()
    for row in rows:
        stable_key = str(row[0])
        if connection.execute(
            "SELECT 1 FROM topic_membership WHERE stable_key = ?", (stable_key,)
        ).fetchone():
            continue
        summary = ""
        revision = connection.execute(
            """
            SELECT r.activation_triggers_json
            FROM memory AS m
            JOIN memory_revision AS r ON r.revision_id = m.current_revision_id
            WHERE m.stable_key = ? AND m.status = 'active' AND r.status = 'current'
            ORDER BY m.updated_at DESC, m.memory_id
            LIMIT 1
            """,
            (stable_key,),
        ).fetchone()
        if revision is not None:
            try:
                triggers = json.loads(revision[0])
            except (TypeError, json.JSONDecodeError):
                triggers = []
            if isinstance(triggers, list) and triggers and isinstance(triggers[0], str):
                summary = triggers[0]
        topic_id = _upsert_topic(
            connection, _default_topic(stable_key, str(row[1]), summary)
        )
        connection.execute(
            """
            INSERT INTO topic_membership(stable_key, topic_id, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (stable_key, topic_id, timestamp, timestamp),
        )
    return {
        "topics_created": connection.execute("SELECT COUNT(*) FROM topic").fetchone()[0]
        - created_topics_before,
        "memberships_created": connection.execute(
            "SELECT COUNT(*) FROM topic_membership"
        ).fetchone()[0]
        - memberships_before,
    }


def topic_catalog(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    ensure_topic_assignments(connection)
    topics: list[dict[str, Any]] = []
    for row in connection.execute(
        """
        SELECT * FROM topic WHERE status = 'active' ORDER BY updated_at DESC, slug
        """
    ):
        keys = [
            item[0]
            for item in connection.execute(
                "SELECT stable_key FROM topic_membership WHERE topic_id = ? ORDER BY stable_key",
                (row["topic_id"],),
            )
        ]
        if keys:
            topics.append({**dict(row), "stable_keys": keys})
    return topics


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    allowed_root = {"schema_version", "baseline_memory_hash", "actions"}
    if set(plan) != allowed_root or plan.get("schema_version") != TOPIC_PLAN_VERSION:
        raise ValueError("invalid topic maintenance plan envelope")
    baseline = plan.get("baseline_memory_hash")
    actions = plan.get("actions")
    if not isinstance(baseline, str) or not baseline or not isinstance(actions, list):
        raise ValueError("invalid topic maintenance plan values")
    normalized: list[dict[str, Any]] = []
    for raw in actions:
        if not isinstance(raw, Mapping):
            raise ValueError("topic action must be an object")
        action = dict(raw)
        kind = action.get("kind")
        if kind in {"rename_topic", "update_summary"}:
            expected = {"kind", "topic_slug", "value"}
        elif kind == "merge_topics":
            expected = {"kind", "source_slugs", "target"}
        elif kind == "split_topic":
            expected = {"kind", "source_slug", "targets"}
        else:
            raise ValueError(f"unsupported topic action: {kind}")
        if set(action) != expected:
            raise ValueError(f"invalid fields for topic action {kind}")
        normalized.append(action)
    return {
        "schema_version": TOPIC_PLAN_VERSION,
        "baseline_memory_hash": baseline,
        "actions": normalized,
    }


def _topic_by_slug(connection: sqlite3.Connection, slug: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM topic WHERE slug = ? AND status = 'active'", (slug,)
    ).fetchone()
    if row is None:
        raise ValueError(f"active topic not found: {slug}")
    return row


def _target_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("topic target must be an object")
    result = dict(value)
    if set(result) != {"slug", "name", "description", "family", "stable_keys"}:
        raise ValueError("invalid topic target fields")
    if not _SLUG.fullmatch(str(result["slug"])):
        raise ValueError("invalid target slug")
    if not isinstance(result["stable_keys"], list) or any(
        not isinstance(item, str) or not item for item in result["stable_keys"]
    ):
        raise ValueError("target stable_keys must be non-empty strings")
    return result


def _move_keys(
    connection: sqlite3.Connection, stable_keys: Sequence[str], topic_id: str
) -> None:
    known = {
        row[0]
        for row in connection.execute("SELECT DISTINCT stable_key FROM memory")
    }
    if not set(stable_keys) <= known:
        raise ValueError("topic plan references an unknown stable key")
    for stable_key in stable_keys:
        cursor = connection.execute(
            """
            UPDATE topic_membership SET topic_id = ?, updated_at = ?
            WHERE stable_key = ?
            """,
            (topic_id, now_iso(), stable_key),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"stable key has no unique owner: {stable_key}")


def apply_topic_plan(connection: sqlite3.Connection, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a reviewed Curator plan while proving formal semantics are unchanged."""

    ensure_topic_assignments(connection)
    normalized = _validate_plan(plan)
    before_hash = formal_semantic_hash(connection)
    if normalized["baseline_memory_hash"] != before_hash:
        raise ValueError("topic plan baseline is stale")
    plan_hash = _hash(normalized, "topic_plan_")
    prior = connection.execute(
        "SELECT status FROM topic_maintenance_run WHERE plan_hash = ?", (plan_hash,)
    ).fetchone()
    if prior is not None:
        return {"status": "noop", "plan_hash": plan_hash, "reason": "plan_already_applied"}

    run_id = "topicrun_" + uuid4().hex
    started = now_iso()
    actions_applied = 0
    try:
        with connection:
            for action in normalized["actions"]:
                kind = action["kind"]
                if kind in {"rename_topic", "update_summary"}:
                    topic = _topic_by_slug(connection, str(action["topic_slug"]))
                    value = str(action["value"]).strip()
                    if not value or len(value) > 500:
                        raise ValueError("topic text is empty or too long")
                    column = "name" if kind == "rename_topic" else "description"
                    connection.execute(
                        f"UPDATE topic SET {column} = ?, state_version = state_version + 1, "
                        "updated_at = ? WHERE topic_id = ?",
                        (value, now_iso(), topic["topic_id"]),
                    )
                elif kind == "merge_topics":
                    sources = action["source_slugs"]
                    if not isinstance(sources, list) or len(sources) < 2:
                        raise ValueError("merge_topics requires at least two source topics")
                    target = _target_spec(action["target"])
                    source_rows = [_topic_by_slug(connection, str(slug)) for slug in sources]
                    source_keys = {
                        row[0]
                        for source in source_rows
                        for row in connection.execute(
                            "SELECT stable_key FROM topic_membership WHERE topic_id = ?",
                            (source["topic_id"],),
                        )
                    }
                    if source_keys != set(target["stable_keys"]):
                        raise ValueError("merge target must preserve the exact source key set")
                    target_id = _upsert_topic(connection, target)
                    _move_keys(connection, sorted(source_keys), target_id)
                    for source in source_rows:
                        if source["topic_id"] != target_id:
                            connection.execute(
                                "UPDATE topic SET status = 'retired', state_version = state_version + 1, updated_at = ? WHERE topic_id = ?",
                                (now_iso(), source["topic_id"]),
                            )
                elif kind == "split_topic":
                    source = _topic_by_slug(connection, str(action["source_slug"]))
                    targets_raw = action["targets"]
                    if not isinstance(targets_raw, list) or len(targets_raw) < 2:
                        raise ValueError("split_topic requires at least two target topics")
                    targets = [_target_spec(item) for item in targets_raw]
                    source_keys = {
                        row[0]
                        for row in connection.execute(
                            "SELECT stable_key FROM topic_membership WHERE topic_id = ?",
                            (source["topic_id"],),
                        )
                    }
                    target_keys = [key for target in targets for key in target["stable_keys"]]
                    if set(target_keys) != source_keys or len(target_keys) != len(set(target_keys)):
                        raise ValueError("split targets must partition the exact source key set")
                    for target in targets:
                        target_id = _upsert_topic(connection, target)
                        _move_keys(connection, target["stable_keys"], target_id)
                    if all(target["slug"] != source["slug"] for target in targets):
                        connection.execute(
                            "UPDATE topic SET status = 'retired', state_version = state_version + 1, updated_at = ? WHERE topic_id = ?",
                            (now_iso(), source["topic_id"]),
                        )
                actions_applied += 1

            owned = connection.execute(
                "SELECT COUNT(DISTINCT stable_key), COUNT(*) FROM topic_membership"
            ).fetchone()
            formal_keys = connection.execute(
                "SELECT COUNT(DISTINCT stable_key) FROM memory"
            ).fetchone()[0]
            if owned[0] != formal_keys or owned[1] != formal_keys:
                raise ValueError("every formal stable key must have exactly one topic owner")
            after_hash = formal_semantic_hash(connection)
            if after_hash != before_hash:
                raise ValueError("topic maintenance changed formal memory semantics")
            completed = now_iso()
            status = "complete" if actions_applied else "noop"
            connection.execute(
                """
                INSERT INTO topic_maintenance_run(
                    run_id, plan_hash, baseline_memory_hash, plan_json, status,
                    started_at, completed_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (run_id, plan_hash, before_hash, _json(normalized), status, started, completed),
            )
        return {
            "status": status,
            "run_id": run_id,
            "plan_hash": plan_hash,
            "actions_applied": actions_applied,
            "formal_semantic_hash": before_hash,
        }
    except Exception:
        raise
