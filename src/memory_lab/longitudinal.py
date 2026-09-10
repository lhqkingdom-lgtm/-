"""Open longitudinal evaluation ledger for the standalone experiment.

This module does not decide whether a memory is correct.  It freezes external
expectations and records later observations so Writer/Reviewer agreement can
never be mistaken for ground truth.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .store import now_iso


LONGITUDINAL_SCHEMA_VERSION = "longitudinal-eval-v1"
DEFAULT_TARGETS = {
    "minimum_days": 14,
    "minimum_projects": 3,
    "minimum_root_sessions": 24,
    "minimum_completed_turns": 80,
    "recall_at_5": 0.90,
    "unrelated_empty_rate": 0.95,
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS longitudinal_run (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    visibility TEXT NOT NULL CHECK (visibility IN ('open', 'blind', 'hybrid')),
    status TEXT NOT NULL CHECK (status IN ('active', 'complete', 'stopped')),
    started_at TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    completed_at TEXT,
    targets_json TEXT NOT NULL,
    baseline_json TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_longitudinal_run
ON longitudinal_run(status) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS longitudinal_memory_feedback (
    feedback_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    worth_storing INTEGER NOT NULL CHECK (worth_storing IN (0, 1)),
    scope_correct INTEGER NOT NULL CHECK (scope_correct IN (0, 1)),
    future_useful INTEGER NOT NULL CHECK (future_useful IN (0, 1)),
    note TEXT,
    recorded_at TEXT NOT NULL,
    UNIQUE (run_id, memory_id, revision_id),
    FOREIGN KEY (run_id) REFERENCES longitudinal_run(run_id)
);

CREATE TABLE IF NOT EXISTS longitudinal_recall_case (
    case_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    query TEXT NOT NULL,
    context_json TEXT NOT NULL,
    expected_keys_json TEXT NOT NULL,
    expect_empty INTEGER NOT NULL CHECK (expect_empty IN (0, 1)),
    frozen_at TEXT NOT NULL,
    frozen_hash TEXT NOT NULL,
    actual_keys_json TEXT,
    evaluated_at TEXT,
    hit INTEGER,
    unexpected_count INTEGER,
    FOREIGN KEY (run_id) REFERENCES longitudinal_run(run_id)
);

CREATE TRIGGER IF NOT EXISTS longitudinal_recall_gold_no_update
BEFORE UPDATE OF query, context_json, expected_keys_json, expect_empty, frozen_at,
                 frozen_hash ON longitudinal_recall_case
BEGIN
    SELECT RAISE(ABORT, 'longitudinal recall gold is immutable');
END;
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)


def _counts(connection: sqlite3.Connection, started_at_ms: int = 0) -> dict[str, int]:
    projects = connection.execute(
        """
        SELECT COUNT(DISTINCT a.project_key)
        FROM conversation_turn AS t
        JOIN event_annotation AS a ON a.event_id = t.primary_event_id
        JOIN event AS e ON e.event_id = t.primary_event_id
        WHERE COALESCE(t.user_timestamp, e.timestamp, 0) >= ?
          AND t.current_state IN ('completed', 'failed')
        """,
        (started_at_ms,),
    ).fetchone()[0]
    sessions = connection.execute(
        """
        SELECT COUNT(DISTINCT t.session_id)
        FROM conversation_turn AS t
        JOIN event AS e ON e.event_id = t.primary_event_id
        WHERE COALESCE(t.user_timestamp, e.timestamp, 0) >= ?
          AND t.current_state IN ('completed', 'failed')
          AND (e.parent_session_id IS NULL OR TRIM(e.parent_session_id) = '')
        """,
        (started_at_ms,),
    ).fetchone()[0]
    turns = connection.execute(
        """
        SELECT COUNT(*) FROM conversation_turn AS t
        JOIN event AS e ON e.event_id = t.primary_event_id
        WHERE COALESCE(t.user_timestamp, e.timestamp, 0) >= ?
          AND t.current_state IN ('completed', 'failed')
          AND (e.parent_session_id IS NULL OR TRIM(e.parent_session_id) = '')
        """,
        (started_at_ms,),
    ).fetchone()[0]
    return {"projects": int(projects), "root_sessions": int(sessions), "completed_turns": int(turns)}


def start_run(
    connection: sqlite3.Connection,
    name: str,
    *,
    visibility: str = "open",
    targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_schema(connection)
    if visibility not in {"open", "blind", "hybrid"}:
        raise ValueError("unsupported longitudinal visibility")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("longitudinal run name is required")
    active = connection.execute(
        "SELECT run_id FROM longitudinal_run WHERE status = 'active'"
    ).fetchone()
    if active is not None:
        raise ValueError(f"an active longitudinal run already exists: {active[0]}")
    merged_targets = {**DEFAULT_TARGETS, **dict(targets or {})}
    started = datetime.now(timezone.utc)
    started_ms = int(started.timestamp() * 1000)
    baseline = _counts(connection, 0)
    baseline.update(
        {
            "memories": connection.execute("SELECT COUNT(*) FROM memory").fetchone()[0],
            "revisions": connection.execute("SELECT COUNT(*) FROM memory_revision").fetchone()[0],
            "evidence": connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
        }
    )
    run_id = "soak_" + uuid4().hex
    connection.execute(
        """
        INSERT INTO longitudinal_run(
            run_id, name, schema_version, visibility, status, started_at,
            started_at_ms, targets_json, baseline_json
        ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)
        """,
        (
            run_id,
            name.strip(),
            LONGITUDINAL_SCHEMA_VERSION,
            visibility,
            started.isoformat(timespec="milliseconds"),
            started_ms,
            _json(merged_targets),
            _json(baseline),
        ),
    )
    return {"run_id": run_id, "status": "active", "visibility": visibility, "targets": merged_targets, "baseline": baseline}


def freeze_recall_case(
    connection: sqlite3.Connection,
    run_id: str,
    query: str,
    context: Mapping[str, Any],
    expected_keys: Sequence[str],
    *,
    expect_empty: bool = False,
) -> dict[str, Any]:
    ensure_schema(connection)
    if not query.strip():
        raise ValueError("recall query is required")
    expected = sorted(set(expected_keys))
    if expect_empty and expected:
        raise ValueError("an empty recall case cannot carry expected keys")
    frozen = {
        "query": query,
        "context": dict(context),
        "expected_keys": expected,
        "expect_empty": bool(expect_empty),
    }
    case_id = "recallcase_" + uuid4().hex
    connection.execute(
        """
        INSERT INTO longitudinal_recall_case(
            case_id, run_id, query, context_json, expected_keys_json,
            expect_empty, frozen_at, frozen_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            run_id,
            query,
            _json(dict(context)),
            _json(expected),
            int(expect_empty),
            now_iso(),
            _hash(frozen),
        ),
    )
    return {"case_id": case_id, "frozen_hash": _hash(frozen)}


def evaluate_recall_case(
    connection: sqlite3.Connection, case_id: str, actual_keys: Sequence[str]
) -> dict[str, Any]:
    ensure_schema(connection)
    row = connection.execute(
        "SELECT * FROM longitudinal_recall_case WHERE case_id = ?", (case_id,)
    ).fetchone()
    if row is None:
        raise ValueError("recall case not found")
    if row["evaluated_at"] is not None:
        raise ValueError("recall case is already evaluated")
    expected = set(json.loads(row["expected_keys_json"]))
    actual = set(actual_keys)
    hit = (not actual) if row["expect_empty"] else expected <= actual
    unexpected = len(actual - expected)
    connection.execute(
        """
        UPDATE longitudinal_recall_case
        SET actual_keys_json = ?, evaluated_at = ?, hit = ?, unexpected_count = ?
        WHERE case_id = ?
        """,
        (_json(sorted(actual)), now_iso(), int(hit), unexpected, case_id),
    )
    return {
        "case_id": case_id,
        "hit": bool(hit),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
    }


def record_memory_feedback(
    connection: sqlite3.Connection,
    run_id: str,
    memory_id: str,
    revision_id: str,
    *,
    worth_storing: bool,
    scope_correct: bool,
    future_useful: bool,
    note: str | None = None,
) -> str:
    ensure_schema(connection)
    feedback_id = "feedback_" + uuid4().hex
    connection.execute(
        """
        INSERT INTO longitudinal_memory_feedback(
            feedback_id, run_id, memory_id, revision_id, worth_storing,
            scope_correct, future_useful, note, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            feedback_id,
            run_id,
            memory_id,
            revision_id,
            int(worth_storing),
            int(scope_correct),
            int(future_useful),
            note,
            now_iso(),
        ),
    )
    return feedback_id


def run_status(connection: sqlite3.Connection, run_id: str | None = None) -> dict[str, Any]:
    ensure_schema(connection)
    if run_id is None:
        row = connection.execute(
            "SELECT * FROM longitudinal_run ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT * FROM longitudinal_run WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise ValueError("longitudinal run not found")
    targets = json.loads(row["targets_json"])
    counts = _counts(connection, int(row["started_at_ms"]))
    started = datetime.fromisoformat(row["started_at"])
    days = max(0.0, (datetime.now(timezone.utc) - started).total_seconds() / 86400)
    recall = connection.execute(
        """
        SELECT COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits,
               COALESCE(SUM(CASE WHEN expect_empty = 1 THEN 1 ELSE 0 END), 0) AS empty_total,
               COALESCE(SUM(CASE WHEN expect_empty = 1 AND hit = 1 THEN 1 ELSE 0 END), 0) AS empty_hits
        FROM longitudinal_recall_case WHERE run_id = ? AND evaluated_at IS NOT NULL
        """,
        (row["run_id"],),
    ).fetchone()
    feedback = connection.execute(
        """
        SELECT COUNT(*) AS total, COALESCE(SUM(worth_storing), 0) AS worth,
               COALESCE(SUM(scope_correct), 0) AS scope_ok,
               COALESCE(SUM(future_useful), 0) AS useful
        FROM longitudinal_memory_feedback WHERE run_id = ?
        """,
        (row["run_id"],),
    ).fetchone()
    gates = {
        "duration": days >= float(targets["minimum_days"]),
        "projects": counts["projects"] >= int(targets["minimum_projects"]),
        "root_sessions": counts["root_sessions"] >= int(targets["minimum_root_sessions"]),
        "completed_turns": counts["completed_turns"] >= int(targets["minimum_completed_turns"]),
    }
    recall_rate = recall["hits"] / recall["total"] if recall["total"] else None
    empty_rate = recall["empty_hits"] / recall["empty_total"] if recall["empty_total"] else None
    return {
        "run_id": row["run_id"],
        "name": row["name"],
        "status": row["status"],
        "visibility": row["visibility"],
        "elapsed_days": round(days, 2),
        "counts": counts,
        "recall": {"evaluated": recall["total"], "hit_rate": recall_rate, "unrelated_empty_rate": empty_rate},
        "feedback": dict(feedback),
        "gates": gates,
        "sample_ready": all(gates.values()),
    }
