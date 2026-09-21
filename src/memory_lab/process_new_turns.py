"""Minimal local polling entry point for CP turns.

This module intentionally keeps the first polling MVP small:

* one local CodePulse SQLite source;
* one local Memory Lab SQLite state/cache database;
* one PostgreSQL memory store;
* optional dry-run;
* idempotence by ``turn_id + input_hash + dry_run``.

It does not try to solve multi-device synchronization, source registries, or a
background daemon yet.  Those can be layered on top of this processing shape.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .memory_react_agent import run_memory_react_agent
from .pg_store import (
    DEFAULT_PG_CONFIG,
    _json_literal,
    _run_psql,
    _sql_literal,
    ingest_turn_to_postgres,
    load_pg_config,
)
from .source import source_id
from .store import open_store
from .turns import run_turn_scan


PROCESSING_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS turn_processing_state (
  state_id bigserial PRIMARY KEY,
  turn_id text NOT NULL,
  session_id text NOT NULL,
  input_hash text NOT NULL,
  dry_run boolean NOT NULL,
  status text NOT NULL,
  agent_run_id text,
  last_error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(turn_id, input_hash, dry_run)
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_turn_processing_state(*, pg_config_path: Path = DEFAULT_PG_CONFIG) -> None:
    """Create the minimal processing state table if needed."""

    config = load_pg_config(pg_config_path)
    _run_psql(config, PROCESSING_STATE_TABLE_SQL)


def _pg_processed_keys(
    *,
    pg_config_path: Path,
    candidates: list[Mapping[str, Any]],
    dry_run: bool,
) -> set[tuple[str, str]]:
    if not candidates:
        return set()
    config = load_pg_config(pg_config_path)
    rows = [
        {"turn_id": str(item["turn_id"]), "input_hash": str(item["input_hash"])}
        for item in candidates
    ]
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH wanted AS (
  SELECT *
  FROM jsonb_to_recordset({_json_literal(rows)}) AS x(turn_id text, input_hash text)
)
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'turn_id', s.turn_id,
  'input_hash', s.input_hash
)), '[]'::jsonb)::text
FROM turn_processing_state s
JOIN wanted w
  ON w.turn_id = s.turn_id
 AND w.input_hash = s.input_hash
WHERE s.dry_run = {'true' if dry_run else 'false'}
  AND s.status = 'completed';
"""
    output = _run_psql(config, sql).strip()
    if not output:
        return set()
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"processed state query did not return JSON: {output[:200]}")
    parsed = json.loads(output[start : end + 1])
    return {(str(item["turn_id"]), str(item["input_hash"])) for item in parsed}


def mark_turn_processing_state(
    *,
    pg_config_path: Path,
    turn_id: str,
    session_id: str,
    input_hash: str,
    dry_run: bool,
    status: str,
    agent_run_id: str | None = None,
    last_error: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Upsert one turn processing state row."""

    if status not in {"processing", "completed", "failed", "skipped"}:
        raise ValueError(f"unsupported processing status: {status}")
    config = load_pg_config(pg_config_path)
    now_sql = "now()"
    metadata_obj = dict(metadata or {})
    sql = f"""
INSERT INTO turn_processing_state(
  turn_id, session_id, input_hash, dry_run, status,
  agent_run_id, last_error, started_at, finished_at, metadata
)
VALUES (
  {_sql_literal(turn_id)},
  {_sql_literal(session_id)},
  {_sql_literal(input_hash)},
  {'true' if dry_run else 'false'},
  {_sql_literal(status)},
  {_sql_literal(agent_run_id) if agent_run_id else 'NULL'},
  {_sql_literal(last_error[:4000]) if last_error else 'NULL'},
  CASE WHEN {_sql_literal(status)} = 'processing' THEN {now_sql} ELSE NULL END,
  CASE WHEN {_sql_literal(status)} IN ('completed','failed','skipped') THEN {now_sql} ELSE NULL END,
  {_json_literal(metadata_obj)}
)
ON CONFLICT(turn_id, input_hash, dry_run) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  status = EXCLUDED.status,
  agent_run_id = COALESCE(EXCLUDED.agent_run_id, turn_processing_state.agent_run_id),
  last_error = EXCLUDED.last_error,
  started_at = COALESCE(turn_processing_state.started_at, EXCLUDED.started_at),
  finished_at = EXCLUDED.finished_at,
  metadata = turn_processing_state.metadata || EXCLUDED.metadata;
"""
    _run_psql(config, sql)


def load_terminal_turn_candidates(
    *,
    lab_database: Path,
    source_identifier: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return recent terminal CP turns from the local lab database."""

    with closing(open_store(lab_database)) as connection:
        rows = connection.execute(
            """
            SELECT
              turn_id,
              session_id,
              user_message_id,
              user_timestamp,
              current_state,
              last_structural_signature,
              assistant_count,
              part_count,
              active_tool_count
            FROM conversation_turn
            WHERE source_id = ?
              AND current_state IN ('completed', 'failed', 'interrupted')
            ORDER BY
              COALESCE(user_timestamp, 0) DESC,
              turn_id DESC
            LIMIT ?
            """,
            (source_identifier, int(limit)),
        ).fetchall()
    return [
        {
            "turn_id": str(row["turn_id"]),
            "session_id": str(row["session_id"]),
            "user_message_id": str(row["user_message_id"]),
            "user_timestamp": row["user_timestamp"],
            "state": str(row["current_state"]),
            "input_hash": str(row["last_structural_signature"] or ""),
            "assistant_count": int(row["assistant_count"] or 0),
            "part_count": int(row["part_count"] or 0),
            "active_tool_count": int(row["active_tool_count"] or 0),
        }
        for row in rows
    ]


def process_new_turns(
    *,
    source_database: Path,
    lab_database: Path,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path,
    limit: int = 5,
    settle_ms: int = 1000,
    memory_limit: int = 80,
    topic_limit: int = 5,
    max_iterations: int = 3,
    dry_run: bool = True,
    reprocess: bool = False,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Scan local CP turns and process the newest unprocessed terminal turns."""

    ensure_turn_processing_state(pg_config_path=pg_config_path)
    scan_report = run_turn_scan(source_database, lab_database, settle_ms=settle_ms)
    source_identifier = source_id(source_database)
    candidates = load_terminal_turn_candidates(
        lab_database=lab_database,
        source_identifier=source_identifier,
        limit=max(1, int(limit)) * (3 if reprocess else 5),
    )
    processed = set() if reprocess else _pg_processed_keys(
        pg_config_path=pg_config_path,
        candidates=candidates,
        dry_run=dry_run,
    )
    pending: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in candidates:
        key = (item["turn_id"], item["input_hash"])
        if key in processed:
            skipped.append({**item, "reason": "already_completed"})
            continue
        pending.append(item)
        if len(pending) >= int(limit):
            break

    results: list[dict[str, Any]] = []
    for item in pending:
        turn_id = item["turn_id"]
        session_id = item["session_id"]
        input_hash = item["input_hash"]
        mark_turn_processing_state(
            pg_config_path=pg_config_path,
            turn_id=turn_id,
            session_id=session_id,
            input_hash=input_hash,
            dry_run=dry_run,
            status="processing",
            metadata={"started_by": "process-new-turns", "started_at": _utc_now_iso()},
        )
        try:
            ingest_report = ingest_turn_to_postgres(
                source_database,
                lab_database,
                pg_config_path=pg_config_path,
                turn_id=turn_id,
                settle_ms=0,
            )
            react_report = run_memory_react_agent(
                source_database=source_database,
                turn_id=turn_id,
                pg_config_path=pg_config_path,
                config_path=config_path,
                memory_limit=memory_limit,
                topic_limit=topic_limit,
                max_iterations=max_iterations,
                dry_run=dry_run,
                log_dir=log_dir,
            )
            plan = react_report.get("plan") or {}
            changes = plan.get("changes") or []
            action_counts: dict[str, int] = {}
            iter_changes = changes if isinstance(changes, list) else []
            for change in iter_changes:
                if isinstance(change, Mapping):
                    action = str(change.get("action") or "unknown")
                    action_counts[action] = action_counts.get(action, 0) + 1
            agent_run_id = str(react_report.get("run_id") or "")
            mark_turn_processing_state(
                pg_config_path=pg_config_path,
                turn_id=turn_id,
                session_id=session_id,
                input_hash=input_hash,
                dry_run=dry_run,
                status="completed",
                agent_run_id=agent_run_id or None,
                metadata={
                    "ingest": ingest_report,
                    "react_status": react_report.get("status"),
                    "action_counts": action_counts,
                    "completed_at": _utc_now_iso(),
                },
            )
            results.append(
                {
                    **item,
                    "status": "completed",
                    "agent_run_id": agent_run_id or None,
                    "action_counts": action_counts,
                    "react_status": react_report.get("status"),
                    "dry_run": dry_run,
                }
            )
        except Exception as error:  # Keep processing following turns.
            mark_turn_processing_state(
                pg_config_path=pg_config_path,
                turn_id=turn_id,
                session_id=session_id,
                input_hash=input_hash,
                dry_run=dry_run,
                status="failed",
                last_error=f"{type(error).__name__}: {error}",
                metadata={"failed_at": _utc_now_iso()},
            )
            results.append(
                {
                    **item,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "dry_run": dry_run,
                }
            )

    return {
        "status": "complete",
        "mode": "dry_run" if dry_run else "commit",
        "source_db": str(source_database),
        "source_id": source_identifier,
        "limit": int(limit),
        "reprocess": bool(reprocess),
        "scan": {
            "run_id": scan_report.get("run_id"),
            "counts": scan_report.get("counts"),
            "turn_count": scan_report.get("turn_count"),
            "tracked_turn_count": scan_report.get("tracked_turn_count"),
        },
        "candidate_count": len(candidates),
        "pending_count": len(pending),
        "skipped_count": len(skipped),
        "completed_count": sum(1 for item in results if item.get("status") == "completed"),
        "failed_count": sum(1 for item in results if item.get("status") == "failed"),
        "skipped": skipped[:20],
        "turns": results,
    }


__all__ = [
    "PROCESSING_STATE_TABLE_SQL",
    "ensure_turn_processing_state",
    "load_terminal_turn_candidates",
    "mark_turn_processing_state",
    "process_new_turns",
]
