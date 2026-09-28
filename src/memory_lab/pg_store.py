"""PostgreSQL MVP storage for turn evidence pools.

This module intentionally uses the bundled ``psql.exe`` instead of adding a
Python database dependency.  It is the first MVP step: persist one CP turn and
its three evidence layers into PostgreSQL so the user can inspect the tables.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from .evidence_pool import build_evidence_pool
from .phase import classify_turn_phase
from .source import source_id
from .store import open_store
from .turn_evidence import build_turn_evidence
from .turns import run_turn_scan
from .memory_chain_state import (
    MEMORY_CHAIN_SCHEMA_VERSION,
    MemoryChainStateValidationError,
    validate_chain_state,
    validate_chain_transition,
)


DEFAULT_PG_CONFIG = Path(__file__).resolve().parents[2] / "pg-memory-lab.local.json"
DEFAULT_PSQL = Path("D:/PostgreSQL/17/bin/psql.exe")
PG_CONFIG_SCHEMA = "memory-lab-postgres-config-v1"
DEFAULT_USER_ID = "local-default-user"
CONFLICT_RECORD_STATUSES = (
    "needs_review",
    "resolved_by_update",
    "resolved_by_merge",
    "discarded_as_stale",
    "carried_forward",
)
ACTIVE_CONFLICT_RECORD_STATUSES = ("needs_review", "carried_forward")
TRUST_LEVELS = ("user_confirmed", "verified", "observed", "inferred", "speculative")
MEMORY_RELATION_TYPES = ("supersedes", "merges", "conflicts_with")


class PostgresConfigError(ValueError):
    pass


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PostgresConfigError(f"{name} must be an object")
    return dict(value)


def load_pg_config(path: Path = DEFAULT_PG_CONFIG) -> dict[str, Any]:
    try:
        root = _object(json.loads(path.read_text(encoding="utf-8")), "pg config")
    except FileNotFoundError as error:
        raise PostgresConfigError(f"PostgreSQL config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise PostgresConfigError(f"invalid PostgreSQL config JSON: {error}") from error
    if root.get("schema_version") != PG_CONFIG_SCHEMA:
        raise PostgresConfigError("unsupported PostgreSQL config schema_version")
    postgres = _object(root.get("postgres"), "postgres")
    user = _object(root.get("user"), "user")
    required = {"host", "port", "database", "user", "password", "sslmode"}
    missing = required - set(postgres)
    if missing:
        raise PostgresConfigError(f"postgres config missing fields: {sorted(missing)}")
    if not isinstance(user.get("user_id"), str) or not user["user_id"].strip():
        raise PostgresConfigError("user.user_id must be a non-empty string")
    return {"postgres": postgres, "user": user}


def _psql_command(config: Mapping[str, Any]) -> list[str]:
    postgres = _object(config.get("postgres"), "postgres")
    psql = Path(str(postgres.get("psql_path") or DEFAULT_PSQL))
    if not psql.is_file():
        raise PostgresConfigError(f"psql.exe not found: {psql}")
    return [
        str(psql),
        "-h",
        str(postgres["host"]),
        "-p",
        str(postgres["port"]),
        "-U",
        str(postgres["user"]),
        "-d",
        str(postgres["database"]),
        "-v",
        "ON_ERROR_STOP=1",
    ]


def _run_psql(config: Mapping[str, Any], sql: str) -> str:
    postgres = _object(config.get("postgres"), "postgres")
    env = os.environ.copy()
    env["PGPASSWORD"] = str(postgres["password"])
    process = subprocess.run(
        _psql_command(config),
        input=sql,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr or ""
        stdout = process.stdout or ""
        raise RuntimeError(
            "psql failed: " + (stderr.strip() or stdout.strip())
        )
    return process.stdout or ""


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _json_literal(value: Any) -> str:
    return _sql_literal(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ) + "::jsonb"


def _memory_versioning_migration_sql(*, user_id: str) -> str:
    """Return the idempotent PostgreSQL migration for lightweight versioning."""

    del user_id  # Reserved for future per-user migration filtering.
    return """
ALTER TABLE memory_topics
  ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE memories
  ADD COLUMN IF NOT EXISTS trust_level text NOT NULL DEFAULT 'inferred';

ALTER TABLE memories
  DROP CONSTRAINT IF EXISTS memories_trust_level_check;

ALTER TABLE memories
  ADD CONSTRAINT memories_trust_level_check
  CHECK (trust_level IN ('user_confirmed', 'verified', 'observed', 'inferred', 'speculative'));

UPDATE memories
SET trust_level = CASE
  WHEN metadata ->> 'trust_level' IN ('user_confirmed', 'verified', 'observed', 'inferred', 'speculative')
    THEN metadata ->> 'trust_level'
  ELSE COALESCE(NULLIF(trust_level, ''), 'inferred')
END
WHERE trust_level IS NULL
   OR trust_level = ''
   OR metadata ? 'trust_level';

CREATE TABLE IF NOT EXISTS memory_revisions (
  revision_id text PRIMARY KEY,
  user_id text NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  memory_id text NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  revision_number integer NOT NULL,
  summary text NOT NULL,
  content text NOT NULL,
  confidence numeric(4,3),
  trust_level text NOT NULL DEFAULT 'inferred'
    CHECK (trust_level IN ('user_confirmed', 'verified', 'observed', 'inferred', 'speculative')),
  status text NOT NULL DEFAULT 'current',
  change_type text NOT NULL,
  reason text NOT NULL,
  source_turn_id text REFERENCES turns(turn_id) ON DELETE SET NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, memory_id, revision_number)
);

CREATE INDEX IF NOT EXISTS idx_memory_revisions_memory
  ON memory_revisions(user_id, memory_id, revision_number DESC);

CREATE INDEX IF NOT EXISTS idx_memory_revisions_turn
  ON memory_revisions(user_id, source_turn_id);

CREATE TABLE IF NOT EXISTS memory_relations (
  relation_id text PRIMARY KEY,
  user_id text NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  from_memory_id text NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  to_memory_id text NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  relation_type text NOT NULL
    CHECK (relation_type IN ('supersedes', 'merges', 'conflicts_with')),
  reason text NOT NULL,
  source_turn_id text REFERENCES turns(turn_id) ON DELETE SET NULL,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (from_memory_id <> to_memory_id),
  UNIQUE (user_id, from_memory_id, to_memory_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_memory_relations_from
  ON memory_relations(user_id, from_memory_id, relation_type);

CREATE INDEX IF NOT EXISTS idx_memory_relations_to
  ON memory_relations(user_id, to_memory_id, relation_type);

INSERT INTO memory_revisions(
  revision_id, user_id, memory_id, revision_number, summary, content,
  confidence, trust_level, status, change_type, reason, source_turn_id, metadata,
  created_at
)
SELECT
  'rev_' || substr(md5(m.user_id || '|' || m.memory_id || '|initial'), 1, 16),
  m.user_id,
  m.memory_id,
  1,
  m.summary,
  m.content,
  m.confidence,
  m.trust_level,
  'current',
  'initial',
  '迁移生成：把现有 memories 当前内容补为初始版本。',
  NULL,
  jsonb_build_object('migration', 'memory_versioning_v1'),
  COALESCE(m.created_at, now())
FROM memories m
WHERE NOT EXISTS (
  SELECT 1
  FROM memory_revisions r
  WHERE r.user_id = m.user_id
    AND r.memory_id = m.memory_id
    AND r.revision_number = 1
);
"""


def migrate_memory_versioning(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
) -> dict[str, Any]:
    """Install lightweight memory versioning tables and backfill initial revisions."""

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    migration_sql = _memory_versioning_migration_sql(user_id=user_id)
    report_sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT jsonb_build_object(
  'topic_count', (SELECT count(*) FROM memory_topics WHERE user_id = {_sql_literal(user_id)}),
  'topic_metadata_count', (
    SELECT count(*)
    FROM memory_topics
    WHERE user_id = {_sql_literal(user_id)}
      AND metadata IS NOT NULL
  ),
  'memory_count', (SELECT count(*) FROM memories WHERE user_id = {_sql_literal(user_id)}),
  'revision_count', (SELECT count(*) FROM memory_revisions WHERE user_id = {_sql_literal(user_id)}),
  'initial_revision_count', (
    SELECT count(*)
    FROM memory_revisions
    WHERE user_id = {_sql_literal(user_id)}
      AND revision_number = 1
      AND change_type = 'initial'
  ),
  'relation_count', (SELECT count(*) FROM memory_relations WHERE user_id = {_sql_literal(user_id)}),
  'trust_level_counts', COALESCE((
    SELECT jsonb_object_agg(trust_level, count)
    FROM (
      SELECT trust_level, count(*) AS count
      FROM memories
      WHERE user_id = {_sql_literal(user_id)}
      GROUP BY trust_level
      ORDER BY trust_level
    ) grouped
  ), '{{}}'::jsonb)
)::text;
"""
    report = _json_from_psql_object(config, "BEGIN;\n" + migration_sql + "\nCOMMIT;\n" + report_sql)
    return {
        "status": "complete",
        "tool": "migrate_memory_versioning",
        "user_id": user_id,
        "trust_levels": list(TRUST_LEVELS),
        "relation_types": list(MEMORY_RELATION_TYPES),
        "report": report,
    }


def _latest_terminal_turn(lab_database: Path, source_identifier: str) -> str:
    with closing(open_store(lab_database)) as connection:
        row = connection.execute(
            """
            SELECT turn_id
            FROM conversation_turn
            WHERE source_id = ?
              AND current_state IN ('completed', 'failed', 'interrupted')
            ORDER BY user_timestamp DESC, turn_id DESC
            LIMIT 1
            """,
            (source_identifier,),
        ).fetchone()
    if row is None:
        raise ValueError("no terminal turn found; run turn scan first")
    return str(row["turn_id"])


def _ms_to_timestamptz_sql(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "NULL"
    return f"to_timestamp({float(value) / 1000.0})"


def _evidence_rows(pool: Mapping[str, Any], user_id: str) -> list[dict[str, Any]]:
    turn_id = str((pool.get("turn") or {}).get("turn_id"))
    rows: list[dict[str, Any]] = []

    def append_common(item: Mapping[str, Any], layer: str, text: str) -> None:
        rows.append(
            {
                "evidence_id": str(item.get("ref")),
                "user_id": user_id,
                "turn_id": turn_id,
                "layer": layer,
                "source_type": str(item.get("source_type") or "unknown"),
                "kind": str(item.get("kind") or "unknown"),
                "text_or_summary": text,
                "raw_pointer_id": item.get("raw_pointer_ref"),
                "message_id": item.get("message_id"),
                "part_id": item.get("part_id"),
                "ordinal": item.get("ordinal"),
                "metadata": {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "ref",
                        "source_type",
                        "kind",
                        "text",
                        "summary",
                        "raw_pointer_ref",
                        "message_id",
                        "part_id",
                        "ordinal",
                    }
                },
            }
        )

    for item in pool.get("core_evidence") or []:
        if isinstance(item, Mapping):
            append_common(item, "core", str(item.get("text") or ""))
    for item in pool.get("candidate_evidence") or []:
        if isinstance(item, Mapping):
            append_common(item, "candidate", str(item.get("summary") or ""))
    for item in pool.get("raw_pointers") or []:
        if isinstance(item, Mapping):
            append_common(item, "pointer", str(item.get("summary") or ""))
    return rows


def _upsert_sql(
    *,
    user_id: str,
    source_database: Path,
    source_identifier: str,
    pool: Mapping[str, Any],
) -> str:
    turn = _object(pool.get("turn"), "pool.turn")
    workspace = _object(pool.get("workspace"), "pool.workspace")
    session = _object(pool.get("session"), "pool.session")
    rule_phase = _object(pool.get("rule_phase"), "pool.rule_phase")
    source_hash = source_id(source_database)
    source_metadata = {
        "source_database_name": source_database.name,
        "path_stored_for_local_debug": True,
    }
    turn_metadata = {
        "rule_phase": rule_phase,
        "stats": pool.get("stats", {}),
        "workspace": workspace,
        "session": session,
    }
    evidence_rows = _evidence_rows(pool, user_id)
    evidence_json = json.dumps(evidence_rows, ensure_ascii=False, sort_keys=True)
    occurred_at_sql = _ms_to_timestamptz_sql(turn.get("occurred_at_ms"))
    sql = f"""
BEGIN;

INSERT INTO users(user_id, email, display_name)
VALUES ({_sql_literal(user_id)}, NULL, 'Local Default User')
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO sources(source_id, user_id, source_kind, database_path, database_fingerprint, metadata)
VALUES (
  {_sql_literal(source_identifier)},
  {_sql_literal(user_id)},
  'cp_sqlite',
  {_sql_literal(str(source_database))},
  {_sql_literal(source_hash)},
  {_json_literal(source_metadata)}
)
ON CONFLICT (source_id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  database_path = EXCLUDED.database_path,
  database_fingerprint = EXCLUDED.database_fingerprint,
  metadata = EXCLUDED.metadata;

INSERT INTO turns(
  turn_id, user_id, source_id, session_id, project_key, state,
  rule_phase, final_phase, occurred_at, source_hash, metadata
)
VALUES (
  {_sql_literal(str(turn["turn_id"]))},
  {_sql_literal(user_id)},
  {_sql_literal(source_identifier)},
  {_sql_literal(str(session.get("session_id") or "unknown-session"))},
  {_sql_literal(str(workspace.get("project_key") or ""))},
  {_sql_literal(str(turn.get("state") or "unknown"))},
  {_sql_literal(str(rule_phase.get("primary_phase") or ""))},
  NULL,
  {occurred_at_sql},
  {_sql_literal(str(turn.get("source_signature") or ""))},
  {_json_literal(turn_metadata)}
)
ON CONFLICT (turn_id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  source_id = EXCLUDED.source_id,
  session_id = EXCLUDED.session_id,
  project_key = EXCLUDED.project_key,
  state = EXCLUDED.state,
  rule_phase = EXCLUDED.rule_phase,
  final_phase = EXCLUDED.final_phase,
  occurred_at = EXCLUDED.occurred_at,
  source_hash = EXCLUDED.source_hash,
  metadata = EXCLUDED.metadata ||
    CASE
      -- A source turn can be replayed under an isolated evaluation user.
      -- Preserve chain state only for the same owner; carrying it across an
      -- ownership change leaks one user's unfinished diagnosis into another.
      WHEN turns.user_id = EXCLUDED.user_id
       AND turns.metadata ? 'memory_chains'
      THEN jsonb_build_object('memory_chains', turns.metadata -> 'memory_chains')
      ELSE '{{}}'::jsonb
    END;

WITH rows AS (
  SELECT *
  FROM jsonb_to_recordset({_json_literal(evidence_rows)}) AS x(
    evidence_id text,
    user_id text,
    turn_id text,
    layer text,
    source_type text,
    kind text,
    text_or_summary text,
    raw_pointer_id text,
    message_id text,
    part_id text,
    ordinal integer,
    metadata jsonb
  )
)
INSERT INTO evidence_items(
  evidence_id, user_id, turn_id, layer, source_type, kind,
  text_or_summary, raw_pointer_id, message_id, part_id, ordinal, metadata
)
SELECT
  evidence_id, user_id, turn_id, layer, source_type, kind,
  text_or_summary, raw_pointer_id, message_id, part_id, ordinal,
  COALESCE(metadata, '{{}}'::jsonb)
FROM rows
ON CONFLICT (evidence_id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  turn_id = EXCLUDED.turn_id,
  layer = EXCLUDED.layer,
  source_type = EXCLUDED.source_type,
  kind = EXCLUDED.kind,
  text_or_summary = EXCLUDED.text_or_summary,
  raw_pointer_id = EXCLUDED.raw_pointer_id,
  message_id = EXCLUDED.message_id,
  part_id = EXCLUDED.part_id,
  ordinal = EXCLUDED.ordinal,
  metadata = EXCLUDED.metadata;

COMMIT;
"""
    del evidence_json
    return sql


def ingest_turn_to_postgres(
    source_database: Path,
    lab_database: Path,
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    turn_id: str | None = None,
    settle_ms: int = 0,
) -> dict[str, Any]:
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    run_turn_scan(source_database, lab_database, settle_ms=settle_ms)
    source_identifier = source_id(source_database)
    selected_turn = turn_id or _latest_terminal_turn(lab_database, source_identifier)
    bundle = build_turn_evidence(source_database, lab_database, selected_turn)
    phase = classify_turn_phase(bundle)
    pool = build_evidence_pool(bundle, phase_result=phase)
    sql = _upsert_sql(
        user_id=user_id,
        source_database=source_database,
        source_identifier=source_identifier,
        pool=pool,
    )
    _run_psql(config, sql)
    counts = {
        "core": len(pool.get("core_evidence") or []),
        "candidate": len(pool.get("candidate_evidence") or []),
        "pointer": len(pool.get("raw_pointers") or []),
    }
    return {
        "status": "complete",
        "user_id": user_id,
        "source_id": source_identifier,
        "turn_id": selected_turn,
        "rule_phase": phase["primary_phase"],
        "evidence_counts": counts,
        "postgres": {
            "host": config["postgres"]["host"],
            "port": config["postgres"]["port"],
            "database": config["postgres"]["database"],
        },
    }


def show_postgres_turn(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    turn_id: str,
) -> dict[str, Any]:
    config = load_pg_config(pg_config_path)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH payload AS (
  SELECT jsonb_build_object(
    'turn', to_jsonb(t),
    'evidence', (
      SELECT jsonb_agg(
        jsonb_build_object(
          'layer', e.layer,
          'source_type', e.source_type,
          'kind', e.kind,
          'evidence_id', e.evidence_id,
          'raw_pointer_id', e.raw_pointer_id,
          'preview', left(e.text_or_summary, 260)
        )
        ORDER BY
          CASE e.layer WHEN 'core' THEN 1 WHEN 'candidate' THEN 2 WHEN 'pointer' THEN 3 ELSE 4 END,
          e.ordinal NULLS LAST,
          e.evidence_id
      )
      FROM evidence_items AS e
      WHERE e.turn_id = t.turn_id
    )
  ) AS body
  FROM turns AS t
  WHERE t.turn_id = {_sql_literal(turn_id)}
)
SELECT body::text FROM payload;
"""
    output = _run_psql(config, sql).strip()
    if not output:
        raise ValueError(f"turn not found in PostgreSQL: {turn_id}")
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"turn query did not return JSON: {output[:200]}")
    return {"status": "complete", "turn": json.loads(output[start : end + 1])}


def get_turn_evidence(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    turn_id: str,
    layers: list[str] | None = None,
    max_chars: int = 1200,
) -> dict[str, Any]:
    """Read one turn's evidence pool as explicit agent-readable layers."""

    allowed_layers = {"core", "candidate", "pointer"}
    requested_layers = layers or ["core", "candidate", "pointer"]
    invalid_layers = [layer for layer in requested_layers if layer not in allowed_layers]
    if invalid_layers:
        raise ValueError(f"invalid evidence layers: {invalid_layers}")
    if not requested_layers:
        raise ValueError("at least one evidence layer is required")
    max_chars = max(80, min(int(max_chars), 8000))

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    layer_values = ", ".join(_sql_literal(layer) for layer in requested_layers)
    empty_arrays = ",\n      ".join(
        f"{_sql_literal(layer)}, '[]'::jsonb" for layer in requested_layers
    )
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH selected_turn AS (
  SELECT *
  FROM turns
  WHERE turn_id = {_sql_literal(turn_id)}
    AND user_id = {_sql_literal(user_id)}
),
requested(layer) AS (
  VALUES {", ".join(f"({_sql_literal(layer)})" for layer in requested_layers)}
),
grouped AS (
  SELECT
    e.layer,
    jsonb_agg(
      jsonb_build_object(
        'evidence_id', e.evidence_id,
        'layer', e.layer,
        'source_type', e.source_type,
        'kind', e.kind,
        'raw_pointer_id', e.raw_pointer_id,
        'message_id', e.message_id,
        'part_id', e.part_id,
        'ordinal', e.ordinal,
        'text', left(e.text_or_summary, {max_chars}),
        'truncated', length(e.text_or_summary) > {max_chars}
      )
      ORDER BY e.ordinal NULLS LAST, e.evidence_id
    ) AS items
  FROM evidence_items e
  JOIN selected_turn t ON t.turn_id = e.turn_id
  WHERE e.layer IN ({layer_values})
  GROUP BY e.layer
),
payload AS (
  SELECT jsonb_build_object(
    'found', true,
    'turn', jsonb_build_object(
      'turn_id', t.turn_id,
      'user_id', t.user_id,
      'source_id', t.source_id,
      'session_id', t.session_id,
      'project_key', t.project_key,
      'state', t.state,
      'rule_phase', t.rule_phase,
      'final_phase', t.final_phase,
      'phase_hint', COALESCE(t.final_phase, t.rule_phase),
      'occurred_at', t.occurred_at
    ),
    'requested_layers', to_jsonb(ARRAY[{layer_values}]),
    'layers', (
      SELECT jsonb_build_object({empty_arrays}) ||
             COALESCE(jsonb_object_agg(r.layer, COALESCE(g.items, '[]'::jsonb)), '{{}}'::jsonb)
      FROM requested r
      LEFT JOIN grouped g ON g.layer = r.layer
    )
  ) AS body
  FROM selected_turn t
)
SELECT COALESCE(
  (SELECT body FROM payload),
  jsonb_build_object(
    'found', false,
    'turn', null,
    'requested_layers', to_jsonb(ARRAY[{layer_values}]),
    'layers', jsonb_build_object({empty_arrays})
  )
)::text;
"""
    evidence = _json_from_psql_object(config, sql)
    return {
        "status": "complete",
        "tool": "get_turn_evidence",
        "turn_id": turn_id,
        "found": bool(evidence.get("found")),
        "evidence": evidence,
    }


def list_previous_session_turns(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    current_turn_id: str,
    limit: int = 10,
    preview_chars: int = 320,
) -> dict[str, Any]:
    """List bounded summaries of turns preceding ``current_turn_id``.

    The current turn is the security anchor.  Only earlier turns owned by the
    configured user and belonging to the same source, session and project are
    returned.  This is intentionally a discovery tool: callers must use
    :func:`read_session_turn_evidence` before citing historical evidence.
    """

    current_turn_id = str(current_turn_id or "").strip()
    if not current_turn_id:
        return {
            "status": "rejected",
            "tool": "list_previous_session_turns",
            "error_code": "empty_current_turn_id",
            "turns": [],
            "count": 0,
            "guidance": "current_turn_id 不能为空。",
        }
    try:
        normalized_limit = max(1, min(int(limit), 20))
        normalized_preview_chars = max(80, min(int(preview_chars), 800))
    except (TypeError, ValueError):
        return {
            "status": "rejected",
            "tool": "list_previous_session_turns",
            "error_code": "invalid_bounds",
            "turns": [],
            "count": 0,
            "guidance": "limit 和 preview_chars 必须是整数。",
        }

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH anchor AS (
  SELECT turn_id, user_id, source_id, session_id, project_key, occurred_at, created_at
  FROM turns
  WHERE turn_id = {_sql_literal(current_turn_id)}
    AND user_id = {_sql_literal(user_id)}
),
previous AS (
  SELECT t.*
  FROM turns t
  JOIN anchor a
    ON t.user_id = a.user_id
   AND t.source_id = a.source_id
   AND t.session_id = a.session_id
   AND COALESCE(t.project_key, '') = COALESCE(a.project_key, '')
  WHERE t.turn_id <> a.turn_id
    AND (
      COALESCE(t.occurred_at, t.created_at), t.created_at, t.turn_id
    ) < (
      COALESCE(a.occurred_at, a.created_at), a.created_at, a.turn_id
    )
  ORDER BY COALESCE(t.occurred_at, t.created_at) DESC, t.created_at DESC, t.turn_id DESC
  LIMIT {normalized_limit}
),
summaries AS (
  SELECT jsonb_build_object(
    'turn_id', p.turn_id,
    'occurred_at', p.occurred_at,
    'state', p.state,
    'rule_phase', p.rule_phase,
    'final_phase', p.final_phase,
    'phase_hint', COALESCE(p.final_phase, p.rule_phase),
    'evidence_count', (
      SELECT count(*) FROM evidence_items e WHERE e.turn_id = p.turn_id AND e.user_id = p.user_id
    ),
    'layer_counts', COALESCE((
      SELECT jsonb_object_agg(layer, item_count)
      FROM (
        SELECT e.layer, count(*) AS item_count
        FROM evidence_items e
        WHERE e.turn_id = p.turn_id AND e.user_id = p.user_id
        GROUP BY e.layer
      ) layer_totals
    ), '{{}}'::jsonb),
    'evidence_kinds', COALESCE((
      SELECT jsonb_agg(kind ORDER BY kind)
      FROM (
        SELECT DISTINCT e.kind
        FROM evidence_items e
        WHERE e.turn_id = p.turn_id AND e.user_id = p.user_id
      ) kinds
    ), '[]'::jsonb),
    'previews', COALESCE((
      SELECT jsonb_agg(jsonb_build_object(
        'evidence_id', preview.evidence_id,
        'layer', preview.layer,
        'source_type', preview.source_type,
        'kind', preview.kind,
        'text', left(preview.text_or_summary, {normalized_preview_chars})
      ) ORDER BY preview.layer_rank, preview.ordinal NULLS LAST, preview.evidence_id)
      FROM (
        SELECT
          e.*,
          CASE e.layer WHEN 'core' THEN 1 WHEN 'candidate' THEN 2 ELSE 3 END AS layer_rank
        FROM evidence_items e
        WHERE e.turn_id = p.turn_id
          AND e.user_id = p.user_id
          AND e.layer IN ('core', 'candidate')
        ORDER BY
          CASE e.layer WHEN 'core' THEN 1 WHEN 'candidate' THEN 2 ELSE 3 END,
          e.ordinal NULLS LAST,
          e.evidence_id
        LIMIT 6
      ) preview
    ), '[]'::jsonb)
  ) AS item,
  COALESCE(p.occurred_at, p.created_at) AS sort_time,
  p.created_at,
  p.turn_id
  FROM previous p
)
SELECT jsonb_build_object(
  'anchor_found', EXISTS(SELECT 1 FROM anchor),
  'turns', COALESCE((
    SELECT jsonb_agg(item ORDER BY sort_time DESC, created_at DESC, turn_id DESC)
    FROM summaries
  ), '[]'::jsonb)
)::text;
"""
    payload = _json_from_psql_object(config, sql)
    if not payload.get("anchor_found"):
        return {
            "status": "not_found",
            "tool": "list_previous_session_turns",
            "error_code": "current_turn_not_found",
            "current_turn_id": current_turn_id,
            "turns": [],
            "count": 0,
            "guidance": "当前 turn 尚未写入 PostgreSQL，不能查询其会话历史。",
        }
    turns = [item for item in payload.get("turns") or [] if isinstance(item, Mapping)]
    return {
        "status": "complete",
        "tool": "list_previous_session_turns",
        "current_turn_id": current_turn_id,
        "turns": turns,
        "count": len(turns),
        "limits": {"limit": normalized_limit, "preview_chars": normalized_preview_chars},
        "guidance": (
            "这里只返回历史 turn 摘要；如需引用其证据，必须继续调用 read_session_turn_evidence。"
        ),
    }


def read_session_turn_evidence(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    current_turn_id: str,
    target_turn_id: str,
    layers: list[str] | None = None,
    max_chars: int = 1200,
) -> dict[str, Any]:
    """Read evidence for an earlier turn in the current turn's conversation.

    Authorization is derived from PostgreSQL ownership and chronology, never
    from model-provided session/project values.
    """

    current_turn_id = str(current_turn_id or "").strip()
    target_turn_id = str(target_turn_id or "").strip()
    if not current_turn_id or not target_turn_id:
        return {
            "status": "rejected",
            "tool": "read_session_turn_evidence",
            "error_code": "empty_turn_id",
            "current_turn_id": current_turn_id,
            "target_turn_id": target_turn_id,
            "guidance": "current_turn_id 和 target_turn_id 都不能为空。",
        }
    if current_turn_id == target_turn_id:
        return {
            "status": "rejected",
            "tool": "read_session_turn_evidence",
            "error_code": "target_is_current_turn",
            "current_turn_id": current_turn_id,
            "target_turn_id": target_turn_id,
            "guidance": "当前 turn 应使用 get_current_turn_evidence_summary，不属于历史 turn。",
        }

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH anchor AS (
  SELECT turn_id, user_id, source_id, session_id, project_key, occurred_at, created_at
  FROM turns
  WHERE turn_id = {_sql_literal(current_turn_id)}
    AND user_id = {_sql_literal(user_id)}
),
target AS (
  SELECT turn_id, user_id, source_id, session_id, project_key, occurred_at, created_at
  FROM turns
  WHERE turn_id = {_sql_literal(target_turn_id)}
    AND user_id = {_sql_literal(user_id)}
)
SELECT jsonb_build_object(
  'anchor_found', EXISTS(SELECT 1 FROM anchor),
  'target_found', EXISTS(SELECT 1 FROM target),
  'authorized', EXISTS(
    SELECT 1
    FROM anchor a
    JOIN target t
      ON t.user_id = a.user_id
     AND t.source_id = a.source_id
     AND t.session_id = a.session_id
     AND COALESCE(t.project_key, '') = COALESCE(a.project_key, '')
    WHERE (
      COALESCE(t.occurred_at, t.created_at), t.created_at, t.turn_id
    ) < (
      COALESCE(a.occurred_at, a.created_at), a.created_at, a.turn_id
    )
  )
)::text;
"""
    authorization = _json_from_psql_object(config, sql)
    if not authorization.get("anchor_found"):
        return {
            "status": "not_found",
            "tool": "read_session_turn_evidence",
            "error_code": "current_turn_not_found",
            "current_turn_id": current_turn_id,
            "target_turn_id": target_turn_id,
            "guidance": "当前 turn 尚未写入 PostgreSQL。",
        }
    if not authorization.get("target_found"):
        return {
            "status": "not_found",
            "tool": "read_session_turn_evidence",
            "error_code": "target_turn_not_found",
            "current_turn_id": current_turn_id,
            "target_turn_id": target_turn_id,
            "guidance": "目标历史 turn 不存在或不属于当前用户。",
        }
    if not authorization.get("authorized"):
        return {
            "status": "denied",
            "tool": "read_session_turn_evidence",
            "error_code": "target_outside_current_conversation",
            "current_turn_id": current_turn_id,
            "target_turn_id": target_turn_id,
            "guidance": "目标 turn 不是当前会话中更早的 turn，拒绝读取。",
        }

    detail = get_turn_evidence(
        pg_config_path=pg_config_path,
        turn_id=target_turn_id,
        layers=layers or ["core", "candidate", "pointer"],
        max_chars=max_chars,
    )
    evidence = detail.get("evidence") if isinstance(detail.get("evidence"), Mapping) else {}
    return {
        "status": "complete" if detail.get("found") else "not_found",
        "tool": "read_session_turn_evidence",
        "current_turn_id": current_turn_id,
        "target_turn_id": target_turn_id,
        "found": bool(detail.get("found")),
        "evidence": evidence,
        "guidance": (
            "已读取同会话历史 turn；只有本次返回的 evidence_id 才能进入跨 turn 证据授权集合。"
        ),
    }


def _memory_chain_id(
    *,
    user_id: str,
    session_id: str,
    project_key: str,
    chain_type: str,
    anchor_turn_id: str,
) -> str:
    material = "|".join(
        (user_id, session_id, project_key, chain_type, anchor_turn_id)
    )
    return "chain_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def save_memory_chain_state(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    current_turn_id: str,
    chain_state: Mapping[str, Any],
    final_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Persist one validated technical chain in existing ``turns.metadata``."""

    current_turn_id = str(current_turn_id or "").strip()
    observed_turn_ids = {
        str(item)
        for item in (chain_state.get("related_turn_ids") or [])
        if str(item) and str(item) != current_turn_id
    }
    try:
        normalized = validate_chain_state(
            chain_state,
            current_turn_id=current_turn_id,
            observed_turn_ids=observed_turn_ids,
        )
    except MemoryChainStateValidationError as error:
        return {
            "status": "rejected",
            "tool": "save_memory_chain_state",
            "error_code": "invalid_chain_state",
            "errors": error.errors,
            "guidance": "修正 chain_state 后重试；未通过校验的状态不会写入。",
        }
    if normalized["chain_type"] == "none":
        return {
            "status": "skipped",
            "tool": "save_memory_chain_state",
            "result": "not_applicable",
            "guidance": "非技术故障 turn 不保存故障链状态。",
        }

    normalized_final_ids: list[str] = []
    for memory_id in final_memory_ids or []:
        memory_id = str(memory_id or "").strip()
        if memory_id and memory_id not in normalized_final_ids:
            normalized_final_ids.append(memory_id)
    if len(normalized_final_ids) > 10:
        return {
            "status": "rejected",
            "tool": "save_memory_chain_state",
            "error_code": "too_many_final_memories",
            "guidance": "一条故障链最多关联 10 条最终记忆。",
        }
    if normalized["status"] == "open" and normalized_final_ids:
        return {
            "status": "rejected",
            "tool": "save_memory_chain_state",
            "error_code": "open_chain_has_final_memory",
            "guidance": "未闭环状态不能关联正式记忆。",
        }

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    requested_turn_ids = normalized["related_turn_ids"]
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH anchor AS (
  SELECT * FROM turns
  WHERE turn_id = {_sql_literal(current_turn_id)}
    AND user_id = {_sql_literal(user_id)}
),
requested AS (
  SELECT value #>> '{{}}' AS turn_id
  FROM jsonb_array_elements({_json_literal(requested_turn_ids)})
),
authorized AS (
  SELECT t.turn_id, t.session_id, COALESCE(t.project_key, '') AS project_key,
         t.occurred_at, t.created_at, t.metadata
  FROM turns t
  JOIN anchor a
    ON t.user_id = a.user_id
   AND t.source_id = a.source_id
   AND t.session_id = a.session_id
   AND COALESCE(t.project_key, '') = COALESCE(a.project_key, '')
  JOIN requested r ON r.turn_id = t.turn_id
  WHERE (
    COALESCE(t.occurred_at, t.created_at), t.created_at, t.turn_id
  ) <= (
    COALESCE(a.occurred_at, a.created_at), a.created_at, a.turn_id
  )
),
conversation_chains AS (
  SELECT t.turn_id AS stored_on_turn_id, chain.value AS chain
  FROM turns t
  JOIN anchor a
    ON t.user_id = a.user_id
   AND t.source_id = a.source_id
   AND t.session_id = a.session_id
   AND COALESCE(t.project_key, '') = COALESCE(a.project_key, '')
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE
      WHEN jsonb_typeof(t.metadata -> 'memory_chains') = 'array'
      THEN t.metadata -> 'memory_chains'
      ELSE '[]'::jsonb
    END
  ) AS chain(value)
)
SELECT jsonb_build_object(
  'current_found', EXISTS(SELECT 1 FROM anchor),
  'session_id', COALESCE((SELECT session_id FROM anchor), ''),
  'project_key', COALESCE((SELECT project_key FROM anchor), ''),
  'turns', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
      'turn_id', turn_id,
      'memory_chains', CASE
        WHEN jsonb_typeof(metadata -> 'memory_chains') = 'array'
        THEN metadata -> 'memory_chains'
        ELSE '[]'::jsonb
      END
    ) ORDER BY COALESCE(occurred_at, created_at), created_at, turn_id)
    FROM authorized
  ), '[]'::jsonb),
  'conversation_chains', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
      'stored_on_turn_id', stored_on_turn_id,
      'chain', chain
    ))
    FROM conversation_chains
  ), '[]'::jsonb)
)::text;
"""
    context = _json_from_psql_object(config, sql)
    if not context.get("current_found"):
        return {
            "status": "not_found",
            "tool": "save_memory_chain_state",
            "error_code": "current_turn_not_found",
            "guidance": "当前 turn 尚未入库，不能保存故障链状态。",
        }
    authorized_turns = [
        item for item in context.get("turns") or [] if isinstance(item, Mapping)
    ]
    authorized_ids = {str(item.get("turn_id") or "") for item in authorized_turns}
    unauthorized = [item for item in requested_turn_ids if item not in authorized_ids]
    if unauthorized:
        return {
            "status": "denied",
            "tool": "save_memory_chain_state",
            "error_code": "related_turn_outside_conversation",
            "unauthorized_turn_ids": unauthorized,
            "guidance": "关联 turn 必须属于同一用户、来源、会话和项目，且不能晚于当前 turn。",
        }

    related_set = set(requested_turn_ids)
    requested_chain_id = str(normalized.get("chain_id") or "")
    candidates: list[dict[str, Any]] = []
    for item in context.get("conversation_chains") or []:
        if not isinstance(item, Mapping) or not isinstance(item.get("chain"), Mapping):
            continue
        chain = dict(item["chain"])
        chain_turns = {str(value) for value in chain.get("related_turn_ids") or []}
        exact_id_match = (
            bool(requested_chain_id)
            and str(chain.get("chain_id") or "") == requested_chain_id
        )
        overlap_match = (
            not requested_chain_id
            and chain.get("chain_type") == normalized["chain_type"]
            and bool(chain_turns.intersection(related_set))
        )
        if exact_id_match or overlap_match:
            candidates.append(
                {**chain, "stored_on_turn_id": str(item.get("stored_on_turn_id") or "")}
            )
    if requested_chain_id and not candidates:
        return {
            "status": "rejected",
            "tool": "save_memory_chain_state",
            "error_code": "unknown_chain_id",
            "chain_id": requested_chain_id,
            "guidance": "指定的故障链不存在或不属于当前会话；重新调用 list_open_memory_chains 后再判断。",
        }
    if len(candidates) > 1:
        return {
            "status": "rejected",
            "tool": "save_memory_chain_state",
            "error_code": "ambiguous_existing_chain",
            "candidate_chain_ids": [str(item.get("chain_id") or "") for item in candidates],
            "guidance": "当前证据同时命中多条故障链；需要模型读取更多历史 turn 后明确归属。",
        }

    existing = candidates[0] if candidates else None
    if existing and not requested_chain_id:
        return {
            "status": "rejected",
            "tool": "save_memory_chain_state",
            "error_code": "chain_id_required_for_existing_chain",
            "candidate_chain_ids": [str(existing.get("chain_id") or "")],
            "guidance": "当前证据与已有故障链重叠；必须携带 list_open_memory_chains 返回的 chain_id，不能静默续接。",
        }
    if existing:
        try:
            validate_chain_transition(existing, normalized)
        except MemoryChainStateValidationError as error:
            return {
                "status": "rejected",
                "tool": "save_memory_chain_state",
                "error_code": "invalid_chain_transition",
                "errors": error.errors,
                "guidance": "不能静默倒退或丢弃已有故障链证据；必要时创建新链。",
            }
        anchor_turn_id = str(existing.get("anchor_turn_id") or existing["stored_on_turn_id"])
        chain_id = str(existing.get("chain_id") or "")
    else:
        anchor_turn_id = str(authorized_turns[0]["turn_id"])
        chain_id = _memory_chain_id(
            user_id=user_id,
            session_id=str(context.get("session_id") or ""),
            project_key=str(context.get("project_key") or ""),
            chain_type=normalized["chain_type"],
            anchor_turn_id=anchor_turn_id,
        )

    record = {
        "schema_version": MEMORY_CHAIN_SCHEMA_VERSION,
        "chain_id": chain_id,
        "chain_type": normalized["chain_type"],
        "state": normalized["state"],
        "anchor_turn_id": anchor_turn_id,
        "related_turn_ids": normalized["related_turn_ids"],
        "missing_evidence": normalized["missing_evidence"],
        "status": normalized["status"],
        "updated_by_turn_id": current_turn_id,
        "final_memory_ids": normalized_final_ids,
    }
    if existing and all(
        existing.get(key) == record.get(key)
        for key in (
            "chain_id", "chain_type", "state", "anchor_turn_id",
            "related_turn_ids", "missing_evidence", "status",
            "updated_by_turn_id", "final_memory_ids",
        )
    ):
        return {
            "status": "complete",
            "tool": "save_memory_chain_state",
            "result": "skipped_existing",
            "chain": record,
            "guidance": "相同故障链状态已经保存，无需重复写入。",
        }

    anchor_row = next(
        item for item in authorized_turns if str(item.get("turn_id") or "") == anchor_turn_id
    )
    old_chains = [
        dict(item)
        for item in anchor_row.get("memory_chains") or []
        if isinstance(item, Mapping)
    ]
    new_chains = [item for item in old_chains if str(item.get("chain_id") or "") != chain_id]
    new_chains.append(record)
    update_sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH updated AS (
  UPDATE turns
  SET metadata = jsonb_set(
    COALESCE(metadata, '{{}}'::jsonb),
    '{{memory_chains}}',
    {_json_literal(new_chains)},
    true
  )
  WHERE turn_id = {_sql_literal(anchor_turn_id)}
    AND user_id = {_sql_literal(user_id)}
    AND COALESCE(metadata -> 'memory_chains', '[]'::jsonb) = {_json_literal(old_chains)}
  RETURNING turn_id
)
SELECT jsonb_build_object('updated', EXISTS(SELECT 1 FROM updated))::text;
"""
    update_result = _json_from_psql_object(config, update_sql)
    if not update_result.get("updated"):
        return {
            "status": "retry",
            "tool": "save_memory_chain_state",
            "error_code": "concurrent_chain_update",
            "chain_id": chain_id,
            "guidance": "故障链状态在写入前已被其他进程修改；重新读取后再判断。",
        }
    return {
        "status": "complete",
        "tool": "save_memory_chain_state",
        "result": "updated" if existing else "created",
        "chain": record,
        "guidance": "故障链状态已保存到 anchor turn 的 metadata，不会进入正式记忆召回。",
    }


def list_open_memory_chains(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    current_turn_id: str,
    limit: int = 10,
) -> dict[str, Any]:
    """List open technical chains in the current turn's conversation.

    Only chain summaries and turn previews are returned. Evidence IDs remain
    unavailable until ``read_session_turn_evidence`` is called.
    """

    current_turn_id = str(current_turn_id or "").strip()
    normalized_limit = max(1, min(int(limit), 20))
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH anchor AS (
  SELECT * FROM turns
  WHERE turn_id = {_sql_literal(current_turn_id)}
    AND user_id = {_sql_literal(user_id)}
),
chain_rows AS (
  SELECT t.turn_id AS stored_on_turn_id, chain.value AS chain
  FROM turns t
  JOIN anchor a
    ON t.user_id = a.user_id
   AND t.source_id = a.source_id
   AND t.session_id = a.session_id
   AND COALESCE(t.project_key, '') = COALESCE(a.project_key, '')
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE
      WHEN jsonb_typeof(t.metadata -> 'memory_chains') = 'array'
      THEN t.metadata -> 'memory_chains'
      ELSE '[]'::jsonb
    END
  ) AS chain(value)
  WHERE (chain.value ->> 'status') = 'open'
    AND (
      COALESCE(t.occurred_at, t.created_at), t.created_at, t.turn_id
    ) <= (
      COALESCE(a.occurred_at, a.created_at), a.created_at, a.turn_id
    )
  ORDER BY COALESCE(t.occurred_at, t.created_at) DESC, t.turn_id DESC
  LIMIT {normalized_limit}
)
SELECT jsonb_build_object(
  'current_found', EXISTS(SELECT 1 FROM anchor),
  'chains', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
      'chain_id', chain ->> 'chain_id',
      'chain_type', chain ->> 'chain_type',
      'state', chain ->> 'state',
      'status', chain ->> 'status',
      'anchor_turn_id', chain ->> 'anchor_turn_id',
      'related_turn_ids', COALESCE(chain -> 'related_turn_ids', '[]'::jsonb),
      'missing_evidence', COALESCE(chain -> 'missing_evidence', '[]'::jsonb),
      'updated_by_turn_id', chain ->> 'updated_by_turn_id',
      'related_turns', COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
          'turn_id', rt.turn_id,
          'rule_phase', rt.rule_phase,
          'occurred_at', rt.occurred_at,
          'preview', COALESCE((
            SELECT left(e.text_or_summary, 320)
            FROM evidence_items e
            WHERE e.user_id = rt.user_id
              AND e.turn_id = rt.turn_id
              AND e.layer = 'core'
            ORDER BY e.ordinal NULLS LAST, e.evidence_id
            LIMIT 1
          ), '')
        ) ORDER BY COALESCE(rt.occurred_at, rt.created_at), rt.created_at, rt.turn_id)
        FROM turns rt
        WHERE rt.user_id = {_sql_literal(user_id)}
          AND rt.turn_id IN (
            SELECT jsonb_array_elements_text(COALESCE(chain -> 'related_turn_ids', '[]'::jsonb))
          )
      ), '[]'::jsonb)
    ))
    FROM chain_rows
  ), '[]'::jsonb)
)::text;
"""
    payload = _json_from_psql_object(config, sql)
    if not payload.get("current_found"):
        return {
            "status": "not_found",
            "tool": "list_open_memory_chains",
            "error_code": "current_turn_not_found",
            "current_turn_id": current_turn_id,
            "chains": [],
            "count": 0,
            "guidance": "当前 turn 尚未入库，不能查询未闭环故障链。",
        }
    chains = [item for item in payload.get("chains") or [] if isinstance(item, Mapping)]
    return {
        "status": "complete",
        "tool": "list_open_memory_chains",
        "current_turn_id": current_turn_id,
        "chains": chains,
        "count": len(chains),
        "limit": normalized_limit,
        "guidance": (
            "这里只返回未闭环链和 turn 摘要；引用历史证据前仍须调用 read_session_turn_evidence。"
        ),
    }


def list_memory_topics(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    domain: str | None = None,
    project_key: str | None = None,
    include_inactive: bool = False,
    include_test_data: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    """List lightweight memory topic index entries without reading memory bodies."""

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    limit = max(1, min(int(limit), 500))
    filters = [f"mt.user_id = {_sql_literal(user_id)}"]
    if not include_inactive:
        filters.append("mt.status = 'active'")
    if not include_test_data:
        filters.append(_exclude_test_topic_sql("mt"))
    scope_filters = ["mt.scope ->> 'level' = 'global'"]
    if domain:
        scope_filters.append(
            f"(mt.scope ->> 'level' = 'domain' AND mt.scope ->> 'domain' = {_sql_literal(domain)})"
        )
    if project_key:
        scope_filters.append(
            f"(mt.scope ->> 'level' = 'project' AND mt.scope ->> 'project_key' = {_sql_literal(project_key)})"
        )
    if domain or project_key:
        filters.append("(" + " OR ".join(scope_filters) + ")")
    where_sql = " AND ".join(filters)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT COALESCE(jsonb_agg(
  jsonb_build_object(
    'topic_id', mt.topic_id,
    'topic_key', mt.topic_key,
    'title', mt.title,
    'summary', mt.summary,
    'topic_type', mt.topic_type,
    'scope', mt.scope,
    'status', mt.status,
    'memory_count', (
      SELECT count(*)
      FROM memories m
      WHERE m.topic_id = mt.topic_id
        AND m.user_id = mt.user_id
        AND ({'true' if include_inactive else "m.status = 'active'"})
    ),
    'created_at', mt.created_at,
    'updated_at', mt.updated_at
  )
  ORDER BY mt.updated_at DESC, mt.topic_key
), '[]'::jsonb)::text
FROM (
  SELECT *
  FROM memory_topics mt
  WHERE {where_sql}
  ORDER BY mt.updated_at DESC, mt.topic_key
  LIMIT {limit}
) mt;
"""
    topics = _json_from_psql_array(config, sql)
    return {
        "status": "complete",
        "tool": "list_memory_topics",
        "filters": {
            "domain": domain,
            "project_key": project_key,
            "include_inactive": include_inactive,
            "include_test_data": include_test_data,
            "limit": limit,
        },
        "topics": topics,
        "count": len(topics),
    }


def read_topic_detail(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    topic_id: str,
    include_inactive: bool = False,
    include_test_data: bool = False,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Read one topic and its memories after the topic has been selected."""

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    max_chars = max(200, min(int(max_chars), 12000))
    topic_status_filter = "true" if include_inactive else "mt.status = 'active'"
    memory_status_filter = "true" if include_inactive else "m.status = 'active'"
    topic_test_filter = "true" if include_test_data else _exclude_test_topic_sql("mt")
    memory_test_filter = "true" if include_test_data else _exclude_test_memory_sql("m", None)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH selected_topic AS (
  SELECT *
  FROM memory_topics mt
  WHERE mt.topic_id = {_sql_literal(topic_id)}
    AND mt.user_id = {_sql_literal(user_id)}
    AND {topic_status_filter}
    AND {topic_test_filter}
)
SELECT COALESCE((
  SELECT jsonb_build_object(
    'found', true,
    'topic', jsonb_build_object(
      'topic_id', mt.topic_id,
      'topic_key', mt.topic_key,
      'title', mt.title,
      'summary', mt.summary,
      'topic_type', mt.topic_type,
      'scope', mt.scope,
      'status', mt.status,
      'created_at', mt.created_at,
      'updated_at', mt.updated_at
    ),
    'memories', COALESCE((
      SELECT jsonb_agg(
        jsonb_build_object(
          'memory_id', m.memory_id,
          'stable_key', m.stable_key,
          'memory_type', m.memory_type,
          'scope', m.scope,
          'summary', m.summary,
          'content', left(m.content, {max_chars}),
          'content_truncated', length(m.content) > {max_chars},
          'status', m.status,
          'confidence', m.confidence,
          'trust_level', m.trust_level,
          'metadata', m.metadata,
          'created_at', m.created_at,
          'updated_at', m.updated_at
        )
        ORDER BY m.updated_at DESC, m.stable_key
      )
      FROM memories m
      WHERE m.topic_id = mt.topic_id
        AND m.user_id = mt.user_id
        AND {memory_status_filter}
        AND {memory_test_filter}
    ), '[]'::jsonb)
  )
  FROM selected_topic mt
), jsonb_build_object('found', false, 'topic', null, 'memories', '[]'::jsonb))::text;
"""
    detail = _json_from_psql_object(config, sql)
    return {
        "status": "complete",
        "tool": "read_topic_detail",
        "topic_id": topic_id,
        "found": bool(detail.get("found")),
        "detail": detail,
    }


def _search_terms(query: str, *, max_terms: int = 12) -> list[str]:
    """Build bounded, explainable search terms for lightweight SQL recall."""

    text = re.sub(r"\s+", " ", str(query or "").strip())[:240]
    if not text:
        return []
    raw_terms = re.findall(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", text)
    terms: list[str] = []
    for term in raw_terms:
        if len(term) < 2:
            continue
        terms.append(term)
        if re.fullmatch(r"[\u4e00-\u9fff]{5,}", term):
            for size in (2, 3):
                for index in range(0, len(term) - size + 1):
                    terms.append(term[index : index + size])
    if not terms and len(text) >= 2:
        terms.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
        if len(deduped) >= max_terms:
            break
    return deduped


def _exclude_test_memory_sql(memory_alias: str = "m", topic_alias: str | None = "mt") -> str:
    """SQL predicate that hides acceptance/test memory rows from normal recall."""

    memory_parts = [
        f"COALESCE({memory_alias}.metadata ->> 'is_test', 'false') = 'true'",
        f"{memory_alias}.metadata ? 'test_namespace'",
        f"{memory_alias}.stable_key LIKE 'acceptance.%'",
        f"{memory_alias}.stable_key LIKE 'test.%'",
    ]
    topic_parts: list[str] = []
    if topic_alias:
        topic_parts = [
            f"COALESCE({topic_alias}.metadata ->> 'is_test', 'false') = 'true'",
            f"{topic_alias}.metadata ? 'test_namespace'",
            f"{topic_alias}.topic_key LIKE 'acceptance.%'",
            f"{topic_alias}.topic_key LIKE 'test.%'",
        ]
    return "NOT (" + " OR ".join(memory_parts + topic_parts) + ")"


def _exclude_test_topic_sql(topic_alias: str = "mt") -> str:
    """SQL predicate that hides acceptance/test topics from normal topic indexes."""

    return "NOT (" + " OR ".join(
        [
            f"COALESCE({topic_alias}.metadata ->> 'is_test', 'false') = 'true'",
            f"{topic_alias}.metadata ? 'test_namespace'",
            f"{topic_alias}.topic_key LIKE 'acceptance.%'",
            f"{topic_alias}.topic_key LIKE 'test.%'",
        ]
    ) + ")"


def search_related_memories(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    query: str,
    memory_type: str | None = None,
    domain: str | None = None,
    project_key: str | None = None,
    include_inactive: bool = False,
    include_test_data: bool = False,
    limit: int = 8,
) -> dict[str, Any]:
    """Search topic titles/summaries and memory summaries/content for candidates.

    This is a recall tool only.  Results are intentionally summaries plus IDs;
    callers must use ``read_memory_detail`` or ``read_topic_detail`` before
    modifying any existing memory.
    """

    normalized_query = re.sub(r"\s+", " ", str(query or "").strip())[:240]
    terms = _search_terms(normalized_query)
    limit = max(1, min(int(limit), 20))
    if not terms:
        return {
            "status": "rejected",
            "tool": "search_related_memories",
            "error_code": "empty_query",
            "query": normalized_query,
            "matches": [],
            "count": 0,
            "guidance": "query 为空或没有可搜索词；请根据当前 turn 的症状、主题或关键词重新搜索。",
        }

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    values_sql = ", ".join(f"({_sql_literal(term)})" for term in terms)
    min_term_matches = 2 if len(terms) >= 3 else 1
    filters = [f"m.user_id = {_sql_literal(user_id)}"]
    if not include_inactive:
        filters.extend(["m.status = 'active'", "mt.status = 'active'"])
    if not include_test_data:
        filters.append(_exclude_test_memory_sql("m", "mt"))
    if memory_type:
        filters.append(f"m.memory_type = {_sql_literal(str(memory_type).strip()[:120])}")
    scope_filters: list[str] = []
    if domain:
        scope_filters.append(
            f"(m.scope ->> 'level' = 'domain' AND m.scope ->> 'domain' = {_sql_literal(str(domain).strip()[:120])})"
        )
        scope_filters.append(
            f"(mt.scope ->> 'level' = 'domain' AND mt.scope ->> 'domain' = {_sql_literal(str(domain).strip()[:120])})"
        )
    if project_key:
        scope_filters.append(
            f"(m.scope ->> 'level' = 'project' AND m.scope ->> 'project_key' = {_sql_literal(str(project_key).strip()[:160])})"
        )
        scope_filters.append(
            f"(mt.scope ->> 'level' = 'project' AND mt.scope ->> 'project_key' = {_sql_literal(str(project_key).strip()[:160])})"
        )
    if scope_filters:
        filters.append("(" + " OR ".join(scope_filters) + ")")
    where_sql = " AND ".join(filters)
    any_match_sql = """
        mt.title ILIKE '%' || t.term || '%'
        OR mt.summary ILIKE '%' || t.term || '%'
        OR m.summary ILIKE '%' || t.term || '%'
        OR m.content ILIKE '%' || t.term || '%'
    """
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH terms(term) AS (VALUES {values_sql}),
ranked AS (
  SELECT
    m.memory_id,
    m.topic_id,
    mt.topic_key,
    mt.title AS topic_title,
    mt.summary AS topic_summary,
    mt.topic_type,
    mt.scope AS topic_scope,
    m.stable_key,
    m.memory_type,
    m.scope AS memory_scope,
    m.summary AS memory_summary,
    m.status AS memory_status,
    m.confidence,
    m.trust_level AS trust_level,
    COALESCE(m.metadata ->> 'source_rank', 'unknown') AS source_rank,
    COALESCE(m.metadata ->> 'verification_status', 'unknown') AS verification_status,
    m.updated_at,
    (
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE m.summary ILIKE '%' || t.term || '%') THEN 8 ELSE 0 END +
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE mt.title ILIKE '%' || t.term || '%') THEN 6 ELSE 0 END +
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE mt.summary ILIKE '%' || t.term || '%') THEN 4 ELSE 0 END +
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE m.content ILIKE '%' || t.term || '%') THEN 2 ELSE 0 END +
      (SELECT count(*) FROM terms t WHERE {any_match_sql})
    ) AS score,
    to_jsonb(ARRAY(
      SELECT DISTINCT t.term
      FROM terms t
      WHERE {any_match_sql}
      ORDER BY t.term
    )) AS matched_terms,
    to_jsonb(array_remove(ARRAY[
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE mt.title ILIKE '%' || t.term || '%') THEN 'topic.title' END,
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE mt.summary ILIKE '%' || t.term || '%') THEN 'topic.summary' END,
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE m.summary ILIKE '%' || t.term || '%') THEN 'memory.summary' END,
      CASE WHEN EXISTS (SELECT 1 FROM terms t WHERE m.content ILIKE '%' || t.term || '%') THEN 'memory.content' END
    ], NULL)) AS matched_fields
  FROM memories m
  JOIN memory_topics mt ON mt.topic_id = m.topic_id AND mt.user_id = m.user_id
  WHERE {where_sql}
    AND (SELECT count(*) FROM terms t WHERE {any_match_sql}) >= {min_term_matches}
)
SELECT COALESCE(jsonb_agg(
  jsonb_build_object(
    'memory_id', memory_id,
    'topic_id', topic_id,
    'topic_key', topic_key,
    'topic_title', topic_title,
    'topic_summary', topic_summary,
    'topic_type', topic_type,
    'topic_scope', topic_scope,
    'stable_key', stable_key,
    'memory_type', memory_type,
    'memory_scope', memory_scope,
    'summary', memory_summary,
    'status', memory_status,
    'confidence', confidence,
    'trust_level', trust_level,
    'source_rank', source_rank,
    'verification_status', verification_status,
    'updated_at', updated_at,
    'matched_terms', matched_terms,
    'matched_fields', matched_fields,
    'score', score,
    'score_reason', '命中字段=' || matched_fields::text || '；命中词=' || matched_terms::text,
    'why_matched', '命中字段=' || matched_fields::text || '；命中词=' || matched_terms::text,
    'next_steps', jsonb_build_array(
      '搜索结果只用于召回候选，不代表已经读取正文。',
      '如果只需查看完整内容，请调用 read_memory_detail。',
      '如果需要判断合并、更新或冲突，请先调用 read_memory_detail 或 read_topic_detail。'
    ),
    'next_step', '如需合并、更新或冲突处理，请先调用 read_memory_detail 读取 memory 正文。'
  )
  ORDER BY score DESC, updated_at DESC, stable_key
), '[]'::jsonb)::text
FROM (
  SELECT *
  FROM ranked
  ORDER BY score DESC, updated_at DESC, stable_key
  LIMIT {limit}
) limited;
"""
    matches = _json_from_psql_array(config, sql)
    guidance = (
        "已返回候选 memory；搜索结果只可用于发现候选，修改前必须 read_memory_detail。"
        if matches
        else "未找到候选；可尝试换用症状词、工具链词、主题词，或读取 turn pointer 补充原始证据。"
    )
    return {
        "status": "complete",
        "tool": "search_related_memories",
        "query": normalized_query,
        "terms": terms,
        "filters": {
            "memory_type": memory_type,
            "domain": domain,
            "project_key": project_key,
            "include_inactive": include_inactive,
            "include_test_data": include_test_data,
            "limit": limit,
            "min_term_matches": min_term_matches,
        },
        "matches": matches,
        "count": len(matches),
        "guidance": guidance,
    }


def get_current_turn_evidence_summary(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    turn_id: str,
    layers: list[str] | None = None,
    max_items_per_layer: int = 12,
    max_chars: int = 800,
) -> dict[str, Any]:
    """Return a compact read-only summary of the current turn evidence pool.

    This tool is meant for ReAct runs that have executed several other tools and
    need to re-check the current turn before writing memory changes.  It does not
    expand raw pointers; it only returns bounded evidence summaries and legal IDs.
    """

    max_items_per_layer = max(1, min(int(max_items_per_layer), 50))
    max_chars = max(80, min(int(max_chars), 4000))
    detail = get_turn_evidence(
        pg_config_path=pg_config_path,
        turn_id=turn_id,
        layers=layers or ["core", "candidate", "pointer"],
        max_chars=max_chars,
    )
    evidence = detail.get("evidence") if isinstance(detail.get("evidence"), Mapping) else {}
    found = bool(detail.get("found"))
    raw_layers = evidence.get("layers") if isinstance(evidence.get("layers"), Mapping) else {}
    compact_layers: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    evidence_ids: list[str] = []
    pointer_ids: list[str] = []
    for layer in (layers or ["core", "candidate", "pointer"]):
        items = raw_layers.get(layer) if isinstance(raw_layers, Mapping) else []
        if not isinstance(items, list):
            items = []
        counts[layer] = len(items)
        compact: list[dict[str, Any]] = []
        for item in items[:max_items_per_layer]:
            if not isinstance(item, Mapping):
                continue
            evidence_id = str(item.get("evidence_id") or "")
            raw_pointer_id = str(item.get("raw_pointer_id") or "")
            if evidence_id:
                evidence_ids.append(evidence_id)
            if layer == "pointer" and evidence_id:
                pointer_ids.append(evidence_id)
            compact.append(
                {
                    "evidence_id": evidence_id,
                    "layer": item.get("layer") or layer,
                    "source_type": item.get("source_type"),
                    "kind": item.get("kind"),
                    "text": item.get("text"),
                    "raw_pointer_id": raw_pointer_id or None,
                    "message_id": item.get("message_id"),
                    "part_id": item.get("part_id"),
                    "ordinal": item.get("ordinal"),
                    "truncated": bool(item.get("truncated")),
                }
            )
        compact_layers[layer] = compact
    guidance = (
        "已返回当前 turn 的紧凑证据摘要；写入 memory_changes 时只能引用 evidence_ids 中真实存在的 evidence_id。"
        if found
        else "未找到当前 turn；请确认 turn_id 是否已经入库。"
    )
    return {
        "status": "complete" if found else "not_found",
        "tool": "get_current_turn_evidence_summary",
        "turn_id": turn_id,
        "found": found,
        "turn": evidence.get("turn") if isinstance(evidence, Mapping) else None,
        "counts": counts,
        "layers": compact_layers,
        "evidence_ids": evidence_ids,
        "pointer_ids": pointer_ids,
        "limits": {
            "max_items_per_layer": max_items_per_layer,
            "max_chars": max_chars,
        },
        "guidance": guidance,
    }


def _json_from_psql_object(config: Mapping[str, Any], sql: str) -> dict[str, Any]:
    output = _run_psql(config, sql).strip()
    if not output:
        raise ValueError("query returned no rows")
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        if "Output format is unaligned" in output and "{" not in output:
            raise ValueError("query returned no rows")
        raise ValueError(f"query did not return JSON object: {output[:200]}")
    value = json.loads(output[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("query returned non-object JSON")
    return value


def _json_from_psql_array(config: Mapping[str, Any], sql: str) -> list[dict[str, Any]]:
    output = _run_psql(config, sql).strip()
    if not output:
        return []
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"query did not return JSON array: {output[:200]}")
    value = json.loads(output[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("query returned non-array JSON")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def build_writer_context(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    turn_id: str,
    memory_limit: int = 80,
) -> dict[str, Any]:
    """Build the fixed Writer input: turn evidence + lightweight memory index."""

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    topic_test_filter = _exclude_test_topic_sql("mt")
    memory_test_filter = _exclude_test_memory_sql("m", "mt")
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH selected_turn AS (
  SELECT *
  FROM turns
  WHERE turn_id = {_sql_literal(turn_id)}
    AND user_id = {_sql_literal(user_id)}
),
turn_evidence AS (
  SELECT jsonb_build_object(
    'turn_id', t.turn_id,
    'user_id', t.user_id,
    'project_key', t.project_key,
    'state', t.state,
    'phase_hint', COALESCE(t.final_phase, t.rule_phase),
    'core', COALESCE((
      SELECT jsonb_agg(
        jsonb_build_object(
          'evidence_id', e.evidence_id,
          'kind', e.kind,
          'source_type', e.source_type,
          'text', e.text_or_summary,
          'raw_pointer_id', e.raw_pointer_id
        )
        ORDER BY e.ordinal NULLS LAST, e.evidence_id
      )
      FROM evidence_items e
      WHERE e.turn_id = t.turn_id AND e.layer = 'core'
    ), '[]'::jsonb),
    'candidate', COALESCE((
      SELECT jsonb_agg(
        jsonb_build_object(
          'evidence_id', e.evidence_id,
          'kind', e.kind,
          'source_type', e.source_type,
          'summary', e.text_or_summary,
          'raw_pointer_id', e.raw_pointer_id
        )
        ORDER BY e.ordinal NULLS LAST, e.evidence_id
      )
      FROM evidence_items e
      WHERE e.turn_id = t.turn_id AND e.layer = 'candidate'
    ), '[]'::jsonb)
  ) AS body
  FROM selected_turn t
),
memory_index AS (
  SELECT jsonb_build_object(
    'topics', COALESCE((
      SELECT jsonb_agg(
        jsonb_build_object(
          'topic_id', mt.topic_id,
          'topic_key', mt.topic_key,
          'title', mt.title,
          'summary', mt.summary,
          'topic_type', mt.topic_type,
          'scope', mt.scope,
          'status', mt.status,
          'updated_at', mt.updated_at,
          'memory_count', (
            SELECT count(*) FROM memories m
            WHERE m.topic_id = mt.topic_id AND m.status = 'active'
          )
        )
        ORDER BY mt.updated_at DESC, mt.topic_key
      )
      FROM memory_topics mt
      WHERE mt.user_id = {_sql_literal(user_id)}
        AND mt.status = 'active'
        AND {topic_test_filter}
    ), '[]'::jsonb),
    'memories', COALESCE((
      SELECT jsonb_agg(
        jsonb_build_object(
          'memory_id', m.memory_id,
          'topic_id', m.topic_id,
          'stable_key', m.stable_key,
          'memory_type', m.memory_type,
          'summary', m.summary,
          'scope', m.scope,
          'status', m.status,
          'confidence', m.confidence,
          'trust_level', m.trust_level,
          'updated_at', m.updated_at
        )
        ORDER BY m.updated_at DESC, m.stable_key
      )
      FROM (
        SELECT m.*
        FROM memories m
        JOIN memory_topics mt ON mt.topic_id = m.topic_id AND mt.user_id = m.user_id
        WHERE m.user_id = {_sql_literal(user_id)}
          AND m.status = 'active'
          AND mt.status = 'active'
          AND {memory_test_filter}
        ORDER BY m.updated_at DESC, m.stable_key
        LIMIT {int(memory_limit)}
      ) m
    ), '[]'::jsonb)
  ) AS body
)
SELECT jsonb_build_object(
  'schema_version', 'memory-lab-writer-context-v1',
  'turn_evidence', (SELECT body FROM turn_evidence),
  'memory_index', (SELECT body FROM memory_index),
  'available_tools', jsonb_build_array('expand_pointer', 'read_memory_detail')
)::text;
"""
    result = _json_from_psql_object(config, sql)
    if result.get("turn_evidence") is None:
        raise ValueError(f"turn not found in PostgreSQL: {turn_id}")
    return {"status": "complete", "context": result}


def expand_pointer(
    source_database: Path,
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    raw_pointer_id: str,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Expand one raw pointer through PostgreSQL metadata back to CP SQLite."""

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    pointer_sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT jsonb_build_object(
  'raw_pointer_id', e.evidence_id,
  'turn_id', e.turn_id,
  'source_type', e.source_type,
  'kind', e.kind,
  'message_id', e.message_id,
  'part_id', e.part_id,
  'preview', e.text_or_summary
)::text
FROM evidence_items e
WHERE e.evidence_id = {_sql_literal(raw_pointer_id)}
  AND e.layer = 'pointer'
  AND e.user_id = {_sql_literal(user_id)}
LIMIT 1;
"""
    try:
        pointer = _json_from_psql_object(config, pointer_sql)
    except ValueError as error:
        raise ValueError(f"pointer not found: {raw_pointer_id}") from error
    part_id = str(pointer.get("part_id") or "")
    if not part_id:
        raise ValueError(f"pointer has no part_id: {raw_pointer_id}")
    with sqlite3.connect(source_database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, message_id, session_id, time_created, time_updated, data FROM part WHERE id = ?",
            (part_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"CP SQLite part not found: {part_id}")
    data = str(row["data"] or "")
    truncated = len(data) > max_chars
    return {
        "status": "complete",
        "tool": "expand_pointer",
        "raw_pointer_id": raw_pointer_id,
        "message_id": pointer.get("message_id"),
        "part_id": part_id,
        "source_type": pointer.get("source_type"),
        "kind": pointer.get("kind"),
        "content": data[: max(0, int(max_chars))],
        "truncated": truncated,
        "sqlite": {
            "session_id": row["session_id"],
            "time_created": row["time_created"],
            "time_updated": row["time_updated"],
        },
    }


def read_memory_detail(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    memory_id: str,
) -> dict[str, Any]:
    """Read one lightweight current-state memory with its topic."""

    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT jsonb_build_object(
  'memory_id', m.memory_id,
  'user_id', m.user_id,
  'topic', jsonb_build_object(
    'topic_id', mt.topic_id,
    'topic_key', mt.topic_key,
    'title', mt.title,
    'summary', mt.summary,
    'topic_type', mt.topic_type,
    'scope', mt.scope,
    'status', mt.status
  ),
  'stable_key', m.stable_key,
  'memory_type', m.memory_type,
  'scope', m.scope,
  'summary', m.summary,
  'content', m.content,
  'status', m.status,
  'confidence', m.confidence,
  'trust_level', m.trust_level,
  'metadata', m.metadata,
  'created_at', m.created_at,
  'updated_at', m.updated_at
)::text
FROM memories m
JOIN memory_topics mt ON mt.topic_id = m.topic_id
WHERE m.memory_id = {_sql_literal(memory_id)}
  AND m.user_id = {_sql_literal(user_id)}
LIMIT 1;
"""
    try:
        memory = _json_from_psql_object(config, sql)
    except ValueError as error:
        raise ValueError(f"memory not found: {memory_id}") from error
    return {"status": "complete", "tool": "read_memory_detail", "memory": memory}


def list_topic_memories(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    topic_id: str,
    include_inactive: bool = False,
    include_test_data: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """List bounded current-state memory summaries under one topic."""

    limit = max(1, min(int(limit), 50))
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    topic_status_filter = "true" if include_inactive else "mt.status = 'active'"
    memory_status_filter = "true" if include_inactive else "m.status = 'active'"
    topic_test_filter = "true" if include_test_data else _exclude_test_topic_sql("mt")
    memory_test_filter = "true" if include_test_data else _exclude_test_memory_sql("m", None)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH selected_topic AS (
  SELECT *
  FROM memory_topics mt
  WHERE mt.topic_id = {_sql_literal(topic_id)}
    AND mt.user_id = {_sql_literal(user_id)}
    AND {topic_status_filter}
    AND {topic_test_filter}
)
SELECT COALESCE((
  SELECT jsonb_build_object(
    'found', true,
    'topic', jsonb_build_object(
      'topic_id', mt.topic_id,
      'topic_key', mt.topic_key,
      'title', mt.title,
      'summary', mt.summary,
      'topic_type', mt.topic_type,
      'scope', mt.scope,
      'status', mt.status,
      'updated_at', mt.updated_at
    ),
    'memories', COALESCE((
      SELECT jsonb_agg(
        jsonb_build_object(
          'memory_id', m.memory_id,
          'stable_key', m.stable_key,
          'memory_type', m.memory_type,
          'scope', m.scope,
          'summary', left(m.summary, 500),
          'summary_truncated', length(m.summary) > 500,
          'status', m.status,
          'confidence', m.confidence,
          'trust_level', m.trust_level,
          'updated_at', m.updated_at
        )
        ORDER BY m.updated_at DESC, m.stable_key
      )
      FROM (
        SELECT *
        FROM memories m
        WHERE m.topic_id = mt.topic_id
          AND m.user_id = mt.user_id
          AND {memory_status_filter}
          AND {memory_test_filter}
        ORDER BY m.updated_at DESC, m.stable_key
        LIMIT {limit}
      ) m
    ), '[]'::jsonb),
    'total_active_memory_count', (
      SELECT count(*)
      FROM memories m
      WHERE m.topic_id = mt.topic_id
        AND m.user_id = mt.user_id
        AND {memory_status_filter}
        AND {memory_test_filter}
    )
  )
  FROM selected_topic mt
), jsonb_build_object(
  'found', false,
  'topic', null,
  'memories', '[]'::jsonb,
  'total_active_memory_count', 0
))::text;
"""
    detail = _json_from_psql_object(config, sql)
    found = bool(detail.get("found"))
    return {
        "status": "complete" if found else "not_found",
        "tool": "list_topic_memories",
        "topic_id": topic_id,
        "found": found,
        "include_inactive": include_inactive,
        "limit": limit,
        "topic": detail.get("topic"),
        "memories": detail.get("memories") or [],
        "count": len(detail.get("memories") or []),
        "total_active_memory_count": int(detail.get("total_active_memory_count") or 0),
        "guidance": (
            "如需修改某条 memory，请先调用 read_memory_detail 读取完整正文。"
            if found
            else "未找到 topic_id；请先通过 list_memory_topics 或 search_related_memories 获取有效 topic_id。"
        ),
    }


def read_memory_history(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    memory_id: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Read bounded revision history for one memory."""

    limit = max(1, min(int(limit), 50))
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH selected_memory AS (
  SELECT m.*, mt.topic_key, mt.title AS topic_title
  FROM memories m
  JOIN memory_topics mt ON mt.topic_id = m.topic_id AND mt.user_id = m.user_id
  WHERE m.memory_id = {_sql_literal(memory_id)}
    AND m.user_id = {_sql_literal(user_id)}
),
revision_rows AS (
  SELECT COALESCE(jsonb_agg(
    jsonb_build_object(
      'revision_id', r.revision_id,
      'revision_number', r.revision_number,
      'summary', left(r.summary, 500),
      'summary_truncated', length(r.summary) > 500,
      'content_preview', left(r.content, 1000),
      'content_truncated', length(r.content) > 1000,
      'confidence', r.confidence,
      'trust_level', r.trust_level,
      'status', r.status,
      'change_type', r.change_type,
      'reason', left(r.reason, 500),
      'source_turn_id', r.source_turn_id,
      'created_at', r.created_at
    )
    ORDER BY r.revision_number DESC
  ), '[]'::jsonb) AS revisions
  FROM (
    SELECT *
    FROM memory_revisions r
    WHERE r.memory_id = {_sql_literal(memory_id)}
      AND r.user_id = {_sql_literal(user_id)}
    ORDER BY r.revision_number DESC
    LIMIT {limit}
  ) r
)
SELECT COALESCE((
  SELECT jsonb_build_object(
    'found', true,
    'memory', jsonb_build_object(
      'memory_id', m.memory_id,
      'topic_id', m.topic_id,
      'topic_key', m.topic_key,
      'topic_title', m.topic_title,
      'stable_key', m.stable_key,
      'memory_type', m.memory_type,
      'summary', m.summary,
      'status', m.status,
      'confidence', m.confidence,
      'trust_level', m.trust_level,
      'updated_at', m.updated_at
    ),
    'revisions', (SELECT revisions FROM revision_rows),
    'total_revision_count', (
      SELECT count(*)
      FROM memory_revisions r
      WHERE r.memory_id = m.memory_id
        AND r.user_id = m.user_id
    )
  )
  FROM selected_memory m
), jsonb_build_object(
  'found', false,
  'memory', null,
  'revisions', '[]'::jsonb,
  'total_revision_count', 0
))::text;
"""
    detail = _json_from_psql_object(config, sql)
    found = bool(detail.get("found"))
    return {
        "status": "complete" if found else "not_found",
        "tool": "read_memory_history",
        "memory_id": memory_id,
        "found": found,
        "limit": limit,
        "memory": detail.get("memory"),
        "revisions": detail.get("revisions") or [],
        "count": len(detail.get("revisions") or []),
        "total_revision_count": int(detail.get("total_revision_count") or 0),
        "guidance": (
            "history 只用于判断时间线和变更原因；修改 memory 前仍需 read_memory_detail。"
            if found
            else "未找到 memory_id；请先通过 search_related_memories 或 list_topic_memories 获取有效 memory_id。"
        ),
    }


def list_memory_conflicts(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    memory_id: str | None = None,
    topic_id: str | None = None,
    include_test_data: bool = False,
    status_filter: str | None = None,
    include_resolved: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """List bounded conflict decisions for one memory or one topic."""

    memory_id = str(memory_id or "").strip() or None
    topic_id = str(topic_id or "").strip() or None
    status_filter = str(status_filter or "").strip() or None
    if status_filter and status_filter not in set(CONFLICT_RECORD_STATUSES):
        return {
            "status": "rejected",
            "tool": "list_memory_conflicts",
            "error_code": "invalid_conflict_status",
            "memory_id": memory_id,
            "topic_id": topic_id,
            "status_filter": status_filter,
            "include_resolved": include_resolved,
            "conflicts": [],
            "count": 0,
            "guidance": f"status 只能是 {', '.join(CONFLICT_RECORD_STATUSES)}。旧版状态可用 include_resolved 审计查询，但不能作为过滤条件。",
        }
    if bool(memory_id) == bool(topic_id):
        return {
            "status": "rejected",
            "tool": "list_memory_conflicts",
            "error_code": "invalid_target",
            "memory_id": memory_id,
            "topic_id": topic_id,
            "conflicts": [],
            "count": 0,
            "guidance": "必须且只能提供 memory_id 或 topic_id 其中一个。",
        }
    limit = max(1, min(int(limit), 50))
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    if status_filter:
        conflict_status_filter = f"c.status = {_sql_literal(status_filter)}"
    elif include_resolved:
        conflict_status_filter = "true"
    else:
        active_values = ", ".join(_sql_literal(item) for item in ACTIVE_CONFLICT_RECORD_STATUSES)
        conflict_status_filter = f"c.status IN ({active_values})"
    target_filter = (
        f"c.memory_id = {_sql_literal(memory_id)}"
        if memory_id
        else f"c.topic_id = {_sql_literal(topic_id or '')}"
    )
    target_exists_sql = (
        f"""
        SELECT EXISTS(
          SELECT 1
          FROM memories m
          JOIN memory_topics mt ON mt.topic_id = m.topic_id AND mt.user_id = m.user_id
          WHERE m.user_id = {_sql_literal(user_id)}
            AND m.memory_id = {_sql_literal(memory_id or '')}
            AND {'true' if include_test_data else _exclude_test_memory_sql('m', 'mt')}
        )
        """
        if memory_id
        else f"""
        SELECT EXISTS(
          SELECT 1 FROM memory_topics mt
          WHERE mt.user_id = {_sql_literal(user_id)}
            AND mt.topic_id = {_sql_literal(topic_id or '')}
            AND {'true' if include_test_data else _exclude_test_topic_sql('mt')}
        )
        """
    )
    conflict_test_filter = (
        "true"
        if include_test_data
        else f"""
        (
          (c.memory_id IS NULL OR EXISTS (
            SELECT 1
            FROM memories m
            JOIN memory_topics mt ON mt.topic_id = m.topic_id AND mt.user_id = m.user_id
            WHERE m.user_id = c.user_id
              AND m.memory_id = c.memory_id
              AND {_exclude_test_memory_sql('m', 'mt')}
          ))
          AND
          (c.topic_id IS NULL OR EXISTS (
            SELECT 1
            FROM memory_topics mt
            WHERE mt.user_id = c.user_id
              AND mt.topic_id = c.topic_id
              AND {_exclude_test_topic_sql('mt')}
          ))
        )
        """
    )
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH target AS (
  {target_exists_sql} AS found
),
conflict_rows AS (
  SELECT COALESCE(jsonb_agg(
    jsonb_build_object(
      'conflict_id', c.conflict_id,
      'memory_id', c.memory_id,
      'topic_id', c.topic_id,
      'conflict_type', c.conflict_type,
      'old_claim', left(c.old_claim, 800),
      'new_claim', left(c.new_claim, 800),
      'decision', left(c.decision, 800),
      'reason', left(c.reason, 800),
      'status', c.status,
      'turn_id', c.turn_id,
      'evidence_refs', c.evidence_refs,
      'created_at', c.created_at
    )
    ORDER BY c.created_at DESC, c.conflict_id
  ), '[]'::jsonb) AS conflicts
  FROM (
    SELECT *
    FROM memory_conflict_records c
    WHERE c.user_id = {_sql_literal(user_id)}
      AND {target_filter}
      AND {conflict_test_filter}
      AND {conflict_status_filter}
    ORDER BY c.created_at DESC, c.conflict_id
    LIMIT {limit}
  ) c
)
SELECT jsonb_build_object(
  'found', (SELECT found FROM target),
  'conflicts', (SELECT conflicts FROM conflict_rows),
  'total_conflict_count', (
    SELECT count(*)
    FROM memory_conflict_records c
    WHERE c.user_id = {_sql_literal(user_id)}
      AND {target_filter}
      AND {conflict_test_filter}
      AND {conflict_status_filter}
  )
)::text;
"""
    detail = _json_from_psql_object(config, sql)
    found = bool(detail.get("found"))
    conflicts = detail.get("conflicts") or []
    return {
        "status": "complete" if found else "not_found",
        "tool": "list_memory_conflicts",
        "memory_id": memory_id,
        "topic_id": topic_id,
        "found": found,
        "limit": limit,
        "include_test_data": include_test_data,
        "status_filter": status_filter,
        "include_resolved": include_resolved,
        "active_conflict_statuses": list(ACTIVE_CONFLICT_RECORD_STATUSES),
        "conflicts": conflicts,
        "count": len(conflicts),
        "total_conflict_count": int(detail.get("total_conflict_count") or 0),
        "guidance": (
            "若已有相同冲突记录，优先避免重复记录；必要时补充证据或更新决策。"
            if found
            else "目标不存在；请先通过 read_memory_detail 或 read_topic_detail 确认 ID。"
        ),
    }


def trace_memory_relations(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    memory_id: str,
    direction: str = "both",
    max_depth: int = 1,
    include_test_data: bool = False,
) -> dict[str, Any]:
    """Read one memory's lightweight version history and direct relations."""

    direction = str(direction or "both")
    if direction not in {"backward", "forward", "both"}:
        return {
            "status": "rejected",
            "tool": "trace_memory_relations",
            "memory_id": memory_id,
            "error_code": "invalid_direction",
            "guidance": "direction 只能是 backward、forward 或 both。",
        }
    max_depth = max(1, min(int(max_depth), 5))
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    include_backward = direction in {"backward", "both"}
    include_forward = direction in {"forward", "both"}
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
WITH selected_memory AS (
  SELECT m.*, mt.title AS topic_title, mt.topic_key
  FROM memories m
  JOIN memory_topics mt ON mt.topic_id = m.topic_id AND mt.user_id = m.user_id
  WHERE m.memory_id = {_sql_literal(memory_id)}
    AND m.user_id = {_sql_literal(user_id)}
    AND {'true' if include_test_data else _exclude_test_memory_sql('m', 'mt')}
),
revision_rows AS (
  SELECT COALESCE(jsonb_agg(
    jsonb_build_object(
      'revision_id', r.revision_id,
      'revision_number', r.revision_number,
      'summary', r.summary,
      'confidence', r.confidence,
      'trust_level', r.trust_level,
      'status', r.status,
      'change_type', r.change_type,
      'reason', r.reason,
      'source_turn_id', r.source_turn_id,
      'created_at', r.created_at
    )
    ORDER BY r.revision_number DESC
  ), '[]'::jsonb) AS items
  FROM memory_revisions r
  JOIN selected_memory m ON m.memory_id = r.memory_id AND m.user_id = r.user_id
),
backward_rows AS (
  SELECT COALESCE(jsonb_agg(
    jsonb_build_object(
      'relation_id', rel.relation_id,
      'relation_type', rel.relation_type,
      'reason', rel.reason,
      'source_turn_id', rel.source_turn_id,
      'evidence_refs', rel.evidence_refs,
      'created_at', rel.created_at,
      'from_memory', jsonb_build_object(
        'memory_id', src.memory_id,
        'summary', src.summary,
        'status', src.status,
        'trust_level', src.trust_level,
        'confidence', src.confidence
      ),
      'to_memory_id', rel.to_memory_id
    )
    ORDER BY rel.created_at DESC, rel.relation_id
  ), '[]'::jsonb) AS items
  FROM memory_relations rel
  JOIN selected_memory m ON m.memory_id = rel.to_memory_id AND m.user_id = rel.user_id
  JOIN memories src ON src.memory_id = rel.from_memory_id AND src.user_id = rel.user_id
  JOIN memory_topics src_mt ON src_mt.topic_id = src.topic_id AND src_mt.user_id = src.user_id
  WHERE {str(include_backward).lower()}
    AND {'true' if include_test_data else _exclude_test_memory_sql('src', 'src_mt')}
),
forward_rows AS (
  SELECT COALESCE(jsonb_agg(
    jsonb_build_object(
      'relation_id', rel.relation_id,
      'relation_type', rel.relation_type,
      'reason', rel.reason,
      'source_turn_id', rel.source_turn_id,
      'evidence_refs', rel.evidence_refs,
      'created_at', rel.created_at,
      'from_memory_id', rel.from_memory_id,
      'to_memory', jsonb_build_object(
        'memory_id', dst.memory_id,
        'summary', dst.summary,
        'status', dst.status,
        'trust_level', dst.trust_level,
        'confidence', dst.confidence
      )
    )
    ORDER BY rel.created_at DESC, rel.relation_id
  ), '[]'::jsonb) AS items
  FROM memory_relations rel
  JOIN selected_memory m ON m.memory_id = rel.from_memory_id AND m.user_id = rel.user_id
  JOIN memories dst ON dst.memory_id = rel.to_memory_id AND dst.user_id = rel.user_id
  JOIN memory_topics dst_mt ON dst_mt.topic_id = dst.topic_id AND dst_mt.user_id = dst.user_id
  WHERE {str(include_forward).lower()}
    AND {'true' if include_test_data else _exclude_test_memory_sql('dst', 'dst_mt')}
)
SELECT COALESCE((
  SELECT jsonb_build_object(
    'found', true,
    'memory', jsonb_build_object(
      'memory_id', m.memory_id,
      'topic_id', m.topic_id,
      'topic_key', m.topic_key,
      'topic_title', m.topic_title,
      'stable_key', m.stable_key,
      'memory_type', m.memory_type,
      'summary', m.summary,
      'status', m.status,
      'confidence', m.confidence,
      'trust_level', m.trust_level,
      'updated_at', m.updated_at
    ),
    'revisions', (SELECT items FROM revision_rows),
    'backward', (SELECT items FROM backward_rows),
    'forward', (SELECT items FROM forward_rows)
  )
  FROM selected_memory m
), jsonb_build_object('found', false, 'memory', null, 'revisions', '[]'::jsonb, 'backward', '[]'::jsonb, 'forward', '[]'::jsonb))::text;
"""
    detail = _json_from_psql_object(config, sql)
    guidance = (
        "backward 表示哪些旧记忆指向当前记忆；forward 表示当前记忆后来指向哪些新记忆。"
        if detail.get("found")
        else "未找到 memory_id；请先通过 search_related_memories 或 read_topic_detail 获取有效 memory_id。"
    )
    return {
        "status": "complete" if detail.get("found") else "not_found",
        "tool": "trace_memory_relations",
        "memory_id": memory_id,
        "direction": direction,
        "max_depth": max_depth,
        "include_test_data": include_test_data,
        "found": bool(detail.get("found")),
        "detail": detail,
        "guidance": guidance,
    }


def _query_json(config: Mapping[str, Any], sql: str) -> Any:
    output = _run_psql(config, sql).strip()
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        start = output.find("{")
        end = output.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"query did not return JSON: {output[:200]}")
    return json.loads(output[start : end + 1])


def _load_pg_turns(config: Mapping[str, Any], user_id: str, limit: int) -> list[dict[str, Any]]:
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT COALESCE(jsonb_agg(body ORDER BY occurred_at), '[]'::jsonb)::text
FROM (
  SELECT
    t.occurred_at,
    jsonb_build_object(
      'turn_id', t.turn_id,
      'rule_phase', t.rule_phase,
      'state', t.state,
      'project_key', t.project_key,
      'evidence', (
        SELECT COALESCE(jsonb_agg(
          jsonb_build_object(
            'evidence_id', e.evidence_id,
            'layer', e.layer,
            'source_type', e.source_type,
            'kind', e.kind,
            'text_or_summary', e.text_or_summary,
            'raw_pointer_id', e.raw_pointer_id,
            'message_id', e.message_id,
            'part_id', e.part_id,
            'ordinal', e.ordinal
          )
          ORDER BY
            CASE e.layer WHEN 'core' THEN 1 WHEN 'candidate' THEN 2 WHEN 'pointer' THEN 3 ELSE 4 END,
            e.ordinal NULLS LAST,
            e.evidence_id
        ), '[]'::jsonb)
        FROM evidence_items AS e
        WHERE e.turn_id = t.turn_id
      )
    ) AS body
  FROM turns AS t
  WHERE t.user_id = {_sql_literal(user_id)}
  ORDER BY t.occurred_at DESC NULLS LAST, t.turn_id DESC
  LIMIT {int(limit)}
) AS limited;
"""
    value = _query_json(config, sql)
    if not isinstance(value, list):
        raise ValueError("PostgreSQL turn query returned non-list JSON")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _sqlite_part_exists(connection: sqlite3.Connection, part_id: str | None) -> bool:
    if not part_id:
        return False
    row = connection.execute("SELECT 1 FROM part WHERE id = ? LIMIT 1", (part_id,)).fetchone()
    return row is not None


def _layer_counts(evidence: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"core": 0, "candidate": 0, "pointer": 0}
    for item in evidence:
        layer = str(item.get("layer") or "")
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def _phase_sufficiency(turn: Mapping[str, Any], evidence: list[Mapping[str, Any]]) -> dict[str, Any]:
    phase = str(turn.get("rule_phase") or "")
    core_kinds = {str(item.get("kind") or "") for item in evidence if item.get("layer") == "core"}
    candidate_kinds = {str(item.get("kind") or "") for item in evidence if item.get("layer") == "candidate"}
    problems: list[str] = []
    if not core_kinds:
        problems.append("缺少 core 证据")
    if phase == "requirement_alignment":
        if not ({"requirement_signal", "user_contract"} & core_kinds):
            problems.append("需求对齐缺少用户需求/约束 core")
        if "requirement_signal" not in candidate_kinds:
            problems.append("需求对齐缺少候选需求选项")
    elif phase == "compilation":
        if not ({"build_or_tool_result", "compile_conclusion"} & core_kinds):
            problems.append("编译阶段缺少构建结果或编译结论 core")
    elif phase == "runtime_symptom_diagnosis":
        if "runtime_symptom" not in core_kinds:
            problems.append("运行诊断缺少用户现象 core")
        if not ({"diagnosis_or_solution", "assistant_conclusion"} & core_kinds):
            problems.append("运行诊断缺少助手诊断/结论 core")
    elif phase == "fix_confirmation":
        if "user_confirmation" not in core_kinds:
            problems.append("修复确认缺少用户验证 core")
        if not ({"diagnosis_or_solution", "assistant_conclusion"} & core_kinds):
            problems.append("修复确认缺少修复回顾/结论 core")
    return {
        "ok": not problems,
        "problems": problems,
        "core_kinds": sorted(core_kinds),
        "candidate_kinds": sorted(candidate_kinds),
    }


def _candidate_cleanliness(evidence: list[Mapping[str, Any]]) -> dict[str, Any]:
    noisy_tokens = (
        " completed {",
        "<path>",
        "</path>",
        "<content>",
        "</content>",
        "<WORKSPACE>",
        "<ABS_PATH>",
        "read completed",
        "write completed",
        "edit completed",
        "glob completed",
        "question completed",
    )
    candidates = [item for item in evidence if item.get("layer") == "candidate"]
    noisy: list[dict[str, str]] = []
    for item in candidates:
        text = str(item.get("text_or_summary") or "")
        if any(token in text for token in noisy_tokens):
            noisy.append(
                {
                    "evidence_id": str(item.get("evidence_id") or ""),
                    "preview": text[:180],
                }
            )
    return {
        "candidate_count": len(candidates),
        "noisy_count": len(noisy),
        "ok": len(noisy) == 0,
        "noisy_examples": noisy[:5],
    }


def _pointer_integrity(
    source_database: Path,
    turns: list[Mapping[str, Any]],
) -> dict[str, Any]:
    all_evidence: list[Mapping[str, Any]] = []
    for turn in turns:
        evidence = turn.get("evidence")
        if isinstance(evidence, list):
            all_evidence.extend(item for item in evidence if isinstance(item, Mapping))
    pointers = {
        str(item.get("evidence_id")): item
        for item in all_evidence
        if item.get("layer") == "pointer"
    }
    linked = [
        item
        for item in all_evidence
        if item.get("layer") in {"core", "candidate"}
    ]
    missing_link = [item for item in linked if not item.get("raw_pointer_id")]
    missing_pointer = [
        item for item in linked if item.get("raw_pointer_id") and str(item.get("raw_pointer_id")) not in pointers
    ]
    sqlite_missing: list[dict[str, str]] = []
    with sqlite3.connect(source_database) as connection:
        for item in linked:
            pointer_id = item.get("raw_pointer_id")
            pointer = pointers.get(str(pointer_id))
            if not pointer:
                continue
            if not _sqlite_part_exists(connection, str(pointer.get("part_id") or "")):
                sqlite_missing.append(
                    {
                        "evidence_id": str(item.get("evidence_id") or ""),
                        "raw_pointer_id": str(pointer_id or ""),
                        "part_id": str(pointer.get("part_id") or ""),
                    }
                )
    return {
        "linked_core_candidate": len(linked),
        "with_raw_pointer_id": len(linked) - len(missing_link),
        "pointer_row_found": len(linked) - len(missing_link) - len(missing_pointer),
        "sqlite_part_found": len(linked) - len(missing_link) - len(missing_pointer) - len(sqlite_missing),
        "ok": not missing_link and not missing_pointer and not sqlite_missing,
        "missing_link_examples": [
            {"evidence_id": str(item.get("evidence_id") or ""), "layer": str(item.get("layer") or "")}
            for item in missing_link[:5]
        ],
        "missing_pointer_examples": [
            {"evidence_id": str(item.get("evidence_id") or ""), "raw_pointer_id": str(item.get("raw_pointer_id") or "")}
            for item in missing_pointer[:5]
        ],
        "sqlite_missing_examples": sqlite_missing[:5],
    }


def postgres_quality_report(
    source_database: Path,
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    limit: int = 20,
) -> dict[str, Any]:
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    turns = _load_pg_turns(config, user_id, limit)
    turn_reports: list[dict[str, Any]] = []
    aggregate_layers = {"core": 0, "candidate": 0, "pointer": 0}
    for turn in turns:
        evidence = turn.get("evidence") if isinstance(turn.get("evidence"), list) else []
        evidence = [item for item in evidence if isinstance(item, Mapping)]
        counts = _layer_counts(evidence)
        for key, value in counts.items():
            aggregate_layers[key] = aggregate_layers.get(key, 0) + value
        sufficiency = _phase_sufficiency(turn, evidence)
        cleanliness = _candidate_cleanliness(evidence)
        core_preview = [
            {
                "kind": str(item.get("kind") or ""),
                "preview": str(item.get("text_or_summary") or "")[:220],
            }
            for item in evidence
            if item.get("layer") == "core"
        ][:4]
        candidate_preview = [
            {
                "kind": str(item.get("kind") or ""),
                "preview": str(item.get("text_or_summary") or "")[:220],
                "raw_pointer_id": str(item.get("raw_pointer_id") or ""),
            }
            for item in evidence
            if item.get("layer") == "candidate"
        ][:6]
        turn_reports.append(
            {
                "turn_id": turn.get("turn_id"),
                "phase": turn.get("rule_phase"),
                "state": turn.get("state"),
                "project_key": turn.get("project_key"),
                "counts": counts,
                "information_sufficient": sufficiency,
                "candidate_cleanliness": cleanliness,
                "core_preview": core_preview,
                "candidate_preview": candidate_preview,
            }
        )
    pointer = _pointer_integrity(source_database, turns)
    sufficient_turns = sum(1 for item in turn_reports if item["information_sufficient"]["ok"])
    clean_turns = sum(1 for item in turn_reports if item["candidate_cleanliness"]["ok"])
    return {
        "status": "complete",
        "user_id": user_id,
        "turn_count": len(turn_reports),
        "aggregate_layers": aggregate_layers,
        "summary": {
            "evidence_pool_ready": bool(turn_reports) and clean_turns == len(turn_reports) and pointer["ok"],
            "ready_for_next_chain": bool(turn_reports) and clean_turns == len(turn_reports) and pointer["ok"],
            "candidate_clean_turns": clean_turns,
            "pointer_integrity_ok": pointer["ok"],
            "information_sufficient_turns": sufficient_turns,
            "information_sufficiency_is_blocking": False,
            "warning": (
                "阶段充分性只作为提示；信息不足时由后续 Writer 通过 raw_pointer_id/turn_id 请求补证据。"
            ),
        },
        "pointer_integrity": pointer,
        "turns": turn_reports,
    }


__all__ = [
    "DEFAULT_PG_CONFIG",
    "get_turn_evidence",
    "ingest_turn_to_postgres",
    "list_previous_session_turns",
    "list_open_memory_chains",
    "list_memory_conflicts",
    "load_pg_config",
    "list_topic_memories",
    "read_session_turn_evidence",
    "save_memory_chain_state",
    "read_memory_history",
    "postgres_quality_report",
    "show_postgres_turn",
]
