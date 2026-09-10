"""DeepSeek shadow evaluation isolated from authoritative admission and memory."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .config import DEFAULT_CONFIG, load_llm_config
from .consolidation import (
    canonicalize_scope,
    latest_admission_snapshot,
    semantic_value_for,
)
from .judge import (
    INPUT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    PROMPT_VERSION,
    ApiJudge,
    JudgeError,
    JudgeValidationError,
    MemoryJudge,
    request_hash,
    validate_judge_result,
)
from .store import now_iso, open_store


SHADOW_VERSION = "llm-shadow-v1"
FORMAL_TABLES = ("memory", "memory_revision", "evidence", "conflict")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def formal_state_snapshot(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Fingerprint every formal row, not just table counts."""

    snapshot: dict[str, dict[str, Any]] = {}
    for table in FORMAL_TABLES:
        table_info = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        primary_keys = [
            row["name"]
            for row in sorted(table_info, key=lambda item: item["pk"])
            if row["pk"]
        ]
        order = ", ".join(f'"{column}"' for column in primary_keys) or "rowid"
        rows = [
            dict(row)
            for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY {order}')
        ]
        encoded = _json(rows).encode("utf-8")
        snapshot[table] = {
            "row_count": len(rows),
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return snapshot


def _eligible_shadow_rows(
    connection: sqlite3.Connection, admission_run_id: str
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT d.*, e.source_id, e.session_id, e.message_id, e.part_id,
               e.directory, e.timestamp, e.content, e.parent_session_id,
               e.event_type, e.eligibility_status, a.semantic_content,
               sc.project_key
        FROM admission_decision AS d
        JOIN event AS e ON e.event_id = d.event_id
        JOIN event_annotation AS a ON a.event_id = e.event_id
        JOIN session_context AS sc
          ON sc.source_id = e.source_id AND sc.session_id = e.session_id
        WHERE d.run_id = ?
          AND e.event_type = 'user_text'
          AND e.eligibility_status = 'eligible'
          AND e.parent_session_id IS NULL
          AND EXISTS(
              SELECT 1
              FROM turn_event AS te
              JOIN conversation_turn AS turn ON turn.turn_id = te.turn_id
              WHERE te.event_id = e.event_id
                AND turn.current_state = 'completed'
          )
        ORDER BY e.timestamp, e.event_id, d.ordinal
        """,
        (admission_run_id,),
    ).fetchall()


def _judge_scope(raw_scope: Mapping[str, Any]) -> dict[str, str] | None:
    canonical = canonicalize_scope(dict(raw_scope))
    if canonical is None:
        return None
    if canonical.get("conditions"):
        # Judge v1 has no condition fields.  Omitting them would silently merge
        # version/toolchain/chip-specific memories, so leave them out of shadow
        # routing until the schema can represent the full identity.
        return None
    level = canonical["level"]
    if level == "global":
        return {"level": "global"}
    if level == "domain":
        return {"level": "domain", "domain": str(canonical["domain"])}
    if level == "project":
        return {"level": "project", "project_key": str(canonical["project_key"])}
    return None


def _scope_limit(raw_scope_json: str) -> dict[str, Any]:
    try:
        raw_scope = json.loads(raw_scope_json)
    except (TypeError, json.JSONDecodeError):
        return {"allowed_levels": [], "reason": "invalid_scope_evidence"}
    if raw_scope.get("level") in {"unknown", "unresolved"}:
        return {"allowed_levels": [], "reason": "unresolved_scope_evidence"}
    scope = _judge_scope(raw_scope)
    if scope is None:
        return {"allowed_levels": [], "reason": "unsupported_scope_evidence"}
    result: dict[str, Any] = {"allowed_levels": [scope["level"]]}
    if scope["level"] == "domain":
        result["domain"] = scope["domain"]
    elif scope["level"] == "project":
        result["project_key"] = scope["project_key"]
    return result


def _memory_targets(connection: sqlite3.Connection, stable_key: str | None) -> list[dict]:
    if not stable_key:
        return []
    rows = connection.execute(
        """
        SELECT m.memory_id, m.stable_key, m.scope_json, m.current_revision_id,
               r.semantic_value, r.statement
        FROM memory AS m
        JOIN memory_revision AS r ON r.revision_id = m.current_revision_id
        WHERE m.status = 'active' AND m.stable_key = ?
        ORDER BY m.updated_at DESC, m.memory_id
        LIMIT 5
        """,
        (stable_key,),
    ).fetchall()
    targets: list[dict] = []
    for row in rows:
        scope = _judge_scope(json.loads(row["scope_json"]))
        if scope is None:
            continue
        conflicts = [
            item[0]
            for item in connection.execute(
                """
                SELECT incoming_semantic_value FROM conflict
                WHERE memory_id = ? AND status = 'open'
                ORDER BY created_at, conflict_id
                """,
                (row["memory_id"],),
            )
        ]
        targets.append(
            {
                "memory_id": row["memory_id"],
                "stable_key": row["stable_key"],
                "scope": scope,
                "revision_id": row["current_revision_id"],
                "value": row["semantic_value"],
                "statement": row["statement"],
                "open_conflict_values": conflicts,
            }
        )
    return targets[:5]


def _request_for_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    prior_rule_targets: list[dict],
) -> dict:
    signals = json.loads(row["signals_json"])
    coverage = json.loads(row["instruction_coverage_json"])
    stable_key = row["stable_key_suggestion"]
    pending = [
        item for item in reversed(prior_rule_targets) if item["stable_key"] == stable_key
    ][:5]
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "request_id": "llmreq_" + row["candidate_key"],
        "claim": {
            "candidate_key": row["candidate_key"],
            "event_id": row["event_id"],
            "part_id": row["part_id"],
            "source_text": row["semantic_content"],
            "exact_quote": row["exact_quote"],
            "start_offset": row["start_offset"],
            "end_offset": row["end_offset"],
        },
        "context": {
            "session_id": row["session_id"],
            "project_key": row["project_key"],
            "directory": row["directory"],
            "occurred_at": row["timestamp"] or 0,
        },
        "hard_signals": {
            "long_term": list(signals.get("long_term", [])),
            "temporary": list(signals.get("temporary", [])),
            "correction": list(signals.get("correction", [])),
            "instruction_coverage": {
                "status": coverage.get("status", "none"),
                "scope": None,
            },
            "sensitive": row["disposition"] == "sensitive_blocked",
            "scope_limit": _scope_limit(row["scope_json"]),
        },
        "current_memories": _memory_targets(connection, stable_key),
        "pending_proposals": pending,
    }


def _blocked_request_hash(row: sqlite3.Row) -> str:
    """Create an audit key without putting sensitive content in a model request."""

    identity = f"{SHADOW_VERSION}\0{row['decision_id']}\0{row['candidate_key']}"
    return "blocked_req_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _rule_target(row: sqlite3.Row) -> dict | None:
    stable_key = row["stable_key_suggestion"]
    if not stable_key:
        return None
    try:
        raw_scope = json.loads(row["scope_json"])
    except (TypeError, json.JSONDecodeError):
        return None
    scope = _judge_scope(raw_scope)
    semantic_value = semantic_value_for(stable_key, row["exact_quote"])
    if scope is None or semantic_value is None:
        return None
    return {
        "proposal_id": "rule_target_" + row["decision_id"],
        "stable_key": stable_key,
        "scope": scope,
        "value": semantic_value,
        "statement": semantic_value,
    }


def _cached_result(
    connection: sqlite3.Connection,
    digest: str,
    model: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT judgment_id, result_json
        FROM llm_judgment
        WHERE request_hash = ? AND model = ? AND prompt_version = ?
          AND input_schema_version = ? AND output_schema_version = ?
          AND status IN ('valid', 'replayed')
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """,
        (
            digest,
            model,
            PROMPT_VERSION,
            INPUT_SCHEMA_VERSION,
            OUTPUT_SCHEMA_VERSION,
        ),
    ).fetchone()


def _safe_error(error: Exception) -> tuple[str, str]:
    code = type(error).__name__
    message = str(error)
    return code[:80], message[:500]


def _insert_judgment(
    connection: sqlite3.Connection,
    *,
    shadow_run_id: str,
    admission_run_id: str,
    row: sqlite3.Row,
    digest: str,
    model: str,
    status: str,
    execution_mode: str,
    result: Mapping[str, Any] | None = None,
    error: Exception | None = None,
    replayed_from: str | None = None,
) -> dict:
    judgment_id = "llmjudgment_" + uuid4().hex
    error_code, error_message = _safe_error(error) if error else (None, None)
    relation = result.get("relation") if result else None
    value = result.get("value") if result else None
    connection.execute(
        """
        INSERT INTO llm_judgment(
            judgment_id, shadow_run_id, admission_run_id, decision_id,
            candidate_key, request_hash, model, prompt_version,
            input_schema_version, output_schema_version, status, result_json,
            execution_mode,
            should_store, memory_type, stable_key, scope_json, semantic_value,
            relation, target_kind, target_id, confidence, error_code,
            error_message, replayed_from_judgment_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            judgment_id,
            shadow_run_id,
            admission_run_id,
            row["decision_id"],
            row["candidate_key"],
            digest,
            model,
            PROMPT_VERSION,
            INPUT_SCHEMA_VERSION,
            OUTPUT_SCHEMA_VERSION,
            status,
            _json(result) if result else None,
            execution_mode,
            int(bool(result["should_store"])) if result else None,
            result.get("type") if result else None,
            result.get("key") if result else None,
            _json(result.get("scope")) if result and result.get("scope") else None,
            value.get("normalized") if isinstance(value, Mapping) else None,
            relation.get("kind") if isinstance(relation, Mapping) else None,
            relation.get("target_kind") if isinstance(relation, Mapping) else None,
            relation.get("target_id") if isinstance(relation, Mapping) else None,
            result.get("confidence") if result else None,
            error_code,
            error_message,
            replayed_from,
            now_iso(),
        ),
    )
    return {
        "judgment_id": judgment_id,
        "candidate_key": row["candidate_key"],
        "status": status,
        "execution_mode": execution_mode,
        "result": dict(result) if result else None,
        "error_code": error_code,
        "error_message": error_message,
        "replayed_from_judgment_id": replayed_from,
    }


def run_llm_shadow(
    lab_database: Path,
    judge: MemoryJudge,
    *,
    provider: str,
    model: str,
    refresh: bool = False,
) -> dict:
    """Judge the latest complete admission snapshot without touching formal memory."""

    shadow_run_id = "llmshadow_" + uuid4().hex
    started_at = now_iso()
    with closing(open_store(lab_database)) as connection:
        admission, _snapshot_rows, admission_snapshot_hash = latest_admission_snapshot(
            connection
        )
        if admission is None:
            raise ValueError("no complete admission snapshot exists; run admit first")
        if admission_snapshot_hash is None:
            raise ValueError("latest complete admission snapshot is structurally incomplete")
        latest_admission_attempt = connection.execute(
            """
            SELECT run_id, status, completed_at FROM admission_run
            ORDER BY started_at DESC, rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if (
            latest_admission_attempt is None
            or latest_admission_attempt["run_id"] != admission["run_id"]
        ):
            raise ValueError(
                "latest admission attempt is not a complete valid snapshot; rerun admit"
            )
        rows = _eligible_shadow_rows(connection, admission["run_id"])
        if len(rows) != len(_snapshot_rows):
            raise ValueError(
                "latest admission snapshot cannot be joined to the complete shadow input set"
            )
        before = formal_state_snapshot(connection)
        connection.execute(
            """
            INSERT INTO llm_shadow_run(
                shadow_run_id, shadow_version, admission_run_id,
                admission_snapshot_hash, provider, model,
                prompt_version, input_schema_version, output_schema_version,
                started_at, status, candidate_count, formal_state_before_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                shadow_run_id,
                SHADOW_VERSION,
                admission["run_id"],
                admission_snapshot_hash,
                provider,
                model,
                PROMPT_VERSION,
                INPUT_SCHEMA_VERSION,
                OUTPUT_SCHEMA_VERSION,
                started_at,
                len(rows),
                _json(before),
            ),
        )
        connection.commit()

        results: list[dict] = []
        prior_rule_targets: list[dict] = []
        for row in rows:
            if row["disposition"] == "sensitive_blocked":
                item = _insert_judgment(
                    connection,
                    shadow_run_id=shadow_run_id,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=_blocked_request_hash(row),
                    model=model,
                    status="blocked",
                    execution_mode="local_block",
                    error=JudgeValidationError(
                        "candidate blocked locally by the sensitive-data veto"
                    ),
                )
                connection.commit()
                results.append(item)
                continue
            digest = _blocked_request_hash(row).replace("blocked_req_", "local_req_", 1)
            try:
                request = _request_for_row(connection, row, prior_rule_targets)
                digest = request_hash(request)
                cached = None if refresh else _cached_result(connection, digest, model)
                if cached is not None:
                    result = validate_judge_result(json.loads(cached["result_json"]), request)
                    item = _insert_judgment(
                        connection,
                        shadow_run_id=shadow_run_id,
                        admission_run_id=admission["run_id"],
                        row=row,
                        digest=digest,
                        model=model,
                        status="replayed",
                        execution_mode="cache_replay",
                        result=result,
                        replayed_from=cached["judgment_id"],
                    )
                else:
                    result = judge.judge(request)
                    item = _insert_judgment(
                        connection,
                        shadow_run_id=shadow_run_id,
                        admission_run_id=admission["run_id"],
                        row=row,
                        digest=digest,
                        model=model,
                        status="valid",
                        execution_mode="judge_call",
                        result=result,
                    )
            except JudgeError as error:
                item = _insert_judgment(
                    connection,
                    shadow_run_id=shadow_run_id,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=digest,
                    model=model,
                    status="invalid",
                    execution_mode="judge_call",
                    error=error,
                )
            except Exception as error:
                item = _insert_judgment(
                    connection,
                    shadow_run_id=shadow_run_id,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=digest,
                    model=model,
                    status="invalid",
                    execution_mode="judge_call",
                    error=JudgeError(
                        f"unexpected local judge failure: {type(error).__name__}"
                    ),
                )
            connection.commit()
            results.append(item)
            target = _rule_target(row)
            if target is not None:
                prior_rule_targets.append(target)

        after = formal_state_snapshot(connection)
        if before != after:
            error = "formal memory state changed during shadow evaluation"
            connection.execute(
                """
                UPDATE llm_shadow_run
                SET completed_at = ?, status = 'failed', formal_state_after_json = ?, error = ?
                WHERE shadow_run_id = ?
                """,
                (now_iso(), _json(after), error, shadow_run_id),
            )
            connection.commit()
            raise RuntimeError(error)

        counts = Counter(item["status"] for item in results)
        status = "complete" if not counts.get("invalid") else "complete_with_errors"
        connection.execute(
            """
            UPDATE llm_shadow_run
            SET completed_at = ?, status = ?, success_count = ?, error_count = ?,
                replay_count = ?, counts_json = ?, formal_state_after_json = ?
            WHERE shadow_run_id = ?
            """,
            (
                now_iso(),
                status,
                counts.get("valid", 0) + counts.get("replayed", 0),
                counts.get("invalid", 0),
                counts.get("replayed", 0),
                _json(dict(sorted(counts.items()))),
                _json(after),
                shadow_run_id,
            ),
        )
        connection.commit()

    return {
        "shadow_run_id": shadow_run_id,
        "admission_run_id": admission["run_id"],
        "admission_snapshot_hash": admission_snapshot_hash,
        "provider": provider,
        "model": model,
        "status": status,
        "candidate_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "formal_state_before": before,
        "formal_state_after": after,
        "formal_state_unchanged": before == after,
        "judgments": results,
    }


def run_deepseek_shadow(
    lab_database: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    refresh: bool = False,
) -> dict:
    config = load_llm_config(config_path)
    if config.provider != "deepseek":
        raise ValueError("the configured provider is not deepseek")
    return run_llm_shadow(
        lab_database,
        ApiJudge.from_config(config_path),
        provider=config.provider,
        model=config.model,
        refresh=refresh,
    )
