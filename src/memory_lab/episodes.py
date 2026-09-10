"""Build privacy-bounded, non-semantic engineering trace episodes.

Only turns already stabilized by :mod:`memory_lab.turns` are eligible.  Source
assistant messages and tool payloads are inspected through a read-only SQLite
connection, reduced to structural evidence, and then discarded.  In
particular, assistant prose, reasoning, commands, paths, logs, and errors are
never copied into the lab database.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any
from uuid import uuid4

from .prepare import project_identity
from .safety import require_distinct_source_and_lab, require_memory_lab_database
from .source import open_read_only, source_id
from .store import now_iso, open_store


EXTRACTOR_VERSION = "cp-trace-episode-v3"
TERMINAL_TURN_STATES = {"completed", "failed"}
SAFE_TOOL_STATUSES = {"pending", "running", "completed", "error"}
SAFE_TOOL_NAMES = {
    "bash",
    "keil_build",
    "json_api",
    "ssot",
    "edit",
    "write",
    "read",
    "glob",
    "grep",
    "rag",
}
EXPLICIT_READ_ONLY_TOOL_NAMES = {"read", "glob", "grep", "rag"}
SAFE_FAILURE_KINDS = {"source", "build_runner_missing"}
DOMAIN_OUTCOMES = {
    "verified_success",
    "verified_failure",
    "ambiguous",
    "unverified",
}


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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_integer(value: Any, *, nonnegative: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < -(2**63) or value > 2**63 - 1:
        return None
    if nonnegative and value < 0:
        return None
    return value


def _safe_boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _safe_tool_name(value: Any) -> str:
    if not isinstance(value, str):
        return "other"
    normalized = value.casefold()
    return normalized if normalized in SAFE_TOOL_NAMES else "other"


def _safe_status(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = value.casefold()
    return normalized if normalized in SAFE_TOOL_STATUSES else "other"


def _safe_failure_kind(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.casefold()
    if normalized in SAFE_FAILURE_KINDS:
        return normalized
    return "other" if value else None


def _safe_metadata(tool_name: str, value: Any) -> dict[str, bool | int | str]:
    """Copy the explicit tool-specific allow-list and nothing else."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, bool | int | str] = {}
    if tool_name == "bash":
        exit_code = _safe_integer(value.get("exit"))
        if exit_code is not None:
            result["exit"] = exit_code
    elif tool_name == "keil_build":
        success = _safe_boolean(value.get("success"))
        error_count = _safe_integer(value.get("errorCount"), nonnegative=True)
        warning_count = _safe_integer(value.get("warningCount"), nonnegative=True)
        failure_kind = _safe_failure_kind(value.get("failureKind"))
        build_failure_kind = _safe_failure_kind(value.get("buildFailureKind"))
        if success is not None:
            result["success"] = success
        if error_count is not None:
            result["errorCount"] = error_count
        if warning_count is not None:
            result["warningCount"] = warning_count
        if failure_kind is not None:
            result["failureKind"] = failure_kind
        if build_failure_kind is not None:
            result["buildFailureKind"] = build_failure_kind
    elif tool_name in {"json_api", "ssot"}:
        success = _safe_boolean(value.get("success"))
        if success is not None:
            result["success"] = success
    return result


def _tool_execution_verdict(
    state_status: str, *, has_error_payload: bool
) -> str:
    """Tool execution is deliberately separate from domain verification."""
    return (
        "failure"
        if state_status == "error" or has_error_payload
        else "unverified"
    )


def _event_signature_projection(event: dict[str, Any]) -> dict[str, Any]:
    """Return the only fields allowed to influence a v3 trace signature."""
    return {
        "ordinal": event["ordinal"],
        "source_assistant_message_id": event["source_assistant_message_id"],
        "source_part_id": event["source_part_id"],
        "tool_name": event["tool_name"],
        "state_status": event["state_status"],
        "started_at_ms": event["started_at_ms"],
        "ended_at_ms": event["ended_at_ms"],
        "has_error_payload": event["has_error_payload"],
        "metadata": event["metadata"],
    }


def _tool_event(
    assistant_message_id: str,
    part_id: str,
    ordinal: int,
    data: dict[str, Any],
) -> dict[str, Any]:
    tool_name = _safe_tool_name(data.get("tool"))
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    state_status = _safe_status(state.get("status"))
    state_time = state.get("time") if isinstance(state.get("time"), dict) else {}
    metadata = _safe_metadata(tool_name, state.get("metadata"))
    has_error_payload = state.get("error") is not None
    event: dict[str, Any] = {
        "ordinal": ordinal,
        "source_assistant_message_id": assistant_message_id,
        "source_part_id": part_id,
        "tool_name": tool_name,
        "state_status": state_status,
        "started_at_ms": _safe_integer(state_time.get("start"), nonnegative=True),
        "ended_at_ms": _safe_integer(state_time.get("end"), nonnegative=True),
        "has_error_payload": has_error_payload,
        "metadata": metadata,
        # Legacy columns remain in SQLite, but v3 never fingerprints raw
        # payloads: unsalted hashes and exact lengths can disclose low-entropy
        # commands, paths, or secrets through offline guessing.
        "input_hash": None,
        "input_length": None,
        "output_hash": None,
        "output_length": None,
        "error_hash": None,
        "error_length": None,
    }
    event["evidence_verdict"] = _tool_execution_verdict(
        state_status, has_error_payload=has_error_payload
    )
    event["event_signature"] = _sha256_text(
        _canonical_json(_event_signature_projection(event))
    )
    return event


def _keil_domain_signal(event: dict[str, Any]) -> str:
    if event["tool_name"] != "keil_build":
        return "unverified"
    if event["state_status"] != "completed":
        return "unverified"
    metadata = event["metadata"]
    has_claim = any(
        key in metadata
        for key in ("success", "errorCount", "failureKind", "buildFailureKind")
    )
    started_at_ms = event["started_at_ms"]
    ended_at_ms = event["ended_at_ms"]
    if (
        started_at_ms is None
        or ended_at_ms is None
        or ended_at_ms < started_at_ms
        or event["has_error_payload"]
    ):
        return "ambiguous" if has_claim else "unverified"

    success = metadata.get("success")
    error_count = metadata.get("errorCount")
    warning_count = metadata.get("warningCount")
    has_failure_kind = (
        "failureKind" in metadata or "buildFailureKind" in metadata
    )
    if (
        not isinstance(success, bool)
        or not isinstance(error_count, int)
        or isinstance(error_count, bool)
        or not isinstance(warning_count, int)
        or isinstance(warning_count, bool)
    ):
        return "ambiguous" if has_claim else "unverified"
    if success is True:
        if error_count == 0 and not has_failure_kind:
            return "verified_success"
        return "ambiguous"
    if (
        isinstance(error_count, int)
        and not isinstance(error_count, bool)
        and error_count > 0
    ) or has_failure_kind:
        return "verified_failure"
    return "ambiguous"


def _domain_verification_outcome(events: list[dict[str, Any]]) -> str:
    signals = [
        (index, _keil_domain_signal(event))
        for index, event in enumerate(events)
        if event["tool_name"] == "keil_build"
    ]
    decisive = [
        (index, signal)
        for index, signal in signals
        if signal in {"verified_success", "verified_failure"}
    ]
    if any(signal == "ambiguous" for _, signal in signals):
        return "ambiguous"
    distinct = {signal for _, signal in decisive}
    # Without target/config identity, E1 cannot claim that a later pass
    # recovered an earlier failure; mixed domain evidence is contested.
    if len(distinct) > 1:
        return "ambiguous"
    if not distinct:
        return "unverified"

    last_verification = max(index for index, _ in decisive)
    if any(
        index > last_verification
        and event["tool_name"] not in EXPLICIT_READ_ONLY_TOOL_NAMES
        for index, event in enumerate(events)
    ):
        return "unverified"
    outcome = next(iter(distinct))
    if outcome not in DOMAIN_OUTCOMES:
        raise AssertionError("unexpected domain verification outcome")
    return outcome


def _legacy_trace_classification(
    turn_state: str, events: list[dict[str, Any]]
) -> tuple[str, dict[str, int]]:
    """Populate v1-compatible operational fields, never a domain result."""
    verdict_counts = Counter(event["evidence_verdict"] for event in events)
    # The column remains for pre-v3 schema compatibility only. All v3 rows use a
    # neutral value so downstream code cannot mistake lifecycle/tool errors for
    # a verified engineering result.
    evidence_class = "unverified"
    return evidence_class, {
        "positive_event_count": 0,
        "negative_event_count": verdict_counts["failure"],
        "unverified_event_count": verdict_counts["unverified"],
        "recovered_tool_count": 0,
    }


def _message_order(row: sqlite3.Row) -> tuple[int, str]:
    timestamp = row["time_created"]
    return (timestamp if isinstance(timestamp, int) else -1, str(row["id"]))


def _read_turn_trace(
    source: sqlite3.Connection, turn: dict[str, Any]
) -> dict[str, Any]:
    message_rows = source.execute(
        """
        SELECT id, time_created, data
        FROM message
        WHERE session_id = ?
        ORDER BY time_created, id
        """,
        (turn["session_id"],),
    ).fetchall()
    assistants = []
    for row in message_rows:
        decoded = _safe_json(row["data"])
        if (
            decoded.get("role") == "assistant"
            and decoded.get("parentID") == turn["user_message_id"]
        ):
            assistants.append((row, decoded))
    assistants.sort(key=lambda item: _message_order(item[0]))
    assistant_order = {row["id"]: index for index, (row, _) in enumerate(assistants)}

    part_rows = source.execute(
        """
        SELECT id, message_id, time_created, data
        FROM part
        WHERE session_id = ?
        ORDER BY time_created, id
        """,
        (turn["session_id"],),
    ).fetchall()
    selected_parts = [
        row for row in part_rows if row["message_id"] in assistant_order
    ]
    selected_parts.sort(
        key=lambda row: (
            assistant_order[row["message_id"]],
            row["time_created"] if isinstance(row["time_created"], int) else -1,
            str(row["id"]),
        )
    )

    events: list[dict[str, Any]] = []
    for part in selected_parts:
        decoded = _safe_json(part["data"])
        # Text and reasoning parts are intentionally ignored rather than
        # summarized or hashed.  Only tool payloads cross this boundary.
        if decoded.get("type") != "tool":
            continue
        events.append(
            _tool_event(
                str(part["message_id"]),
                str(part["id"]),
                len(events),
                decoded,
            )
        )

    evidence_class, evidence_counts = _legacy_trace_classification(
        turn["current_state"], events
    )
    trace_payload = {
        "assistant_count": len(assistants),
        "turn_state": turn["current_state"],
        "events": [_event_signature_projection(event) for event in events],
    }
    return {
        "assistant_count": len(assistants),
        "events": events,
        "trace_signature": _sha256_text(_canonical_json(trace_payload)),
        "evidence_class": evidence_class,
        "domain_verification_outcome": _domain_verification_outcome(events),
        **evidence_counts,
    }


def _capture_traces(
    source_database: Path, turns: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    with closing(open_read_only(source_database)) as source:
        source.execute("BEGIN")
        try:
            for turn in turns:
                traces[turn["turn_id"]] = _read_turn_trace(source, turn)
        finally:
            source.rollback()
    return traces


def _wait_between_samples(settle_ms: int) -> None:
    time.sleep(settle_ms / 1000)


def _load_terminal_turns(
    connection: sqlite3.Connection, source_identifier: str
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            t.turn_id,
            t.source_id,
            t.session_id,
            t.user_message_id,
            t.primary_event_id,
            t.current_state,
            t.last_structural_signature,
            COALESCE(a.project_key, sc.project_key) AS project_key,
            e.directory
        FROM conversation_turn AS t
        JOIN event AS e ON e.event_id = t.primary_event_id
        LEFT JOIN event_annotation AS a ON a.event_id = t.primary_event_id
        LEFT JOIN session_context AS sc
          ON sc.source_id = t.source_id AND sc.session_id = t.session_id
        WHERE t.source_id = ?
          AND t.current_state IN ('completed', 'failed')
        ORDER BY t.user_timestamp, t.turn_id
        """,
        (source_identifier,),
    ).fetchall()
    turns: list[dict[str, Any]] = []
    for row in rows:
        turn = dict(row)
        if not turn["project_key"]:
            if turn["directory"]:
                turn["project_key"] = project_identity(turn["directory"])[0]
            else:
                fallback = f"{source_identifier}\0{turn['session_id']}"
                turn["project_key"] = "unknown:" + _sha256_text(fallback)[:12]
        signature = turn["last_structural_signature"]
        if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
            # Legacy/nonstandard structural values are never copied verbatim.
            turn["last_structural_signature"] = _sha256_text(str(signature or "missing"))
        turn.pop("directory", None)
        turns.append(turn)
    return turns


def _identity(turn: dict[str, Any], trace_signature: str) -> tuple[str, str]:
    payload = {
        "extractor_version": EXTRACTOR_VERSION,
        "source_id": turn["source_id"],
        "turn_id": turn["turn_id"],
        "source_signature": turn["last_structural_signature"],
        "trace_signature": trace_signature,
    }
    key = _sha256_text(_canonical_json(payload))
    return key, "episode_" + key


def _insert_episode(
    connection: sqlite3.Connection,
    run_id: str,
    turn: dict[str, Any],
    trace: dict[str, Any],
    created_at: str,
) -> tuple[str, str, str, str]:
    idempotency_key, episode_id = _identity(turn, trace["trace_signature"])
    current = connection.execute(
        """
        SELECT episode_id, idempotency_key, trace_signature,
               domain_verification_outcome
        FROM trace_episode
        WHERE source_id = ? AND turn_id = ? AND extractor_version = ?
        ORDER BY created_at, episode_id
        LIMIT 1
        """,
        (turn["source_id"], turn["turn_id"], EXTRACTOR_VERSION),
    ).fetchone()
    if current is not None:
        if (
            current["idempotency_key"] == idempotency_key
            and current["trace_signature"] == trace["trace_signature"]
        ):
            return (
                current["episode_id"],
                "existing",
                current["domain_verification_outcome"],
                "idempotent_current_episode",
            )
        return (
            current["episode_id"],
            "skipped",
            "unverified",
            "current_extractor_version_already_has_different_trace",
        )

    connection.execute(
        """
        INSERT INTO trace_episode(
            episode_id, idempotency_key, extraction_run_id, extractor_version,
            source_id, turn_id, primary_event_id, project_key,
            source_signature, trace_signature, turn_state, evidence_class,
            domain_verification_outcome, semantic_status,
            assistant_count, tool_event_count,
            positive_event_count, negative_event_count, unverified_event_count,
            recovered_tool_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'not_evaluated', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            episode_id,
            idempotency_key,
            run_id,
            EXTRACTOR_VERSION,
            turn["source_id"],
            turn["turn_id"],
            turn["primary_event_id"],
            turn["project_key"],
            turn["last_structural_signature"],
            trace["trace_signature"],
            turn["current_state"],
            trace["evidence_class"],
            trace["domain_verification_outcome"],
            trace["assistant_count"],
            len(trace["events"]),
            trace["positive_event_count"],
            trace["negative_event_count"],
            trace["unverified_event_count"],
            trace["recovered_tool_count"],
            created_at,
        ),
    )

    for event in trace["events"]:
        event_id_payload = f"{episode_id}\0{event['ordinal']}\0{event['source_part_id']}"
        episode_event_id = "episode_event_" + _sha256_text(event_id_payload)
        connection.execute(
            """
            INSERT INTO trace_episode_event(
                episode_event_id, episode_id, ordinal,
                source_assistant_message_id, source_part_id, tool_name,
                state_status, started_at_ms, ended_at_ms, has_error_payload,
                metadata_json,
                input_hash, input_length, output_hash, output_length,
                error_hash, error_length, evidence_verdict, event_signature,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_event_id,
                episode_id,
                event["ordinal"],
                event["source_assistant_message_id"],
                event["source_part_id"],
                event["tool_name"],
                event["state_status"],
                event["started_at_ms"],
                event["ended_at_ms"],
                int(event["has_error_payload"]),
                _canonical_json(event["metadata"]),
                event["input_hash"],
                event["input_length"],
                event["output_hash"],
                event["output_length"],
                event["error_hash"],
                event["error_length"],
                event["evidence_verdict"],
                event["event_signature"],
                created_at,
            ),
        )
    return (
        episode_id,
        "created",
        trace["domain_verification_outcome"],
        "stable_whitelist_trace_created",
    )


def _record_run_result(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    turn_id: str,
    episode_id: str | None,
    disposition: str,
    first_signature: str,
    second_signature: str,
    domain_outcome: str,
    reason: str,
    created_at: str,
) -> None:
    result_key = _sha256_text(f"{run_id}\0{turn_id}")
    connection.execute(
        """
        INSERT INTO run_episode_result(
            result_id, run_id, turn_id, episode_id, disposition,
            sample_1_trace_signature, sample_2_trace_signature,
            domain_verification_outcome, reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "episode_result_" + result_key,
            run_id,
            turn_id,
            episode_id,
            disposition,
            first_signature,
            second_signature,
            domain_outcome,
            reason,
            created_at,
        ),
    )


def run_episode_extraction(
    source_database: Path, lab_database: Path, settle_ms: int = 1000
) -> dict[str, Any]:
    """Create one immutable v3 trace episode per stable terminal turn."""
    if settle_ms < 0 or settle_ms > 60000:
        raise ValueError("settle_ms must be between zero and 60000")
    source_database = Path(source_database)
    lab_database = Path(lab_database)
    require_distinct_source_and_lab(source_database, lab_database)
    require_memory_lab_database(lab_database)
    source_identifier = source_id(source_database)

    # Validate the source schema through a query-only connection before an
    # extraction run is registered in the writable lab.
    with closing(open_read_only(source_database)) as validation:
        validation.execute("SELECT id, session_id, data FROM message LIMIT 0")
        validation.execute("SELECT id, message_id, session_id, data FROM part LIMIT 0")

    run_id = "episode_run_" + uuid4().hex
    started_at = now_iso()
    with closing(open_store(lab_database)) as connection, connection:
        registered = connection.execute(
            "SELECT 1 FROM source WHERE source_id = ?", (source_identifier,)
        ).fetchone()
        if registered is None:
            raise ValueError("source has not been registered in this Memory Lab")
        connection.execute(
            """
            INSERT INTO episode_extraction_run(
                run_id, source_id, extractor_version, started_at, status, settle_ms
            ) VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (run_id, source_identifier, EXTRACTOR_VERSION, started_at, settle_ms),
        )

    try:
        with closing(open_store(lab_database)) as connection:
            turns = _load_terminal_turns(connection, source_identifier)
        first = _capture_traces(source_database, turns)
        if turns:
            _wait_between_samples(settle_ms)
        second = _capture_traces(source_database, turns)

        domain_counts: Counter[str] = Counter()
        disposition_counts: Counter[str] = Counter()
        results: list[dict[str, Any]] = []
        completed_at = now_iso()
        created_total = 0
        existing_total = 0
        unstable_total = 0
        skipped_total = 0
        with closing(open_store(lab_database)) as connection, connection:
            for turn in turns:
                first_trace = first[turn["turn_id"]]
                second_trace = second[turn["turn_id"]]
                signatures_match = (
                    first_trace["trace_signature"]
                    == second_trace["trace_signature"]
                )
                if not signatures_match:
                    episode_id = None
                    disposition = "unstable"
                    domain_outcome = "unverified"
                    reason = "whitelist_trace_changed_between_samples"
                    unstable_total += 1
                else:
                    (
                        episode_id,
                        disposition,
                        domain_outcome,
                        reason,
                    ) = _insert_episode(
                        connection, run_id, turn, second_trace, completed_at
                    )
                if disposition == "created":
                    created_total += 1
                elif disposition == "existing":
                    existing_total += 1
                elif disposition == "skipped":
                    skipped_total += 1

                domain_counts[domain_outcome] += 1
                disposition_counts[disposition] += 1
                _record_run_result(
                    connection,
                    run_id=run_id,
                    turn_id=turn["turn_id"],
                    episode_id=episode_id,
                    disposition=disposition,
                    first_signature=first_trace["trace_signature"],
                    second_signature=second_trace["trace_signature"],
                    domain_outcome=domain_outcome,
                    reason=reason,
                    created_at=completed_at,
                )
                results.append(
                    {
                        "episode_id": episode_id,
                        "turn_id": turn["turn_id"],
                        "primary_event_id": turn["primary_event_id"],
                        "project_key": turn["project_key"],
                        "turn_state": turn["current_state"],
                        "domain_verification_outcome": domain_outcome,
                        "assistant_count": second_trace["assistant_count"],
                        "tool_event_count": len(second_trace["events"]),
                        "tool_execution_error_count": second_trace[
                            "negative_event_count"
                        ],
                        "trace_stable": signatures_match,
                        "disposition": disposition,
                        "reason": reason,
                    }
                )
            connection.execute(
                """
                UPDATE episode_extraction_run
                SET completed_at = ?, status = 'completed', eligible_turns = ?,
                    created_episodes = ?, existing_episodes = ?,
                    unstable_turns = ?, skipped_turns = ?, counts_json = ?
                WHERE run_id = ?
                """,
                (
                    completed_at,
                    len(turns),
                    created_total,
                    existing_total,
                    unstable_total,
                    skipped_total,
                    _canonical_json(
                        {
                            "dispositions": dict(sorted(disposition_counts.items())),
                            "domain_verification": dict(sorted(domain_counts.items())),
                        }
                    ),
                    run_id,
                ),
            )
        return {
            "status": "completed",
            "run_id": run_id,
            "source_id": source_identifier,
            "extractor_version": EXTRACTOR_VERSION,
            "settle_ms": settle_ms,
            "eligible_turns": len(turns),
            "episodes_created": created_total,
            "episodes_existing": existing_total,
            "unstable_turns": unstable_total,
            "skipped_turns": skipped_total,
            "episodes_skipped": unstable_total + skipped_total,
            "domain_verification_counts": dict(sorted(domain_counts.items())),
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "episodes": results,
            "semantic_extraction": False,
            "compatibility_boundary": (
                "Immutable v1/v2 episodes remain historical records and are not "
                "reused as current v3 verification outcomes."
            ),
            "privacy_boundary": (
                "Only allow-listed tool identity/status/time/metadata are persisted. "
                "Assistant text, reasoning, tool payloads, commands, paths, logs, raw "
                "payload hashes, and exact payload lengths are excluded from v3; "
                "legacy v1 direct payload hash/length columns are cleared by store "
                "migration, while v2 rows remain immutable history."
            ),
        }
    except Exception as error:
        with closing(open_store(lab_database)) as connection, connection:
            connection.execute(
                """
                UPDATE episode_extraction_run
                SET completed_at = ?, status = 'failed', error = ?
                WHERE run_id = ?
                """,
                (now_iso(), type(error).__name__, run_id),
            )
        raise
