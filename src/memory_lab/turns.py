"""Detect completed CodePulse turns from the source SQLite database.

The detector deliberately stores only structural metadata.  Assistant text,
reasoning, tool inputs, and tool outputs are read transiently from the source
database and are never copied into the Memory Lab turn tables.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any
from uuid import uuid4

from .source import open_read_only, source_id
from .store import now_iso, open_store, register_source


DETECTOR_VERSION = "cp-db-turn-detector-v1"
ACTIVE_TOOL_STATES = {"pending", "running"}
# CodePulse receives the AI SDK's unified finish values.  Its local providers
# emit stop/length/content-filter/tool-calls/other and compaction can emit
# error.  Some imported/provider-specific histories use end_turn. Treat only
# this explicit set as terminal; an unexpected provider string must not
# accidentally release a turn for extraction.
KNOWN_TERMINAL_FINISHES = {
    "stop",
    "length",
    "content-filter",
    "other",
    "error",
    "end_turn",
}
SAFE_FINISH_VALUES = {"stop", "tool-calls", "unknown", "other"}


def _normalize_finish(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    normalized = str(value).casefold()
    if normalized in SAFE_FINISH_VALUES:
        return normalized
    return "other"


def _is_known_terminal_finish(value: Any) -> bool:
    return isinstance(value, str) and value.casefold() in KNOWN_TERMINAL_FINISHES


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


def _turn_id(source_identifier: str, session_id: str, message_id: str) -> str:
    payload = f"{source_identifier}\0{session_id}\0{message_id}".encode("utf-8")
    return "turn_" + hashlib.sha256(payload).hexdigest()


def _message_order(row: dict[str, Any]) -> tuple[int, str]:
    value = row.get("time_created")
    return (value if isinstance(value, int) else -1, str(row.get("id") or ""))


def _is_human_user_message(
    message: dict[str, Any], parts: list[dict[str, Any]]
) -> bool:
    """Mirror the ingestion boundary just enough to identify a later user turn."""
    if message["decoded"].get("role") != "user":
        return False
    for part in parts:
        data = part["decoded"]
        if data.get("type") != "text":
            continue
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        lowered = text.lstrip().casefold()
        if lowered.startswith("<system-reminder") or lowered.startswith("<system"):
            continue
        return True
    return False


def _time_descriptor(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {
        "start": data.get("start"),
        "end": data.get("end"),
    }


def _part_descriptor(part: dict[str, Any]) -> dict[str, Any]:
    """Return content-free structure used only to calculate a stability hash."""
    data = part["decoded"]
    part_type = str(data.get("type") or "unknown")
    descriptor: dict[str, Any] = {
        "id": part["id"],
        "created": part.get("time_created"),
        "updated": part.get("time_updated"),
        "type": part_type,
        # Field names expose schema evolution without persisting field values.
        "fields": sorted(
            key
            for key in data
            if key not in {"text", "reasoning", "input", "output", "error", "raw"}
        ),
        "time": _time_descriptor(data.get("time")),
    }
    state = data.get("state")
    if isinstance(state, dict):
        descriptor["state"] = {
            "status": state.get("status"),
            "fields": sorted(
                key
                for key in state
                if key not in {"input", "output", "error", "metadata", "raw"}
            ),
            "time": _time_descriptor(state.get("time")),
            "has_output": "output" in state and state.get("output") is not None,
            "has_error": "error" in state and state.get("error") is not None,
        }
    # A final visible text part can be appended after the terminal message row.
    # Its length catches that write without copying or hashing the prose itself.
    if part_type == "text":
        text = data.get("text")
        descriptor["text_length"] = len(text) if isinstance(text, str) else -1
    return descriptor


def _preliminary_state(snapshot: dict[str, Any]) -> str:
    trusted_terminal_state = None
    if snapshot["terminal_error"]:
        trusted_terminal_state = "failed"
    elif snapshot["terminal_completed"] and snapshot["terminal_finish_is_known_terminal"]:
        trusted_terminal_state = "completed"
    # A lingering running row blocks a claimed terminal while CP is still
    # settling.  Without a trustworthy terminal, however, a later human user
    # message is stronger evidence that the earlier turn was interrupted.
    if trusted_terminal_state is not None:
        if snapshot["active_tool_count"]:
            return "pending"
        return trusted_terminal_state
    if snapshot["next_user_message_id"]:
        return "interrupted"
    if snapshot["active_tool_count"]:
        return "pending"
    return "pending"


def _snapshot_for_turn(
    candidate: dict[str, Any],
    messages: list[dict[str, Any]],
    parts_by_message: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    user_message_id = candidate["user_message_id"]
    message_by_id = {item["id"]: item for item in messages}
    user_message = message_by_id.get(user_message_id)
    assistants = [
        item
        for item in messages
        if item["decoded"].get("role") == "assistant"
        and item["decoded"].get("parentID") == user_message_id
    ]
    assistants.sort(key=_message_order)

    descriptors: list[dict[str, Any]] = []
    active_tool_count = 0
    part_count = 0
    for assistant in assistants:
        data = assistant["decoded"]
        assistant_parts = sorted(
            parts_by_message.get(assistant["id"], []),
            key=lambda part: (
                part.get("time_created")
                if isinstance(part.get("time_created"), int)
                else -1,
                part["id"],
            ),
        )
        part_count += len(assistant_parts)
        for part in assistant_parts:
            state = part["decoded"].get("state")
            status = state.get("status") if isinstance(state, dict) else None
            if part["decoded"].get("type") == "tool" and status in ACTIVE_TOOL_STATES:
                active_tool_count += 1
        message_time = data.get("time")
        completed = (
            isinstance(message_time, dict)
            and "completed" in message_time
            and message_time.get("completed") is not None
        )
        descriptors.append(
            {
                "id": assistant["id"],
                "created": assistant.get("time_created"),
                "updated": assistant.get("time_updated"),
                "finish": _normalize_finish(data.get("finish")),
                "known_terminal_finish": _is_known_terminal_finish(data.get("finish")),
                "completed": completed,
                "completed_at": message_time.get("completed")
                if isinstance(message_time, dict)
                else None,
                "error": "error" in data and data.get("error") is not None,
                "parts": [_part_descriptor(part) for part in assistant_parts],
            }
        )

    next_user_message_id = None
    ordered_messages = sorted(messages, key=_message_order)
    current_order = _message_order(user_message) if user_message else (
        candidate.get("user_timestamp") or -1,
        user_message_id,
    )
    for message in ordered_messages:
        if message["id"] == user_message_id or _message_order(message) <= current_order:
            continue
        if _is_human_user_message(message, parts_by_message.get(message["id"], [])):
            next_user_message_id = message["id"]
            break

    terminal = assistants[-1] if assistants else None
    terminal_data = terminal["decoded"] if terminal else {}
    terminal_time = terminal_data.get("time")
    terminal_completed = (
        isinstance(terminal_time, dict)
        and "completed" in terminal_time
        and terminal_time.get("completed") is not None
    )
    terminal_error = "error" in terminal_data and terminal_data.get("error") is not None
    signature_payload = {
        "user_message_id": user_message_id,
        "source_user_present": bool(user_message),
        "source_user_updated": user_message.get("time_updated") if user_message else None,
        "assistants": descriptors,
        "next_user_message_id": next_user_message_id,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    snapshot = {
        "structural_signature": signature,
        "source_user_present": bool(user_message),
        "assistant_count": len(assistants),
        "part_count": part_count,
        "active_tool_count": active_tool_count,
        "terminal_assistant_id": terminal["id"] if terminal else None,
        "terminal_finish": _normalize_finish(terminal_data.get("finish")),
        "terminal_finish_is_known_terminal": _is_known_terminal_finish(
            terminal_data.get("finish")
        ),
        "terminal_completed": terminal_completed,
        "terminal_error": terminal_error,
        "next_user_message_id": next_user_message_id,
    }
    snapshot["preliminary_state"] = _preliminary_state(snapshot)
    return snapshot


def _capture_turn_snapshots(
    source_database: Path, candidates: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_session[candidate["session_id"]].append(candidate)

    result: dict[str, dict[str, Any]] = {}
    with closing(open_read_only(source_database)) as connection:
        connection.execute("BEGIN")
        message_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(message)")
        }
        part_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(part)")
        }
        message_updated = (
            "time_updated" if "time_updated" in message_columns else "NULL AS time_updated"
        )
        part_updated = (
            "time_updated" if "time_updated" in part_columns else "NULL AS time_updated"
        )
        for session_id, session_candidates in by_session.items():
            message_rows = connection.execute(
                f"""
                SELECT id, session_id, time_created, {message_updated}, data
                FROM message
                WHERE session_id = ?
                ORDER BY time_created, id
                """,
                (session_id,),
            ).fetchall()
            part_rows = connection.execute(
                f"""
                SELECT id, message_id, session_id, time_created, {part_updated}, data
                FROM part
                WHERE session_id = ?
                ORDER BY time_created, id
                """,
                (session_id,),
            ).fetchall()
            messages = [
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "time_created": row["time_created"],
                    "time_updated": row["time_updated"],
                    "decoded": _safe_json(row["data"]),
                }
                for row in message_rows
            ]
            parts_by_message: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in part_rows:
                parts_by_message[row["message_id"]].append(
                    {
                        "id": row["id"],
                        "message_id": row["message_id"],
                        "time_created": row["time_created"],
                        "time_updated": row["time_updated"],
                        "decoded": _safe_json(row["data"]),
                    }
                )
            for candidate in session_candidates:
                result[candidate["turn_id"]] = _snapshot_for_turn(
                    candidate, messages, parts_by_message
                )
        connection.rollback()
    return result


def _wait_between_samples(settle_ms: int) -> None:
    time.sleep(settle_ms / 1000)


def _final_state(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[str, bool, str]:
    stable = first["structural_signature"] == second["structural_signature"]
    preliminary = second["preliminary_state"]
    if preliminary in {"completed", "failed"}:
        if stable:
            return preliminary, True, "terminal_state_and_structure_stable"
        return (
            "terminal_waiting_stability",
            False,
            "terminal_state_seen_but_structure_changed_between_samples",
        )
    if preliminary == "interrupted":
        return "interrupted", stable, "later_user_turn_seen_without_terminal_assistant"
    if second["active_tool_count"]:
        return "pending", stable, "assistant_tool_is_pending_or_running"
    return "pending", stable, "no_terminal_assistant_yet"


def _load_candidates(connection, source_identifier: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT e.event_id, e.session_id, e.message_id, e.timestamp
        FROM event AS e
        LEFT JOIN conversation_turn AS t
          ON t.source_id = e.source_id
         AND t.session_id = e.session_id
         AND t.user_message_id = e.message_id
        WHERE e.source_id = ?
          AND e.eligibility_status = 'eligible'
          AND (
              t.turn_id IS NULL
              OR t.current_state IN ('pending', 'terminal_waiting_stability')
          )
        ORDER BY e.timestamp, e.event_id
        """,
        (source_identifier,),
    ).fetchall()
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["session_id"], row["message_id"])
        candidate = grouped.setdefault(
            key,
            {
                "turn_id": _turn_id(source_identifier, row["session_id"], row["message_id"]),
                "source_id": source_identifier,
                "session_id": row["session_id"],
                "user_message_id": row["message_id"],
                "primary_event_id": row["event_id"],
                "user_timestamp": row["timestamp"],
                "event_ids": [],
            },
        )
        candidate["event_ids"].append(row["event_id"])
        if row["timestamp"] is not None and (
            candidate["user_timestamp"] is None
            or row["timestamp"] < candidate["user_timestamp"]
        ):
            candidate["user_timestamp"] = row["timestamp"]
            candidate["primary_event_id"] = row["event_id"]
    return list(grouped.values())


def _record_observation(
    connection,
    run_id: str,
    turn_id: str,
    sample_index: int,
    snapshot: dict[str, Any],
) -> None:
    observed_at = now_iso()
    observed_at_ms = int(time.time() * 1000)
    connection.execute(
        """
        INSERT INTO turn_observation(
            observation_id, run_id, turn_id, sample_index, observed_at,
            observed_at_ms, structural_signature, source_user_present,
            assistant_count, part_count, active_tool_count,
            terminal_assistant_id, terminal_finish, terminal_completed,
            terminal_error, next_user_message_id, preliminary_state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid4().hex,
            run_id,
            turn_id,
            sample_index,
            observed_at,
            observed_at_ms,
            snapshot["structural_signature"],
            int(snapshot["source_user_present"]),
            snapshot["assistant_count"],
            snapshot["part_count"],
            snapshot["active_tool_count"],
            snapshot["terminal_assistant_id"],
            snapshot["terminal_finish"],
            int(snapshot["terminal_completed"]),
            int(snapshot["terminal_error"]),
            snapshot["next_user_message_id"],
            snapshot["preliminary_state"],
        ),
    )


def run_turn_scan(
    source_database: Path, lab_database: Path, settle_ms: int = 1000
) -> dict[str, Any]:
    """Sample CP twice and classify every eligible normalized user turn."""
    if settle_ms < 0:
        raise ValueError("settle_ms must be zero or greater")
    source_database = Path(source_database)
    lab_database = Path(lab_database)
    source_identifier = source_id(source_database)
    run_id = uuid4().hex
    started_at = now_iso()

    with closing(open_store(lab_database)) as connection, connection:
        register_source(connection, source_identifier, source_database)
        connection.execute(
            """
            INSERT INTO turn_detection_run(
                run_id, source_id, detector_version, started_at, settle_ms, status
            ) VALUES (?, ?, ?, ?, ?, 'running')
            """,
            (run_id, source_identifier, DETECTOR_VERSION, started_at, settle_ms),
        )
        candidates = _load_candidates(connection, source_identifier)
        for candidate in candidates:
            connection.execute(
                """
                INSERT INTO conversation_turn(
                    turn_id, source_id, session_id, user_message_id,
                    primary_event_id, user_timestamp, current_state,
                    first_detected_at, last_detected_at, last_detection_run_id,
                    last_structural_signature
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, '')
                ON CONFLICT(source_id, session_id, user_message_id) DO NOTHING
                """,
                (
                    candidate["turn_id"],
                    source_identifier,
                    candidate["session_id"],
                    candidate["user_message_id"],
                    candidate["primary_event_id"],
                    candidate["user_timestamp"],
                    started_at,
                    started_at,
                    run_id,
                ),
            )
            for event_id in candidate["event_ids"]:
                connection.execute(
                    "INSERT OR IGNORE INTO turn_event(turn_id, event_id) VALUES (?, ?)",
                    (candidate["turn_id"], event_id),
                )
        tracked_turn_count = connection.execute(
            "SELECT COUNT(*) FROM conversation_turn WHERE source_id = ?",
            (source_identifier,),
        ).fetchone()[0]
        skipped_terminal_turns = connection.execute(
            """
            SELECT COUNT(*)
            FROM conversation_turn
            WHERE source_id = ?
              AND current_state IN ('completed', 'failed', 'interrupted')
            """,
            (source_identifier,),
        ).fetchone()[0]

    try:
        first = _capture_turn_snapshots(source_database, candidates)
        if candidates:
            _wait_between_samples(settle_ms)
        second = _capture_turn_snapshots(source_database, candidates)
        outcomes: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        decided_at = now_iso()
        with closing(open_store(lab_database)) as connection, connection:
            for candidate in candidates:
                turn_id = candidate["turn_id"]
                first_snapshot = first[turn_id]
                second_snapshot = second[turn_id]
                state, stable, reason = _final_state(first_snapshot, second_snapshot)
                _record_observation(connection, run_id, turn_id, 1, first_snapshot)
                _record_observation(connection, run_id, turn_id, 2, second_snapshot)
                connection.execute(
                    """
                    INSERT INTO turn_detection_result(
                        result_id, run_id, turn_id, state, signature_stable,
                        sample_1_signature, sample_2_signature, reason, decided_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid4().hex,
                        run_id,
                        turn_id,
                        state,
                        int(stable),
                        first_snapshot["structural_signature"],
                        second_snapshot["structural_signature"],
                        reason,
                        decided_at,
                    ),
                )
                connection.execute(
                    """
                    UPDATE conversation_turn
                    SET primary_event_id = ?, user_timestamp = ?, current_state = ?,
                        last_detected_at = ?, last_detection_run_id = ?,
                        last_structural_signature = ?, assistant_count = ?,
                        part_count = ?, active_tool_count = ?,
                        terminal_assistant_id = ?, terminal_finish = ?,
                        terminal_error = ?, next_user_message_id = ?
                    WHERE turn_id = ?
                    """,
                    (
                        candidate["primary_event_id"],
                        candidate["user_timestamp"],
                        state,
                        decided_at,
                        run_id,
                        second_snapshot["structural_signature"],
                        second_snapshot["assistant_count"],
                        second_snapshot["part_count"],
                        second_snapshot["active_tool_count"],
                        second_snapshot["terminal_assistant_id"],
                        second_snapshot["terminal_finish"],
                        int(second_snapshot["terminal_error"]),
                        second_snapshot["next_user_message_id"],
                        turn_id,
                    ),
                )
                counts[state] += 1
                outcomes.append(
                    {
                        "turn_id": turn_id,
                        "session_id": candidate["session_id"],
                        "user_message_id": candidate["user_message_id"],
                        "event_ids": candidate["event_ids"],
                        "state": state,
                        "signature_stable": stable,
                        "assistant_count": second_snapshot["assistant_count"],
                        "part_count": second_snapshot["part_count"],
                        "active_tool_count": second_snapshot["active_tool_count"],
                        "terminal_finish": second_snapshot["terminal_finish"],
                        "terminal_error": second_snapshot["terminal_error"],
                        "next_user_message_id": second_snapshot["next_user_message_id"],
                        "reason": reason,
                    }
                )
            connection.execute(
                """
                UPDATE turn_detection_run
                SET completed_at = ?, status = 'completed', candidate_turns = ?,
                    counts_json = ?
                WHERE run_id = ?
                """,
                (
                    decided_at,
                    len(candidates),
                    json.dumps(dict(sorted(counts.items())), sort_keys=True),
                    run_id,
                ),
            )
        return {
            "run_id": run_id,
            "source_id": source_identifier,
            "detector_version": DETECTOR_VERSION,
            "settle_ms": settle_ms,
            "turn_count": len(candidates),
            "tracked_turn_count": tracked_turn_count,
            "skipped_terminal_turns": skipped_terminal_turns,
            "counts": dict(sorted(counts.items())),
            "turns": outcomes,
            "scan_mode": "manual_incremental_two_sample",
            "watcher_enabled": False,
            "terminal_turns_rescanned": False,
            "privacy_boundary": (
                "Turn observations contain structural hashes and status metadata only; "
                "assistant text, reasoning, tool inputs, and tool outputs are not persisted."
            ),
        }
    except Exception as error:
        with closing(open_store(lab_database)) as connection, connection:
            connection.execute(
                """
                UPDATE turn_detection_run
                SET completed_at = ?, status = 'failed', error = ?
                WHERE run_id = ?
                """,
                (now_iso(), str(error), run_id),
            )
        raise
