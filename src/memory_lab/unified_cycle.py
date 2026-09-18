"""End-to-end orchestration for the generic conversation-memory path.

This module is intentionally independent from the CodePulse process.  It
reads CP SQLite through the existing adapters, processes at most one ordered
batch, and records a success cursor only after the formal database, topics,
and immutable Markdown projection are all ready.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .change_report import build_change_report, write_change_report
from .formal_markdown import publish_formal_memory
from .ingest import run_ingestion
from .shadow import formal_state_snapshot
from .source import source_id
from .store import now_iso, open_store
from .topics import ensure_topic_assignments
from .turn_evidence import build_turn_evidence, fit_turn_evidence_to_model_limits
from .turns import run_turn_scan
from .unified_model import UnifiedInputError, build_unified_model_request


CYCLE_VERSION = "memory-lab-unified-cycle-v1"
MAX_BATCH_SIZE = 32
MAX_CYCLE_BATCH_OBSERVATION_CHARS = 50_000


_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS unified_cycle_run (
    run_id TEXT PRIMARY KEY,
    cycle_version TEXT NOT NULL,
    source_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    selected_turn_count INTEGER NOT NULL DEFAULT 0,
    processed_turn_count INTEGER NOT NULL DEFAULT 0,
    selected_turns_json TEXT NOT NULL DEFAULT '[]',
    stages_json TEXT NOT NULL DEFAULT '{}',
    formal_state_before_json TEXT NOT NULL DEFAULT '{}',
    formal_state_after_json TEXT,
    actions_json TEXT NOT NULL DEFAULT '[]',
    review_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS unified_processed_turn (
    source_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    bundle_id TEXT NOT NULL,
    bundle_hash TEXT NOT NULL,
    occurred_at_ms INTEGER,
    cycle_run_id TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY (source_id, turn_id),
    FOREIGN KEY (cycle_run_id) REFERENCES unified_cycle_run(run_id)
);

CREATE INDEX IF NOT EXISTS unified_processed_turn_run_idx
    ON unified_processed_turn(cycle_run_id, processed_at);

CREATE TABLE IF NOT EXISTS unified_cycle_checkpoint (
    checkpoint_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    formal_state_before_json TEXT NOT NULL,
    formal_state_after_json TEXT NOT NULL,
    actions_json TEXT NOT NULL,
    review_json TEXT NOT NULL,
    commit_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unified_cycle_checkpoint_turn (
    checkpoint_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    bundle_id TEXT NOT NULL,
    bundle_hash TEXT NOT NULL,
    occurred_at_ms INTEGER,
    PRIMARY KEY (checkpoint_id, turn_id),
    UNIQUE (source_id, turn_id),
    FOREIGN KEY (checkpoint_id) REFERENCES unified_cycle_checkpoint(checkpoint_id)
        ON DELETE CASCADE
);
"""


class UnifiedCycleError(RuntimeError):
    """The unified batch failed without being marked successfully processed."""


class Planner(Protocol):
    def plan(
        self,
        turns: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        existing_memories: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]: ...


class Reviewer(Protocol):
    def review(
        self,
        writer_plan: Mapping[str, Any],
        turns: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        existing_memories: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]: ...


def _default_commit(
    lab_database: Path, reviewer_result: Mapping[str, Any]
) -> Mapping[str, Any]:
    # Imported lazily so the evidence/model/cycle modules remain independently
    # testable while the formal executor evolves.
    from .unified_commit import commit_reviewed_unified_plan

    return commit_reviewed_unified_plan(lab_database, reviewer_result)


def _default_topics(lab_database: Path) -> Mapping[str, Any]:
    with closing(open_store(lab_database)) as connection, connection:
        return ensure_topic_assignments(connection)


def _default_build_report(
    lab_database: Path,
    pipeline_report: Mapping[str, Any],
    markdown_report: Mapping[str, Any],
    stages: Mapping[str, Any],
) -> Mapping[str, Any]:
    return build_change_report(
        lab_database,
        pipeline_report,
        markdown_report=markdown_report,
        stages=stages,
    )


@dataclass(frozen=True)
class CycleDependencies:
    """Injectable boundaries used by deterministic failure and idempotency tests."""

    ingest: Callable[[Path, Path, str], Mapping[str, Any]] = run_ingestion
    scan_turns: Callable[[Path, Path, int], Mapping[str, Any]] = run_turn_scan
    build_evidence: Callable[..., Mapping[str, Any]] = build_turn_evidence
    commit_reviewed_plan: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] = (
        _default_commit
    )
    ensure_topics: Callable[[Path], Mapping[str, Any]] = _default_topics
    publish: Callable[[Path, Path], Mapping[str, Any]] = publish_formal_memory
    build_report: Callable[
        [Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
        Mapping[str, Any],
    ] = _default_build_report
    write_report: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] = (
        write_change_report
    )


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _ensure_audit_schema(lab_database: Path) -> None:
    with closing(open_store(lab_database)) as connection, connection:
        connection.executescript(_AUDIT_SCHEMA)


def _formal_snapshot(lab_database: Path) -> dict[str, Any]:
    with closing(open_store(lab_database)) as connection:
        return formal_state_snapshot(connection)


def _pending_turns(
    lab_database: Path, source_identifier: str, batch_size: int
) -> list[dict[str, Any]]:
    with closing(open_store(lab_database)) as connection:
        connection.executescript(_AUDIT_SCHEMA)
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT t.turn_id, t.user_timestamp, t.current_state
                FROM conversation_turn AS t
                WHERE t.source_id = ?
                  AND t.current_state IN ('completed', 'failed', 'interrupted')
                  AND NOT EXISTS (
                      SELECT 1 FROM unified_processed_turn AS p
                      WHERE p.source_id = t.source_id AND p.turn_id = t.turn_id
                  )
                ORDER BY (t.user_timestamp IS NULL), t.user_timestamp, t.turn_id
                LIMIT ?
                """,
                (source_identifier, batch_size),
            )
        ]


def _update_run_selection(
    lab_database: Path,
    *,
    run_id: str,
    turns: Sequence[Mapping[str, Any]],
) -> None:
    with closing(open_store(lab_database)) as connection, connection:
        connection.execute(
            """
            UPDATE unified_cycle_run
            SET selected_turn_count = ?, selected_turns_json = ?
            WHERE run_id = ?
            """,
            (
                len(turns),
                _json([item["turn_id"] for item in turns]),
                run_id,
            ),
        )


def _longest_model_compatible_prefix(
    bundles: Sequence[Mapping[str, Any]],
    existing_memories: Sequence[Mapping[str, Any]],
) -> int:
    """Select the longest chronological prefix accepted by the model contract.

    The lower-level model contract is a hard transport limit.  The cycle uses
    a smaller soft budget when batching multiple turns so long CP sessions do
    not starve later turns or send an overly broad context window to the model.
    A single accepted turn is still allowed even when it exceeds this soft
    budget because otherwise an oversized but compacted turn could never make
    progress.
    """

    if not bundles:
        return 0
    accepted = 0
    for end in range(1, len(bundles) + 1):
        try:
            _, catalog, _ = build_unified_model_request(
                bundles[:end], existing_memories=existing_memories
            )
        except UnifiedInputError as error:
            message = str(error)
            if (
                "observation limit" not in message
                and "total text limit" not in message
                and "between 1 and" not in message
            ):
                raise
            break
        total_chars = sum(
            len(str(item.get("text") or ""))
            for item in catalog.values()
            if isinstance(item, Mapping)
        )
        if end > 1 and total_chars > MAX_CYCLE_BATCH_OBSERVATION_CHARS:
            break
        accepted = end
    if accepted < 1:
        raise UnifiedCycleError(
            "oldest pending turn does not fit the unified model contract"
        )
    return accepted


def _load_checkpoint(
    lab_database: Path, source_identifier: str
) -> dict[str, Any] | None:
    with closing(open_store(lab_database)) as connection:
        connection.executescript(_AUDIT_SCHEMA)
        batch = connection.execute(
            """
            SELECT * FROM unified_cycle_checkpoint
            WHERE source_id = ? ORDER BY created_at, checkpoint_id LIMIT 1
            """,
            (source_identifier,),
        ).fetchone()
        if batch is None:
            return None
        turns = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM unified_cycle_checkpoint_turn
                WHERE checkpoint_id = ?
                ORDER BY (occurred_at_ms IS NULL), occurred_at_ms, turn_id
                """,
                (batch["checkpoint_id"],),
            )
        ]
    if not turns:
        raise UnifiedCycleError("commit checkpoint has no turns")
    try:
        return {
            "checkpoint_id": batch["checkpoint_id"],
            "turns": turns,
            "formal_state_before": json.loads(batch["formal_state_before_json"]),
            "formal_state_after": json.loads(batch["formal_state_after_json"]),
            "actions": json.loads(batch["actions_json"]),
            "review": json.loads(batch["review_json"]),
            "commit": json.loads(batch["commit_json"]),
        }
    except (TypeError, json.JSONDecodeError) as error:
        raise UnifiedCycleError("commit checkpoint is malformed") from error


def _checkpoint_commit(
    lab_database: Path,
    *,
    checkpoint_id: str,
    source_identifier: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
    commit_report: Mapping[str, Any],
    bundles: Sequence[Mapping[str, Any]],
) -> None:
    with closing(open_store(lab_database)) as connection, connection:
        connection.executescript(_AUDIT_SCHEMA)
        connection.execute(
            """
            INSERT INTO unified_cycle_checkpoint(
                checkpoint_id, source_id, created_at, formal_state_before_json,
                formal_state_after_json, actions_json, review_json, commit_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint_id,
                source_identifier,
                now_iso(),
                _json(before),
                _json(after),
                _json(list(actions)),
                _json(list(review)),
                _json(dict(commit_report)),
            ),
        )
        for bundle in bundles:
            turn = bundle["turn"]
            connection.execute(
                """
                INSERT INTO unified_cycle_checkpoint_turn(
                    checkpoint_id, source_id, turn_id, bundle_id, bundle_hash,
                    occurred_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    source_identifier,
                    turn["turn_id"],
                    bundle["bundle_id"],
                    bundle["bundle_hash"],
                    turn.get("occurred_at_ms"),
                ),
            )


def _memory_type_for_model(family: str | None, memory_type: str, stable_key: str) -> str:
    if memory_type in {
        "user",
        "preference",
        "workflow",
        "engineering",
        "project",
        "reference",
        "knowledge",
        "other",
    }:
        return memory_type
    if family == "engineering_memory":
        return "engineering"
    if stable_key.startswith("preference."):
        return "preference"
    if stable_key.startswith("workflow.") or stable_key.startswith("toolchain."):
        return "workflow"
    if family == "user_contract":
        return "user"
    return "other"


def _existing_memories(lab_database: Path) -> list[dict[str, Any]]:
    """Load a bounded semantic catalog; no raw evidence enters the model."""
    with closing(open_store(lab_database)) as connection:
        rows = connection.execute(
            """
            SELECT m.stable_key, m.family, m.memory_type, m.scope_json,
                   r.statement, m.updated_at
            FROM memory AS m
            JOIN memory_revision AS r ON r.revision_id = m.current_revision_id
            WHERE m.status = 'active' AND r.status = 'current'
            ORDER BY m.updated_at DESC, m.stable_key, m.memory_id
            LIMIT 200
            """
        ).fetchall()
    result = []
    for row in rows:
        try:
            scope = json.loads(row["scope_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise UnifiedCycleError("formal memory contains an invalid scope") from error
        statement = str(row["statement"])
        result.append(
            {
                "stable_key": str(row["stable_key"]),
                "type": _memory_type_for_model(
                    row["family"], str(row["memory_type"]), str(row["stable_key"])
                ),
                "scope": scope,
                "summary": statement[:320],
                "statement": statement,
            }
        )
    return result


def _review_rows(
    reviewer_result: Mapping[str, Any],
    fallback_plan: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    outcome = str(reviewer_result.get("outcome") or "reject")
    reason = str(
        reviewer_result.get("reason")
        or reviewer_result.get("review_reason")
        or "independent_review"
    )[:300]
    plan = reviewer_result.get("reviewed_plan")
    if not isinstance(plan, Mapping):
        plan = fallback_plan
    candidates = plan.get("candidates", []) if isinstance(plan, Mapping) else []
    rows = []
    for raw in candidates if isinstance(candidates, list) else []:
        if not isinstance(raw, Mapping):
            continue
        memory = raw.get("memory")
        if not isinstance(memory, Mapping):
            continue
        scope = memory.get("scope") if isinstance(memory.get("scope"), Mapping) else {}
        content = memory.get("content") if isinstance(memory.get("content"), Mapping) else {}
        hint = raw.get("action_hint") if isinstance(raw.get("action_hint"), Mapping) else {}
        rows.append(
            {
                "stable_key": str(memory.get("stable_key") or "")[:200],
                "operation": str(hint.get("kind") or "create_or_match")[:80],
                "outcome": outcome,
                "review_reason": reason,
                "statement": str(content.get("statement") or "")[:400],
                "scope": dict(scope),
                "project_key": scope.get("project_key"),
            }
        )
    return rows


def _commit_actions(commit_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_actions = commit_report.get("actions", [])
    if not isinstance(raw_actions, list):
        return []
    result = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            continue
        result.append(
            {
                "action": str(raw.get("action") or raw.get("operation") or "unknown")[:80],
                "reason": str(raw.get("reason") or "deterministic_unified_commit")[:300],
                "memory_id": raw.get("memory_id"),
                "before_revision_id": raw.get("before_revision_id"),
                "after_revision_id": raw.get("after_revision_id"),
            }
        )
    return result


def _source_summary(
    bundles: Sequence[Mapping[str, Any]],
    *,
    candidates: int = 0,
    accepted: int = 0,
    rejected: int = 0,
) -> dict[str, Any]:
    projects: dict[str, dict[str, str]] = {}
    user_messages = 0
    assistant_finals = 0
    tool_events = 0
    for bundle in bundles:
        workspace = bundle.get("workspace")
        if isinstance(workspace, Mapping):
            project_key = str(workspace.get("project_key") or "unknown")[:500]
            projects[project_key] = {
                "project_key": project_key,
                "display_name": str(workspace.get("display_name") or "unknown")[:160],
            }
        user = bundle.get("user")
        if isinstance(user, Mapping) and user.get("message_id"):
            user_messages += 1
        assistant = bundle.get("assistant")
        if isinstance(assistant, Mapping) and assistant.get("message_id"):
            assistant_finals += 1
        tools = bundle.get("tools")
        manifest = bundle.get("tool_event_manifest")
        manifest_total = (
            manifest.get("total_count") if isinstance(manifest, Mapping) else None
        )
        if isinstance(manifest_total, int) and manifest_total >= 0:
            tool_events += manifest_total
        elif isinstance(tools, list):
            tool_events += len(tools)
    return {
        "turns_processed": len(bundles),
        "projects": [projects[key] for key in sorted(projects)],
        "user_messages": user_messages,
        "assistant_finals": assistant_finals,
        "tool_events": tool_events,
        "candidates": candidates,
        "accepted": accepted,
        "rejected": rejected,
    }


def _start_run(
    lab_database: Path,
    *,
    run_id: str,
    source_identifier: str,
    turns: Sequence[Mapping[str, Any]],
    before: Mapping[str, Any],
) -> None:
    with closing(open_store(lab_database)) as connection, connection:
        connection.executescript(_AUDIT_SCHEMA)
        connection.execute(
            """
            INSERT INTO unified_cycle_run(
                run_id, cycle_version, source_id, started_at, status,
                selected_turn_count, selected_turns_json, formal_state_before_json
            ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
            """,
            (
                run_id,
                CYCLE_VERSION,
                source_identifier,
                now_iso(),
                len(turns),
                _json([item["turn_id"] for item in turns]),
                _json(before),
            ),
        )


def _finish_run(
    lab_database: Path,
    *,
    run_id: str,
    status: str,
    stages: Mapping[str, Any],
    after: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]] = (),
    mark_success: bool = False,
    checkpoint_id: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    with closing(open_store(lab_database)) as connection, connection:
        connection.executescript(_AUDIT_SCHEMA)
        if mark_success:
            for bundle in bundles:
                turn = bundle["turn"]
                connection.execute(
                    """
                    INSERT INTO unified_processed_turn(
                        source_id, turn_id, bundle_id, bundle_hash,
                        occurred_at_ms, cycle_run_id, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, turn_id) DO NOTHING
                    """,
                    (
                        turn["source_id"],
                        turn["turn_id"],
                        bundle["bundle_id"],
                        bundle["bundle_hash"],
                        turn.get("occurred_at_ms"),
                        run_id,
                        now_iso(),
                    ),
                )
            if checkpoint_id is not None:
                connection.execute(
                    "DELETE FROM unified_cycle_checkpoint_turn WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )
                connection.execute(
                    "DELETE FROM unified_cycle_checkpoint WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )
        processed = connection.execute(
            "SELECT COUNT(*) FROM unified_processed_turn WHERE cycle_run_id = ?",
            (run_id,),
        ).fetchone()[0]
        connection.execute(
            """
            UPDATE unified_cycle_run
            SET completed_at = ?, status = ?, processed_turn_count = ?,
                stages_json = ?, formal_state_after_json = ?, actions_json = ?,
                review_json = ?, error_code = ?, error_message = ?
            WHERE run_id = ?
            """,
            (
                now_iso(),
                status,
                processed,
                _json(stages),
                _json(after),
                _json(list(actions)),
                _json(list(review)),
                error_code,
                error_message,
                run_id,
            ),
        )


def _safe_failure(error: Exception) -> tuple[str, str]:
    # Provider exceptions can contain prompts, local paths, or credentials.
    # Keep only bounded local validator messages for actionable audits.
    message = str(error).replace("\n", " ").strip()
    if type(error).__name__ in {
        "UnifiedOutputError",
        "UnifiedInputError",
        "UnifiedTransportError",
    } and message:
        return type(error).__name__[:120], message[:500]
    return type(error).__name__[:120], "unified_cycle_stage_failed"


def _resume_committed_checkpoint(
    source_database: Path,
    lab_database: Path,
    memory_root: Path,
    report_directory: Path | None,
    *,
    run_id: str,
    source_identifier: str,
    checkpoint: Mapping[str, Any],
    dependencies: CycleDependencies,
    stages: dict[str, Any],
) -> dict[str, Any]:
    """Finish projection/report without asking either model a second time."""
    turns = list(checkpoint["turns"])
    before = dict(checkpoint["formal_state_before"])
    actions = list(checkpoint["actions"])
    review_rows = list(checkpoint["review"])
    _start_run(
        lab_database,
        run_id=run_id,
        source_identifier=source_identifier,
        turns=turns,
        before=before,
    )
    bundles: list[Mapping[str, Any]] = []
    current_stage = "recovery"
    try:
        current = _formal_snapshot(lab_database)
        if current != checkpoint["formal_state_after"]:
            raise UnifiedCycleError("formal state no longer matches commit checkpoint")
        for frozen in turns:
            bundle = fit_turn_evidence_to_model_limits(
                dependencies.build_evidence(
                    source_database, lab_database, frozen["turn_id"]
                )
            )
            if (
                bundle.get("bundle_id") != frozen["bundle_id"]
                or bundle.get("bundle_hash") != frozen["bundle_hash"]
            ):
                raise UnifiedCycleError("source turn changed after formal commit")
            bundles.append(bundle)
        stages["recovery"] = {
            "status": "committed_checkpoint",
            "checkpoint_id": checkpoint["checkpoint_id"],
        }
        current_stage = "topics"
        topic_report = dependencies.ensure_topics(lab_database)
        stages["topics"] = {
            "status": str(topic_report.get("status") or "complete")
            if isinstance(topic_report, Mapping)
            else "complete"
        }
        current_stage = "markdown"
        markdown_report = dependencies.publish(lab_database, memory_root)
        if not isinstance(markdown_report, Mapping):
            raise UnifiedCycleError("publisher returned a non-object report")
        stages["markdown"] = {
            "status": str(markdown_report.get("status") or "published"),
            "generation_id": markdown_report.get("generation_id"),
        }
        after = _formal_snapshot(lab_database)
        generation_matches = after == checkpoint["formal_state_after"]
        markdown_report = {
            **dict(markdown_report),
            "matches_run_state": generation_matches,
            "generation_matches_current_database": generation_matches,
        }
        source_summary = _source_summary(
            bundles,
            candidates=len(review_rows),
            accepted=len(review_rows),
        )
        report_files: Mapping[str, Any] | None = None
        if report_directory is not None:
            current_stage = "report"
            report_input = {
                "run_id": run_id,
                "status": "complete",
                "formal_state_before": before,
                "formal_state_after": after,
                "formal_state_changed": before != after,
                "actions": actions,
                "review": review_rows,
                "source_summary": source_summary,
            }
            report = dependencies.build_report(
                lab_database, report_input, markdown_report, stages
            )
            if not isinstance(report, Mapping):
                raise UnifiedCycleError("report builder returned a non-object report")
            report_files = dependencies.write_report(report_directory, report)
            if not isinstance(report_files, Mapping):
                raise UnifiedCycleError("report writer returned a non-object result")
            stages["report"] = {"status": "complete"}
        _finish_run(
            lab_database,
            run_id=run_id,
            status="complete",
            stages=stages,
            after=after,
            actions=actions,
            review=review_rows,
            bundles=bundles,
            mark_success=True,
            checkpoint_id=str(checkpoint["checkpoint_id"]),
        )
        return {
            "run_id": run_id,
            "status": "complete",
            "resumed_from_checkpoint": checkpoint["checkpoint_id"],
            "formal_state_before": before,
            "formal_state_after": after,
            "formal_state_changed": before != after,
            "actions": actions,
            "review": review_rows,
            "selected_turn_count": len(turns),
            "processed_turn_count": len(bundles),
            "stages": stages,
            "markdown": dict(markdown_report),
            "commit": dict(checkpoint["commit"]),
            "source_summary": source_summary,
            "report_files": dict(report_files or {}),
        }
    except Exception as error:
        after = _formal_snapshot(lab_database)
        code, message = _safe_failure(error)
        stages[current_stage] = {"status": "failed", "error_code": code}
        _finish_run(
            lab_database,
            run_id=run_id,
            status="failed",
            stages=stages,
            after=after,
            actions=actions,
            review=review_rows,
            error_code=code,
            error_message=message,
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "error_code": code,
            "error_message": message,
            "resumed_from_checkpoint": checkpoint["checkpoint_id"],
            "formal_state_before": before,
            "formal_state_after": after,
            "formal_state_changed": before != after,
            "formal_state_unchanged": before == after,
            "actions": actions,
            "review": review_rows,
            "selected_turn_count": len(turns),
            "processed_turn_count": 0,
            "stages": stages,
            "source_summary": _source_summary(bundles),
        }


def run_unified_cycle(
    source_database: Path,
    lab_database: Path,
    memory_root: Path,
    *,
    planner: Planner,
    reviewer: Reviewer,
    batch_size: int = MAX_BATCH_SIZE,
    settle_ms: int = 1_000,
    report_directory: Path | None = None,
    dependencies: CycleDependencies | None = None,
) -> dict[str, Any]:
    """Process the next ordered batch and publish its formal Markdown state."""
    if isinstance(batch_size, bool) or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
    if isinstance(settle_ms, bool) or not isinstance(settle_ms, int) or settle_ms < 0:
        raise ValueError("settle_ms must be a non-negative integer")
    if planner is reviewer:
        raise ValueError("planner and reviewer must be independent instances")
    source_database = Path(source_database)
    lab_database = Path(lab_database)
    memory_root = Path(memory_root)
    report_directory = Path(report_directory) if report_directory is not None else None
    deps = dependencies or CycleDependencies()
    source_identifier = source_id(source_database)
    run_id = "unifiedcycle_" + uuid4().hex
    stages: dict[str, Any] = {}

    deps.ingest(source_database, lab_database, "ingest")
    stages["ingest"] = {"status": "complete"}
    deps.scan_turns(source_database, lab_database, settle_ms)
    stages["turn_scan"] = {"status": "complete"}
    _ensure_audit_schema(lab_database)
    checkpoint = _load_checkpoint(lab_database, source_identifier)
    if checkpoint is not None:
        return _resume_committed_checkpoint(
            source_database,
            lab_database,
            memory_root,
            report_directory,
            run_id=run_id,
            source_identifier=source_identifier,
            checkpoint=checkpoint,
            dependencies=deps,
            stages=stages,
        )
    turns = _pending_turns(lab_database, source_identifier, batch_size)
    before = _formal_snapshot(lab_database)
    _start_run(
        lab_database,
        run_id=run_id,
        source_identifier=source_identifier,
        turns=turns,
        before=before,
    )

    if not turns:
        source_summary = _source_summary([])
        report_files: Mapping[str, Any] | None = None
        if report_directory is not None:
            try:
                report_input = {
                    "run_id": run_id,
                    "status": "noop",
                    "formal_state_before": before,
                    "formal_state_after": before,
                    "formal_state_changed": False,
                    "actions": [],
                    "review": [],
                    "source_summary": source_summary,
                }
                report = deps.build_report(
                    lab_database,
                    report_input,
                    {"status": "not_run"},
                    stages,
                )
                if not isinstance(report, Mapping):
                    raise UnifiedCycleError("report builder returned a non-object report")
                report_files = deps.write_report(report_directory, report)
                if not isinstance(report_files, Mapping):
                    raise UnifiedCycleError("report writer returned a non-object result")
                stages["report"] = {"status": "complete"}
            except Exception as error:
                code, message = _safe_failure(error)
                stages["report"] = {"status": "failed", "error_code": code}
                _finish_run(
                    lab_database,
                    run_id=run_id,
                    status="failed",
                    stages=stages,
                    after=before,
                    actions=[],
                    review=[],
                    error_code=code,
                    error_message=message,
                )
                return {
                    "run_id": run_id,
                    "status": "failed",
                    "error_code": code,
                    "error_message": message,
                    "formal_state_before": before,
                    "formal_state_after": before,
                    "formal_state_changed": False,
                    "formal_state_unchanged": True,
                    "actions": [],
                    "review": [],
                    "selected_turn_count": 0,
                    "processed_turn_count": 0,
                    "stages": stages,
                    "source_summary": source_summary,
                }
        _finish_run(
            lab_database,
            run_id=run_id,
            status="noop",
            stages=stages,
            after=before,
            actions=[],
            review=[],
        )
        return {
            "run_id": run_id,
            "status": "noop",
            "formal_state_before": before,
            "formal_state_after": before,
            "formal_state_changed": False,
            "actions": [],
            "review": [],
            "selected_turn_count": 0,
            "processed_turn_count": 0,
            "stages": stages,
            "source_summary": source_summary,
            "report_files": dict(report_files or {}),
        }

    bundles: list[Mapping[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    candidate_count = 0
    accepted_count = 0
    rejected_count = 0
    current_stage = "evidence"
    try:
        for turn in turns:
            bundles.append(
                fit_turn_evidence_to_model_limits(
                    deps.build_evidence(
                        source_database, lab_database, turn["turn_id"]
                    )
                )
            )
        existing = _existing_memories(lab_database)
        candidate_turn_count = len(turns)
        selected_count = _longest_model_compatible_prefix(bundles, existing)
        turns = turns[:selected_count]
        bundles = bundles[:selected_count]
        _update_run_selection(lab_database, run_id=run_id, turns=turns)
        stages["evidence"] = {
            "status": "complete",
            "candidate_count": candidate_turn_count,
            "count": len(bundles),
            "deferred_count": candidate_turn_count - len(bundles),
            "selection": "longest_model_compatible_prefix",
        }
        current_stage = "writer"
        writer_plan = planner.plan(bundles, existing_memories=existing)
        if not isinstance(writer_plan, Mapping):
            raise UnifiedCycleError("planner returned a non-object plan")
        candidate_count = int(writer_plan.get("candidate_count", 0) or 0)
        stages["writer"] = {
            "status": "complete",
            "candidate_count": candidate_count,
        }
        current_stage = "reviewer"
        reviewer_result = reviewer.review(
            writer_plan, bundles, existing_memories=existing
        )
        if not isinstance(reviewer_result, Mapping):
            raise UnifiedCycleError("reviewer returned a non-object decision")
        review_rows = _review_rows(reviewer_result, writer_plan)
        outcome = reviewer_result.get("outcome")
        stages["reviewer"] = {"status": str(outcome or "invalid")}
        if outcome == "reject":
            rejected_count = candidate_count
            after = _formal_snapshot(lab_database)
            if after != before:
                raise UnifiedCycleError("formal state changed during rejected review")
            source_summary = _source_summary(
                bundles,
                candidates=candidate_count,
                rejected=rejected_count,
            )
            report_files: Mapping[str, Any] | None = None
            if report_directory is not None:
                current_stage = "report"
                report_input = {
                    "run_id": run_id,
                    "status": "complete_with_rejections",
                    "formal_state_before": before,
                    "formal_state_after": after,
                    "formal_state_changed": False,
                    "actions": [],
                    "review": review_rows,
                    "source_summary": source_summary,
                }
                report = deps.build_report(
                    lab_database,
                    report_input,
                    {"status": "not_run"},
                    stages,
                )
                if not isinstance(report, Mapping):
                    raise UnifiedCycleError("report builder returned a non-object report")
                report_files = deps.write_report(report_directory, report)
                if not isinstance(report_files, Mapping):
                    raise UnifiedCycleError("report writer returned a non-object result")
                stages["report"] = {"status": "complete"}
            _finish_run(
                lab_database,
                run_id=run_id,
                status="complete_with_rejections",
                stages=stages,
                after=after,
                actions=[],
                review=review_rows,
                bundles=bundles,
                mark_success=True,
            )
            return {
                "run_id": run_id,
                "status": "complete_with_rejections",
                "formal_state_before": before,
                "formal_state_after": after,
                "formal_state_changed": False,
                "actions": [],
                "review": review_rows,
                "selected_turn_count": len(turns),
                "processed_turn_count": len(bundles),
                "stages": stages,
                "source_summary": source_summary,
                "report_files": dict(report_files or {}),
            }
        if outcome not in {"accept", "revise"} or not reviewer_result.get(
            "commit_allowed"
        ):
            raise UnifiedCycleError("reviewer decision is not commit-authorized")

        accepted_count = candidate_count
        current_stage = "commit"
        commit_report = deps.commit_reviewed_plan(lab_database, reviewer_result)
        if not isinstance(commit_report, Mapping):
            raise UnifiedCycleError("commit returned a non-object report")
        actions = _commit_actions(commit_report)
        stages["commit"] = {
            "status": str(commit_report.get("status") or "complete"),
            "action_count": len(actions),
        }
        after_commit = _formal_snapshot(lab_database)
        _checkpoint_commit(
            lab_database,
            checkpoint_id=run_id,
            source_identifier=source_identifier,
            before=before,
            after=after_commit,
            actions=actions,
            review=review_rows,
            commit_report=commit_report,
            bundles=bundles,
        )
        current_stage = "topics"
        topic_report = deps.ensure_topics(lab_database)
        stages["topics"] = {
            "status": str(topic_report.get("status") or "complete")
            if isinstance(topic_report, Mapping)
            else "complete"
        }
        current_stage = "markdown"
        markdown_report = deps.publish(lab_database, memory_root)
        if not isinstance(markdown_report, Mapping):
            raise UnifiedCycleError("publisher returned a non-object report")
        stages["markdown"] = {
            "status": str(markdown_report.get("status") or "published"),
            "generation_id": markdown_report.get("generation_id"),
        }
        after = _formal_snapshot(lab_database)
        generation_matches = after == after_commit
        markdown_report = {
            **dict(markdown_report),
            "matches_run_state": generation_matches,
            "generation_matches_current_database": generation_matches,
        }
        source_summary = _source_summary(
            bundles,
            candidates=candidate_count,
            accepted=accepted_count,
        )
        report_files: Mapping[str, Any] | None = None
        if report_directory is not None:
            current_stage = "report"
            report_input = {
                "run_id": run_id,
                "status": "complete",
                "formal_state_before": before,
                "formal_state_after": after,
                "formal_state_changed": before != after,
                "actions": actions,
                "review": review_rows,
                "source_summary": source_summary,
            }
            report = deps.build_report(
                lab_database, report_input, markdown_report, stages
            )
            if not isinstance(report, Mapping):
                raise UnifiedCycleError("report builder returned a non-object report")
            report_files = deps.write_report(report_directory, report)
            if not isinstance(report_files, Mapping):
                raise UnifiedCycleError("report writer returned a non-object result")
            stages["report"] = {"status": "complete"}
        _finish_run(
            lab_database,
            run_id=run_id,
            status="complete",
            stages=stages,
            after=after,
            actions=actions,
            review=review_rows,
            bundles=bundles,
            mark_success=True,
            checkpoint_id=run_id,
        )
        return {
            "run_id": run_id,
            "status": "complete",
            "formal_state_before": before,
            "formal_state_after": after,
            "formal_state_changed": before != after,
            "actions": actions,
            "review": review_rows,
            "selected_turn_count": len(turns),
            "processed_turn_count": len(bundles),
            "stages": stages,
            "markdown": dict(markdown_report),
            "commit": dict(commit_report),
            "source_summary": source_summary,
            "report_files": dict(report_files or {}),
        }
    except Exception as error:
        after = _formal_snapshot(lab_database)
        error_code, error_message = _safe_failure(error)
        stages[current_stage] = {"status": "failed", "error_code": error_code}
        _finish_run(
            lab_database,
            run_id=run_id,
            status="failed",
            stages=stages,
            after=after,
            actions=actions,
            review=review_rows,
            error_code=error_code,
            error_message=error_message,
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "error_code": error_code,
            "error_message": error_message,
            "formal_state_before": before,
            "formal_state_after": after,
            "formal_state_changed": before != after,
            "formal_state_unchanged": before == after,
            "actions": actions,
            "review": review_rows,
            "selected_turn_count": len(turns),
            "processed_turn_count": 0,
            "stages": stages,
            "source_summary": _source_summary(
                bundles,
                candidates=candidate_count,
                accepted=accepted_count,
                rejected=rejected_count,
            ),
        }


__all__ = [
    "CYCLE_VERSION",
    "CycleDependencies",
    "MAX_BATCH_SIZE",
    "UnifiedCycleError",
    "run_unified_cycle",
]
