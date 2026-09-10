"""Prepare immutable ingested events for controlled memory experiments."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from pathlib import Path, PureWindowsPath

from .store import now_iso, open_store


EXPERIMENT_PREFIX = re.compile(r"^【(?P<case_id>ML-[^】]+)】\s*")


def normalize_directory(directory: str) -> str:
    return str(PureWindowsPath(directory)).rstrip("\\").casefold()


def project_identity(directory: str) -> tuple[str, str]:
    normalized = normalize_directory(directory)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    display_name = PureWindowsPath(directory).name or directory
    return f"directory:{digest}", display_name


def split_experiment_prefix(content: str) -> tuple[str | None, str]:
    match = EXPERIMENT_PREFIX.match(content)
    if not match:
        return None, content.strip()
    return match.group("case_id"), content[match.end() :].strip()


def prepare_dataset(lab_database: Path) -> dict:
    prepared = 0
    existing = 0
    sessions: dict[str, dict] = {}
    projects: dict[str, dict] = {}
    cases: list[dict] = []

    with closing(open_store(lab_database)) as connection, connection:
        eligibility_counts = {
            row["eligibility_status"]: row["event_count"]
            for row in connection.execute(
                """
                SELECT eligibility_status, COUNT(*) AS event_count
                FROM event
                GROUP BY eligibility_status
                """
            )
        }
        isolated_events = sum(
            count
            for status, count in eligibility_counts.items()
            if status != "eligible"
        )
        # Older lab versions could already have prepared a late event. The
        # normalized event remains as immutable audit evidence, while its
        # derived annotation is explicitly removed so downstream candidate
        # stages that consume prepared rows exclude it by default.
        isolated_annotation_cursor = connection.execute(
            """
            DELETE FROM event_annotation
            WHERE event_id IN (
                SELECT event_id
                FROM event
                WHERE eligibility_status != 'eligible'
            )
            """
        )
        events = connection.execute(
            """
            SELECT event_id, source_id, session_id, directory, content, timestamp
            FROM event
            WHERE eligibility_status = 'eligible'
            ORDER BY timestamp, event_id
            """
        ).fetchall()

        for event in events:
            directory = event["directory"] or "<unknown>"
            project_key, display_name = project_identity(directory)
            case_id, semantic_content = split_experiment_prefix(event["content"])
            current_time = now_iso()

            connection.execute(
                """
                INSERT INTO project_context(
                    project_key, directory, display_name, identity_source, created_at
                ) VALUES (?, ?, ?, 'session.directory', ?)
                ON CONFLICT(project_key) DO UPDATE SET
                    directory = excluded.directory,
                    display_name = excluded.display_name
                """,
                (project_key, directory, display_name, current_time),
            )
            connection.execute(
                """
                INSERT INTO session_context(
                    source_id, session_id, project_key, directory, context_source,
                    first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, 'session.directory', ?, ?)
                ON CONFLICT(source_id, session_id) DO UPDATE SET
                    project_key = excluded.project_key,
                    directory = excluded.directory,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    event["source_id"],
                    event["session_id"],
                    project_key,
                    directory,
                    current_time,
                    current_time,
                ),
            )
            cursor = connection.execute(
                """
                INSERT INTO event_annotation(
                    event_id, experiment_case_id, semantic_content, project_key,
                    annotation_source, created_at
                ) VALUES (?, ?, ?, ?, 'deterministic-prefix-v1', ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (event["event_id"], case_id, semantic_content, project_key, current_time),
            )
            if cursor.rowcount:
                prepared += 1
            else:
                existing += 1

            sessions[event["session_id"]] = {
                "session_id": event["session_id"],
                "project_key": project_key,
                "directory": directory,
            }
            projects[project_key] = {
                "project_key": project_key,
                "display_name": display_name,
                "directory": directory,
            }
            cases.append(
                {
                    "case_id": case_id,
                    "session_id": event["session_id"],
                    "project_key": project_key,
                    "semantic_content": semantic_content,
                }
            )

    return {
        "lab_database": str(lab_database.resolve()),
        "events_total": sum(eligibility_counts.values()),
        "events_seen": len(cases),
        "isolated_events": isolated_events,
        "isolated_annotations_removed": isolated_annotation_cursor.rowcount,
        "eligibility_counts": dict(sorted(eligibility_counts.items())),
        "annotations_created": prepared,
        "annotations_existing": existing,
        "session_count": len(sessions),
        "project_count": len(projects),
        "projects": list(projects.values()),
        "sessions": list(sessions.values()),
        "cases": cases,
    }
