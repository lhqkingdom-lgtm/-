"""Read-only CodePulse SQLite ingestion and event normalization."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SourcePart:
    session_id: str
    parent_session_id: str | None
    directory: str | None
    session_title: str | None
    project_id: str | None
    session_version: str | None
    message_id: str
    message_time: int | None
    message_data: dict
    part_id: str
    part_time: int | None
    part_data: dict


def source_id(database: Path) -> str:
    normalized = str(database.resolve()).casefold().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def open_read_only(database: Path) -> sqlite3.Connection:
    if not database.is_file():
        raise FileNotFoundError(f"CodePulse database does not exist: {database}")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def source_activity_token(database: Path) -> str:
    """Return a cheap read-only token that changes with observable CP activity.

    The token is only a wake-up hint; downstream part IDs, structural samples
    and formal CAS checks remain the authority for correctness.
    """

    projections: list[tuple] = []
    with open_read_only(database) as connection:
        connection.execute("BEGIN")
        for table in ("session", "message", "part"):
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            updated = "MAX(time_updated)" if "time_updated" in columns else "NULL"
            created = "MAX(time_created)" if "time_created" in columns else "NULL"
            data_size = "SUM(LENGTH(data))" if "data" in columns else "NULL"
            projections.append(
                tuple(
                    connection.execute(
                        f"SELECT COUNT(*), {created}, {updated}, {data_size} FROM {table}"
                    ).fetchone()
                )
            )
        connection.rollback()
    payload = json.dumps(projections, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def iter_parts(connection: sqlite3.Connection) -> Iterator[SourcePart]:
    query = """
        SELECT
            s.id AS session_id,
            s.parent_id AS parent_session_id,
            s.directory AS directory,
            s.title AS session_title,
            s.project_id AS project_id,
            s.version AS session_version,
            m.id AS message_id,
            m.time_created AS message_time,
            m.data AS message_data,
            p.id AS part_id,
            p.time_created AS part_time,
            p.data AS part_data
        FROM part AS p
        JOIN message AS m ON m.id = p.message_id
        JOIN session AS s ON s.id = p.session_id
        ORDER BY
            COALESCE(p.time_created, m.time_created, s.time_created),
            s.id,
            m.id,
            p.id
    """
    for row in connection.execute(query):
        try:
            message_data = json.loads(row["message_data"])
        except (TypeError, json.JSONDecodeError) as error:
            message_data = {"_parse_error": str(error), "_raw": row["message_data"]}
        try:
            part_data = json.loads(row["part_data"])
        except (TypeError, json.JSONDecodeError) as error:
            part_data = {"_parse_error": str(error), "_raw": row["part_data"]}
        yield SourcePart(
            session_id=row["session_id"],
            parent_session_id=row["parent_session_id"],
            directory=row["directory"],
            session_title=row["session_title"],
            project_id=row["project_id"],
            session_version=row["session_version"],
            message_id=row["message_id"],
            message_time=row["message_time"],
            message_data=message_data,
            part_id=row["part_id"],
            part_time=row["part_time"],
            part_data=part_data,
        )


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def classify(part: SourcePart) -> tuple[str, str]:
    """Return (disposition, reason) without doing semantic memory extraction."""
    if "_parse_error" in part.message_data:
        return "filtered", "invalid_message_json"
    if "_parse_error" in part.part_data:
        return "filtered", "invalid_part_json"
    if part.message_data.get("role") != "user":
        return "filtered", "not_user_message"
    if part.part_data.get("type") != "text":
        return "filtered", "not_user_text"
    text = part.part_data.get("text")
    if not isinstance(text, str) or not text.strip():
        return "filtered", "empty_user_text"
    stripped = text.lstrip().casefold()
    if stripped.startswith("<system-reminder") or stripped.startswith("<system"):
        return "filtered", "system_injected_text"
    return "event", "user_text"


def to_event(part: SourcePart, source_identifier: str) -> dict:
    content = normalize_text(part.part_data["text"])
    message_agent = part.message_data.get("agent")
    timestamp = part.part_time or part.message_time
    return {
        "event_id": f"sqlite:{source_identifier}:{part.part_id}",
        "event_type": "user_text",
        "session_id": part.session_id,
        "parent_session_id": part.parent_session_id,
        "message_id": part.message_id,
        "part_id": part.part_id,
        "directory": part.directory,
        "session_title": part.session_title,
        "project_id": part.project_id,
        "session_version": part.session_version,
        "agent": message_agent,
        "timestamp": timestamp,
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "source_kind": "sqlite",
        "source_id": source_identifier,
    }
