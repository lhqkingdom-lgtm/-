"""Filesystem guards that keep the CodePulse source database read-only."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path


LAB_SENTINEL_TABLES = {"source", "ingestion_run", "processed_part", "event"}


def paths_refer_to_same_file(first: Path, second: Path) -> bool:
    if str(first.resolve()).casefold() == str(second.resolve()).casefold():
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            pass
    return False


def require_distinct_source_and_lab(source_database: Path, lab_database: Path) -> None:
    if paths_refer_to_same_file(source_database, lab_database):
        raise ValueError("lab database must not be the CodePulse source database")


def require_memory_lab_database(lab_database: Path) -> None:
    """Validate an existing lab without opening it in writable/create mode."""
    if not lab_database.is_file():
        raise FileNotFoundError(f"lab database does not exist: {lab_database}")
    uri = lab_database.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
    except sqlite3.Error as error:
        raise ValueError(f"lab database is not valid SQLite: {lab_database}") from error
    missing = LAB_SENTINEL_TABLES - tables
    if missing:
        raise ValueError(
            "existing --lab-db is not a Memory Lab database; missing tables: "
            + ", ".join(sorted(missing))
        )
