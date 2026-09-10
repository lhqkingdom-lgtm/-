"""Display event text as Unicode escapes to separate storage from console issues."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path


DATABASE = Path(__file__).resolve().parents[1] / "data" / "memory_lab.db"


with closing(sqlite3.connect(DATABASE)) as connection:
    rows = connection.execute(
        "SELECT session_id, directory, part_id, content FROM event ORDER BY timestamp"
    ).fetchall()
    for session_id, directory, part_id, content in rows:
        print(
            json.dumps(
                {
                    "session_id": session_id,
                    "directory": directory,
                    "part_id": part_id,
                    "content": content,
                },
                ensure_ascii=True,
            )
        )
