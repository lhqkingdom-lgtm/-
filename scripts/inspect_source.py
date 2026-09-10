"""Read-only diagnostics for the CodePulse conversation database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--samples", action="store_true")
    args = parser.parse_args()

    uri = args.database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        for (table_name,) in rows:
            columns = connection.execute(
                f'PRAGMA table_info("{table_name.replace(chr(34), chr(34) * 2)}")'
            ).fetchall()
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name.replace(chr(34), chr(34) * 2)}"'
            ).fetchone()[0]
            print(f"{table_name} ({count} rows)")
            print("  " + ", ".join(column[1] for column in columns))

        if args.samples:
            print("\nRecent messages and parts")
            samples = connection.execute(
                """
                SELECT m.id, m.session_id, m.time_created, m.data,
                       p.id, p.time_created, p.data
                FROM message AS m
                JOIN part AS p ON p.message_id = m.id
                ORDER BY COALESCE(p.time_created, m.time_created) DESC
                LIMIT 8
                """
            ).fetchall()
            for row in samples:
                message_data = json.loads(row[3])
                part_data = json.loads(row[6])
                print(
                    json.dumps(
                        {
                            "message_id": row[0],
                            "session_id": row[1],
                            "message_time": row[2],
                            "message_data": message_data,
                            "part_id": row[4],
                            "part_time": row[5],
                            "part_data": part_data,
                        },
                        ensure_ascii=False,
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
