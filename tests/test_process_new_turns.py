from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import memory_lab.process_new_turns as process_module
from memory_lab.process_new_turns import (
    PROCESSING_STATE_TABLE_SQL,
    load_terminal_turn_candidates,
    process_new_turns,
)
from memory_lab.store import open_store


class ProcessNewTurnsTests(unittest.TestCase):
    def test_processing_state_table_keeps_dry_run_separate(self) -> None:
        self.assertIn("turn_processing_state", PROCESSING_STATE_TABLE_SQL)
        self.assertIn("dry_run boolean NOT NULL", PROCESSING_STATE_TABLE_SQL)
        self.assertIn("UNIQUE(turn_id, input_hash, dry_run)", PROCESSING_STATE_TABLE_SQL)

    def test_load_terminal_turn_candidates_orders_recent_and_filters_non_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            db = Path(raw_tmp) / "lab.db"
            with closing(open_store(db)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO source(source_id, source_kind, source_path, created_at)
                    VALUES ('src', 'cp_sqlite', 'x.db', '2026-09-21T00:00:00Z')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO turn_detection_run(
                      run_id, source_id, detector_version, started_at, settle_ms, status
                    )
                    VALUES ('run', 'src', 'test', '2026-09-21T00:00:00Z', 0, 'completed')
                    """
                )
                for turn_id, state, ts, signature in [
                    ("turn_old", "completed", 1000, "hash_old"),
                    ("turn_pending", "pending", 3000, "hash_pending"),
                    ("turn_new", "failed", 2000, "hash_new"),
                ]:
                    connection.execute(
                        """
                        INSERT INTO event(
                          event_id, source_id, session_id, message_id, part_id,
                          event_type, timestamp, content, content_hash,
                          source_kind, ingested_at, raw_json
                        )
                        VALUES (?, 'src', 'session', ?, ?, 'user_message', ?, 'x', ?, 'user', 'now', '{}')
                        """,
                        (
                            f"event_{turn_id}",
                            f"msg_{turn_id}",
                            f"part_{turn_id}",
                            ts,
                            f"hash_event_{turn_id}",
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO conversation_turn(
                          turn_id, source_id, session_id, user_message_id,
                          primary_event_id, user_timestamp, current_state,
                          first_detected_at, last_detected_at, last_detection_run_id,
                          last_structural_signature
                        )
                        VALUES (?, 'src', 'session', ?, ?, ?, ?, 'a', 'b', 'run', ?)
                        """,
                        (turn_id, f"user_{turn_id}", f"event_{turn_id}", ts, state, signature),
                    )
                connection.commit()

            candidates = load_terminal_turn_candidates(
                lab_database=db,
                source_identifier="src",
                limit=10,
            )

        self.assertEqual([item["turn_id"] for item in candidates], ["turn_new", "turn_old"])
        self.assertEqual(candidates[0]["input_hash"], "hash_new")

    def test_process_new_turns_executes_selected_pending_turns_oldest_first(self) -> None:
        calls: list[str] = []
        candidates = [
            {
                "turn_id": "turn_new",
                "session_id": "session",
                "user_message_id": "msg_new",
                "user_timestamp": 3000,
                "state": "completed",
                "input_hash": "hash_new",
                "assistant_count": 1,
                "part_count": 2,
                "active_tool_count": 0,
            },
            {
                "turn_id": "turn_mid",
                "session_id": "session",
                "user_message_id": "msg_mid",
                "user_timestamp": 2000,
                "state": "completed",
                "input_hash": "hash_mid",
                "assistant_count": 1,
                "part_count": 2,
                "active_tool_count": 0,
            },
            {
                "turn_id": "turn_old",
                "session_id": "session",
                "user_message_id": "msg_old",
                "user_timestamp": 1000,
                "state": "completed",
                "input_hash": "hash_old",
                "assistant_count": 1,
                "part_count": 2,
                "active_tool_count": 0,
            },
        ]

        def fake_react(**kwargs):
            turn_id = kwargs["turn_id"]
            calls.append(turn_id)
            return {
                "run_id": f"react_{turn_id}",
                "status": "complete",
                "plan": {
                    "changes": [
                        {
                            "action": "noop",
                            "reason": "test",
                        }
                    ]
                },
            }

        with (
            patch.object(process_module, "ensure_turn_processing_state"),
            patch.object(process_module, "run_ingestion", return_value={"run_id": "ingest"}),
            patch.object(process_module, "run_turn_scan", return_value={"run_id": "scan"}),
            patch.object(process_module, "source_id", return_value="src"),
            patch.object(process_module, "load_terminal_turn_candidates", return_value=candidates),
            patch.object(process_module, "_pg_processed_keys", return_value=set()),
            patch.object(process_module, "mark_turn_processing_state"),
            patch.object(process_module, "ingest_turn_to_postgres", return_value={"ok": True}),
            patch.object(process_module, "run_memory_react_agent", side_effect=fake_react),
        ):
            report = process_new_turns(
                source_database=Path("source.db"),
                lab_database=Path("lab.db"),
                pg_config_path=Path("pg.json"),
                config_path=Path("config.json"),
                limit=3,
            )

        self.assertEqual(calls, ["turn_old", "turn_mid", "turn_new"])
        self.assertEqual(
            [item["turn_id"] for item in report["turns"]],
            ["turn_old", "turn_mid", "turn_new"],
        )


if __name__ == "__main__":
    unittest.main()
