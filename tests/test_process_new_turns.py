from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.process_new_turns import (
    PROCESSING_STATE_TABLE_SQL,
    load_terminal_turn_candidates,
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


if __name__ == "__main__":
    unittest.main()
