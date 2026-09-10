from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from memory_lab.ingest import run_ingestion
from memory_lab.turns import run_turn_scan


class TurnScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.source = root / "source.db"
        self.store = root / "memory_lab.db"
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE session (
                    id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, directory TEXT,
                    title TEXT, version TEXT, time_created INTEGER
                );
                CREATE TABLE message (
                    id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT
                );
                CREATE TABLE part (
                    id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
                    time_created INTEGER, data TEXT
                );
                INSERT INTO session VALUES
                    ('ses_1', 'global', NULL, 'D:\\project-a', 'turn test', '1.0', 1000);
                """
            )
        run_ingestion(self.source, self.store, "baseline")
        # Compact fixture timestamps are deliberately deterministic; make them
        # post-baseline for these detector-focused tests.
        with closing(sqlite3.connect(self.store)) as connection, connection:
            connection.execute("UPDATE source SET baseline_cutoff_ms = 0")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_user(self, message_id: str, timestamp: int, text: str = "用户任务") -> None:
        self.add_message(
            message_id,
            timestamp,
            {"role": "user", "time": {"created": timestamp}},
        )
        self.add_part(
            "prt_" + message_id,
            message_id,
            timestamp + 1,
            {"type": "text", "text": text},
        )

    def add_assistant(
        self,
        message_id: str,
        parent_id: str,
        timestamp: int,
        *,
        finish: str | None = None,
        completed: bool = True,
        error: bool = False,
    ) -> None:
        data: dict = {
            "role": "assistant",
            "parentID": parent_id,
            "time": {"created": timestamp},
        }
        if finish is not None:
            data["finish"] = finish
        if completed:
            data["time"]["completed"] = timestamp + 10
        if error:
            data["error"] = {"name": "APIError", "message": "not persisted by lab"}
        self.add_message(message_id, timestamp, data)

    def add_message(self, message_id: str, timestamp: int, data: dict) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO message VALUES (?, 'ses_1', ?, ?)",
                (message_id, timestamp, json.dumps(data, ensure_ascii=False)),
            )

    def add_part(self, part_id: str, message_id: str, timestamp: int, data: dict) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO part VALUES (?, ?, 'ses_1', ?, ?)",
                (part_id, message_id, timestamp, json.dumps(data, ensure_ascii=False)),
            )

    def update_part(self, part_id: str, data: dict) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "UPDATE part SET data = ? WHERE id = ?",
                (json.dumps(data, ensure_ascii=False), part_id),
            )

    def update_part_with_timestamp(
        self, part_id: str, data: dict, time_updated: int
    ) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "UPDATE part SET data = ?, time_updated = ? WHERE id = ?",
                (json.dumps(data, ensure_ascii=False), time_updated, part_id),
            )

    def ingest(self) -> None:
        run_ingestion(self.source, self.store, "ingest")

    def by_message(self, report: dict) -> dict[str, dict]:
        return {item["user_message_id"]: item for item in report["turns"]}

    def test_tool_loop_with_final_stop_is_one_completed_turn(self) -> None:
        self.add_user("usr_1", 2000)
        self.add_assistant("ast_1", "usr_1", 2010, finish="tool-calls")
        self.add_part(
            "tool_1",
            "ast_1",
            2011,
            {"type": "tool", "state": {"status": "completed", "output": "secret"}},
        )
        self.add_assistant("ast_2", "usr_1", 2020, finish="tool-calls")
        self.add_part("text_mid", "ast_2", 2021, {"type": "text", "text": "处理中"})
        self.add_assistant("ast_3", "usr_1", 2030, finish="stop")
        self.add_part("text_final", "ast_3", 2031, {"type": "text", "text": "完成"})
        self.ingest()

        report = run_turn_scan(self.source, self.store, settle_ms=0)

        self.assertEqual(report["turn_count"], 1)
        turn = report["turns"][0]
        self.assertEqual(turn["state"], "completed")
        self.assertEqual(turn["assistant_count"], 3)
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM conversation_turn").fetchone()[0], 1)

    def test_tool_calls_finish_is_non_terminal_and_running_tool_blocks_terminal(self) -> None:
        self.add_user("usr_1", 2000)
        self.add_assistant("ast_1", "usr_1", 2010, finish="tool-calls")
        self.add_part(
            "tool_done",
            "ast_1",
            2011,
            {"type": "tool", "state": {"status": "completed", "output": "done"}},
        )
        self.ingest()
        first = run_turn_scan(self.source, self.store, settle_ms=0)
        self.assertEqual(first["turns"][0]["state"], "pending")

        self.add_assistant("ast_2", "usr_1", 2020, finish="stop")
        self.add_part(
            "tool_running",
            "ast_2",
            2021,
            {"type": "tool", "state": {"status": "running", "input": {"secret": 1}}},
        )
        second = run_turn_scan(self.source, self.store, settle_ms=0)
        self.assertEqual(second["turns"][0]["state"], "pending")
        self.assertEqual(second["turns"][0]["active_tool_count"], 1)

    def test_final_text_written_between_samples_waits_for_next_stable_scan(self) -> None:
        self.add_user("usr_1", 2000)
        self.add_assistant("ast_1", "usr_1", 2010, finish="stop")
        self.add_part("text_final", "ast_1", 2011, {"type": "text", "text": "短"})
        self.ingest()

        with mock.patch(
            "memory_lab.turns._wait_between_samples",
            side_effect=lambda _milliseconds: self.update_part(
                "text_final", {"type": "text", "text": "最终可见文本稍后写入"}
            ),
        ):
            changing = run_turn_scan(self.source, self.store, settle_ms=0)

        self.assertEqual(changing["turns"][0]["state"], "terminal_waiting_stability")
        self.assertFalse(changing["turns"][0]["signature_stable"])
        stable = run_turn_scan(self.source, self.store, settle_ms=0)
        self.assertEqual(stable["turns"][0]["state"], "completed")

    def test_same_length_text_update_is_seen_through_optional_time_updated(self) -> None:
        self.add_user("usr_1", 2000)
        self.add_assistant("ast_1", "usr_1", 2010, finish="stop")
        self.add_part("text_final", "ast_1", 2011, {"type": "text", "text": "甲乙"})
        self.ingest()
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute("ALTER TABLE message ADD COLUMN time_updated INTEGER")
            connection.execute("ALTER TABLE part ADD COLUMN time_updated INTEGER")
            connection.execute("UPDATE message SET time_updated = time_created")
            connection.execute("UPDATE part SET time_updated = time_created")

        with mock.patch(
            "memory_lab.turns._wait_between_samples",
            side_effect=lambda _milliseconds: self.update_part_with_timestamp(
                "text_final", {"type": "text", "text": "丙丁"}, 9999
            ),
        ):
            changing = run_turn_scan(self.source, self.store, settle_ms=0)

        self.assertEqual(changing["turns"][0]["state"], "terminal_waiting_stability")
        self.assertFalse(changing["turns"][0]["signature_stable"])
        stable = run_turn_scan(self.source, self.store, settle_ms=0)
        self.assertEqual(stable["turns"][0]["state"], "completed")

    def test_terminal_assistant_error_is_failed(self) -> None:
        self.add_user("usr_1", 2000)
        self.add_assistant(
            "ast_error", "usr_1", 2010, completed=False, error=True
        )
        self.ingest()

        report = run_turn_scan(self.source, self.store, settle_ms=0)

        self.assertEqual(report["turns"][0]["state"], "failed")
        self.assertTrue(report["turns"][0]["terminal_error"])

    def test_later_user_message_interrupts_unterminated_turn(self) -> None:
        self.add_user("usr_1", 2000, "第一轮")
        self.add_assistant("ast_1", "usr_1", 2010, finish="tool-calls")
        self.add_part(
            "tool_stale",
            "ast_1",
            2011,
            {"type": "tool", "state": {"status": "running", "input": {}}},
        )
        self.add_user("usr_2", 3000, "第二轮")
        self.ingest()

        report = run_turn_scan(self.source, self.store, settle_ms=0)
        turns = self.by_message(report)

        self.assertEqual(turns["usr_1"]["state"], "interrupted")
        self.assertEqual(turns["usr_1"]["next_user_message_id"], "usr_2")
        self.assertEqual(turns["usr_2"]["state"], "pending")

    def test_unrecognized_finish_is_not_terminal_and_is_redacted_to_other(self) -> None:
        self.add_user("usr_1", 2000)
        self.add_assistant(
            "ast_unknown", "usr_1", 2010, finish="provider-private-finish"
        )
        self.ingest()

        report = run_turn_scan(self.source, self.store, settle_ms=0)

        self.assertEqual(report["turns"][0]["state"], "pending")
        self.assertEqual(report["turns"][0]["terminal_finish"], "other")
        with closing(sqlite3.connect(self.store)) as connection:
            stored = connection.execute(
                "SELECT terminal_finish FROM conversation_turn"
            ).fetchone()[0]
        self.assertEqual(stored, "other")
        self.assertNotIn("provider-private-finish", json.dumps(report))

        # `length` is one of CP/AI SDK's known final values, but is redacted to
        # the public `other` bucket in reports and turn audit rows.
        self.add_assistant("ast_length", "usr_1", 2020, finish="length")
        known_terminal = run_turn_scan(self.source, self.store, settle_ms=0)
        self.assertEqual(known_terminal["turns"][0]["state"], "completed")
        self.assertEqual(known_terminal["turns"][0]["terminal_finish"], "other")

        # Historical/imported providers can use `end_turn`; it is a known
        # terminal semantic but still redacted to the public `other` bucket.
        self.add_user("usr_2", 3000)
        self.add_assistant("ast_end_turn", "usr_2", 3010, finish="end_turn")
        self.ingest()
        imported_terminal = run_turn_scan(self.source, self.store, settle_ms=0)
        self.assertEqual(imported_terminal["turn_count"], 1)
        self.assertEqual(imported_terminal["turns"][0]["state"], "completed")
        self.assertEqual(imported_terminal["turns"][0]["terminal_finish"], "other")

    def test_repeated_scan_keeps_turn_identity_but_appends_audit(self) -> None:
        self.add_user("usr_1", 2000)
        self.add_assistant("ast_1", "usr_1", 2010, finish="stop")
        self.ingest()

        first = run_turn_scan(self.source, self.store, settle_ms=0)
        second = run_turn_scan(self.source, self.store, settle_ms=0)

        self.assertEqual(first["turn_count"], 1)
        self.assertEqual(second["turn_count"], 0)
        self.assertEqual(second["tracked_turn_count"], 1)
        self.assertEqual(second["skipped_terminal_turns"], 1)
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM conversation_turn").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM turn_detection_run").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM turn_detection_result").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM turn_observation").fetchone()[0], 2)
            raw = "\n".join(
                str(row[0])
                for row in connection.execute(
                    "SELECT structural_signature FROM turn_observation"
                )
            )
            self.assertNotIn("用户任务", raw)


if __name__ == "__main__":
    unittest.main()
