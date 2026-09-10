from __future__ import annotations

import sqlite3
import unittest

from memory_lab.longitudinal import (
    evaluate_recall_case,
    freeze_recall_case,
    run_status,
    start_run,
)


def fixture() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE memory(memory_id TEXT);
        CREATE TABLE memory_revision(revision_id TEXT);
        CREATE TABLE evidence(evidence_id TEXT);
        CREATE TABLE event(event_id TEXT, parent_session_id TEXT, timestamp INTEGER);
        CREATE TABLE event_annotation(event_id TEXT, project_key TEXT);
        CREATE TABLE conversation_turn(
          turn_id TEXT, session_id TEXT, primary_event_id TEXT,
          user_timestamp INTEGER, current_state TEXT
        );
        """
    )
    return connection


class LongitudinalTests(unittest.TestCase):
    def test_freezes_gold_and_scores_recall(self) -> None:
        connection = fixture()
        run = start_run(connection, "open test")
        frozen = freeze_recall_case(
            connection,
            run["run_id"],
            "生成代码",
            {"project_key": "p1"},
            ["workflow.code_generation.confirmation"],
        )
        result = evaluate_recall_case(
            connection,
            frozen["case_id"],
            ["workflow.code_generation.confirmation"],
        )
        self.assertTrue(result["hit"])
        with self.assertRaises(sqlite3.DatabaseError):
            connection.execute(
                "UPDATE longitudinal_recall_case SET query = 'changed' WHERE case_id = ?",
                (frozen["case_id"],),
            )

    def test_status_exposes_sample_gates(self) -> None:
        connection = fixture()
        run = start_run(connection, "open test")
        status = run_status(connection, run["run_id"])
        self.assertFalse(status["sample_ready"])
        self.assertEqual(status["visibility"], "open")


if __name__ == "__main__":
    unittest.main()
