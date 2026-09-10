from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from memory_lab.cli import build_parser, main
from memory_lab.episodes import EXTRACTOR_VERSION, run_episode_extraction
from memory_lab.ingest import run_ingestion
from memory_lab.prepare import prepare_dataset
from memory_lab.turns import run_turn_scan


class EpisodeExtractionTests(unittest.TestCase):
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
                    ('ses_1', 'project-a', NULL, 'D:\\project-a', 'episode test', '1.0', 1000);
                """
            )
        run_ingestion(self.source, self.store, "baseline")
        with closing(sqlite3.connect(self.store)) as connection, connection:
            connection.execute("UPDATE source SET baseline_cutoff_ms = 0")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_message(self, message_id: str, timestamp: int, data: dict) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO message VALUES (?, 'ses_1', ?, ?)",
                (message_id, timestamp, json.dumps(data, ensure_ascii=False)),
            )

    def add_part(
        self, part_id: str, message_id: str, timestamp: int, data: dict
    ) -> None:
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

    def add_user(self, name: str, timestamp: int) -> str:
        message_id = "usr_" + name
        self.add_message(
            message_id,
            timestamp,
            {"role": "user", "time": {"created": timestamp}},
        )
        self.add_part(
            "user_part_" + name,
            message_id,
            timestamp + 1,
            {"type": "text", "text": "工程任务 " + name},
        )
        return message_id

    def add_assistant(
        self,
        name: str,
        parent_id: str,
        timestamp: int,
        *,
        failed: bool = False,
        finish: str = "stop",
    ) -> str:
        message_id = "ast_" + name
        data: dict = {
            "role": "assistant",
            "parentID": parent_id,
            "finish": finish,
            "time": {"created": timestamp},
        }
        if failed:
            data["error"] = {
                "name": "ProviderError",
                "message": "ASSISTANT_ERROR_SECRET",
            }
        else:
            data["time"]["completed"] = timestamp + 20
        self.add_message(message_id, timestamp, data)
        return message_id

    def finalize_turns(self) -> dict:
        run_ingestion(self.source, self.store, "ingest")
        prepare_dataset(self.store)
        return run_turn_scan(self.source, self.store, settle_ms=0)

    def seed_terminal_examples(self) -> None:
        success_user = self.add_user("success", 2000)
        success_assistant = self.add_assistant("success", success_user, 2010)
        self.add_part(
            "success_text",
            success_assistant,
            2011,
            {"type": "text", "text": "ASSISTANT_TEXT_SECRET"},
        )
        self.add_part(
            "success_reasoning",
            success_assistant,
            2012,
            {"type": "reasoning", "text": "HIDDEN_REASONING_SECRET"},
        )
        self.add_part(
            "success_bash",
            success_assistant,
            2013,
            {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "time": {"start": 2013, "end": 2014},
                    "input": {"command": "echo INPUT_SECRET D:\\private\\file"},
                    "output": "OUTPUT_SECRET",
                    "metadata": {
                        "exit": 0,
                        "cwd": "D:\\private",
                        "credential": "METADATA_SECRET",
                    },
                },
            },
        )
        self.add_part(
            "success_json",
            success_assistant,
            2014,
            {
                "type": "tool",
                "tool": "json_api",
                "state": {
                    "status": "completed",
                    "metadata": {"success": True, "token": "JSON_TOKEN_SECRET"},
                },
            },
        )
        self.add_part(
            "success_keil",
            success_assistant,
            2015,
            {
                "type": "tool",
                "tool": "keil_build",
                "state": {
                    "status": "completed",
                    "time": {"start": 2015, "end": 2016},
                    "metadata": {
                        "success": True,
                        "errorCount": 0,
                        "warningCount": 0,
                    },
                },
            },
        )

        recovered_user = self.add_user("recovered", 3000)
        recovered_assistant = self.add_assistant("recovered", recovered_user, 3010)
        self.add_part(
            "keil_failed",
            recovered_assistant,
            3011,
            {
                "type": "tool",
                "tool": "keil_build",
                "state": {
                    "status": "completed",
                    "time": {"start": 3011, "end": 3012},
                    "input": {"project": "D:\\secret\\firmware.uvprojx"},
                    "output": "LONG_PRIVATE_BUILD_LOG",
                    "metadata": {
                        "success": False,
                        "errorCount": 2,
                        "warningCount": 1,
                        "failureKind": "source",
                        "buildFailureKind": "D:\\must-not-be-stored",
                        "projectPath": "D:\\secret\\firmware.uvprojx",
                    },
                },
            },
        )
        self.add_part(
            "keil_succeeded",
            recovered_assistant,
            3012,
            {
                "type": "tool",
                "tool": "keil_build",
                "state": {
                    "status": "completed",
                    "time": {"start": 3012, "end": 3013},
                    "metadata": {
                        "success": True,
                        "errorCount": 0,
                        "warningCount": 3,
                        "private": "KEIL_METADATA_SECRET",
                    },
                },
            },
        )

        failed_user = self.add_user("failed", 4000)
        self.add_assistant("failed", failed_user, 4010, failed=True)

        unverified_user = self.add_user("unverified", 5000)
        unverified_assistant = self.add_assistant(
            "unverified", unverified_user, 5010
        )
        self.add_part(
            "edit_completed",
            unverified_assistant,
            5011,
            {
                "type": "tool",
                "tool": "edit",
                "state": {
                    "status": "completed",
                    "input": {"path": "D:\\private\\source.c"},
                    "output": "EDIT_OUTPUT_SECRET",
                    "metadata": {"success": True, "path": "D:\\private"},
                },
            },
        )
        self.add_part(
            "unverified_claim",
            unverified_assistant,
            5012,
            {"type": "text", "text": "已经成功完成并验证。SELF_CLAIM_SECRET"},
        )

        tool_failure_user = self.add_user("tool_failure", 6000)
        tool_failure_assistant = self.add_assistant(
            "tool_failure", tool_failure_user, 6010
        )
        self.add_part(
            "bash_error",
            tool_failure_assistant,
            6011,
            {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "error",
                    "input": "PRIVATE_COMMAND",
                    "error": {"message": "TOOL_ERROR_SECRET"},
                    "metadata": {"exit": 1},
                },
            },
        )

        # This assistant is in the same session but is not a child of any
        # eligible user turn and must not enter an episode.
        rogue_assistant = self.add_assistant("rogue", "unrelated_parent", 7000)
        self.add_part(
            "rogue_tool",
            rogue_assistant,
            7001,
            {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "output": "ROGUE_TOOL_SECRET",
                    "metadata": {"exit": 0},
                },
            },
        )
        scan = self.finalize_turns()
        self.assertEqual(scan["counts"], {"completed": 4, "failed": 1})

    def episode_storage_dump(self) -> str:
        with closing(sqlite3.connect(self.store)) as connection:
            connection.row_factory = sqlite3.Row
            payload = {
                "episodes": [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM trace_episode ORDER BY episode_id"
                    )
                ],
                "events": [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM trace_episode_event ORDER BY episode_event_id"
                    )
                ],
            }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def test_extracts_conservative_classes_and_only_allowlisted_evidence(self) -> None:
        self.seed_terminal_examples()

        report = run_episode_extraction(self.source, self.store, settle_ms=0)

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["extractor_version"], EXTRACTOR_VERSION)
        self.assertEqual(report["eligible_turns"], 5)
        self.assertEqual(report["episodes_created"], 5)
        self.assertEqual(report["unstable_turns"], 0)
        self.assertEqual(report["episodes_skipped"], 0)
        self.assertEqual(
            report["domain_verification_counts"],
            {"ambiguous": 1, "unverified": 3, "verified_success": 1},
        )
        self.assertTrue(report["episodes"][0]["project_key"].startswith("directory:"))
        self.assertTrue(report["episodes"][0]["primary_event_id"].startswith("sqlite:"))
        self.assertNotIn("evidence_class", json.dumps(report))

        with closing(sqlite3.connect(self.store)) as connection:
            connection.row_factory = sqlite3.Row
            bash = connection.execute(
                "SELECT * FROM trace_episode_event WHERE source_part_id = 'success_bash'"
            ).fetchone()
            self.assertEqual(json.loads(bash["metadata_json"]), {"exit": 0})
            self.assertEqual(bash["state_status"], "completed")
            self.assertEqual(bash["started_at_ms"], 2013)
            self.assertEqual(bash["ended_at_ms"], 2014)
            self.assertEqual(bash["evidence_verdict"], "unverified")
            for field in (
                "input_hash",
                "input_length",
                "output_hash",
                "output_length",
                "error_hash",
                "error_length",
            ):
                self.assertIsNone(bash[field])

            failed_keil = connection.execute(
                "SELECT * FROM trace_episode_event WHERE source_part_id = 'keil_failed'"
            ).fetchone()
            self.assertEqual(
                json.loads(failed_keil["metadata_json"]),
                {
                    "buildFailureKind": "other",
                    "errorCount": 2,
                    "failureKind": "source",
                    "success": False,
                    "warningCount": 1,
                },
            )
            self.assertEqual(failed_keil["evidence_verdict"], "unverified")
            json_signal = connection.execute(
                "SELECT * FROM trace_episode_event WHERE source_part_id = 'success_json'"
            ).fetchone()
            self.assertEqual(json_signal["evidence_verdict"], "unverified")
            edit = connection.execute(
                "SELECT * FROM trace_episode_event WHERE source_part_id = 'edit_completed'"
            ).fetchone()
            self.assertEqual(edit["evidence_verdict"], "unverified")
            self.assertEqual(json.loads(edit["metadata_json"]), {})
            self.assertIsNone(edit["input_hash"])
            self.assertIsNone(edit["output_hash"])
            self.assertEqual(
                connection.execute(
                    """
                    SELECT ep.domain_verification_outcome
                    FROM trace_episode AS ep
                    JOIN conversation_turn AS t ON t.turn_id = ep.turn_id
                    WHERE t.user_message_id = 'usr_unverified'
                    """
                ).fetchone()[0],
                "unverified",
            )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT ep.domain_verification_outcome
                    FROM trace_episode AS ep
                    JOIN conversation_turn AS t ON t.turn_id = ep.turn_id
                    WHERE t.user_message_id = 'usr_recovered'
                    """
                ).fetchone()[0],
                "ambiguous",
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM trace_episode_event WHERE source_part_id = 'rogue_tool'"
                ).fetchone()
            )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM trace_episode_event
                    WHERE input_hash IS NOT NULL OR input_length IS NOT NULL
                       OR output_hash IS NOT NULL OR output_length IS NOT NULL
                       OR error_hash IS NOT NULL OR error_length IS NOT NULL
                    """
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT GROUP_CONCAT(DISTINCT evidence_class)
                    FROM trace_episode WHERE extractor_version = ?
                    """,
                    (EXTRACTOR_VERSION,),
                ).fetchone()[0],
                "unverified",
            )

        stored = self.episode_storage_dump()
        for forbidden in (
            "ASSISTANT_TEXT_SECRET",
            "HIDDEN_REASONING_SECRET",
            "ASSISTANT_ERROR_SECRET",
            "SELF_CLAIM_SECRET",
            "INPUT_SECRET",
            "OUTPUT_SECRET",
            "METADATA_SECRET",
            "JSON_TOKEN_SECRET",
            "LONG_PRIVATE_BUILD_LOG",
            "D:\\secret",
            "D:\\private",
            "KEIL_METADATA_SECRET",
            "EDIT_OUTPUT_SECRET",
            "PRIVATE_COMMAND",
            "TOOL_ERROR_SECRET",
            "ROGUE_TOOL_SECRET",
        ):
            self.assertNotIn(forbidden, stored)

    def test_repeated_extraction_is_idempotent_and_trace_rows_are_immutable(self) -> None:
        self.seed_terminal_examples()
        source_before = self._source_shape()

        first = run_episode_extraction(self.source, self.store, settle_ms=0)
        second = run_episode_extraction(self.source, self.store, settle_ms=0)

        self.assertEqual(first["episodes_created"], 5)
        self.assertEqual(second["episodes_created"], 0)
        self.assertEqual(second["episodes_existing"], 5)
        self.assertEqual(
            second["domain_verification_counts"],
            first["domain_verification_counts"],
        )
        self.assertEqual(second["disposition_counts"], {"existing": 5})
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM trace_episode").fetchone()[0],
                5,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM episode_extraction_run"
                ).fetchone()[0],
                2,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(DISTINCT idempotency_key) FROM trace_episode"
                ).fetchone()[0],
                5,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM run_episode_result"
                ).fetchone()[0],
                10,
            )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                connection.execute(
                    "UPDATE trace_episode SET evidence_class = 'success'"
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                connection.execute("DELETE FROM trace_episode_event")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                connection.execute("DELETE FROM run_episode_result")
        self.assertEqual(self._source_shape(), source_before)

    def _source_shape(self) -> tuple[list[tuple], list[tuple]]:
        with closing(sqlite3.connect(self.source)) as connection:
            tables = connection.execute(
                """
                SELECT name, sql FROM sqlite_master
                WHERE type = 'table' ORDER BY name
                """
            ).fetchall()
            counts = [
                (table, connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("session", "message", "part")
            ]
        return tables, counts

    def _outcomes_by_user(self) -> dict[str, str]:
        with closing(sqlite3.connect(self.store)) as connection:
            return {
                row[0]: row[1]
                for row in connection.execute(
                    """
                    SELECT t.user_message_id, ep.domain_verification_outcome
                    FROM trace_episode AS ep
                    JOIN conversation_turn AS t ON t.turn_id = ep.turn_id
                    WHERE ep.extractor_version = ?
                    ORDER BY t.user_message_id
                    """,
                    (EXTRACTOR_VERSION,),
                )
            }

    def test_domain_verification_is_strict_and_never_reports_recovered(self) -> None:
        cases = {
            "pass": {
                "status": "completed",
                "time": {"start": 2011, "end": 2012},
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
            "fail": {
                "status": "completed",
                "time": {"start": 3011, "end": 3012},
                "metadata": {
                    "success": False,
                    "errorCount": 2,
                    "warningCount": 1,
                    "failureKind": "unknown-provider-path:C:\\secret",
                },
            },
            "contradictory": {
                "status": "completed",
                "time": {"start": 4011, "end": 4012},
                "metadata": {"success": True, "errorCount": 3, "warningCount": 0},
            },
            "missing_end": {
                "status": "completed",
                "time": {"start": 5011},
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
            "missing_start": {
                "status": "completed",
                "time": {"end": 6012},
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
            "reversed_time": {
                "status": "completed",
                "time": {"start": 7012, "end": 7011},
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
            "completed_with_error": {
                "status": "completed",
                "time": {"start": 8011, "end": 8012},
                "error": {"message": "COMPLETED_ERROR_SECRET"},
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
        }
        timestamp = 2000
        for name, state in cases.items():
            user = self.add_user(name, timestamp)
            assistant = self.add_assistant(name, user, timestamp + 10)
            self.add_part(
                "tool_" + name,
                assistant,
                timestamp + 11,
                {"type": "tool", "tool": "keil_build", "state": state},
            )
            timestamp += 1000

        mutation_user = self.add_user("mutation_after_pass", timestamp)
        mutation_assistant = self.add_assistant(
            "mutation_after_pass", mutation_user, timestamp + 10
        )
        self.add_part(
            "mutation_pass",
            mutation_assistant,
            timestamp + 11,
            {
                "type": "tool",
                "tool": "keil_build",
                "state": {
                    "status": "completed",
                    "time": {"start": timestamp + 11, "end": timestamp + 12},
                    "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
                },
            },
        )
        self.add_part(
            "mutation_write",
            mutation_assistant,
            timestamp + 13,
            {
                "type": "tool",
                "tool": "write",
                "state": {
                    "status": "completed",
                    "input": {"path": "C:\\SECRET_AFTER_VALIDATION"},
                },
            },
        )
        timestamp += 1000

        generic_user = self.add_user("generic_success_claim", timestamp)
        generic_assistant = self.add_assistant(
            "generic_success_claim", generic_user, timestamp + 10
        )
        self.add_part(
            "generic_bash",
            generic_assistant,
            timestamp + 11,
            {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "time": {"start": timestamp + 11, "end": timestamp + 12},
                    "input": "echo CLAIMED_SUCCESS_SECRET",
                    "metadata": {"exit": 0},
                },
            },
        )
        self.add_part(
            "generic_unknown",
            generic_assistant,
            timestamp + 12,
            {
                "type": "tool",
                "tool": "provider_private_secret_tool",
                "state": {
                    "status": "completed",
                    "metadata": {"success": True, "path": "C:\\secret"},
                },
            },
        )
        self.add_part(
            "generic_claim_text",
            generic_assistant,
            timestamp + 13,
            {"type": "text", "text": "工程验证成功 CLAIM_TEXT_SECRET"},
        )
        timestamp += 1000

        post_bash_user = self.add_user("post_validation_bash", timestamp)
        post_bash_assistant = self.add_assistant(
            "post_validation_bash", post_bash_user, timestamp + 10
        )
        self.add_part(
            "post_bash_keil",
            post_bash_assistant,
            timestamp + 11,
            {
                "type": "tool",
                "tool": "keil_build",
                "state": {
                    "status": "completed",
                    "time": {"start": timestamp + 11, "end": timestamp + 12},
                    "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
                },
            },
        )
        self.add_part(
            "post_bash_unknown_effect",
            post_bash_assistant,
            timestamp + 13,
            {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "time": {"start": timestamp + 13, "end": timestamp + 14},
                    "metadata": {"exit": 0},
                },
            },
        )
        self.finalize_turns()

        report = run_episode_extraction(self.source, self.store, settle_ms=0)

        self.assertEqual(
            report["domain_verification_counts"],
            {
                "ambiguous": 5,
                "unverified": 3,
                "verified_failure": 1,
                "verified_success": 1,
            },
        )
        self.assertNotIn("recovered", json.dumps(report))
        self.assertEqual(
            self._outcomes_by_user(),
            {
                "usr_contradictory": "ambiguous",
                "usr_completed_with_error": "ambiguous",
                "usr_fail": "verified_failure",
                "usr_generic_success_claim": "unverified",
                "usr_missing_end": "ambiguous",
                "usr_missing_start": "ambiguous",
                "usr_mutation_after_pass": "unverified",
                "usr_pass": "verified_success",
                "usr_post_validation_bash": "unverified",
                "usr_reversed_time": "ambiguous",
            },
        )
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT tool_name FROM trace_episode_event WHERE source_part_id = 'generic_unknown'"
                ).fetchone()[0],
                "other",
            )
            self.assertEqual(
                json.loads(
                    connection.execute(
                        "SELECT metadata_json FROM trace_episode_event WHERE source_part_id = 'tool_fail'"
                    ).fetchone()[0]
                )["failureKind"],
                "other",
            )
            completed_error = connection.execute(
                """
                SELECT has_error_payload, error_hash, error_length
                FROM trace_episode_event
                WHERE source_part_id = 'tool_completed_with_error'
                """
            ).fetchone()
            self.assertEqual(completed_error, (1, None, None))
        stored = self.episode_storage_dump()
        self.assertNotIn("CLAIMED_SUCCESS_SECRET", stored)
        self.assertNotIn("CLAIM_TEXT_SECRET", stored)
        self.assertNotIn("SECRET_AFTER_VALIDATION", stored)
        self.assertNotIn("COMPLETED_ERROR_SECRET", stored)

    def test_double_sample_ignores_raw_payload_but_prevents_second_v3_episode(self) -> None:
        user = self.add_user("stable", 2000)
        assistant = self.add_assistant("stable", user, 2010)
        original = {
            "type": "tool",
            "tool": "keil_build",
            "state": {
                "status": "completed",
                "time": {"start": 2011, "end": 2012},
                "input": {"path": "C:\\FIRST_SECRET"},
                "output": "FIRST_LOG_SECRET",
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
        }
        raw_only_change = {
            "type": "tool",
            "tool": "keil_build",
            "state": {
                "status": "completed",
                "time": {"start": 2011, "end": 2012},
                "input": {"path": "C:\\SECOND_SECRET"},
                "output": "SECOND_LOG_SECRET",
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
        }
        changed_domain = {
            "type": "tool",
            "tool": "keil_build",
            "state": {
                "status": "completed",
                "time": {"start": 2011, "end": 2012},
                "metadata": {"success": False, "errorCount": 2, "warningCount": 0},
            },
        }
        self.add_part("stable_keil", assistant, 2011, original)
        self.finalize_turns()

        with mock.patch(
            "memory_lab.episodes._wait_between_samples",
            side_effect=lambda _milliseconds: self.update_part(
                "stable_keil", raw_only_change
            ),
        ):
            first = run_episode_extraction(self.source, self.store, settle_ms=0)

        self.assertEqual(first["episodes_created"], 1)
        self.assertEqual(first["unstable_turns"], 0)
        self.assertEqual(
            first["domain_verification_counts"], {"verified_success": 1}
        )
        self.update_part("stable_keil", changed_domain)
        second = run_episode_extraction(self.source, self.store, settle_ms=0)
        self.assertEqual(second["episodes_created"], 0)
        self.assertEqual(second["episodes_skipped"], 1)
        self.assertEqual(second["skipped_turns"], 1)
        self.assertEqual(second["disposition_counts"], {"skipped": 1})
        self.assertEqual(second["domain_verification_counts"], {"unverified": 1})
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM trace_episode WHERE extractor_version = ?",
                    (EXTRACTOR_VERSION,),
                ).fetchone()[0],
                1,
            )
            result = connection.execute(
                """
                SELECT disposition, episode_id
                FROM run_episode_result
                WHERE run_id = ?
                """,
                (second["run_id"],),
            ).fetchone()
            self.assertEqual(result[0], "skipped")
            self.assertIsNotNone(result[1])
        stored = self.episode_storage_dump()
        for secret in (
            "FIRST_SECRET",
            "FIRST_LOG_SECRET",
            "SECOND_SECRET",
            "SECOND_LOG_SECRET",
        ):
            self.assertNotIn(secret, stored)

    def test_allowlisted_change_between_samples_is_unstable_and_not_persisted(self) -> None:
        user = self.add_user("unstable", 2000)
        assistant = self.add_assistant("unstable", user, 2010)
        initial = {
            "type": "tool",
            "tool": "keil_build",
            "state": {
                "status": "completed",
                "time": {"start": 2011, "end": 2012},
                "metadata": {"success": True, "errorCount": 0, "warningCount": 0},
            },
        }
        changed = {
            "type": "tool",
            "tool": "keil_build",
            "state": {
                "status": "completed",
                "time": {"start": 2011, "end": 2012},
                "metadata": {"success": False, "errorCount": 4, "warningCount": 0},
            },
        }
        self.add_part("unstable_keil", assistant, 2011, initial)
        self.finalize_turns()

        with mock.patch(
            "memory_lab.episodes._wait_between_samples",
            side_effect=lambda _milliseconds: self.update_part(
                "unstable_keil", changed
            ),
        ):
            report = run_episode_extraction(self.source, self.store, settle_ms=0)

        self.assertEqual(report["episodes_created"], 0)
        self.assertEqual(report["unstable_turns"], 1)
        self.assertEqual(report["skipped_turns"], 0)
        self.assertEqual(report["episodes_skipped"], 1)
        self.assertEqual(report["disposition_counts"], {"unstable": 1})
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM trace_episode WHERE extractor_version = ?",
                    (EXTRACTOR_VERSION,),
                ).fetchone()[0],
                0,
            )
            result = connection.execute(
                """
                SELECT disposition, episode_id,
                       sample_1_trace_signature = sample_2_trace_signature
                FROM run_episode_result
                """
            ).fetchone()
            self.assertEqual(result, ("unstable", None, 0))

    def test_only_completed_or_failed_turns_are_extracted(self) -> None:
        terminal_user = self.add_user("terminal", 2000)
        self.add_assistant("terminal", terminal_user, 2010)

        interrupted_user = self.add_user("interrupted", 3000)
        self.add_assistant(
            "interrupted", interrupted_user, 3010, finish="tool-calls"
        )
        pending_user = self.add_user("pending", 4000)
        pending_assistant = self.add_assistant(
            "pending", pending_user, 4010, finish="tool-calls"
        )
        self.add_part(
            "pending_tool",
            pending_assistant,
            4011,
            {
                "type": "tool",
                "tool": "bash",
                "state": {"status": "running", "input": "PENDING_SECRET"},
            },
        )
        scan = self.finalize_turns()
        self.assertEqual(
            scan["counts"],
            {"completed": 1, "interrupted": 1, "pending": 1},
        )

        report = run_episode_extraction(self.source, self.store, settle_ms=0)

        self.assertEqual(report["eligible_turns"], 1)
        self.assertEqual(report["episodes_created"], 1)
        self.assertEqual(report["episodes"][0]["turn_state"], "completed")
        self.assertNotIn("PENDING_SECRET", self.episode_storage_dump())

    def test_turn_load_failure_closes_the_registered_run(self) -> None:
        terminal_user = self.add_user("terminal", 2000)
        self.add_assistant("terminal", terminal_user, 2010)
        self.finalize_turns()

        with mock.patch(
            "memory_lab.episodes._load_terminal_turns",
            side_effect=RuntimeError("sensitive failure detail"),
        ), self.assertRaises(RuntimeError):
            run_episode_extraction(self.source, self.store, settle_ms=0)

        with closing(sqlite3.connect(self.store)) as connection:
            run = connection.execute(
                """
                SELECT status, error FROM episode_extraction_run
                ORDER BY started_at DESC LIMIT 1
                """
            ).fetchone()
        self.assertEqual(run, ("failed", "RuntimeError"))

    def test_legacy_v1_episode_is_preserved_and_not_used_as_v3_outcome(self) -> None:
        terminal_user = self.add_user("terminal", 2000)
        self.add_assistant("terminal", terminal_user, 2010)
        self.finalize_turns()
        with closing(sqlite3.connect(self.store)) as connection, connection:
            turn = connection.execute(
                """
                SELECT turn_id, source_id, primary_event_id,
                       last_structural_signature
                FROM conversation_turn
                """
            ).fetchone()
            connection.execute(
                """
                INSERT INTO episode_extraction_run(
                    run_id, source_id, extractor_version, started_at,
                    completed_at, status
                ) VALUES ('legacy_run', ?, 'cp-trace-episode-v1',
                          '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z',
                          'completed')
                """,
                (turn[1],),
            )
            connection.execute(
                """
                INSERT INTO trace_episode(
                    episode_id, idempotency_key, extraction_run_id,
                    extractor_version, source_id, turn_id, primary_event_id,
                    project_key, source_signature, trace_signature, turn_state,
                    evidence_class, semantic_status, assistant_count,
                    tool_event_count, positive_event_count, negative_event_count,
                    unverified_event_count, recovered_tool_count, created_at
                ) VALUES (
                    'legacy_episode', 'legacy_key', 'legacy_run',
                    'cp-trace-episode-v1', ?, ?, ?, 'legacy-project', ?, ?,
                    'completed', 'success', 'not_evaluated', 1, 0, 1, 0, 0, 0,
                    '2026-01-01T00:00:01Z'
                )
                """,
                (turn[1], turn[0], turn[2], turn[3], "a" * 64),
            )
            connection.execute(
                """
                INSERT INTO trace_episode_event(
                    episode_event_id, episode_id, ordinal,
                    source_assistant_message_id, source_part_id, tool_name,
                    state_status, started_at_ms, ended_at_ms, metadata_json,
                    input_hash, input_length, output_hash, output_length,
                    error_hash, error_length, evidence_verdict,
                    event_signature, created_at
                ) VALUES (
                    'legacy_event', 'legacy_episode', 0, 'legacy_assistant',
                    'legacy_part', 'bash', 'completed', 1, 2, '{}',
                    ?, 11, ?, 22, ?, 33, 'success', ?,
                    '2026-01-01T00:00:01Z'
                )
                """,
                ("b" * 64, "c" * 64, "d" * 64, "e" * 64),
            )
            connection.execute(
                """
                INSERT INTO episode_extraction_run(
                    run_id, source_id, extractor_version, started_at,
                    completed_at, status
                ) VALUES ('v2_run', ?, 'cp-trace-episode-v2',
                          '2026-01-02T00:00:00Z', '2026-01-02T00:00:01Z',
                          'completed')
                """,
                (turn[1],),
            )
            connection.execute(
                """
                INSERT INTO trace_episode(
                    episode_id, idempotency_key, extraction_run_id,
                    extractor_version, source_id, turn_id, primary_event_id,
                    project_key, source_signature, trace_signature, turn_state,
                    evidence_class, domain_verification_outcome, semantic_status,
                    assistant_count, tool_event_count, positive_event_count,
                    negative_event_count, unverified_event_count,
                    recovered_tool_count, created_at
                ) VALUES (
                    'v2_episode', 'v2_key', 'v2_run', 'cp-trace-episode-v2',
                    ?, ?, ?, 'legacy-project', ?, ?, 'completed', 'unverified',
                    'verified_success', 'not_evaluated', 1, 0, 0, 0, 0, 0,
                    '2026-01-02T00:00:01Z'
                )
                """,
                (turn[1], turn[0], turn[2], turn[3], "f" * 64),
            )

        report = run_episode_extraction(self.source, self.store, settle_ms=0)

        self.assertEqual(report["episodes_created"], 1)
        self.assertEqual(report["domain_verification_counts"], {"unverified": 1})
        self.assertIn("v1/v2", report["compatibility_boundary"])
        with closing(sqlite3.connect(self.store)) as connection:
            rows = connection.execute(
                """
                SELECT extractor_version, domain_verification_outcome
                FROM trace_episode ORDER BY extractor_version
                """
            ).fetchall()
            legacy_event = connection.execute(
                """
                SELECT episode_event_id, input_hash, input_length,
                       output_hash, output_length, error_hash, error_length
                FROM trace_episode_event
                WHERE episode_event_id = 'legacy_event'
                """
            ).fetchone()
            immutable_trigger = connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'trigger'
                  AND name = 'trace_episode_event_no_update'
                """
            ).fetchone()[0]
        self.assertEqual(
            rows,
            [
                ("cp-trace-episode-v1", "unverified"),
                ("cp-trace-episode-v2", "verified_success"),
                (EXTRACTOR_VERSION, "unverified"),
            ],
        )
        self.assertEqual(
            legacy_event,
            ("legacy_event", None, None, None, None, None, None),
        )
        self.assertEqual(immutable_trigger, 1)

    def test_cli_redacts_episode_exception_message(self) -> None:
        output = io.StringIO()
        with mock.patch(
            "memory_lab.cli.run_episode_extraction",
            side_effect=RuntimeError("SOURCE_PATH_AND_SECRET_MUST_NOT_PRINT"),
        ), redirect_stdout(output):
            exit_code = main(
                [
                    "--source-db",
                    str(self.source),
                    "--lab-db",
                    str(self.store),
                    "episodes",
                    "--settle-ms",
                    "0",
                ]
            )
        self.assertEqual(exit_code, 1)
        self.assertIn("RuntimeError", output.getvalue())
        self.assertNotIn("SOURCE_PATH_AND_SECRET_MUST_NOT_PRINT", output.getvalue())

    def test_cli_exposes_episodes_command(self) -> None:
        args = build_parser().parse_args(["episodes", "--settle-ms", "0"])
        self.assertEqual(args.command, "episodes")
        self.assertEqual(args.settle_ms, 0)


if __name__ == "__main__":
    unittest.main()
