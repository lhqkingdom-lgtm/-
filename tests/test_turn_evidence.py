from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.ingest import run_ingestion
from memory_lab.prepare import prepare_dataset
from memory_lab.turn_evidence import (
    EvidenceLimits,
    TurnEvidenceError,
    build_turn_evidence,
    fit_turn_evidence_to_model_limits,
)
from memory_lab.turns import run_turn_scan
from memory_lab.unified_model import (
    MAX_OBSERVATIONS,
    MAX_TOTAL_OBSERVATION_CHARS,
    build_unified_model_request,
)


class TurnEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.source = root / "source.db"
        self.lab = root / "memory-lab.db"
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
                    ('ses_root', 'project-a', NULL, 'D:\\Private Root', 'root', '1', 100),
                    ('ses_work', 'project-a', NULL, 'D:\\Private Root\\Firmware App', 'work', '1', 1000);
                """
            )
        run_ingestion(self.source, self.lab, "baseline")
        with closing(sqlite3.connect(self.lab)) as connection, connection:
            connection.execute("UPDATE source SET baseline_cutoff_ms = 0")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_message(self, message_id: str, timestamp: int, data: dict) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO message VALUES (?, 'ses_work', ?, ?)",
                (message_id, timestamp, json.dumps(data, ensure_ascii=False)),
            )

    def add_part(
        self, part_id: str, message_id: str, timestamp: int, data: dict
    ) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO part VALUES (?, ?, 'ses_work', ?, ?)",
                (part_id, message_id, timestamp, json.dumps(data, ensure_ascii=False)),
            )

    def add_user(self, message_id: str, timestamp: int, text: str) -> None:
        self.add_message(
            message_id, timestamp, {"role": "user", "time": {"created": timestamp}}
        )
        self.add_part(
            "part_" + message_id,
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
        finish: str = "stop",
        failed: bool = False,
    ) -> None:
        data = {
            "role": "assistant",
            "parentID": parent_id,
            "finish": finish,
            "time": {"created": timestamp},
        }
        if failed:
            data["error"] = {
                "name": "ProviderError",
                "message": "password=assistant-failure-password D:\\Private Root\\error.log",
            }
        else:
            data["time"]["completed"] = timestamp + 10
        self.add_message(message_id, timestamp, data)

    def finalize(self) -> str:
        run_ingestion(self.source, self.lab, "ingest")
        prepare_dataset(self.lab)
        report = run_turn_scan(self.source, self.lab, settle_ms=0)
        self.assertEqual(report["turn_count"], 1)
        return report["turns"][0]["turn_id"]

    def test_complete_observable_turn_is_ordered_redacted_and_idempotent(self) -> None:
        self.add_user(
            "usr_1",
            2000,
            "请修复 D:\\Private Root\\Firmware App\\src\\main.c，api_key=top-secret-key，"
            "API key is natural-language-key，访问令牌为 chinese-token-value，密码是 chinese-password",
        )
        self.add_assistant("ast_tools", "usr_1", 2010, finish="tool-calls")
        self.add_part(
            "part_thinking_1",
            "ast_tools",
            2011,
            {"type": "thinking", "text": "HIDDEN_THINKING password=never-visible"},
        )
        self.add_part(
            "part_unknown_tool",
            "ast_tools",
            2012,
            {
                "type": "tool",
                "tool": "vendor_magic_tool",
                "state": {
                    "status": "provider-finished",
                    "input": {
                        "project": "D:\\Private Root\\Firmware App\\device.uvprojx",
                        "password": "input-password",
                        "number": float("nan"),
                    },
                    "output": "built /home/alice/private/output.hex token=output-token "
                    + "x" * 200,
                    "error": {"credential": "error-credential", "code": 7},
                },
            },
        )
        self.add_assistant("ast_final", "usr_1", 2020)
        self.add_part(
            "part_reasoning_final",
            "ast_final",
            2021,
            {"type": "reasoning", "text": "HIDDEN_REASONING_SECRET"},
        )
        self.add_part(
            "part_final_text",
            "ast_final",
            2022,
            {
                "type": "text",
                "text": "已完成 D:\\Private Root\\Firmware App\\src\\main.c，secret=final-secret",
            },
        )
        self.add_part(
            "part_second_tool",
            "ast_final",
            2023,
            {
                "type": "tool",
                "tool": "never_registered_before",
                "state": {"status": "completed", "input": {}, "output": {"ok": True}},
            },
        )
        turn_id = self.finalize()
        # A CP fork is represented by session.parent_id.  Changing only this
        # session-level relation does not alter the already-stabilized turn.
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "UPDATE session SET parent_id = 'ses_root' WHERE id = 'ses_work'"
            )

        limits = EvidenceLimits(tool_payload_chars=80)
        before = self.source.read_bytes()
        first = build_turn_evidence(self.source, self.lab, turn_id, limits=limits)
        second = build_turn_evidence(self.source, self.lab, turn_id, limits=limits)
        after = self.source.read_bytes()

        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(first["bundle_id"], "turn_evidence_" + first["bundle_hash"])
        self.assertEqual(first["turn"]["state"], "completed")
        self.assertEqual(first["turn"]["occurred_at_ms"], 2000)
        self.assertEqual(first["session"]["root_session_id"], "ses_root")
        self.assertEqual(first["session"]["branch_depth"], 1)
        self.assertTrue(first["session"]["lineage_complete"])
        self.assertEqual(
            [item["session_id"] for item in first["session"]["lineage"]],
            ["ses_root", "ses_work"],
        )
        self.assertTrue(first["workspace"]["path_alias"].startswith("<WORKSPACE:"))
        self.assertFalse(first["workspace"]["absolute_path_included"])

        self.assertEqual(first["assistant"]["message_id"], "ast_final")
        self.assertEqual(
            first["assistant"]["text_parts"][0]["source_part_id"],
            "part_final_text",
        )
        self.assertEqual(len(first["tools"]), 2)
        self.assertEqual(
            [item["name"] for item in first["tools"]],
            ["vendor_magic_tool", "never_registered_before"],
        )
        self.assertEqual(first["tools"][0]["status"], "provider-finished")
        self.assertEqual(first["tools"][0]["source_message_id"], "ast_tools")
        self.assertEqual(first["tools"][0]["source_part_id"], "part_unknown_tool")
        self.assertEqual(first["tools"][0]["input"]["present"], True)
        self.assertEqual(first["tools"][0]["output"]["truncated"], True)
        self.assertEqual(first["tools"][0]["error"]["present"], True)
        self.assertEqual(first["safety"]["excluded_hidden_parts"], {"reasoning": 1, "thinking": 1})

        rendered = json.dumps(first, ensure_ascii=False, allow_nan=False)
        for forbidden in (
            "D:\\Private Root",
            "/home/alice",
            "top-secret-key",
            "natural-language-key",
            "chinese-token-value",
            "chinese-password",
            "input-password",
            "output-token",
            "error-credential",
            "final-secret",
            "HIDDEN_THINKING",
            "HIDDEN_REASONING",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("[REDACTED]", rendered)
        self.assertIn("<WORKSPACE>", rendered)

    def test_failed_turn_keeps_visible_error_but_redacts_it(self) -> None:
        self.add_user("usr_failed", 3000, "执行并说明失败原因")
        self.add_assistant("ast_failed", "usr_failed", 3010, failed=True)
        turn_id = self.finalize()

        bundle = build_turn_evidence(self.source, self.lab, turn_id)

        self.assertEqual(bundle["turn"]["state"], "failed")
        self.assertTrue(bundle["assistant"]["error"]["present"])
        rendered = json.dumps(bundle, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("assistant-failure-password", rendered)
        self.assertNotIn("D:\\Private Root", rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_pending_turn_and_wrong_source_are_rejected(self) -> None:
        self.add_user("usr_pending", 4000, "尚未完成")
        self.add_assistant("ast_pending", "usr_pending", 4010, finish="tool-calls")
        run_ingestion(self.source, self.lab, "ingest")
        prepare_dataset(self.lab)
        report = run_turn_scan(self.source, self.lab, settle_ms=0)
        turn_id = report["turns"][0]["turn_id"]

        with self.assertRaisesRegex(TurnEvidenceError, "not a stable"):
            build_turn_evidence(self.source, self.lab, turn_id)

        # Make it terminal, rescan, and prove a byte-identical DB at another
        # path is still a different configured CP source identity.
        self.add_assistant("ast_final", "usr_pending", 4020)
        stable = run_turn_scan(self.source, self.lab, settle_ms=0)
        self.assertEqual(stable["turns"][0]["state"], "completed")
        copied_source = self.source.with_name("different-source.db")
        shutil.copyfile(self.source, copied_source)
        with self.assertRaisesRegex(TurnEvidenceError, "different CodePulse"):
            build_turn_evidence(copied_source, self.lab, turn_id)

    def test_missing_turn_and_invalid_limits_fail_closed(self) -> None:
        with self.assertRaisesRegex(TurnEvidenceError, "does not exist"):
            build_turn_evidence(self.source, self.lab, "missing")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            EvidenceLimits(tool_payload_chars=0)
        with self.assertRaisesRegex(ValueError, "turn_id"):
            build_turn_evidence(self.source, self.lab, "")

    def test_pathological_single_turn_is_bounded_without_losing_tool_audit_metadata(self) -> None:
        self.add_user("usr_huge", 5000, "分析整个长任务并沉淀可复用经验")
        self.add_assistant("ast_huge", "usr_huge", 5010)
        with closing(sqlite3.connect(self.source)) as connection, connection:
            for index in range(650):
                connection.execute(
                    "INSERT INTO part VALUES (?, 'ast_huge', 'ses_work', ?, ?)",
                    (
                        f"huge-tool-{index:04d}",
                        5020 + index,
                        json.dumps(
                            {
                                "type": "tool",
                                "tool": f"tool-{index:04d}",
                                "state": {
                                    "status": "completed",
                                    "output": "result-" + str(index) + "-" + "x" * 1000,
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
        self.add_part(
            "huge-final",
            "ast_huge",
            6000,
            {"type": "text", "text": "长任务已完成并通过最终验证。"},
        )
        turn_id = self.finalize()

        bundle = build_turn_evidence(self.source, self.lab, turn_id)
        request, catalog, _ = build_unified_model_request(bundle)

        manifest = bundle["tool_event_manifest"]
        self.assertEqual(manifest["total_count"], 650)
        self.assertEqual(len(manifest["events"]), 650)
        self.assertLess(manifest["model_observed_count"], 650)
        self.assertTrue(
            all(
                event["source_part_id"]
                and event["name"]
                and event["status"] == "completed"
                and event["sanitized_event_hash"]
                and event["output"]["sanitized_hash"]
                for event in manifest["events"]
            )
        )
        self.assertLessEqual(len(catalog), MAX_OBSERVATIONS)
        self.assertLessEqual(
            sum(len(item["text"]) for item in catalog.values()),
            MAX_TOTAL_OBSERVATION_CHARS,
        )
        self.assertEqual(len(request["turns"]), 1)
        self.assertIsNotNone(bundle["safety"]["model_budget"]["content_char_cap"])
        self.assertEqual(fit_turn_evidence_to_model_limits(bundle), bundle)


if __name__ == "__main__":
    unittest.main()
