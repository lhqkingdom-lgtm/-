from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from memory_lab.ingest import run_ingestion
from memory_lab.instructions import (
    clause_fingerprint,
    discover_global_instructions,
    discover_project_instructions,
    parse_clauses,
    run_instruction_coverage,
)
from memory_lab.prepare import prepare_dataset


class IngestionTests(unittest.TestCase):
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
                    ('ses_1', 'global', NULL, 'D:\\project-a', 'test', '1.0', 1000);
                """
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_part(
        self,
        part_id: str,
        text: str,
        *,
        role: str = "user",
        part_type: str = "text",
        session_id: str = "ses_1",
        timestamp: int | None = None,
    ) -> None:
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        message_id = "msg_" + part_id
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?)",
                (message_id, session_id, timestamp - 1, json.dumps({"role": role, "agent": "build"})),
            )
            connection.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                (
                    part_id,
                    message_id,
                    session_id,
                    timestamp,
                    json.dumps({"type": part_type, "text": text}, ensure_ascii=False),
                ),
            )

    def test_baseline_then_incremental_ingest_is_idempotent(self) -> None:
        self.add_part("prt_old", "历史消息")
        baseline = run_ingestion(self.source, self.store, "baseline")
        self.assertEqual(baseline["new_parts"], 1)
        self.assertEqual(baseline["user_events"], 0)

        self.add_part("prt_new", "以后默认使用中文回答")
        first = run_ingestion(self.source, self.store, "ingest")
        second = run_ingestion(self.source, self.store, "ingest")

        self.assertEqual(first["user_events"], 1)
        self.assertEqual(first["events"][0]["part_id"], "prt_new")
        self.assertEqual(second["user_events"], 0)
        self.assertEqual(second["duplicate_parts"], 2)
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event").fetchone()[0], 1)

    def test_explicit_historical_backfill_imports_only_named_session_and_is_idempotent(self) -> None:
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("ses_other", "global", None, r"D:\\project-b", "other", "1.0", 1001),
            )
        self.add_part("prt_selected", "用户确认修复后可以继续", session_id="ses_1")
        self.add_part("prt_other", "不应导入的会话", session_id="ses_other")
        run_ingestion(self.source, self.store, "baseline")

        first = run_ingestion(
            self.source,
            self.store,
            "historical_backfill",
            session_ids={"ses_1"},
        )
        second = run_ingestion(
            self.source,
            self.store,
            "historical_backfill",
            session_ids={"ses_1"},
        )

        self.assertEqual(first["user_events"], 1)
        self.assertEqual(first["events"][0]["part_id"], "prt_selected")
        self.assertEqual(second["user_events"], 0)
        self.assertEqual(second["duplicate_parts"], 1)
        with closing(sqlite3.connect(self.store)) as connection:
            rows = connection.execute(
                "SELECT part_id, eligibility_status FROM event ORDER BY part_id"
            ).fetchall()
        self.assertEqual(rows, [("prt_selected", "eligible")])

    def test_source_database_can_never_be_used_as_lab_database(self) -> None:
        with closing(sqlite3.connect(self.source)) as connection:
            before = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        with self.assertRaisesRegex(ValueError, "must not be the CodePulse source"):
            run_ingestion(self.source, self.source, "baseline")
        with closing(sqlite3.connect(self.source)) as connection:
            after = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertEqual(after, before)

    def test_late_prebaseline_row_inserted_after_baseline_is_isolated(self) -> None:
        baseline = run_ingestion(self.source, self.store, "baseline")
        cutoff_ms = baseline["baseline_cutoff_ms"]
        self.assertIsNotNone(cutoff_ms)

        repeated_baseline = run_ingestion(self.source, self.store, "baseline")
        self.assertEqual(repeated_baseline["baseline_cutoff_ms"], cutoff_ms)

        self.add_part("prt_late", "延迟写入的历史用户消息", timestamp=cutoff_ms - 1)
        self.add_part("prt_current", "基线后的当前用户消息", timestamp=cutoff_ms + 1)
        first = run_ingestion(self.source, self.store, "ingest")
        second = run_ingestion(self.source, self.store, "ingest")

        self.assertEqual(first["user_events"], 1)
        self.assertEqual(first["isolated_user_events"], 1)
        self.assertEqual(first["filtered_parts"], 1)
        self.assertEqual(first["reasons"]["late_arrival_pre_baseline"], 1)
        self.assertEqual([item["part_id"] for item in first["events"]], ["prt_current"])
        self.assertEqual(
            [item["part_id"] for item in first["isolated_events"]], ["prt_late"]
        )
        self.assertEqual(second["user_events"], 0)
        self.assertEqual(second["isolated_user_events"], 0)
        self.assertEqual(second["duplicate_parts"], 2)

        prepared = prepare_dataset(self.store)
        self.assertEqual(prepared["events_total"], 2)
        self.assertEqual(prepared["events_seen"], 1)
        self.assertEqual(prepared["isolated_events"], 1)
        self.assertEqual(
            prepared["eligibility_counts"],
            {"eligible": 1, "late_arrival_pre_baseline": 1},
        )
        self.assertEqual(
            [item["semantic_content"] for item in prepared["cases"]],
            ["基线后的当前用户消息"],
        )
        with closing(sqlite3.connect(self.store)) as connection:
            late = connection.execute(
                """
                SELECT e.eligibility_status, p.disposition, p.reason
                FROM event AS e
                JOIN processed_part AS p
                  ON p.source_id = e.source_id AND p.part_id = e.part_id
                WHERE e.part_id = 'prt_late'
                """
            ).fetchone()
            self.assertEqual(
                late,
                ("late_arrival_pre_baseline", "isolated", "late_arrival_pre_baseline"),
            )

    def test_child_or_fork_session_user_part_is_isolated_from_u1(self) -> None:
        baseline = run_ingestion(self.source, self.store, "baseline")
        cutoff_ms = baseline["baseline_cutoff_ms"]
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "ses_child",
                    "global",
                    "ses_1",
                    r"D:\project-a",
                    "child agent",
                    "1.0",
                    cutoff_ms + 1,
                ),
            )
        self.add_part(
            "prt_child_user",
            "子 Agent 内部生成的用户形态消息",
            session_id="ses_child",
            timestamp=cutoff_ms + 2,
        )

        report = run_ingestion(self.source, self.store, "ingest")

        self.assertEqual(report["user_events"], 0)
        self.assertEqual(report["isolated_user_events"], 1)
        self.assertEqual(
            report["reasons"], {"child_or_fork_session_user_message": 1}
        )
        with closing(sqlite3.connect(self.store)) as connection:
            status = connection.execute(
                "SELECT eligibility_status FROM event WHERE part_id = 'prt_child_user'"
            ).fetchone()[0]
            disposition, reason = connection.execute(
                "SELECT disposition, reason FROM processed_part WHERE part_id = 'prt_child_user'"
            ).fetchone()
        self.assertEqual(status, "child_or_fork_session_user_message")
        self.assertEqual(disposition, "isolated")
        self.assertEqual(reason, "child_or_fork_session_user_message")

    def test_existing_event_is_reclassified_without_deleting_audit(self) -> None:
        baseline = run_ingestion(self.source, self.store, "baseline")
        old_timestamp = baseline["baseline_cutoff_ms"] - 1
        with closing(sqlite3.connect(self.store)) as connection, connection:
            connection.execute(
                """
                INSERT INTO event(
                    event_id, source_id, session_id, message_id, part_id,
                    event_type, directory, timestamp, content, content_hash,
                    source_kind, eligibility_status, ingested_at, raw_json
                ) VALUES ('legacy-event', ?, 'ses_1', 'legacy-message',
                          'prt_prebaseline', 'user_text', 'D:\\project-a', ?,
                          '错误进入事件表的旧消息', 'legacy-hash', 'sqlite',
                          'eligible', '2026-01-01T00:00:00.000+00:00', '{}')
                """,
                (baseline["source_id"], old_timestamp),
            )
        self.assertEqual(prepare_dataset(self.store)["annotations_created"], 1)

        repeated = run_ingestion(self.source, self.store, "baseline")
        self.assertEqual(repeated["reclassified_late_events"], 1)
        prepared = prepare_dataset(self.store)

        self.assertEqual(prepared["events_total"], 1)
        self.assertEqual(prepared["events_seen"], 0)
        self.assertEqual(prepared["isolated_events"], 1)
        self.assertEqual(prepared["isolated_annotations_removed"], 1)
        with closing(sqlite3.connect(self.store)) as connection:
            event_count, status = connection.execute(
                "SELECT COUNT(*), eligibility_status FROM event"
            ).fetchone()
            annotation_count = connection.execute(
                "SELECT COUNT(*) FROM event_annotation"
            ).fetchone()[0]
            self.assertEqual(event_count, 1)
            self.assertEqual(status, "late_arrival_pre_baseline")
            self.assertEqual(annotation_count, 0)

    def test_filters_assistant_and_system_injected_text(self) -> None:
        run_ingestion(self.source, self.store, "baseline")
        self.add_part("prt_assistant", "not a user", role="assistant")
        self.add_part("prt_system", "<system-reminder>hidden</system-reminder>")
        self.add_part("prt_user", "真实用户文本")

        report = run_ingestion(self.source, self.store, "ingest")

        self.assertEqual(report["user_events"], 1)
        self.assertEqual(report["filtered_parts"], 2)
        self.assertEqual(report["reasons"]["not_user_message"], 1)
        self.assertEqual(report["reasons"]["system_injected_text"], 1)

    def test_prepare_groups_sessions_by_directory_and_strips_case_marker(self) -> None:
        run_ingestion(self.source, self.store, "baseline")
        self.add_part("prt_case", "【ML-S1-A1】以后默认使用中文回复。")
        run_ingestion(self.source, self.store, "ingest")

        report = prepare_dataset(self.store)
        repeated = prepare_dataset(self.store)

        self.assertEqual(report["events_seen"], 1)
        self.assertEqual(report["session_count"], 1)
        self.assertEqual(report["project_count"], 1)
        self.assertEqual(report["cases"][0]["case_id"], "ML-S1-A1")
        self.assertEqual(report["cases"][0]["semantic_content"], "以后默认使用中文回复。")
        self.assertEqual(repeated["annotations_created"], 0)
        self.assertEqual(repeated["annotations_existing"], 1)

    def test_instruction_coverage_is_project_scoped_and_scope_aware(self) -> None:
        root = Path(self.temporary_directory.name)
        project_a = root / "project-a"
        project_b = root / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        (project_b / "AGENTS.md").write_text(
            """# 项目规则

## 单片机项目

- 修改代码后必须进行编译检查（Keil/Makefile/IAR/CMake 等项目均适用）
- TG032Plus 项目编译统一使用 Keil C51
""",
            encoding="utf-8",
        )
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "UPDATE session SET directory = ? WHERE id = 'ses_1'",
                (str(project_a),),
            )
            connection.execute(
                "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("ses_2", "global", None, str(project_b), "test-b", "1.0", 1001),
            )

        run_ingestion(self.source, self.store, "baseline")
        self.add_part("prt_a", "【ML-S1-A1】以后默认使用中文回复。")
        self.add_part(
            "prt_b1",
            "【ML-S1-P2-01】以后在所有单片机项目中，修改代码后都必须进行编译检查。",
            session_id="ses_2",
        )
        self.add_part(
            "prt_b2",
            "【ML-S1-P2-02】仅在当前 TG032Plus 项目中，编译统一使用 Keil C51。",
            session_id="ses_2",
        )
        run_ingestion(self.source, self.store, "ingest")
        prepare_dataset(self.store)

        report = run_instruction_coverage(self.store, root / "empty-home")
        repeated = run_instruction_coverage(self.store, root / "empty-home")
        by_case = {item["case_id"]: item for item in report["event_hints"]}

        self.assertEqual(report["scan_count"], 2)
        self.assertEqual(sum(len(scan["files"]) for scan in report["scans"]), 1)
        self.assertEqual(by_case["ML-S1-A1"]["covered_now"], "none")
        self.assertEqual(by_case["ML-S1-P2-01"]["covered_now"], "likely_partial")
        self.assertEqual(by_case["ML-S1-P2-02"]["covered_now"], "likely_full")

        time.sleep(0.005)
        self.add_part(
            "prt_b3",
            "【ML-S1-P2-03】仅在当前 TG032Plus 项目中，编译统一使用 Keil C51。",
            session_id="ses_2",
            timestamp=int(time.time() * 1000),
        )
        run_ingestion(self.source, self.store, "ingest")
        prepare_dataset(self.store)
        self.assertEqual(repeated["scan_count"], 2)
        after_new_event = run_instruction_coverage(self.store, root / "empty-home")
        new_hint = next(
            item for item in after_new_event["event_hints"] if item["case_id"] == "ML-S1-P2-03"
        )
        self.assertEqual(new_hint["covered_now"], "likely_full")
        self.assertEqual(new_hint["temporal_evidence"], "seen_in_pre_event_snapshot")
        with closing(sqlite3.connect(self.store)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM instruction_document_revision").fetchone()[0],
                1,
            )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM instruction_clause").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event_instruction_hint").fetchone()[0], 4)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM event_instruction_assessment").fetchone()[0],
                10,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM instruction_coverage_run").fetchone()[0],
                3,
            )

    def test_project_discovery_uses_first_recursive_file_family(self) -> None:
        root = Path(self.temporary_directory.name) / "instructions"
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir(parents=True)
        (root / "a" / "AGENTS.md").write_text("- rule a", encoding="utf-8")
        (root / "b" / "AGENTS.md").write_text("- rule b", encoding="utf-8")
        (root / "b" / "CLAUDE.md").write_text("- ignored family", encoding="utf-8")

        files = discover_project_instructions(root, "project:test")

        self.assertEqual({item.path.name for item in files}, {"AGENTS.md"})
        self.assertEqual(len(files), 2)
        self.assertTrue(all(item.cp_would_load for item in files))
        self.assertTrue(all(item.scope_root != root for item in files))

    def test_global_discovery_prefers_configured_directory(self) -> None:
        root = Path(self.temporary_directory.name)
        profile = root / "profile"
        default = root / "home" / ".config" / "codepulse"
        profile.mkdir()
        default.mkdir(parents=True)
        (profile / "AGENTS.md").write_text("- profile", encoding="utf-8")
        (default / "AGENTS.md").write_text("- default", encoding="utf-8")

        with mock.patch.dict(os.environ, {"OPENCODE_CONFIG_DIR": str(profile)}):
            files = discover_global_instructions(root / "home")

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].path, (profile / "AGENTS.md").resolve())

    def test_markdown_clause_parser_ignores_code_fences(self) -> None:
        clauses = parse_clauses(
            """# Rules
- real rule
```text
- example only
```
2. another rule
"""
        )

        self.assertEqual([item.exact_text for item in clauses], ["real rule", "another rule"])

    def test_clause_fingerprint_includes_heading_semantics(self) -> None:
        clauses = parse_clauses(
            """# 必须执行
- 使用 Keil
# 禁止事项
- 使用 Keil
"""
        )

        self.assertEqual(len(clauses), 2)
        self.assertNotEqual(clause_fingerprint(clauses[0]), clause_fingerprint(clauses[1]))


if __name__ == "__main__":
    unittest.main()
