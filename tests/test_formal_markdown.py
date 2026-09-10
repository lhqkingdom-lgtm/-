from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from memory_lab.formal_markdown import (
    FormalMarkdownError,
    load_formal_topic_bodies,
    publish_formal_memory,
    recall_formal_memory,
    recall_selected_formal_topics,
    scan_formal_memory_index,
    snapshot_formal_database,
    validate_formal_memory,
)
from memory_lab.topics import ensure_topic_assignments


TIME = "2026-09-10T09:00:00.000+00:00"


class FormalMarkdownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.database = self.base / "formal.db"
        self.memory_root = self.base / "memory"
        self._create_database()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_database(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE memory (
                    memory_id TEXT PRIMARY KEY,
                    stable_key TEXT NOT NULL,
                    scope_fingerprint TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    family TEXT,
                    memory_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_revision_id TEXT,
                    state_version INTEGER NOT NULL DEFAULT 0,
                    verification_status TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE memory_revision (
                    revision_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    revision_number INTEGER NOT NULL,
                    semantic_value TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    activation_triggers_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    supersedes_revision_id TEXT,
                    source_decision_id TEXT NOT NULL,
                    verification_status TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE evidence (
                    evidence_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    exact_quote TEXT NOT NULL
                );
                CREATE TABLE conflict (
                    conflict_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    current_revision_id TEXT NOT NULL,
                    incoming_semantic_value TEXT NOT NULL,
                    incoming_statement TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolution_revision_id TEXT,
                    resolution_reason TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                """
            )
            self._memory(
                connection,
                "response",
                "preference.response.structure",
                {"level": "global"},
                "response_preference",
                "conclusion_then_details",
                "回复时先给结论，再说明步骤或细节。",
                ["回复", "结论优先", "处理步骤"],
            )
            self._memory(
                connection,
                "confirm-global",
                "workflow.code_generation.confirmation",
                {"level": "global"},
                "workflow_rule",
                "required",
                "生成正式代码前必须先征得用户确认。",
                ["生成正式代码", "确认"],
            )
            self._memory(
                connection,
                "confirm-project",
                "workflow.code_generation.confirmation",
                {"level": "project", "project_key": "project-a"},
                "workflow_rule",
                "not_required",
                "在 project-a 生成正式代码前不要求先确认。",
                ["生成正式代码", "确认"],
            )
            self._memory(
                connection,
                "compile",
                "workflow.embedded.code_change.compile_check",
                {"level": "domain", "domain": "embedded"},
                "workflow_rule",
                "required",
                "单片机项目修改代码后必须进行编译检查。",
                ["单片机", "修改代码", "编译检查"],
            )
            self._memory(
                connection,
                "toolchain",
                "toolchain.tg032plus.compiler",
                {"level": "project", "project_key": "tg032plus"},
                "engineering_experience",
                "keil_c51",
                "TG032Plus 项目统一使用 Keil C51 编译。",
                ["TG032Plus", "Keil C51", "编译"],
                family="engineering_memory",
                verification_status="verified",
            )
            self._memory(
                connection,
                "uart-failure",
                "engineering.uart.timeout_recovery",
                {"level": "domain", "domain": "embedded"},
                "engineering_experience",
                "retry_after_reset",
                "UART 超时后复位外设再重试。",
                ["UART", "超时", "复位外设"],
                family="engineering_memory",
                verification_status="unverified",
            )
            connection.execute(
                """
                INSERT INTO conflict VALUES(
                    'conflict-open', 'toolchain', 'rev-toolchain', 'iar',
                    'TG032Plus 项目统一使用 IAR 编译。', 'open',
                    NULL, NULL, ?, NULL
                )
                """,
                (TIME,),
            )

            # Keep the old revision for an auditable resolved conflict.  The
            # memory pointer itself targets the newer English revision.
            connection.execute(
                """
                INSERT INTO memory VALUES(
                    'language', 'preference.response.language', 'global-language',
                    '{"level":"global"}', 'user_contract', 'response_preference',
                    'active', 'rev-language-new', 2, NULL, ?, ?
                )
                """,
                (TIME, "2026-09-10T10:00:00.000+00:00"),
            )
            connection.executemany(
                """
                INSERT INTO memory_revision VALUES(
                    ?, 'language', ?, ?, ?, '["回复","语言"]', ?, ?, 'decision',
                    NULL, ?
                )
                """,
                [
                    (
                        "rev-language-old",
                        1,
                        "zh",
                        "默认使用中文回复。",
                        "superseded",
                        None,
                        TIME,
                    ),
                    (
                        "rev-language-new",
                        2,
                        "en",
                        "默认使用英文回复。",
                        "current",
                        "rev-language-old",
                        "2026-09-10T10:00:00.000+00:00",
                    ),
                ],
            )
            connection.execute(
                "INSERT INTO evidence VALUES('e-language', 'language', 'RAW-EVIDENCE-MUST-NOT-LEAK')"
            )
            connection.execute(
                """
                INSERT INTO conflict VALUES(
                    'conflict-resolved', 'language', 'rev-language-old', 'en',
                    '默认使用英文回复。', 'resolved', 'rev-language-new',
                    '同范围采用用户最新明确要求', ?, '2026-09-10T10:00:00.000+00:00'
                )
                """,
                (TIME,),
            )

    def _memory(
        self,
        connection: sqlite3.Connection,
        memory_id: str,
        stable_key: str,
        scope: dict,
        memory_type: str,
        value: str,
        statement: str,
        triggers: list[str],
        *,
        family: str = "user_contract",
        verification_status: str | None = None,
    ) -> None:
        revision_id = f"rev-{memory_id}"
        connection.execute(
            """
            INSERT INTO memory VALUES(?, ?, ?, ?, ?, ?, 'active', ?, 1, ?, ?, ?)
            """,
            (
                memory_id,
                stable_key,
                hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest(),
                json.dumps(scope, ensure_ascii=False),
                family,
                memory_type,
                revision_id,
                verification_status,
                TIME,
                TIME,
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_revision VALUES(
                ?, ?, 1, ?, ?, ?, 'current', NULL, 'decision', ?, ?
            )
            """,
            (
                revision_id,
                memory_id,
                value,
                statement,
                json.dumps(triggers, ensure_ascii=False),
                verification_status,
                TIME,
            ),
        )
        connection.execute(
            "INSERT INTO evidence VALUES(?, ?, 'RAW-EVIDENCE-MUST-NOT-LEAK')",
            (f"e-{memory_id}", memory_id),
        )

    def _all_published_text(self, generation_id: str) -> str:
        generation = self.memory_root / "generations" / generation_id
        return "\n".join(
            path.read_text(encoding="utf-8")
            for path in generation.rglob("*")
            if path.is_file()
        )

    def test_publish_full_generation_and_never_project_raw_evidence(self) -> None:
        result = publish_formal_memory(self.database, self.memory_root)

        self.assertTrue(result["created_generation"])
        self.assertTrue(result["current_changed"])
        self.assertEqual(result["memory_count"], 7)
        self.assertEqual(result["open_conflict_count"], 1)
        self.assertEqual(result["resolved_conflict_count"], 1)
        validated = validate_formal_memory(self.memory_root)
        self.assertEqual(validated["status"], "valid")
        self.assertGreaterEqual(validated["verified_file_count"], 6)
        published = self._all_published_text(result["generation_id"])
        self.assertNotIn("RAW-EVIDENCE-MUST-NOT-LEAK", published)
        self.assertIn("TG032Plus 项目统一使用 Keil C51 编译。", published)

        index = (
            self.memory_root / "generations" / result["generation_id"] / "MEMORY.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn("TG032Plus 项目统一使用 Keil C51 编译。", index)
        self.assertNotIn("默认使用英文回复。", index)
        self.assertIn("工程经验", index)

    def test_identical_publish_is_noop_and_changed_state_creates_new_generation(self) -> None:
        first = publish_formal_memory(self.database, self.memory_root)
        old_bytes = {
            path.relative_to(self.memory_root).as_posix(): path.read_bytes()
            for path in (self.memory_root / "generations" / first["generation_id"]).rglob("*")
            if path.is_file()
        }
        second = publish_formal_memory(self.database, self.memory_root)
        self.assertEqual(second["generation_id"], first["generation_id"])
        self.assertFalse(second["created_generation"])
        self.assertFalse(second["current_changed"])

        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "INSERT INTO evidence VALUES('e-response-2', 'response', '另一条不应投影的原始证据')"
            )
        third = publish_formal_memory(self.database, self.memory_root)
        self.assertNotEqual(third["generation_id"], first["generation_id"])
        self.assertEqual(third["previous_generation_id"], first["generation_id"])
        self.assertTrue(third["current_changed"])
        for relative, content in old_bytes.items():
            self.assertEqual((self.memory_root / relative).read_bytes(), content)
        validate_formal_memory(self.memory_root, generation_id=first["generation_id"])

    def test_index_scan_reads_routing_headers_and_full_hash_validation_detects_tamper(self) -> None:
        published = publish_formal_memory(self.database, self.memory_root)
        headers = scan_formal_memory_index(self.memory_root)
        self.assertEqual(headers["memory_count"], 7)
        self.assertLessEqual(headers["topic_count"], 200)
        self.assertTrue(all("entries" not in topic for topic in headers["topics"]))

        topic = self.memory_root / "generations" / published["generation_id"] / headers["topics"][0]["path"]
        topic.write_text(topic.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(FormalMarkdownError, "hash/size mismatch"):
            validate_formal_memory(self.memory_root)

    def test_recall_scope_precedence_conflict_blocking_and_selective_topic_load(self) -> None:
        publish_formal_memory(self.database, self.memory_root)

        project = recall_formal_memory(
            self.memory_root,
            "请生成正式代码并说明结果",
            project_key="project-a",
        )
        confirmation = [
            item
            for item in project["selected"]
            if item["stable_key"] == "workflow.code_generation.confirmation"
        ]
        self.assertEqual(len(confirmation), 1)
        self.assertEqual(confirmation[0]["memory_id"], "confirm-project")
        self.assertTrue(
            any(item["memory_id"] == "confirm-global" for item in project["suppressed_by_scope"])
        )
        self.assertLessEqual(project["topic_bodies_loaded"], 5)

        embedded = recall_formal_memory(
            self.memory_root,
            "修改单片机代码以后做编译检查",
            domain="embedded",
        )
        self.assertTrue(
            any(
                item["stable_key"] == "workflow.embedded.code_change.compile_check"
                for item in embedded["selected"]
            )
        )

        conflict = recall_formal_memory(
            self.memory_root,
            "TG032Plus 应该用什么工具编译？",
            project_key="tg032plus",
        )
        self.assertFalse(any(item["memory_id"] == "toolchain" for item in conflict["selected"]))
        self.assertEqual(conflict["unresolved_conflicts"][0]["conflict_id"], "conflict-open")

        unverified = recall_formal_memory(
            self.memory_root,
            "UART 超时后如何复位外设并重试？",
            domain="embedded",
        )
        self.assertFalse(
            any(item["memory_id"] == "uart-failure" for item in unverified["selected"])
        )
        self.assertTrue(
            any(
                item["memory_id"] == "uart-failure"
                and item["reason"] == "engineering_memory_not_verified"
                for item in unverified["excluded_by_verification"]
            )
        )

    def test_external_router_topic_ids_are_checked_before_body_load_and_recall(self) -> None:
        publish_formal_memory(self.database, self.memory_root)
        headers = scan_formal_memory_index(self.memory_root)
        workflow = next(
            item
            for item in headers["topics"]
            if item["topic_id"] == "workflow-code-generation"
        )
        loaded = load_formal_topic_bodies(self.memory_root, [workflow["topic_id"]])
        self.assertEqual(loaded["topic_bodies_loaded"], 1)
        self.assertEqual(loaded["topics"][0]["topic_id"], workflow["topic_id"])
        self.assertNotIn("RAW-EVIDENCE-MUST-NOT-LEAK", json.dumps(loaded, ensure_ascii=False))

        recalled = recall_selected_formal_topics(
            self.memory_root,
            "生成正式代码",
            [workflow["topic_id"]],
            project_key="project-a",
        )
        self.assertEqual(recalled["topic_bodies_loaded"], 1)
        self.assertEqual(recalled["selected"][0]["memory_id"], "confirm-project")
        self.assertIn("external_router_selection", recalled["selected"][0]["match_reasons"])
        with self.assertRaisesRegex(FormalMarkdownError, "unknown topic ids"):
            load_formal_topic_bodies(self.memory_root, ["hallucinated-topic"])

    def test_curator_topic_ownership_is_authoritative_and_incomplete_map_fails(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.row_factory = sqlite3.Row
            ensure_topic_assignments(connection)
            connection.execute(
                """
                UPDATE topic
                SET name = '经过 Curator 命名的代码流程',
                    description = 'Curator 合并与拆分后的正式主题概述',
                    updated_at = '2026-09-10T11:00:00.000+00:00'
                WHERE slug = 'workflow_code_generation'
                """
            )
        snapshot = snapshot_formal_database(self.database)
        topic = next(
            item
            for item in snapshot["topics"]
            if item["spec"]["path"] == "workflow_code_generation.md"
        )
        self.assertEqual(topic["spec"]["name"], "经过 Curator 命名的代码流程")

        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "DELETE FROM topic_membership WHERE stable_key = 'preference.response.structure'"
            )
        with self.assertRaisesRegex(FormalMarkdownError, "no active topic owner"):
            snapshot_formal_database(self.database)

    def test_unknown_project_scope_blocks_global_fallback(self) -> None:
        publish_formal_memory(self.database, self.memory_root)
        recalled = recall_formal_memory(
            self.memory_root,
            "请生成正式代码",
        )
        self.assertFalse(
            any(
                item["stable_key"] == "workflow.code_generation.confirmation"
                for item in recalled["selected"]
            )
        )
        self.assertTrue(
            any(
                item["stable_key"] == "workflow.code_generation.confirmation"
                for item in recalled["unresolved_context"]
            )
        )

    def test_secret_in_formal_statement_fails_closed_before_publication(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "UPDATE memory_revision SET statement = 'token: sk-secretvalue123456' WHERE revision_id = 'rev-response'"
            )
        with self.assertRaisesRegex(FormalMarkdownError, "probable secret"):
            publish_formal_memory(self.database, self.memory_root)
        self.assertFalse((self.memory_root / "CURRENT").exists())

    def test_failed_current_switch_leaves_previous_generation_active(self) -> None:
        first = publish_formal_memory(self.database, self.memory_root)
        pointer_before = (self.memory_root / "CURRENT").read_bytes()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "INSERT INTO evidence VALUES('e-compile-2', 'compile', 'new evidence')"
            )
        with mock.patch(
            "memory_lab.formal_markdown._replace_current",
            side_effect=OSError("simulated pointer failure"),
        ):
            with self.assertRaisesRegex(OSError, "pointer failure"):
                publish_formal_memory(self.database, self.memory_root)
        self.assertEqual((self.memory_root / "CURRENT").read_bytes(), pointer_before)
        self.assertEqual(validate_formal_memory(self.memory_root)["generation_id"], first["generation_id"])

    def test_database_snapshot_is_read_only_and_missing_schema_fails_closed(self) -> None:
        before = self.database.read_bytes()
        snapshot = snapshot_formal_database(self.database)
        self.assertEqual(snapshot["schema"], "memory-lab-formal-state-v1")
        self.assertEqual(self.database.read_bytes(), before)

        incomplete = self.base / "incomplete.db"
        with closing(sqlite3.connect(incomplete)) as connection, connection:
            connection.execute("CREATE TABLE memory(memory_id TEXT)")
        with self.assertRaisesRegex(FormalMarkdownError, "missing columns"):
            snapshot_formal_database(incomplete)

    def test_legacy_engineering_verification_is_derived_from_admission_signals(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "CREATE TABLE admission_decision(decision_id TEXT PRIMARY KEY, signals_json TEXT)"
            )
            connection.execute(
                "INSERT INTO admission_decision VALUES('decision', ?)",
                (json.dumps({"verification_status": "verified"}),),
            )
            connection.execute(
                "UPDATE memory SET verification_status = NULL WHERE memory_id = 'toolchain'"
            )
            connection.execute(
                "UPDATE memory_revision SET verification_status = NULL WHERE revision_id = 'rev-toolchain'"
            )
        snapshot = snapshot_formal_database(self.database)
        toolchain = next(
            entry
            for topic in snapshot["topics"]
            for entry in topic["entries"]
            if entry["memory_id"] == "toolchain"
        )
        self.assertEqual(toolchain["verification_status"], "verified")


if __name__ == "__main__":
    unittest.main()
