from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import closing
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from memory_lab.cli import main as cli_main
from memory_lab.judge import MockJudge
from memory_lab.shadow import run_llm_shadow
from memory_lab.store import open_store


class NeverJudge:
    def judge(self, _request: dict) -> dict:
        raise AssertionError("cached shadow result should have been replayed")


class ShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "memory_lab.db"
        self.quotes = [
            "以后生成正式代码前先确认。",
            "再确认一次，生成正式代码之前要先问我。",
        ]
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO source(source_id, source_kind, source_path, created_at, baseline_cutoff_ms)
                VALUES ('cp', 'sqlite', 'source.db', '2026-01-01T00:00:00.000+00:00', 1)
                """
            )
            connection.execute(
                """
                INSERT INTO project_context(project_key, directory, display_name, identity_source, created_at)
                VALUES ('project_a', 'D:\\project-a', 'project-a', 'test', '2026-01-01T00:00:00.000+00:00')
                """
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at, completed_at,
                    settle_ms, status
                ) VALUES (
                    'turn_run', 'cp', 'test', '2026-01-01T00:00:00.000+00:00',
                    '2026-01-01T00:00:01.000+00:00', 0, 'complete'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, completed_at, status,
                    decision_count, counts_json
                ) VALUES (
                    'admission_current', 'test', '2026-01-01T00:00:00.000+00:00',
                    '2026-01-01T00:00:02.000+00:00', 'complete', 2, '{"eligible":2}'
                )
                """
            )
            for index, quote in enumerate(self.quotes, 1):
                event_id = f"event_{index}"
                session_id = f"session_{index}"
                message_id = f"message_{index}"
                part_id = f"part_{index}"
                candidate_key = f"candidate_{index}"
                decision_id = f"decision_{index}"
                timestamp = index + 10
                connection.execute(
                    """
                    INSERT INTO session_context(
                        source_id, session_id, project_key, directory, context_source,
                        first_seen_at, last_seen_at
                    ) VALUES ('cp', ?, 'project_a', 'D:\\project-a', 'test', ?, ?)
                    """,
                    (session_id, "2026-01-01T00:00:00.000+00:00", "2026-01-01T00:00:00.000+00:00"),
                )
                connection.execute(
                    """
                    INSERT INTO event(
                        event_id, source_id, session_id, parent_session_id, message_id,
                        part_id, event_type, directory, timestamp, content, content_hash,
                        source_kind, eligibility_status, ingested_at, raw_json
                    ) VALUES (?, 'cp', ?, NULL, ?, ?, 'user_text', 'D:\\project-a', ?, ?, ?,
                              'codepulse_sqlite', 'eligible', '2026-01-01T00:00:00.000+00:00', '{}')
                    """,
                    (
                        event_id,
                        session_id,
                        message_id,
                        part_id,
                        timestamp,
                        quote,
                        f"hash_{index}",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO event_annotation(
                        event_id, semantic_content, project_key, annotation_source, created_at
                    ) VALUES (?, ?, 'project_a', 'test', '2026-01-01T00:00:00.000+00:00')
                    """,
                    (event_id, quote),
                )
                connection.execute(
                    """
                    INSERT INTO conversation_turn(
                        turn_id, source_id, session_id, user_message_id, primary_event_id,
                        user_timestamp, current_state, first_detected_at, last_detected_at,
                        last_detection_run_id, last_structural_signature
                    ) VALUES (?, 'cp', ?, ?, ?, ?, 'completed',
                              '2026-01-01T00:00:00.000+00:00',
                              '2026-01-01T00:00:01.000+00:00', 'turn_run', ?)
                    """,
                    (f"turn_{index}", session_id, message_id, event_id, timestamp, f"sig_{index}"),
                )
                connection.execute(
                    "INSERT INTO turn_event(turn_id, event_id) VALUES (?, ?)",
                    (f"turn_{index}", event_id),
                )
                connection.execute(
                    """
                    INSERT INTO admission_decision(
                        decision_id, run_id, candidate_key, event_id, ordinal,
                        exact_quote, start_offset, end_offset, family, memory_type,
                        stable_key_suggestion, scope_json, activation_triggers_json,
                        disposition, signals_json, instruction_coverage_json,
                        reasons_json, decided_at
                    ) VALUES (?, 'admission_current', ?, ?, 0, ?, 0, ?, 'user_contract',
                              'workflow_rule', 'workflow.code_generation.confirmation',
                              '{"level":"global"}', '["生成代码"]', 'eligible',
                              '{"long_term":["以后"],"temporary":[],"correction":[]}',
                              '{"status":"none"}', '[]', ?)
                    """,
                    (
                        decision_id,
                        candidate_key,
                        event_id,
                        quote,
                        len(quote),
                        f"2026-01-01T00:00:0{index}.000+00:00",
                    ),
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def result(self, index: int, relation: str, target_id: str | None) -> dict:
        quote = self.quotes[index - 1]
        return {
            "should_store": True,
            "type": "workflow_rule",
            "key": "workflow.code_generation.confirmation",
            "scope": {"level": "global"},
            "value": {
                "normalized": "required",
                "statement": "生成正式代码前必须先征得用户确认。",
            },
            "relation": {
                "kind": relation,
                "target_kind": "proposal" if target_id else None,
                "target_id": target_id,
            },
            "evidence": [
                {
                    "event_id": f"event_{index}",
                    "part_id": f"part_{index}",
                    "quote": quote,
                    "start_offset": 0,
                    "end_offset": len(quote),
                }
            ],
            "confidence": 0.98,
            "reason": "这是跨会话持续生效的代码生成确认规则。",
        }

    def test_shadow_is_isolated_and_replays_exact_requests(self) -> None:
        judge = MockJudge(
            {
                "llmreq_candidate_1": self.result(1, "new", None),
                "llmreq_candidate_2": self.result(
                    2, "equivalent", "rule_target_decision_1"
                ),
            }
        )
        first = run_llm_shadow(
            self.database, judge, provider="mock", model="mock-model"
        )

        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["counts"], {"valid": 2})
        self.assertTrue(first["formal_state_unchanged"])
        self.assertEqual(
            {table: state["row_count"] for table, state in first["formal_state_after"].items()},
            {"memory": 0, "memory_revision": 0, "evidence": 0, "conflict": 0},
        )
        self.assertTrue(
            all(
                len(state["content_sha256"]) == 64
                for state in first["formal_state_after"].values()
            )
        )
        self.assertEqual(
            first["judgments"][1]["result"]["relation"]["kind"], "equivalent"
        )

        second = run_llm_shadow(
            self.database, NeverJudge(), provider="mock", model="mock-model"
        )
        self.assertEqual(second["counts"], {"replayed": 2})
        self.assertTrue(second["formal_state_unchanged"])

        with closing(open_store(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM llm_judgment").fetchone()[0], 4)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory").fetchone()[0], 0)

    def test_invalid_model_output_is_audited_without_formal_write(self) -> None:
        bad = self.result(1, "new", None)
        bad["evidence"][0]["quote"] = "fabricated"
        report = run_llm_shadow(
            self.database,
            MockJudge(
                {
                    "llmreq_candidate_1": bad,
                    "llmreq_candidate_2": self.result(
                        2, "equivalent", "rule_target_decision_1"
                    ),
                }
            ),
            provider="mock",
            model="bad-model",
        )

        self.assertEqual(report["status"], "complete_with_errors")
        self.assertEqual(report["counts"], {"invalid": 1, "valid": 1})
        self.assertTrue(report["formal_state_unchanged"])

    def test_sensitive_admission_is_blocked_without_calling_judge(self) -> None:
        quote = "以后使用 github_pat_" + "A" * 24
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                UPDATE admission_run
                SET decision_count = 3, counts_json = '{"eligible":2,"sensitive_blocked":1}'
                WHERE run_id = 'admission_current'
                """
            )
            connection.execute(
                """
                INSERT INTO session_context(
                    source_id, session_id, project_key, directory, context_source,
                    first_seen_at, last_seen_at
                ) VALUES (
                    'cp', 'session_sensitive', 'project_a', 'D:\\project-a', 'test',
                    '2026-01-01T00:00:00.000+00:00', '2026-01-01T00:00:00.000+00:00'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO event(
                    event_id, source_id, session_id, parent_session_id, message_id,
                    part_id, event_type, directory, timestamp, content, content_hash,
                    source_kind, eligibility_status, ingested_at, raw_json
                ) VALUES (
                    'event_sensitive', 'cp', 'session_sensitive', NULL, 'message_sensitive',
                    'part_sensitive', 'user_text', 'D:\\project-a', 99, ?, 'hash_sensitive',
                    'codepulse_sqlite', 'eligible', '2026-01-01T00:00:00.000+00:00', '{}'
                )
                """,
                (quote,),
            )
            connection.execute(
                """
                INSERT INTO event_annotation(
                    event_id, semantic_content, project_key, annotation_source, created_at
                ) VALUES (
                    'event_sensitive', ?, 'project_a', 'test',
                    '2026-01-01T00:00:00.000+00:00'
                )
                """,
                (quote,),
            )
            connection.execute(
                """
                INSERT INTO conversation_turn(
                    turn_id, source_id, session_id, user_message_id, primary_event_id,
                    user_timestamp, current_state, first_detected_at, last_detected_at,
                    last_detection_run_id, last_structural_signature
                ) VALUES (
                    'turn_sensitive', 'cp', 'session_sensitive', 'message_sensitive',
                    'event_sensitive', 99, 'completed',
                    '2026-01-01T00:00:00.000+00:00',
                    '2026-01-01T00:00:01.000+00:00', 'turn_run', 'sig_sensitive'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO admission_decision(
                    decision_id, run_id, candidate_key, event_id, ordinal,
                    exact_quote, start_offset, end_offset, family, memory_type,
                    stable_key_suggestion, scope_json, activation_triggers_json,
                    disposition, signals_json, instruction_coverage_json,
                    reasons_json, decided_at
                ) VALUES (
                    'decision_sensitive', 'admission_current', 'candidate_sensitive',
                    'event_sensitive', 0, ?, 0, ?, 'unknown', NULL, NULL,
                    '{"level":"unknown"}', '[]', 'sensitive_blocked',
                    '{"long_term":["以后"],"temporary":[],"correction":[]}',
                    '{"status":"none"}', '["sensitive_pattern_detected"]',
                    '2026-01-01T00:00:09.000+00:00'
                )
                """,
                (quote, len(quote)),
            )
            connection.execute(
                "INSERT INTO turn_event(turn_id, event_id) VALUES ('turn_sensitive', 'event_sensitive')"
            )

        judge = MockJudge(
            {
                "llmreq_candidate_1": self.result(1, "new", None),
                "llmreq_candidate_2": self.result(
                    2, "equivalent", "rule_target_decision_1"
                ),
            }
        )
        report = run_llm_shadow(
            self.database, judge, provider="mock", model="sensitive-test-model"
        )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["counts"], {"blocked": 1, "valid": 2})
        blocked = next(
            item for item in report["judgments"] if item["candidate_key"] == "candidate_sensitive"
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertIsNone(blocked["result"])
        self.assertTrue(report["formal_state_unchanged"])

    def test_cli_returns_nonzero_for_partial_shadow_failure(self) -> None:
        output = StringIO()
        with patch(
            "memory_lab.cli.run_deepseek_shadow",
            return_value={"status": "complete_with_errors", "counts": {"invalid": 1}},
        ), redirect_stdout(output):
            code = cli_main(
                [
                    "--source-db",
                    str(Path(self.temporary_directory.name) / "source.db"),
                    "--lab-db",
                    str(self.database),
                    "shadow",
                ]
            )

        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])

    def test_equal_count_formal_mutation_is_detected_by_content_hash(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO memory(
                    memory_id, stable_key, scope_fingerprint, scope_json, family,
                    memory_type, status, state_version, created_at, updated_at
                ) VALUES (
                    'memory_1', 'workflow.code_generation.confirmation', 'scope_global',
                    '{"level":"global"}', 'user_contract', 'workflow_rule',
                    'active', 0, 'before', 'before'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO memory_revision(
                    revision_id, memory_id, revision_number, semantic_value,
                    statement, activation_triggers_json, status,
                    source_decision_id, created_at
                ) VALUES (
                    'revision_1', 'memory_1', 1, 'required', 'confirm first',
                    '[]', 'current', 'decision_1', 'before'
                )
                """
            )
            connection.execute(
                "UPDATE memory SET current_revision_id = 'revision_1' WHERE memory_id = 'memory_1'"
            )

        delegate = MockJudge(
            {
                "llmreq_candidate_1": self.result(1, "new", None),
                "llmreq_candidate_2": self.result(
                    2, "equivalent", "rule_target_decision_1"
                ),
            }
        )
        database = self.database

        class MutatingJudge:
            changed = False

            def judge(self, request: dict) -> dict:
                if not self.changed:
                    with closing(open_store(database)) as mutation, mutation:
                        mutation.execute(
                            "UPDATE memory SET updated_at = 'after' WHERE memory_id = 'memory_1'"
                        )
                    self.changed = True
                return delegate.judge(request)

        with self.assertRaisesRegex(RuntimeError, "formal memory state changed"):
            run_llm_shadow(
                self.database,
                MutatingJudge(),
                provider="mock",
                model="mutation-test-model",
            )

        with closing(open_store(self.database)) as connection:
            run = connection.execute(
                """
                SELECT status, formal_state_before_json, formal_state_after_json
                FROM llm_shadow_run ORDER BY rowid DESC LIMIT 1
                """
            ).fetchone()
        before = json.loads(run["formal_state_before_json"])
        after = json.loads(run["formal_state_after_json"])
        self.assertEqual(run["status"], "failed")
        self.assertEqual(before["memory"]["row_count"], after["memory"]["row_count"])
        self.assertNotEqual(
            before["memory"]["content_sha256"], after["memory"]["content_sha256"]
        )

    def test_newer_incomplete_admission_blocks_shadow_before_judge(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, status,
                    decision_count, counts_json
                ) VALUES (
                    'admission_stuck', 'test', '2099-01-01T00:00:00Z',
                    'running', 0, '{}'
                )
                """
            )

        with self.assertRaisesRegex(ValueError, "rerun admit"):
            run_llm_shadow(
                self.database,
                NeverJudge(),
                provider="mock",
                model="must-not-run",
            )


if __name__ == "__main__":
    unittest.main()
