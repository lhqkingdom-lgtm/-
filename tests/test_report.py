from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.consolidation import latest_admission_snapshot
from memory_lab.report import evaluate_shadow_run, generate_shadow_report
from memory_lab.store import open_store


class ShadowReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = self.root / "memory_lab.db"
        self.gold = self.root / "gold.json"
        self.gold.write_text(
            json.dumps(
                {
                    "schema_version": "memory-lab-shadow-gold-v1",
                    "suite_id": "test-suite",
                    "description": "test",
                    "generalization_note": "held-out tests are still required",
                    "cases": [
                        {
                            "candidate_key": "candidate_positive",
                            "label": "positive",
                            "should_store": True,
                            "type": "workflow_rule",
                            "stable_key": "workflow.code_generation.confirmation",
                            "scope": {"level": "global"},
                            "value": "required",
                            "relation": "new",
                        },
                        {
                            "candidate_key": "candidate_negative",
                            "label": "negative",
                            "should_store": False,
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._seed()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _result(self, index: int, should_store: bool) -> dict:
        quote = "以后生成代码前先确认。" if index == 1 else "这一次只回复收到。"
        result = {
            "should_store": should_store,
            "type": "workflow_rule" if should_store else None,
            "key": "workflow.code_generation.confirmation" if should_store else None,
            "scope": {"level": "global"} if should_store else None,
            "value": (
                {"normalized": "required", "statement": "生成正式代码前先确认。"}
                if should_store
                else None
            ),
            "relation": {
                "kind": "new" if should_store else "unrelated",
                "target_kind": None,
                "target_id": None,
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
            "confidence": 0.99,
            "reason": "test result",
        }
        return result

    def _seed(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO source(source_id, source_kind, source_path, created_at, baseline_cutoff_ms)
                VALUES ('source', 'sqlite', 'source.db', '2026-01-01T00:00:00Z', 1)
                """
            )
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, completed_at, status,
                    decision_count, counts_json
                ) VALUES (
                    'admission', 'test', '2026-01-01T00:00:00Z',
                    '2026-01-01T00:00:01Z', 'complete', 2, '{"eligible":2}'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at, completed_at,
                    settle_ms, status
                ) VALUES (
                    'turn_run', 'source', 'test', '2026-01-01T00:00:00Z',
                    '2026-01-01T00:00:01Z', 0, 'complete'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO project_context(
                    project_key, directory, display_name, identity_source, created_at
                ) VALUES (
                    'project', 'D:\\project', 'project', 'test',
                    '2026-01-01T00:00:00Z'
                )
                """
            )
            quotes = ["以后生成代码前先确认。", "这一次只回复收到。"]
            keys = ["candidate_positive", "candidate_negative"]
            for index, (quote, key) in enumerate(zip(quotes, keys), 1):
                connection.execute(
                    """
                    INSERT INTO session_context(
                        source_id, session_id, project_key, directory, context_source,
                        first_seen_at, last_seen_at
                    ) VALUES (
                        'source', ?, 'project', 'D:\\project', 'test',
                        '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
                    )
                    """,
                    (f"session_{index}",),
                )
                connection.execute(
                    """
                    INSERT INTO event(
                        event_id, source_id, session_id, message_id, part_id, event_type,
                        timestamp, content, content_hash, source_kind, eligibility_status,
                        ingested_at, raw_json
                    ) VALUES (?, 'source', ?, ?, ?, 'user_text', ?, ?, ?,
                              'codepulse_sqlite', 'eligible', '2026-01-01T00:00:00Z', '{}')
                    """,
                    (
                        f"event_{index}",
                        f"session_{index}",
                        f"message_{index}",
                        f"part_{index}",
                        index,
                        quote,
                        f"hash_{index}",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO event_annotation(
                        event_id, semantic_content, project_key, annotation_source, created_at
                    ) VALUES (?, ?, 'project', 'test', '2026-01-01T00:00:00Z')
                    """,
                    (f"event_{index}", quote),
                )
                connection.execute(
                    """
                    INSERT INTO conversation_turn(
                        turn_id, source_id, session_id, user_message_id,
                        primary_event_id, user_timestamp, current_state,
                        first_detected_at, last_detected_at,
                        last_detection_run_id, last_structural_signature
                    ) VALUES (?, 'source', ?, ?, ?, ?, 'completed',
                              '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z',
                              'turn_run', ?)
                    """,
                    (
                        f"turn_{index}",
                        f"session_{index}",
                        f"message_{index}",
                        f"event_{index}",
                        index,
                        f"signature_{index}",
                    ),
                )
                connection.execute(
                    "INSERT INTO turn_event(turn_id, event_id) VALUES (?, ?)",
                    (f"turn_{index}", f"event_{index}"),
                )
                connection.execute(
                    """
                    INSERT INTO admission_decision(
                        decision_id, run_id, candidate_key, event_id, ordinal,
                        exact_quote, start_offset, end_offset, family, memory_type,
                        stable_key_suggestion, scope_json, activation_triggers_json,
                        disposition, signals_json, instruction_coverage_json,
                        reasons_json, decided_at
                    ) VALUES (?, 'admission', ?, ?, 0, ?, 0, ?, 'user_contract', ?, ?,
                              ?, '[]', ?, '{}', '{"status":"none"}', '[]', ?)
                    """,
                    (
                        f"decision_{index}",
                        key,
                        f"event_{index}",
                        quote,
                        len(quote),
                        "workflow_rule" if index == 1 else None,
                        "workflow.code_generation.confirmation" if index == 1 else None,
                        '{"level":"global"}' if index == 1 else '{"level":"unknown"}',
                        "eligible" if index == 1 else "temporary",
                        f"2026-01-01T00:00:0{index}Z",
                    ),
                )
            self._insert_run(connection, "run_good", "2026-01-01T00:00:10Z", True)

    def _insert_run(
        self,
        connection,
        run_id: str,
        completed_at: str,
        positive_prediction: bool,
        replayed: bool = False,
        block_negative: bool = False,
    ) -> None:
        empty_state = json.dumps(
            {
                table: {"row_count": 0, "content_sha256": "0" * 64}
                for table in ("memory", "memory_revision", "evidence", "conflict")
            },
            sort_keys=True,
        )
        _admission, _rows, admission_snapshot_hash = latest_admission_snapshot(connection)
        self.assertIsNotNone(admission_snapshot_hash)
        connection.execute(
            """
            INSERT INTO llm_shadow_run(
                shadow_run_id, shadow_version, admission_run_id,
                admission_snapshot_hash, provider, model,
                prompt_version, input_schema_version, output_schema_version,
                started_at, completed_at, status, candidate_count, success_count,
                error_count, replay_count, counts_json, formal_state_before_json,
                formal_state_after_json
            ) VALUES (?, 'shadow-v1', 'admission', ?, 'mock', 'mock-model', 'prompt-v1',
                      'input-v1', 'output-v1', ?, ?, 'complete', 2, 2, 0, ?,
                      ?, ?, ?)
            """,
            (
                run_id,
                admission_snapshot_hash,
                completed_at,
                completed_at,
                2 if replayed else 0,
                '{"replayed":2}' if replayed else '{"valid":2}',
                empty_state,
                empty_state,
            ),
        )
        for index, key in enumerate(("candidate_positive", "candidate_negative"), 1):
            should_store = positive_prediction if index == 1 else False
            result = None if block_negative and index == 2 else self._result(index, should_store)
            status = (
                "blocked"
                if result is None
                else "replayed"
                if replayed
                else "valid"
            )
            execution_mode = (
                "local_block"
                if result is None
                else "cache_replay"
                if replayed
                else "judge_call"
            )
            connection.execute(
                """
                INSERT INTO llm_judgment(
                    judgment_id, shadow_run_id, admission_run_id, decision_id,
                    candidate_key, request_hash, model, prompt_version,
                    input_schema_version, output_schema_version, status, result_json,
                    execution_mode,
                    should_store, memory_type, stable_key, scope_json, semantic_value,
                    relation, confidence, created_at
                ) VALUES (?, ?, 'admission', ?, ?, ?, 'mock-model', 'prompt-v1',
                          'input-v1', 'output-v1', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.99, ?)
                """,
                (
                    f"judgment_{run_id}_{index}",
                    run_id,
                    f"decision_{index}",
                    key,
                    f"hash_{run_id}_{index}",
                    status,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    execution_mode,
                    int(should_store) if result is not None else None,
                    result["type"] if result is not None else None,
                    result["key"] if result is not None else None,
                    json.dumps(result["scope"]) if result and result["scope"] else None,
                    result["value"]["normalized"] if result and result["value"] else None,
                    result["relation"]["kind"] if result is not None else None,
                    completed_at,
                ),
            )

    def test_passes_exact_gold_and_writes_matching_artifacts(self) -> None:
        report = evaluate_shadow_run(self.database, gold_path=self.gold)

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["metrics"]["true_positive"], 1)
        self.assertEqual(report["metrics"]["true_negative"], 1)
        self.assertEqual(report["metrics"]["false_positive"], 0)
        self.assertEqual(report["metrics"]["false_negative"], 0)
        self.assertEqual(report["metrics"]["semantic_fields_correct"], 5)
        self.assertEqual(report["metrics"]["exact_evidence_cases"], 2)

        output = self.root / "reports"
        summary = generate_shadow_report(
            self.database, gold_path=self.gold, output_directory=output
        )
        self.assertEqual(summary["status"], "complete")
        parsed = json.loads((output / "latest-shadow.json").read_text(encoding="utf-8"))
        markdown = (output / "latest-shadow.md").read_text(encoding="utf-8")
        self.assertTrue(parsed["gate"]["passed"])
        self.assertIn("结论：通过", markdown)
        self.assertNotIn("api_key", markdown)

    def test_latest_wrong_prediction_fails_gate(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            self._insert_run(connection, "run_bad", "2026-01-01T00:00:20Z", False)

        report = evaluate_shadow_run(self.database, gold_path=self.gold)

        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(report["metrics"]["false_negative"], 1)
        self.assertEqual(report["metrics"]["recall"], 0.0)
        self.assertFalse(report["cases"][0]["correct"])

    def test_default_report_uses_latest_fresh_run_and_replay_cannot_pass(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            self._insert_run(
                connection,
                "run_replay",
                "2026-01-01T00:00:30Z",
                True,
                replayed=True,
            )

        default_report = evaluate_shadow_run(self.database, gold_path=self.gold)
        replay_report = evaluate_shadow_run(
            self.database, gold_path=self.gold, shadow_run_id="run_replay"
        )

        self.assertEqual(default_report["run"]["shadow_run_id"], "run_good")
        self.assertTrue(default_report["gate"]["passed"])
        self.assertFalse(replay_report["gate"]["fresh_model_outputs"])
        self.assertFalse(replay_report["gate"]["passed"])

    def test_expected_local_sensitive_block_is_a_passing_safety_case(self) -> None:
        gold = json.loads(self.gold.read_text(encoding="utf-8"))
        gold["cases"][1]["expected_status"] = "blocked"
        self.gold.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        with closing(open_store(self.database)) as connection, connection:
            self._insert_run(
                connection,
                "run_blocked",
                "2026-01-01T00:00:40Z",
                True,
                block_negative=True,
            )

        report = evaluate_shadow_run(self.database, gold_path=self.gold)

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["metrics"]["blocked_cases"], 1)
        self.assertEqual(report["metrics"]["blocked_cases_correct"], 1)
        self.assertEqual(report["metrics"]["local_blocks"], 1)
        self.assertEqual(report["metrics"]["judge_invocations"], 1)

    def test_newer_incomplete_admission_invalidates_old_green_report(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, status,
                    decision_count, counts_json
                ) VALUES ('admission_new', 'test', '2026-01-01T00:01:00Z',
                          'running', 0, '{}')
                """
            )

        report = evaluate_shadow_run(self.database, gold_path=self.gold)

        self.assertFalse(report["gate"]["admission_snapshot_current"])
        self.assertFalse(report["gate"]["passed"])

    def test_newer_running_fresh_shadow_cannot_fall_back_to_old_run(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO llm_shadow_run(
                    shadow_run_id, shadow_version, admission_run_id,
                    admission_snapshot_hash, provider, model, prompt_version,
                    input_schema_version, output_schema_version, started_at, status,
                    candidate_count, counts_json, formal_state_before_json
                ) VALUES (
                    'run_stuck', 'shadow-v1', 'admission', 'snapshot_pending',
                    'mock', 'mock-model', 'prompt-v1', 'input-v1', 'output-v1',
                    '2026-01-01T00:02:00Z', 'running', 2, '{}', '{}'
                )
                """
            )

        with self.assertRaisesRegex(ValueError, "not finalized"):
            evaluate_shadow_run(self.database, gold_path=self.gold)

    def test_failed_generation_overwrites_stale_green_latest_artifacts(self) -> None:
        output = self.root / "reports"
        generate_shadow_report(
            self.database, gold_path=self.gold, output_directory=output
        )
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO llm_shadow_run(
                    shadow_run_id, shadow_version, admission_run_id,
                    admission_snapshot_hash, provider, model, prompt_version,
                    input_schema_version, output_schema_version, started_at, status,
                    candidate_count, counts_json, formal_state_before_json
                ) VALUES (
                    'run_stuck', 'shadow-v1', 'admission', 'snapshot_pending',
                    'mock', 'mock-model', 'prompt-v1', 'input-v1', 'output-v1',
                    '2026-01-01T00:02:00Z', 'running', 2, '{}', '{}'
                )
                """
            )

        with self.assertRaises(ValueError):
            generate_shadow_report(
                self.database, gold_path=self.gold, output_directory=output
            )

        failed_json = json.loads(
            (output / "latest-shadow.json").read_text(encoding="utf-8")
        )
        failed_markdown = (output / "latest-shadow.md").read_text(encoding="utf-8")
        self.assertEqual(failed_json["status"], "failed")
        self.assertFalse(failed_json["gate"]["passed"])
        self.assertIn("报告生成失败", failed_markdown)
        self.assertNotIn("结论：通过", failed_markdown)

    def test_holdout_case_can_be_frozen_by_exact_quote_before_candidate_id_exists(self) -> None:
        gold = json.loads(self.gold.read_text(encoding="utf-8"))
        gold["cases"][1].pop("candidate_key")
        gold["cases"][1]["exact_quote"] = "这一次只回复收到。"
        self.gold.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")

        report = evaluate_shadow_run(self.database, gold_path=self.gold)

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["cases"][1]["candidate_key"], "candidate_negative")

    def test_holdout_gold_modified_after_model_run_fails_timing_gate(self) -> None:
        gold = json.loads(self.gold.read_text(encoding="utf-8"))
        gold["provenance"] = "holdout"
        gold["frozen_at"] = "2025-01-01T00:00:00Z"
        self.gold.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        os.utime(self.gold, (1_799_000_000, 1_799_000_000))

        report = evaluate_shadow_run(self.database, gold_path=self.gold)

        self.assertFalse(report["gate"]["gold_frozen_before_run"])
        self.assertFalse(report["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
