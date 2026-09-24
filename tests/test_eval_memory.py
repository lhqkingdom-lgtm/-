from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from memory_lab.eval_memory import (
    run_eval_memory_plan,
    run_eval_memory_review_existing,
    run_eval_memory_replay,
)
from memory_lab.store import open_store


class EvalMemoryPlanTests(unittest.TestCase):
    def test_dry_run_plan_validates_cases_scenarios_and_turns(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            lab = tmp / "lab.db"
            source_db = tmp / "source.db"
            source_db.write_text("", encoding="utf-8")
            evalset = tmp / "evalset"
            evalset.mkdir()
            (evalset / "cases.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "case_id": "case_1",
                                "source_db": str(source_db),
                                "turn_id": "turn_1",
                                "case_type": "runtime_issue",
                                "expected_action": ["create_memory"],
                            }
                        ),
                        json.dumps(
                            {
                                "case_id": "case_2",
                                "source_db": str(source_db),
                                "turn_id": "turn_2",
                                "case_type": "low_value_continue",
                                "expected_action": ["noop"],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            (evalset / "scenarios.json").write_text(
                json.dumps(
                    {
                        "evalset_id": "test",
                        "scenarios": [
                            {
                                "scenario_id": "scenario_1",
                                "title": "test scenario",
                                "turn_ids": ["turn_1", "turn_missing"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with closing(open_store(lab)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO source(source_id, source_kind, source_path, created_at)
                    VALUES ('src', 'cp_sqlite', 'source.db', 'now')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO event(
                      event_id, source_id, session_id, message_id, part_id,
                      event_type, timestamp, content, content_hash,
                      source_kind, ingested_at, raw_json
                    )
                    VALUES ('event_1', 'src', 'session', 'msg_1', 'part_1',
                            'user_text', 1, 'hello', 'hash', 'sqlite', 'now', '{}')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO turn_detection_run(
                      run_id, source_id, detector_version, started_at, settle_ms, status
                    )
                    VALUES ('run', 'src', 'test', 'now', 0, 'completed')
                    """
                )
                for turn_id, state in [
                    ("turn_1", "completed"),
                    ("turn_2", "failed"),
                ]:
                    connection.execute(
                        """
                        INSERT INTO conversation_turn(
                          turn_id, source_id, session_id, user_message_id,
                          primary_event_id, user_timestamp, current_state,
                          first_detected_at, last_detected_at, last_detection_run_id,
                          last_structural_signature, assistant_count, part_count, terminal_error
                        )
                        VALUES (?, 'src', 'session', ?, 'event_1', 1, ?,
                                'a', 'b', 'run', ?, 1, 2, ?)
                        """,
                        (
                            turn_id,
                            f"msg_{turn_id}",
                            state,
                            f"signature_{turn_id}",
                            int(state == "failed"),
                        ),
                    )

            report = run_eval_memory_plan(
                evalset_path=evalset,
                lab_database=lab,
                dry_run_plan=True,
            )

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["case_count"], 2)
            self.assertEqual(report["scenario_count"], 1)
            self.assertEqual(report["found_turn_count"], 2)
            self.assertEqual(report["missing_turn_ids"], ["turn_missing"])
            self.assertEqual(report["case_type_counts"]["runtime_issue"], 1)
            self.assertEqual(report["expected_action_counts"]["noop"], 1)
            self.assertIn("case_parameter_quality", report)
            self.assertEqual(report["case_parameter_quality"]["invalid_case_count"], 2)

    def test_dry_run_plan_validates_eval_case_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            lab = tmp / "lab.db"
            source_db = tmp / "source.db"
            source_db.write_text("", encoding="utf-8")
            evalset = tmp / "evalset"
            evalset.mkdir()
            (evalset / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "case_1",
                        "source_db": str(source_db),
                        "turn_id": "turn_1",
                        "case_type": "runtime_issue",
                        "expected_action": ["create_memory"],
                        "allowed_actions": ["create_memory"],
                        "forbidden_actions": [],
                        "expected_keywords": ["ADC"],
                        "expected_process": {"load_memory_index": True},
                        "expected_memory_type": "runtime_failure_experience",
                        "expected_scope": "domain_or_project",
                        "expected_relation": "create_when_new_root_cause",
                        "semantic_expectation": "沉淀可复用运行故障经验。",
                        "should_extract": ["ADC"],
                        "should_not_extract": ["项目路径"],
                    }
                ),
                encoding="utf-8",
            )
            (evalset / "scenarios.json").write_text(
                json.dumps(
                    {
                        "evalset_id": "test",
                        "scenarios": [
                            {
                                "scenario_id": "scenario_1",
                                "title": "测试",
                                "turn_ids": ["turn_1"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with closing(open_store(lab)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO source(source_id, source_kind, source_path, created_at)
                    VALUES ('src', 'cp_sqlite', 'source.db', 'now')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO turn_detection_run(
                      run_id, source_id, detector_version, started_at, settle_ms, status
                    )
                    VALUES ('run', 'src', 'test', 'now', 0, 'completed')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO event(
                      event_id, source_id, session_id, message_id, part_id,
                      event_type, timestamp, content, content_hash,
                      source_kind, ingested_at, raw_json
                    )
                    VALUES ('event_1', 'src', 's1', 'm1', 'part_1',
                            'user_text', 1, 'hello', 'hash', 'sqlite', 'now', '{}')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO conversation_turn(
                      turn_id, source_id, session_id, user_message_id,
                      primary_event_id, user_timestamp, current_state, first_detected_at,
                      last_detected_at, last_detection_run_id,
                      last_structural_signature, assistant_count, part_count
                    )
                    VALUES('turn_1', 'src', 's1', 'm1', 'event_1', 1, 'completed',
                           'a', 'b', 'run', 'sig', 1, 1)
                    """
                )

            report = run_eval_memory_plan(
                evalset_path=evalset,
                lab_database=lab,
                dry_run_plan=True,
            )

            self.assertEqual(report["case_parameter_quality"]["status"], "pass")

    def test_replay_writes_generated_memories_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            lab = tmp / "lab.db"
            lab.write_text("", encoding="utf-8")
            source_db = tmp / "source.db"
            source_db.write_text("", encoding="utf-8")
            pg_config = tmp / "pg.json"
            pg_config.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-postgres-config-v1",
                        "postgres": {
                            "host": "localhost",
                            "port": 5432,
                            "database": "memory_lab",
                            "user": "root",
                            "password": "123456",
                            "sslmode": "disable",
                        },
                        "user": {"user_id": "base-user", "email": ""},
                    }
                ),
                encoding="utf-8",
            )
            evalset = tmp / "evalset"
            evalset.mkdir()
            (evalset / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "case_1",
                        "source_db": str(source_db),
                        "turn_id": "turn_1",
                        "case_type": "runtime_issue",
                        "expected_action": ["create_memory"],
                    }
                ),
                encoding="utf-8",
            )

            fake_agent_result = {
                "status": "complete",
                "run_id": "react-test",
                "committed": False,
                "log": {"run_log_path": str(tmp / "react.json")},
                "plan": {
                    "changes": [
                        {
                            "action": "create_memory",
                            "reason": "这是可复用的运行问题定位经验。",
                            "memory": {
                                "stable_key": "embedded.test.memory",
                                "memory_type": "runtime_failure_pattern",
                                "summary": "测试摘要",
                                "content": "症状：测试；原因：测试；解决：测试。",
                            },
                            "topic": {
                                "topic_key": "embedded_test",
                                "title": "嵌入式测试主题",
                                "summary": "主题摘要",
                            },
                            "evidence_refs": ["e1"],
                        }
                    ]
                },
            }

            with (
                patch(
                    "memory_lab.eval_memory.ingest_turn_to_postgres",
                    return_value={"status": "complete"},
                ),
                patch(
                    "memory_lab.eval_memory.run_memory_react_agent",
                    return_value=fake_agent_result,
                ),
            ):
                report = run_eval_memory_replay(
                    evalset_path=evalset,
                    lab_database=lab,
                    pg_config_path=pg_config,
                    report_directory=tmp / "reports",
                    run_name="unit-report",
                    mode="single-isolated",
                    limit=1,
                    dry_run=True,
                )

            generated = Path(report["report_directory"]) / "generated-memories.md"
            self.assertTrue(generated.is_file())
            text = generated.read_text(encoding="utf-8")
            self.assertIn("case_1", text)
            self.assertIn("create_memory", text)
            self.assertIn("测试摘要", text)
            quality = Path(report["report_directory"]) / "quality-report.md"
            self.assertTrue(quality.is_file())
            quality_text = quality.read_text(encoding="utf-8")
            self.assertIn("记忆质量评测报告", quality_text)
            self.assertIn("过程指标", quality_text)
            self.assertIn("结果硬指标", quality_text)
            self.assertIn("process_metrics", report["quality"])
            self.assertIn("result_metrics", report["quality"])
            self.assertEqual(
                report["quality"]["process_metrics"]["action_allowed"]["ok"], 1
            )

    def test_scenario_sequential_reuses_one_user_per_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            lab = tmp / "lab.db"
            lab.write_text("", encoding="utf-8")
            source_db = tmp / "source.db"
            source_db.write_text("", encoding="utf-8")
            pg_config = tmp / "pg.json"
            pg_config.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-postgres-config-v1",
                        "postgres": {
                            "host": "localhost",
                            "port": 5432,
                            "database": "memory_lab",
                            "user": "root",
                            "password": "123456",
                            "sslmode": "disable",
                        },
                        "user": {"user_id": "base-user", "email": ""},
                    }
                ),
                encoding="utf-8",
            )
            evalset = tmp / "evalset"
            evalset.mkdir()
            (evalset / "cases.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "case_id": "case_1",
                                "source_db": str(source_db),
                                "turn_id": "turn_1",
                                "case_type": "runtime_issue",
                                "expected_action": ["create_memory"],
                            }
                        ),
                        json.dumps(
                            {
                                "case_id": "case_2",
                                "source_db": str(source_db),
                                "turn_id": "turn_2",
                                "case_type": "runtime_issue",
                                "expected_action": ["update_memory"],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            (evalset / "scenarios.json").write_text(
                json.dumps(
                    {
                        "evalset_id": "test",
                        "scenarios": [
                            {
                                "scenario_id": "scenario_1",
                                "title": "顺序测试",
                                "turn_ids": ["turn_1", "turn_2"],
                                "expected": {"quality_target": "A"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            seen_pg_configs: list[Path] = []

            def fake_agent(**kwargs):
                seen_pg_configs.append(Path(kwargs["pg_config_path"]))
                action = "create_memory" if len(seen_pg_configs) == 1 else "update_memory"
                return {
                    "status": "complete",
                    "run_id": f"react-{len(seen_pg_configs)}",
                    "committed": True,
                    "log": {"run_log_path": str(tmp / f"react-{len(seen_pg_configs)}.json")},
                    "plan": {
                        "changes": [
                            {
                                "action": action,
                                "reason": "测试顺序回放。",
                                "memory": {
                                    "stable_key": "embedded.test.memory",
                                    "memory_type": "runtime_failure_pattern",
                                    "summary": f"第{len(seen_pg_configs)}轮摘要",
                                },
                                "evidence_refs": ["e1"],
                            }
                        ]
                    },
                }

            with (
                patch(
                    "memory_lab.eval_memory.ingest_turn_to_postgres",
                    return_value={"status": "complete"},
                ),
                patch("memory_lab.eval_memory.run_memory_react_agent", side_effect=fake_agent),
            ):
                report = run_eval_memory_replay(
                    evalset_path=evalset,
                    lab_database=lab,
                    pg_config_path=pg_config,
                    report_directory=tmp / "reports",
                    run_name="scenario-unit",
                    mode="scenario-sequential",
                    limit=1,
                    dry_run=False,
                )

            self.assertEqual(report["scenario_count"], 1)
            self.assertEqual(report["turn_count"], 2)
            self.assertEqual(report["success_count"], 2)
            self.assertEqual(report["action_counts"], {"create_memory": 1, "update_memory": 1})
            self.assertEqual(len(set(seen_pg_configs)), 1)
            generated = Path(report["report_directory"]) / "generated-memories.md"
            text = generated.read_text(encoding="utf-8")
            self.assertIn("scenario_1", text)
            self.assertIn("turn_1", text)
            self.assertIn("turn_2", text)
            quality = Path(report["report_directory"]) / "quality-report.md"
            self.assertTrue(quality.is_file())
            self.assertIn("记忆质量评测报告", quality.read_text(encoding="utf-8"))

    def test_review_existing_rejudges_with_latest_evalset_cases(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            evalset = tmp / "evalset"
            evalset.mkdir()
            (evalset / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "case_1",
                        "source_db": str(tmp / "source.db"),
                        "turn_id": "turn_1",
                        "case_type": "requirement_parameter",
                        "expected_action": ["noop"],
                        "allowed_actions": ["noop"],
                        "forbidden_actions": [
                            "create_memory",
                            "update_memory",
                            "merge_memory",
                            "update_topic",
                        ],
                        "expected_keywords": [],
                        "expected_process": {
                            "load_memory_index": True,
                            "evidence_required": False,
                            "final_reason_required": True,
                        },
                        "expected_memory_type": "none",
                        "expected_scope": "none",
                        "expected_relation": "no_memory_change",
                        "semantic_expectation": "一次性参数不沉淀。",
                        "should_extract": [],
                        "should_not_extract": ["一次性参数"],
                    }
                ),
                encoding="utf-8",
            )
            report_dir = tmp / "report"
            report_dir.mkdir()
            (report_dir / "eval-detail.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "tool": "eval_memory_replay",
                        "evalset_id": "test",
                        "evalset_path": str(evalset),
                        "run_id": "run_1",
                        "mode": "scenario-sequential",
                        "dry_run": False,
                        "scenarios": [
                            {
                                "scenario_id": "scenario_1",
                                "turns": [
                                    {
                                        "turn_index": 1,
                                        "turn_id": "turn_1",
                                        "action_counts": {"create_memory": 1},
                                        "changes": [
                                            {
                                                "action": "create_memory",
                                                "reason": "测试原因",
                                                "stable_key": "test.memory",
                                                "memory_type": "runtime_failure_pattern",
                                                "memory_summary": "一次性参数被误存",
                                                "memory_content": "一次性参数",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_eval_memory_review_existing(
                evalset_path=evalset,
                existing_report_directory=report_dir,
            )

            self.assertEqual(report["status"], "complete")
            self.assertTrue((report_dir / "quality-report-rejudged.md").is_file())
            quality = json.loads((report_dir / "quality-rejudged.json").read_text(encoding="utf-8"))
            self.assertGreater(quality["warning_categories"].get("hard_error", 0), 0)


if __name__ == "__main__":
    unittest.main()
