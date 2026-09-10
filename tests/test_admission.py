from __future__ import annotations

import unittest
import json
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

from memory_lab.admission import (
    AtomicClaim,
    _decision_for_claim,
    run_rule_admission,
    split_atomic_claims,
)
from memory_lab.ingest import run_ingestion
from memory_lab.instructions import run_instruction_coverage
from memory_lab.prepare import prepare_dataset
from memory_lab.store import open_store


class AdmissionRuleTests(unittest.TestCase):
    def test_atomic_split_preserves_exact_offsets(self) -> None:
        content = " 以后默认中文；\n生成代码前先确认。 "

        claims = split_atomic_claims(content)

        self.assertEqual(
            [item.exact_quote for item in claims],
            ["以后默认中文", "生成代码前先确认。"],
        )
        self.assertTrue(all(content[item.start_offset : item.end_offset] == item.exact_quote for item in claims))

    def test_temporary_language_wins_over_negative_workflow_wording(self) -> None:
        event = {
            "event_id": "evt_1",
            "session_id": "ses_1",
            "project_key": "project_1",
            "directory": r"D:\project-a",
            "eligibility_status": "eligible",
        }
        claim = AtomicClaim(1, "这一次只解释方案，不要生成代码。", 0, 17)

        decision = _decision_for_claim(event, claim, [])

        self.assertEqual(decision["disposition"], "temporary")

    def test_response_ordering_is_a_global_preference_with_partial_project_coverage(self) -> None:
        event = {
            "event_id": "evt_response_structure",
            "session_id": "ses_response_structure",
            "project_key": "project_response_structure",
            "directory": r"D:\project-response-structure",
            "eligibility_status": "eligible",
        }
        text = "以后无论在哪个项目，回复我时先用一句话给出结论，再说明处理步骤。"
        source = f"  {text}  "
        claim = split_atomic_claims(source)[0]
        clauses = [
            {
                "clause_id": "clause_response_structure",
                "heading_path": "回复要求",
                "line_start": 3,
                "exact_text": "回复我时先用一句话给出结论，再说明处理步骤",
                "canonical_path": r"D:\project-response-structure\AGENTS.md",
                "effective_scope_kind": "project",
            }
        ]

        decision = _decision_for_claim(event, claim, clauses)

        self.assertEqual(decision["exact_quote"], text)
        self.assertEqual(source[decision["start_offset"] : decision["end_offset"]], text)
        self.assertEqual(decision["family"], "user_contract")
        self.assertEqual(decision["memory_type"], "preference")
        self.assertEqual(
            decision["stable_key_suggestion"], "preference.response.structure"
        )
        self.assertEqual(
            decision["activation_triggers"], ["回复", "结论优先", "处理步骤"]
        )
        self.assertEqual(
            decision["scope"],
            {"level": "global", "override_policy": "more_specific_scope_wins"},
        )
        self.assertEqual(decision["disposition"], "eligible")
        self.assertEqual(decision["instruction_coverage"]["status"], "partial")
        self.assertIn("only_partially_covered_by_project_instruction", decision["reasons"])

    def test_temporary_response_ordering_wins_over_supported_identity(self) -> None:
        event = {
            "event_id": "evt_temporary_response_structure",
            "session_id": "ses_temporary_response_structure",
            "project_key": "project_temporary_response_structure",
            "directory": r"D:\project",
            "eligibility_status": "eligible",
        }
        text = "这一次回复时先给出结论，再列出处理步骤。"
        claim = AtomicClaim(1, text, 0, len(text))

        decision = _decision_for_claim(event, claim, [])

        self.assertEqual(
            decision["stable_key_suggestion"], "preference.response.structure"
        )
        self.assertEqual(decision["disposition"], "temporary")

    def test_response_ordering_match_requires_a_conclusion_stage(self) -> None:
        event = {
            "event_id": "evt_other_workflow",
            "session_id": "ses_other_workflow",
            "project_key": "project_other_workflow",
            "directory": r"D:\project",
            "eligibility_status": "eligible",
        }
        text = "以后回复时先核对用户给出的结论，再按处理步骤排查问题。"
        claim = AtomicClaim(1, text, 0, len(text))

        decision = _decision_for_claim(event, claim, [])

        self.assertIsNone(decision["stable_key_suggestion"])
        self.assertEqual(decision["disposition"], "needs_semantic_review")

    def test_partial_project_instruction_does_not_remove_domain_claim(self) -> None:
        event = {
            "event_id": "evt_2",
            "session_id": "ses_2",
            "project_key": "project_2",
            "directory": r"D:\codepulsework",
            "eligibility_status": "eligible",
        }
        text = "以后在所有单片机项目中，修改代码后都必须进行编译检查。"
        claim = AtomicClaim(1, text, 0, len(text))
        clauses = [
            {
                "clause_id": "clause_1",
                "heading_path": "项目规则 > 单片机项目",
                "line_start": 5,
                "exact_text": "修改代码后必须进行编译检查（Keil/Makefile/IAR/CMake 等项目均适用）",
                "canonical_path": r"D:\codepulsework\AGENTS.md",
                "effective_scope_kind": "project",
            }
        ]

        decision = _decision_for_claim(event, claim, clauses)

        self.assertEqual(decision["disposition"], "eligible")
        self.assertEqual(decision["instruction_coverage"]["status"], "partial")
        self.assertIn("only_partially_covered_by_project_instruction", decision["reasons"])

    def test_named_current_project_without_local_support_requires_review(self) -> None:
        event = {
            "event_id": "evt_3",
            "session_id": "ses_3",
            "project_key": "project_3",
            "directory": r"D:\unrelated-window",
            "eligibility_status": "eligible",
        }
        text = "仅在当前 TG032Plus 项目中，编译统一使用 Keil C51。"
        claim = AtomicClaim(1, text, 0, len(text))

        decision = _decision_for_claim(event, claim, [])

        self.assertEqual(decision["disposition"], "needs_scope_review")
        self.assertEqual(decision["scope"]["level"], "unresolved")
        self.assertEqual(decision["scope"]["claimed_project"], "TG032Plus")
        self.assertEqual(decision["scope"]["observed_project_key"], "project_3")

    def test_project_alias_does_not_use_global_rule_or_substring(self) -> None:
        event = {
            "event_id": "evt_4",
            "session_id": "ses_4",
            "project_key": "project_4",
            "directory": r"D:\TG032Plus-old",
            "eligibility_status": "eligible",
        }
        text = "仅在当前 TG032Plus 项目中，编译统一使用 Keil C51。"
        claim = AtomicClaim(1, text, 0, len(text))
        global_clause = {
            "clause_id": "global_1",
            "heading_path": "参考项目",
            "line_start": 1,
            "exact_text": "TG032Plus 项目编译统一使用 Keil C51",
            "canonical_path": r"C:\global\AGENTS.md",
            "effective_scope_kind": "global",
        }

        decision = _decision_for_claim(event, claim, [global_clause])

        self.assertEqual(decision["disposition"], "needs_scope_review")
        self.assertEqual(decision["scope"]["level"], "unresolved")

    def test_explicit_correction_is_not_dropped_without_generic_long_term_word(self) -> None:
        event = {
            "event_id": "evt_5",
            "session_id": "ses_5",
            "project_key": "project_5",
            "directory": r"D:\project",
            "eligibility_status": "eligible",
        }
        text = "更正：生成正式代码不需要问我。"
        claim = AtomicClaim(1, text, 0, len(text))

        decision = _decision_for_claim(event, claim, [])

        self.assertEqual(decision["disposition"], "needs_correction_review")
        self.assertEqual(
            decision["stable_key_suggestion"], "workflow.code_generation.confirmation"
        )


class AdmissionIntegrationTests(unittest.TestCase):
    def test_only_completed_root_turns_enter_admission_snapshot(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = Path(temporary.name) / "memory_lab.db"
        quote = "以后默认使用中文回复。"
        with closing(open_store(store)) as connection, connection:
            connection.execute(
                "INSERT INTO source VALUES ('source', 'sqlite', 'source.db', 'now', NULL, 1)"
            )
            connection.execute(
                "INSERT INTO project_context VALUES ('project', 'D:\\project', 'project', 'test', 'now')"
            )
            connection.execute(
                "INSERT INTO session_context VALUES ('source', 'session', 'project', 'D:\\project', 'test', 'now', 'now')"
            )
            connection.execute(
                """
                INSERT INTO event(
                    event_id, source_id, session_id, message_id, part_id, event_type,
                    timestamp, content, content_hash, source_kind, eligibility_status,
                    ingested_at, raw_json
                ) VALUES (
                    'event', 'source', 'session', 'message', 'part', 'user_text',
                    1, ?, 'hash', 'sqlite', 'eligible', 'now', '{}'
                )
                """,
                (quote,),
            )
            connection.execute(
                "INSERT INTO event_annotation VALUES ('event', NULL, ?, 'project', 'test', 'now')",
                (quote,),
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at, completed_at,
                    settle_ms, status
                ) VALUES ('turn_run', 'source', 'test', 'now', 'now', 0, 'complete')
                """
            )
            connection.execute(
                """
                INSERT INTO conversation_turn(
                    turn_id, source_id, session_id, user_message_id,
                    primary_event_id, current_state, first_detected_at,
                    last_detected_at, last_detection_run_id,
                    last_structural_signature
                ) VALUES (
                    'turn', 'source', 'session', 'message', 'event', 'pending',
                    'now', 'now', 'turn_run', 'signature'
                )
                """
            )
            connection.execute(
                "INSERT INTO turn_event VALUES ('turn', 'event')"
            )

        pending = run_rule_admission(store)
        self.assertEqual(pending["decision_count"], 0)
        self.assertEqual(
            pending["turn_gate"],
            {
                "eligible_root_events": 1,
                "completed_events_admitted": 0,
                "excluded_without_completed_turn": 1,
            },
        )

        with closing(open_store(store)) as connection, connection:
            connection.execute(
                "UPDATE conversation_turn SET current_state = 'completed' WHERE turn_id = 'turn'"
            )
        completed = run_rule_admission(store)
        self.assertEqual(completed["decision_count"], 1)
        self.assertEqual(completed["turn_gate"]["excluded_without_completed_turn"], 0)

    def test_seven_case_gold_set_is_classified_per_atomic_claim(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "source.db"
        store = root / "memory_lab.db"
        project_a = root / "project-a"
        project_b = root / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        (project_b / "AGENTS.md").write_text(
            """# 项目规则

## 单片机项目

- 修改代码后必须进行编译检查（Keil/Makefile/IAR/CMake 等项目均适用）
- TG032Plus 项目编译统一使用 Keil C51
- 回复我时先用一句话给出结论，再说明处理步骤
""",
            encoding="utf-8",
        )
        with closing(sqlite3.connect(source)) as connection, connection:
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
                """
            )
            for session_id, directory in (
                ("ses_a", project_a),
                ("ses_b", project_a),
                ("ses_p2", project_b),
            ):
                connection.execute(
                    "INSERT INTO session VALUES (?, 'global', NULL, ?, 'test', '1.0', 1)",
                    (session_id, str(directory)),
                )

        run_ingestion(source, store, "baseline")
        time.sleep(0.005)
        timestamp = int(time.time() * 1000)
        cases = (
            ("A1", "ses_a", "【ML-S1-A1】以后默认使用中文回复。"),
            ("A2", "ses_a", "【ML-S1-A2】生成正式代码前先向我确认。"),
            ("B1", "ses_b", "【ML-S1-B1】我再确认一次，生成正式代码之前需要先问我。"),
            ("B2", "ses_b", "【ML-S1-B2】这一次只解释方案，不要生成代码。"),
            (
                "P201",
                "ses_p2",
                "【ML-S1-P2-01】以后在所有单片机项目中，修改代码后都必须进行编译检查。",
            ),
            (
                "P202",
                "ses_p2",
                "【ML-S1-P2-02】仅在当前 TG032Plus 项目中，编译统一使用 Keil C51。",
            ),
            (
                "U1",
                "ses_p2",
                "【ML-S1-U1】以后无论在哪个项目，回复我时先用一句话给出结论，再说明处理步骤。",
            ),
        )
        with closing(sqlite3.connect(source)) as connection, connection:
            for offset, (suffix, session_id, text) in enumerate(cases):
                message_id = "msg_" + suffix
                connection.execute(
                    "INSERT INTO message VALUES (?, ?, ?, ?)",
                    (
                        message_id,
                        session_id,
                        timestamp + offset,
                        json.dumps({"role": "user", "agent": "build"}),
                    ),
                )
                connection.execute(
                    "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                    (
                        "prt_" + suffix,
                        message_id,
                        session_id,
                        timestamp + offset,
                        json.dumps({"type": "text", "text": text}, ensure_ascii=False),
                    ),
                )

        run_ingestion(source, store, "ingest")
        prepare_dataset(store)
        with closing(open_store(store)) as connection, connection:
            source_identifier = connection.execute(
                "SELECT source_id FROM source"
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at, completed_at,
                    settle_ms, status
                ) VALUES (
                    'turn_run', ?, 'test', '2026-01-01T00:00:00Z',
                    '2026-01-01T00:00:01Z', 0, 'complete'
                )
                """,
                (source_identifier,),
            )
            events = connection.execute(
                """
                SELECT event_id, source_id, session_id, message_id, timestamp
                FROM event ORDER BY event_id
                """
            ).fetchall()
            for index, event in enumerate(events, 1):
                turn_id = f"turn_{index}"
                connection.execute(
                    """
                    INSERT INTO conversation_turn(
                        turn_id, source_id, session_id, user_message_id,
                        primary_event_id, user_timestamp, current_state,
                        first_detected_at, last_detected_at,
                        last_detection_run_id, last_structural_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, 'completed',
                              '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z',
                              'turn_run', ?)
                    """,
                    (
                        turn_id,
                        event["source_id"],
                        event["session_id"],
                        event["message_id"],
                        event["event_id"],
                        event["timestamp"],
                        f"signature_{index}",
                    ),
                )
                connection.execute(
                    "INSERT INTO turn_event(turn_id, event_id) VALUES (?, ?)",
                    (turn_id, event["event_id"]),
                )
        run_instruction_coverage(store, root / "empty-home")
        report = run_rule_admission(store)
        by_case = {item["case_id"]: item for item in report["decisions"]}

        self.assertEqual(report["decision_count"], 7)
        self.assertEqual(
            report["counts"],
            {"covered_by_cp_instructions": 1, "eligible": 5, "temporary": 1},
        )
        self.assertEqual(report["controlled_gold_subset"], {
            "decision_count": 7,
            "counts": {"covered_by_cp_instructions": 1, "eligible": 5, "temporary": 1},
        })
        self.assertEqual(by_case["ML-S1-A1"]["disposition"], "eligible")
        self.assertEqual(by_case["ML-S1-A2"]["disposition"], "eligible")
        self.assertEqual(by_case["ML-S1-B1"]["disposition"], "eligible")
        self.assertEqual(
            by_case["ML-S1-A2"]["stable_key_suggestion"],
            by_case["ML-S1-B1"]["stable_key_suggestion"],
        )
        self.assertEqual(by_case["ML-S1-B2"]["disposition"], "temporary")
        self.assertEqual(by_case["ML-S1-P2-01"]["disposition"], "eligible")
        self.assertEqual(
            by_case["ML-S1-P2-01"]["instruction_coverage"]["status"], "partial"
        )
        self.assertEqual(
            by_case["ML-S1-P2-02"]["disposition"], "covered_by_cp_instructions"
        )
        self.assertEqual(by_case["ML-S1-U1"]["disposition"], "eligible")
        self.assertEqual(by_case["ML-S1-U1"]["memory_type"], "preference")
        self.assertEqual(
            by_case["ML-S1-U1"]["stable_key_suggestion"],
            "preference.response.structure",
        )
        self.assertEqual(by_case["ML-S1-U1"]["scope"]["level"], "global")
        self.assertEqual(
            by_case["ML-S1-U1"]["instruction_coverage"]["status"], "partial"
        )


if __name__ == "__main__":
    unittest.main()
