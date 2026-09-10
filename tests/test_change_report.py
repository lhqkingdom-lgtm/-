from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.change_report import (
    build_change_report,
    load_auto_memory_run_report,
    rebuild_auto_memory_change_report,
    render_html,
    render_markdown,
    write_change_report,
)
from memory_lab.formal_markdown import publish_formal_memory
from memory_lab.shadow import formal_state_snapshot
from memory_lab.store import open_store


class ChangeReportTests(unittest.TestCase):
    @staticmethod
    def _state(row_count: int = 0) -> dict:
        return {
            table: {"row_count": row_count, "content_sha256": "0" * 64}
            for table in ("memory", "memory_revision", "evidence", "conflict")
        }

    def test_report_contains_no_raw_evidence_or_secret(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "lab.db"
            connection = open_store(database)
            connection.execute(
                "INSERT INTO source VALUES ('s','sqlite','x','t',NULL,NULL)"
            )
            connection.execute(
                "INSERT INTO auto_memory_run(run_id,pipeline_version,provider,model,writer_prompt_version,reviewer_prompt_version,writer_schema_version,reviewer_schema_version,started_at,status,formal_state_before_json) VALUES ('r','v','p','m','w','q','ws','rs','t','complete','{}')"
            )
            connection.commit()
            connection.close()
            report = build_change_report(
                database,
                {
                    "run_id": "r",
                    "status": "complete",
                    "formal_state_before": {},
                    "formal_state_after": {},
                    "actions": [],
                },
            )
            paths = write_change_report(root / "reports", report)
            payload = Path(paths["json"]).read_text(encoding="utf-8")
            self.assertEqual(json.loads(payload)["schema_version"], "memory-change-report-v1")
            self.assertTrue(Path(paths["html"]).is_file())
            self.assertTrue(Path(paths["history_markdown"]).is_file())
            self.assertTrue(Path(paths["history_html"]).is_file())
            self.assertTrue(Path(paths["history_json"]).is_file())
            self.assertEqual(
                Path(paths["history_json"]).read_text(encoding="utf-8"), payload
            )
            self.assertNotIn("exact_quote", payload)

    def test_report_accepts_prevalidated_unified_review_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            database = Path(raw) / "lab.db"
            open_store(database).close()
            report = build_change_report(
                database,
                {
                    "run_id": "unified-run",
                    "status": "complete",
                    "formal_state_before": self._state(0),
                    "formal_state_after": self._state(1),
                    "formal_state_changed": True,
                    "source_summary": {
                        "turns_processed": 2,
                        "projects": [
                            {"project_key": "project-a", "display_name": "项目甲"},
                            {"project_key": "project-b", "display_name": "项目乙"},
                        ],
                        "user_messages": 2,
                        "assistant_finals": 2,
                        "tool_events": 7,
                        "candidates": 1,
                        "accepted": 1,
                        "rejected": 0,
                    },
                    "review": [
                        {
                            "stable_key": "engineering.serial.retry",
                            "operation": "create",
                            "outcome": "accept",
                            "review_reason": "verified against cited turn evidence",
                            "statement": "串口超时后重试一次可恢复通信。",
                        }
                    ],
                    "actions": [],
                },
            )
            self.assertEqual(report["review"][0]["stable_key"], "engineering.serial.retry")
            rendered = render_markdown(report)
            self.assertIn("新建记忆", rendered)
            self.assertIn("串口超时后重试一次可恢复通信", rendered)
            self.assertIn("本轮会话采集", rendered)
            self.assertIn("项目甲、项目乙", rendered)
            rendered_html = render_html(report)
            self.assertIn("本轮会话采集", rendered_html)
            self.assertIn("项目甲、项目乙", rendered_html)

    def test_unsafe_run_id_cannot_escape_report_history(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "unsafe"):
                write_change_report(
                    Path(raw),
                    {
                        "run_id": "../outside",
                        "status": "complete",
                        "before": {},
                        "after": {},
                        "delta": {},
                    },
                )

    def test_human_reports_use_chinese_labels_and_normalized_reasons(self):
        counters = ("memories", "revisions", "evidence", "conflicts")
        report = {
            "run_id": "run-zh",
            "status": "complete_with_rejections",
            "formal_state_changed": True,
            "before": {key: 0 for key in counters},
            "after": {key: 1 for key in counters},
            "delta": {key: 1 for key in counters},
            "review": [
                {
                    "stable_key": "preference.response.structure",
                    "operation": "create_entry",
                    "outcome": "accept",
                    "review_reason": "The plan is safe and correct.",
                    "statement": "回复时先给结论。",
                },
                {
                    "stable_key": "—",
                    "operation": "reject",
                    "outcome": "reject",
                    "review_reason": "local_hard_filter:temporary",
                    "statement": "—",
                },
            ],
            "actions": [
                {
                    "action": "created",
                    "memory_id": "memory-1",
                    "after_revision_id": "revision-1",
                    "reason": "approved_proposal_activated",
                }
            ],
            "markdown": {
                "status": "validated_current_matches_run",
                "generation_id": "gen-1",
                "current_changed": False,
                "matches_run_state": True,
                "generation_matches_current_database": True,
                "topic_count": 1,
                "memory_count": 1,
            },
        }
        for payload in (render_markdown(report), render_html(report)):
            self.assertIn("长期记忆变更报告", payload)
            self.assertIn("审核通过", payload)
            self.assertIn("临时要求，不沉淀为长期记忆", payload)
            self.assertIn("审核通过，候选已激活为正式记忆", payload)
            self.assertNotIn("Latest Memory Change", payload)
            self.assertNotIn("The plan is safe and correct", payload)

    def test_existing_auto_memory_run_reconstructs_persisted_action(self):
        with tempfile.TemporaryDirectory() as raw:
            database = Path(raw) / "lab.db"
            open_store(database).close()
            before = self._state(0)
            after = self._state(1)
            plan = {
                "operation": "create_entry",
                "stable_key": "preference.response.structure",
                "value": {"statement": "回复时先给结论，再说明步骤。"},
            }
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO auto_memory_run(
                        run_id,pipeline_version,provider,model,writer_prompt_version,
                        reviewer_prompt_version,writer_schema_version,
                        reviewer_schema_version,started_at,completed_at,status,
                        candidate_count,accepted_count,counts_json,
                        formal_state_before_json,formal_state_after_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "run-action", "v", "p", "m", "w", "r", "ws", "rs",
                        "t", "t", "complete", 1, 1,
                        json.dumps({"actions": {"created": 1}}),
                        json.dumps(before), json.dumps(after),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memory_writer_result(
                        writer_result_id,run_id,admission_run_id,decision_id,
                        candidate_key,request_hash,plan_hash,provider,model,
                        prompt_version,schema_version,status,execution_mode,
                        plan_json,created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "writer", "run-action", "admission", "decision", "candidate",
                        "request", "plan", "p", "m", "w", "ws", "valid",
                        "writer_call", json.dumps(plan, ensure_ascii=False), "t",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memory_reviewer_result(
                        reviewer_result_id,run_id,writer_result_id,admission_run_id,
                        decision_id,candidate_key,request_hash,reviewed_plan_hash,
                        final_plan_hash,provider,model,prompt_version,schema_version,
                        status,execution_mode,outcome,decision_json,final_plan_json,
                        created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "reviewer", "run-action", "writer", "admission", "decision",
                        "candidate", "review-request", "plan", "plan", "p", "m",
                        "r", "rs", "valid", "reviewer_call", "accept",
                        json.dumps({"reason": "independently checked"}),
                        json.dumps(plan, ensure_ascii=False), "t",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO candidate_review(
                        review_id,proposal_id,decision_id,candidate_key,outcome,
                        reviewer,note,reviewed_at
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        "approval", "proposal", "decision", "candidate", "approve",
                        "ai-reviewer:p:m",
                        json.dumps({"auto_memory_run_id": "run-action"}), "t",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO consolidation_run(
                        run_id,consolidator_version,started_at,completed_at,status,
                        candidate_count,counts_json
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    ("consolidation", "v", "t", "t", "complete", 1, '{}'),
                )
                connection.execute(
                    """
                    INSERT INTO consolidation_action(
                        action_id,run_id,candidate_key,decision_id,proposal_id,
                        review_id,reviewer,action,memory_id,after_revision_id,
                        reason,details_json,created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "action", "consolidation", "candidate", "decision", "proposal",
                        "approval", "ai-reviewer:p:m", "created", "memory",
                        "revision", "approved_proposal_activated", '{}', "t",
                    ),
                )

            pipeline = load_auto_memory_run_report(database, "run-action")
            self.assertEqual(pipeline["actions"][0]["action"], "created")
            self.assertEqual(pipeline["actions"][0]["memory_id"], "memory")
            report = build_change_report(database, pipeline)
            self.assertEqual(report["delta"]["memories"], 1)
            self.assertEqual(report["review"][0]["outcome"], "accept")
            self.assertEqual(report["actions"][0]["after_revision_id"], "revision")

    def test_rebuild_writes_three_formats_and_validates_matching_generation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            database = root / "lab.db"
            with closing(open_store(database)) as connection, connection:
                snapshot = formal_state_snapshot(connection)
                connection.execute(
                    """
                    INSERT INTO auto_memory_run(
                        run_id,pipeline_version,provider,model,writer_prompt_version,
                        reviewer_prompt_version,writer_schema_version,
                        reviewer_schema_version,started_at,completed_at,status,
                        counts_json,formal_state_before_json,formal_state_after_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "run-empty", "v", "p", "m", "w", "r", "ws", "rs",
                        "t", "t", "complete", '{}', json.dumps(snapshot),
                        json.dumps(snapshot),
                    ),
                )
            publication = publish_formal_memory(database, root / "memory")
            result = rebuild_auto_memory_change_report(
                database,
                "run-empty",
                root / "reports",
                memory_root=root / "memory",
            )
            self.assertTrue(result["markdown_matches_run_state"])
            self.assertEqual(
                result["change_report"]["markdown"]["generation_id"],
                publication["generation_id"],
            )
            for name in ("markdown", "html", "json"):
                self.assertTrue(Path(result["report_files"][name]).is_file())


if __name__ == "__main__":
    unittest.main()
