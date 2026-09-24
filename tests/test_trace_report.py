from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memory_lab.trace_report import (
    TRACE_ANNOTATION_TEMPLATE_SCHEMA_VERSION,
    TRACE_REPORT_SCHEMA_VERSION,
    build_annotation_template,
    render_trace_html,
    render_trace_markdown,
    write_trace_report,
)


class TraceReportTests(unittest.TestCase):
    def sample_report(self) -> dict:
        return {
            "schema_version": TRACE_REPORT_SCHEMA_VERSION,
            "status": "complete",
            "generated_at": "2026-09-21T00:00:00+00:00",
            "source_database": "C:/Users/test/.local/share/codepulse/opencode-dev.db",
            "lab_database": "D:/lab/data/memory_lab.db",
            "log_dir": "D:/lab/reports/poll-new-turns-live",
            "turn_count": 1,
            "summary": {
                "turn_count": 1,
                "status_counts": {"completed": 1},
                "action_counts": {"noop": 1},
                "quality_counts": {"✅ 合理跳过": 1},
                "suspicious_turn_count": 0,
            },
            "turns": [
                {
                    "turn_id": "turn_a",
                    "session_id": "ses_a",
                    "processing_status": "completed",
                    "dry_run": True,
                    "agent_run_id": "react_a",
                    "evidence_counts": {"core": 2, "candidate": 3, "pointer": 4},
                    "cp_context": {
                        "status": "complete",
                        "user_text": "用户选择基于现有工程二次开发",
                        "assistant_final_text": "确认：当前工程是已有 Keil C51 工程，应先确认硬件资源。",
                        "cp_tool_call_count": 3,
                    },
                    "react": {
                        "status": "complete",
                        "mode": "dry_run",
                        "iterations": [
                            {
                                "iteration": 1,
                                "decision": "tool_requests",
                                "repaired": False,
                                "requested_tools": [
                                    {"tool": "read_memory_detail", "memory_id": "mem_a"}
                                ],
                                "reason": "需要查看已有记忆正文避免重复沉淀。",
                            },
                            {
                                "iteration": 2,
                                "decision": "final_changes",
                                "repaired": True,
                                "requested_tools": [],
                                "reason": "已有记忆覆盖，最终不写入。",
                            },
                        ],
                        "changes": [
                            {
                                "action": "noop",
                                "reason": "已有记忆覆盖本轮信息，不重复沉淀。",
                                "evidence_refs": [],
                            }
                        ],
                        "memory_diffs": [
                            {
                                "mode": "dry_run",
                                "action": "merge_memory",
                                "target_memory_id": "mem_a",
                                "before_summary": "旧摘要",
                                "after_summary": "新摘要",
                            }
                        ],
                        "observations": [
                            {
                                "status": "complete",
                                "tool": "read_memory_detail",
                                "request": {
                                    "tool": "read_memory_detail",
                                    "parameters": {"memory_id": "mem_a"},
                                    "reason": "检查旧记忆是否已经覆盖本轮偏好。",
                                },
                                "result_status": "complete",
                                "result_summary": "memory_id=mem_a；summary=已有记忆覆盖。",
                            }
                        ],
                    },
                    "committed_revisions": [],
                    "quality_flags": [
                        {
                            "level": "ok",
                            "label": "✅ 合理跳过",
                            "reason": "已有记忆覆盖，不重复沉淀。",
                        }
                    ],
                }
            ],
        }

    def test_markdown_contains_human_trace_sections(self) -> None:
        markdown = render_trace_markdown(self.sample_report())

        self.assertIn("CP 交互到长期记忆链路报告", markdown)
        self.assertIn("## 总览", markdown)
        self.assertIn("## Turn 总览表", markdown)
        self.assertIn("用户选择基于现有工程二次开发", markdown)
        self.assertIn("core=2，candidate=3，pointer=4", markdown)
        self.assertIn("read_memory_detail", markdown)
        self.assertIn("工具调用明细", markdown)
        self.assertIn("CP 原始工具事件数", markdown)
        self.assertIn("已有记忆覆盖", markdown)
        self.assertIn("`noop`", markdown)
        self.assertIn("✅ 合理跳过", markdown)
        self.assertIn("Memory before / after", markdown)
        self.assertIn("计划变更，未写入", markdown)
        self.assertIn("旧摘要", markdown)
        self.assertIn("新摘要", markdown)

    def test_html_escapes_markdown_payload(self) -> None:
        report = self.sample_report()
        report["turns"][0]["cp_context"]["user_text"] = "<script>alert(1)</script>"

        html = render_trace_html(report)

        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_html_contains_visual_dashboard_sections(self) -> None:
        html = render_trace_html(self.sample_report())

        self.assertIn("trace-dashboard", html)
        self.assertIn("Turn 总览", html)
        self.assertIn("本轮重点判断", html)
        self.assertIn("CP 原始交互摘要", html)
        self.assertIn("ReAct 调用流转", html)
        self.assertIn("记忆 Agent 工具调用明细", html)
        self.assertIn("CP 原始工具事件数", html)
        self.assertIn("最终沉淀/计划", html)
        self.assertIn("Memory before / after", html)
        self.assertIn("memory_id&quot;: &quot;mem_a", html)
        self.assertIn("已有记忆覆盖", html)
        self.assertIn("旧摘要", html)
        self.assertIn("新摘要", html)

    def test_write_trace_report_outputs_latest_and_history_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_trace_report(report=self.sample_report(), output_dir=Path(tmp))

            for key in (
                "markdown",
                "html",
                "json",
                "annotation_template",
                "history_markdown",
                "history_html",
                "history_json",
                "history_annotation_template",
            ):
                self.assertTrue(Path(paths[key]).is_file(), key)
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], TRACE_REPORT_SCHEMA_VERSION)
            annotations = json.loads(
                Path(paths["annotation_template"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                annotations["schema_version"],
                TRACE_ANNOTATION_TEMPLATE_SCHEMA_VERSION,
            )
            self.assertEqual(annotations["turn_count"], 1)
            self.assertEqual(annotations["turns"][0]["turn_id"], "turn_a")
            self.assertEqual(annotations["turns"][0]["actual_actions"], ["noop"])
            self.assertEqual(annotations["turns"][0]["quality_labels"], ["✅ 合理跳过"])
            self.assertEqual(annotations["turns"][0]["expected_action"], "")
            self.assertEqual(annotations["turns"][0]["human_label"], "")
            self.assertEqual(annotations["turns"][0]["reviewer"], "")
            self.assertFalse(annotations["turns"][0]["needs_followup"])

    def test_annotation_template_is_human_fillable(self) -> None:
        annotations = build_annotation_template(self.sample_report())

        self.assertIn("expected_action_options", annotations["instructions"])
        self.assertIn("human_label_options", annotations["instructions"])
        self.assertEqual(annotations["turns"][0]["user_summary"], "用户选择基于现有工程二次开发")
        self.assertIn("noop", annotations["instructions"]["expected_action_options"])
        self.assertIn("证据不足", annotations["instructions"]["human_label_options"])


if __name__ == "__main__":
    unittest.main()
