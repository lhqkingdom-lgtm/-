from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.pg_markdown_export import export_pg_markdown


class PgMarkdownExportTests(unittest.TestCase):
    def test_topic_file_includes_conflict_records(self) -> None:
        state = {
            "user_id": "local-default-user",
            "topics": [
                {
                    "topic_id": "topic_language",
                    "topic_key": "user_preference_reply_language",
                    "title": "回复语言偏好",
                    "summary": "用户当前默认英文回复。",
                    "topic_type": "user_preference",
                    "scope": {"level": "domain", "domain": "general"},
                    "status": "active",
                    "updated_at": "2026-09-18T00:00:00Z",
                    "memories": [
                        {
                            "memory_id": "mem_language",
                            "stable_key": "general.user_preference.reply_language",
                            "memory_type": "user_preference",
                            "scope": {"level": "domain", "domain": "general"},
                            "summary": "用户要求默认英文回复。",
                            "content": "用户要求以后在所有对话中统一使用英文回复。",
                            "status": "active",
                            "confidence": 0.95,
                            "metadata": {
                                "evidence_refs": [{"evidence_id": "ev_user"}],
                                "source_turn_ids": ["turn_language"],
                            },
                        }
                    ],
                    "conflicts": [
                        {
                            "conflict_id": "conflict_language",
                            "memory_id": "mem_language",
                            "turn_id": "turn_language",
                            "conflict_type": "user_preference_changed",
                            "old_claim": "用户希望默认中文回复。",
                            "new_claim": "用户希望默认英文回复。",
                            "decision": "采用新的明确偏好。",
                            "reason": "用户明确纠正。",
                            "evidence_refs": [{"evidence_id": "ev_user"}],
                            "status": "resolved",
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "memory_lab.pg_markdown_export._load_state",
            return_value=state,
        ):
            output_dir = Path(tmp)
            report = export_pg_markdown(output_dir=output_dir)
            topic_path = output_dir / "user_preference_reply_language.md"
            text = topic_path.read_text(encoding="utf-8")

        self.assertTrue(report["ok"])
        self.assertIn("## 冲突记录", text)
        self.assertIn("旧说法：用户希望默认中文回复。", text)
        self.assertIn("新说法：用户希望默认英文回复。", text)
        self.assertIn("处理决定：采用新的明确偏好。", text)


if __name__ == "__main__":
    unittest.main()
