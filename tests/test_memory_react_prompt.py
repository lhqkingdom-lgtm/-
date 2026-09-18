from __future__ import annotations

import unittest
from pathlib import Path

from memory_lab.memory_react_agent import REACT_SYSTEM_PROMPT, validate_react_decision


class MemoryReactPromptTests(unittest.TestCase):
    def test_prompt_documents_conflict_maintenance_triplet(self) -> None:
        self.assertIn("组合动作规则", REACT_SYSTEM_PROMPT)
        self.assertIn("update_memory：更新当前有效结论", REACT_SYSTEM_PROMPT)
        self.assertIn("update_topic：如果 topic 标题/概要仍是旧语义", REACT_SYSTEM_PROMPT)
        self.assertIn("record_conflict_decision：记录旧说法", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_search_tool_boundary(self) -> None:
        self.assertIn("search_related_memories", REACT_SYSTEM_PROMPT)
        self.assertIn("搜索结果只用于决定下一步读哪个 memory/topic 详情", REACT_SYSTEM_PROMPT)

    def test_validate_accepts_search_related_memories_request(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "当前摘要里没有看到旧记忆，需要搜索相关候选。",
            "tool_requests": [
                {
                    "tool": "search_related_memories",
                    "query": "Keil C51 编译错误",
                    "reason": "查找是否已有同类编译失败经验。",
                    "memory_type": "compile_failure_pattern",
                    "domain": "embedded",
                    "limit": 5,
                }
            ],
            "changes": [],
        }

        validated = validate_react_decision(
            decision,
            turn_id="turn_1",
            available_ids={
                "topic_ids": set(),
                "memory_ids": set(),
                "pointer_ids": set(),
            },
            pg_config_path=Path("unused.json"),
        )

        self.assertEqual(validated["tool_requests"][0]["tool"], "search_related_memories")

    def test_validate_rejects_search_with_bad_limit(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "需要搜索。",
            "tool_requests": [
                {
                    "tool": "search_related_memories",
                    "query": "回复语言",
                    "reason": "查找相关偏好。",
                    "limit": 99,
                }
            ],
            "changes": [],
        }

        with self.assertRaisesRegex(Exception, "limit must be between 1 and 20"):
            validate_react_decision(
                decision,
                turn_id="turn_1",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": set(),
                    "pointer_ids": set(),
                },
                pg_config_path=Path("unused.json"),
            )

    def test_validate_accepts_current_turn_evidence_summary_request(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "工具调用多轮后需要重新核对当前 turn 的证据。.",
            "tool_requests": [
                {
                    "tool": "get_current_turn_evidence_summary",
                    "reason": "回看当前 turn 的 evidence_id 和核心事实。",
                    "max_items_per_layer": 5,
                    "max_chars": 800,
                }
            ],
            "changes": [],
        }

        validated = validate_react_decision(
            decision,
            turn_id="turn_1",
            available_ids={
                "topic_ids": set(),
                "memory_ids": set(),
                "pointer_ids": set(),
            },
            pg_config_path=Path("unused.json"),
        )

        self.assertEqual(
            validated["tool_requests"][0]["tool"],
            "get_current_turn_evidence_summary",
        )

    def test_validate_rejects_current_turn_summary_bad_bounds(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "需要回看证据。",
            "tool_requests": [
                {
                    "tool": "get_current_turn_evidence_summary",
                    "reason": "回看当前 turn。",
                    "max_items_per_layer": 99,
                    "max_chars": 10,
                }
            ],
            "changes": [],
        }

        with self.assertRaisesRegex(Exception, "max_items_per_layer must be between 1 and 50"):
            validate_react_decision(
                decision,
                turn_id="turn_1",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": set(),
                    "pointer_ids": set(),
                },
                pg_config_path=Path("unused.json"),
            )


if __name__ == "__main__":
    unittest.main()
