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
        self.assertIn("trace_memory_relations", REACT_SYSTEM_PROMPT)
        self.assertIn("record_memory_relation", REACT_SYSTEM_PROMPT)
        self.assertIn("list_topic_memories", REACT_SYSTEM_PROMPT)
        self.assertIn("read_memory_history", REACT_SYSTEM_PROMPT)
        self.assertIn("list_memory_conflicts", REACT_SYSTEM_PROMPT)

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

    def test_validate_accepts_trace_memory_relations_request(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "需要查看旧记忆是否已经被替代。",
            "tool_requests": [
                {
                    "tool": "trace_memory_relations",
                    "memory_id": "mem_language",
                    "reason": "判断这条记忆是否已有去向或冲突关系。",
                    "direction": "both",
                    "max_depth": 2,
                }
            ],
            "changes": [],
        }

        validated = validate_react_decision(
            decision,
            turn_id="turn_1",
            available_ids={
                "topic_ids": set(),
                "memory_ids": {"mem_language"},
                "pointer_ids": set(),
            },
            pg_config_path=Path("unused.json"),
        )

        self.assertEqual(validated["tool_requests"][0]["tool"], "trace_memory_relations")

    def test_validate_rejects_trace_memory_relations_bad_target(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "需要查看关系链。",
            "tool_requests": [
                {
                    "tool": "trace_memory_relations",
                    "memory_id": "mem_missing",
                    "reason": "目标不存在。",
                    "direction": "sideways",
                    "max_depth": 99,
                }
            ],
            "changes": [],
        }

        with self.assertRaisesRegex(Exception, "memory_id does not exist"):
            validate_react_decision(
                decision,
                turn_id="turn_1",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": {"mem_language"},
                    "pointer_ids": set(),
                },
                pg_config_path=Path("unused.json"),
            )

    def test_validate_accepts_new_memory_management_read_tools(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "需要看主题下记忆、记忆历史和已有冲突。",
            "tool_requests": [
                {
                    "tool": "list_topic_memories",
                    "topic_id": "topic_language",
                    "reason": "查看主题下是否已有同类偏好。",
                    "limit": 10,
                },
                {
                    "tool": "read_memory_history",
                    "memory_id": "mem_language",
                    "reason": "查看这条偏好是否已被多次修正。",
                    "limit": 5,
                },
                {
                    "tool": "list_memory_conflicts",
                    "memory_id": "mem_language",
                    "reason": "查看是否已有同类语言偏好冲突记录。",
                    "limit": 5,
                },
            ],
            "changes": [],
        }

        validated = validate_react_decision(
            decision,
            turn_id="turn_1",
            available_ids={
                "topic_ids": {"topic_language"},
                "memory_ids": {"mem_language"},
                "pointer_ids": set(),
            },
            pg_config_path=Path("unused.json"),
        )

        self.assertEqual(
            [item["tool"] for item in validated["tool_requests"]],
            ["list_topic_memories", "read_memory_history", "list_memory_conflicts"],
        )

    def test_validate_rejects_bad_new_memory_management_read_tool_args(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "错误参数。",
            "tool_requests": [
                {
                    "tool": "list_memory_conflicts",
                    "memory_id": "mem_language",
                    "topic_id": "topic_language",
                    "reason": "不能同时给两个目标。",
                    "limit": 99,
                }
            ],
            "changes": [],
        }

        with self.assertRaisesRegex(Exception, "exactly one of memory_id/topic_id"):
            validate_react_decision(
                decision,
                turn_id="turn_1",
                available_ids={
                    "topic_ids": {"topic_language"},
                    "memory_ids": {"mem_language"},
                    "pointer_ids": set(),
                },
                pg_config_path=Path("unused.json"),
            )


if __name__ == "__main__":
    unittest.main()
