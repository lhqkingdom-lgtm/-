from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.memory_react_agent import REACT_SYSTEM_PROMPT, validate_react_decision


class MemoryReactPromptTests(unittest.TestCase):
    def test_prompt_documents_conflict_maintenance_triplet(self) -> None:
        self.assertIn("组合动作规则", REACT_SYSTEM_PROMPT)
        self.assertIn("update_memory：更新当前有效结论", REACT_SYSTEM_PROMPT)
        self.assertIn("update_topic：如果 topic 标题/概要仍是旧语义", REACT_SYSTEM_PROMPT)
        self.assertIn("record_conflict_decision：记录旧说法", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_action_boundaries_and_status_matrix(self) -> None:
        self.assertIn("状态权限矩阵", REACT_SYSTEM_PROMPT)
        self.assertIn("conflict memory：只允许 record_conflict_decision", REACT_SYSTEM_PROMPT)
        self.assertIn("create_memory 不能修改已有 topic", REACT_SYSTEM_PROMPT)
        self.assertIn("update_topic 是唯一允许修改 topic.title/topic.summary 的动作", REACT_SYSTEM_PROMPT)
        self.assertIn("只表示“把当前 turn 证据合并到一个已有 memory”", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_move_memory_topic_boundary(self) -> None:
        self.assertIn("move_memory_topic：移动记忆所属主题", REACT_SYSTEM_PROMPT)
        self.assertIn("不能修改 memory summary/content/confidence", REACT_SYSTEM_PROMPT)
        self.assertIn("from_topic_id 必须匹配当前 memory 所属主题", REACT_SYSTEM_PROMPT)
        self.assertIn("move_memory_topic 不能包含 memory 或 topic 对象", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_merge_topics_boundary(self) -> None:
        self.assertIn("merge_topics：合并重复主题", REACT_SYSTEM_PROMPT)
        self.assertIn("不能修改任何 memory 的 summary/content/confidence", REACT_SYSTEM_PROMPT)
        self.assertIn("不能自动改 target topic 的 title/summary", REACT_SYSTEM_PROMPT)
        self.assertIn("merge_topics 不能包含 memory 或 topic 对象", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_immutable_keys_boundary(self) -> None:
        self.assertIn("stable_key 和 topic_key 是创建时的路由标识", REACT_SYSTEM_PROMPT)
        self.assertIn("不能修改 stable_key", REACT_SYSTEM_PROMPT)
        self.assertIn("不能修改 topic_key", REACT_SYSTEM_PROMPT)
        self.assertIn("不要在 reason 中把“修正 key”作为 update_memory/update_topic 的理由", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_search_tool_boundary(self) -> None:
        self.assertIn("search_related_memories", REACT_SYSTEM_PROMPT)
        self.assertIn("搜索结果只用于决定下一步读哪个 memory/topic 详情", REACT_SYSTEM_PROMPT)
        self.assertIn("trace_memory_relations", REACT_SYSTEM_PROMPT)
        self.assertIn("record_memory_relation", REACT_SYSTEM_PROMPT)
        self.assertIn("list_topic_memories", REACT_SYSTEM_PROMPT)
        self.assertIn("read_memory_history", REACT_SYSTEM_PROMPT)
        self.assertIn("list_memory_conflicts", REACT_SYSTEM_PROMPT)
        self.assertIn("不能因此输出 noop", REACT_SYSTEM_PROMPT)
        self.assertIn("不能把“目标 memory 未读取、暂时没有修改权限”当成 noop 理由", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_compact_merge_memory_policy(self) -> None:
        self.assertIn("merge_memory 不是重新写一篇报告", REACT_SYSTEM_PROMPT)
        self.assertIn("不要复制完整日志、完整执行流、完整错误列表或大段会话", REACT_SYSTEM_PROMPT)
        self.assertIn("新增的事实、验证状态、适用条件或反例", REACT_SYSTEM_PROMPT)
        self.assertIn("新证据必须能用目标 memory 的同一句核心结论解释", REACT_SYSTEM_PROMPT)
        self.assertIn("不要把用户习惯塞进工程流程记忆", REACT_SYSTEM_PROMPT)
        self.assertIn("如果目标 memory 已经很宽泛或内容已经较长", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_atomic_create_memory_policy(self) -> None:
        self.assertIn("新的独立规则", REACT_SYSTEM_PROMPT)
        self.assertIn("不要为了减少条目数强行 merge 到宽泛旧 memory", REACT_SYSTEM_PROMPT)
        self.assertIn("用户协作习惯、工程流程、编译失败经验、运行现象经验应尽量保持原子化", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_verification_status_policy(self) -> None:
        self.assertIn("验证状态写法", REACT_SYSTEM_PROMPT)
        self.assertIn("用户反馈现象”只能证明现象存在", REACT_SYSTEM_PROMPT)
        self.assertIn("助手定位/推理”只能证明候选根因", REACT_SYSTEM_PROMPT)
        self.assertIn("用户确认修复后正常", REACT_SYSTEM_PROMPT)
        self.assertIn("不要因为描述详细就自动提高置信度", REACT_SYSTEM_PROMPT)
        self.assertIn("不要写成“本轮已验证”", REACT_SYSTEM_PROMPT)
        self.assertIn("避免把旧验证状态投射到新现象上", REACT_SYSTEM_PROMPT)

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
                    "status": "needs_review",
                    "include_resolved": False,
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
                    "status": "resolved",
                    "include_resolved": "yes",
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

    def test_validate_requires_read_before_move_memory_topic(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "需要移动归错主题的记忆。",
            "tool_requests": [],
            "changes": {
                "schema_version": "memory-changes-v1",
                "turn_id": "turn_1",
                "reason": "memory 内容正确但主题错误。",
                "changes": [
                    {
                        "action": "move_memory_topic",
                        "reason": "移动到更合适主题。",
                        "target_memory_id": "mem_language",
                        "from_topic_id": "topic_language",
                        "to_topic_id": "topic_user_collaboration",
                        "evidence_refs": [{"evidence_id": "ev_1", "reason": "证据"}],
                    }
                ],
            },
        }

        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            return_value=decision["changes"],
        ):
            with self.assertRaisesRegex(Exception, "must be read by read_memory_detail"):
                validate_react_decision(
                    decision,
                    turn_id="turn_1",
                    available_ids={
                        "topic_ids": {"topic_language", "topic_user_collaboration"},
                        "memory_ids": {"mem_language"},
                        "pointer_ids": set(),
                    },
                    observed_memory_ids=set(),
                    pg_config_path=Path("unused.json"),
                )


if __name__ == "__main__":
    unittest.main()
