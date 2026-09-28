from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from memory_lab.memory_react_agent import (
    REACT_REPAIR_PROMPT,
    REACT_SYSTEM_PROMPT,
    _auto_required_chain_reads,
    _update_turn_access_ids,
    run_memory_react_agent,
    validate_react_decision,
)


class SequenceClient:
    def __init__(self, outputs: list[dict]):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class MemoryReactPromptTests(unittest.TestCase):
    @staticmethod
    def _minimal_context(turn_id: str = "turn_current") -> dict:
        return {
            "status": "complete",
            "turn_id": turn_id,
            "turn_evidence": {
                "turn": {"turn_id": turn_id},
                "layers": {"core": [], "candidate": [], "pointer": []},
            },
            "topic_index": {"topics": []},
            "memory_index": {"topics": [], "memories": []},
            "filters": {"domain": "embedded", "project_key": "project_1"},
        }

    @staticmethod
    def _open_chain_scan_request() -> dict:
        return {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "技术类最终判断前先检查同会话未闭环故障链。",
            "tool_requests": [
                {
                    "tool": "list_open_memory_chains",
                    "reason": "确认本轮是继续旧链还是创建新链。",
                    "limit": 10,
                }
            ],
            "changes": [],
        }

    def test_commit_mode_persists_open_chain_after_noop(self) -> None:
        output = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "修复后还缺用户上板确认。",
            "chain_state": {
                "chain_type": "runtime",
                "state": "waiting_user_validation",
                "related_turn_ids": ["turn_current"],
                "missing_evidence": ["用户重新烧录后的运行确认"],
            },
            "tool_requests": [],
            "changes": {
                "schema_version": "memory-changes-v1",
                "turn_id": "turn_current",
                "reason": "未闭环只保存状态。",
                "changes": [{"action": "noop", "reason": "等待验证。", "evidence_refs": []}],
            },
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "memory_lab.memory_react_agent.build_memory_agent_input",
            return_value=self._minimal_context(),
        ), patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            side_effect=lambda changes, **_kwargs: changes,
        ), patch(
            "memory_lab.memory_react_agent.commit_memory_changes",
            return_value={"ok": True, "counts": {"applied": 0, "rejected": 0}, "applied": []},
        ), patch(
            "memory_lab.memory_react_agent.list_open_memory_chains",
            return_value={
                "status": "complete",
                "tool": "list_open_memory_chains",
                "current_turn_id": "turn_current",
                "chains": [],
            },
        ), patch(
            "memory_lab.memory_react_agent.save_memory_chain_state",
            return_value={
                "status": "complete",
                "result": "created",
                "chain": {"chain_id": "chain_1"},
            },
        ) as save_chain:
            result = run_memory_react_agent(
                source_database=Path("source.db"),
                turn_id="turn_current",
                pg_config_path=Path("unused.json"),
                dry_run=False,
                max_iterations=2,
                log_dir=Path(directory),
                client=SequenceClient([self._open_chain_scan_request(), output]),
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["chain_persistence"]["result"], "created")
        save_chain.assert_called_once()

    def test_closed_chain_is_linked_to_committed_memory(self) -> None:
        plan = {
            "schema_version": "memory-changes-v1",
            "turn_id": "turn_current",
            "reason": "用户确认上板正常。",
            "changes": [
                {
                    "action": "create_memory",
                    "technical_closure": {
                        "chain_type": "runtime",
                        "state": "user_verified",
                        "verification_evidence_ids": ["ev_user"],
                    },
                }
            ],
        }
        output = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "运行故障已闭环。",
            "chain_state": {
                "chain_type": "runtime",
                "state": "user_verified",
                "related_turn_ids": ["turn_current"],
                "missing_evidence": [],
            },
            "tool_requests": [],
            "changes": plan,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "memory_lab.memory_react_agent.build_memory_agent_input",
            return_value=self._minimal_context(),
        ), patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            return_value=plan,
        ), patch(
            "memory_lab.memory_react_agent.commit_memory_changes",
            return_value={
                "ok": True,
                "counts": {"applied": 1, "rejected": 0},
                "applied": [{"action": "create_memory", "memory_id": "mem_closed"}],
            },
        ), patch(
            "memory_lab.memory_react_agent.list_open_memory_chains",
            return_value={
                "status": "complete",
                "tool": "list_open_memory_chains",
                "current_turn_id": "turn_current",
                "chains": [],
            },
        ), patch(
            "memory_lab.memory_react_agent.save_memory_chain_state",
            return_value={
                "status": "complete",
                "result": "updated",
                "chain": {"chain_id": "chain_1"},
            },
        ) as save_chain:
            result = run_memory_react_agent(
                source_database=Path("source.db"),
                turn_id="turn_current",
                pg_config_path=Path("unused.json"),
                dry_run=False,
                max_iterations=2,
                log_dir=Path(directory),
                client=SequenceClient([self._open_chain_scan_request(), output]),
            )
        self.assertTrue(result["committed"])
        self.assertEqual(save_chain.call_args.kwargs["final_memory_ids"], ["mem_closed"])

    def test_open_compile_chain_does_not_link_same_turn_preference_memory(self) -> None:
        plan = {
            "schema_version": "memory-changes-v1",
            "turn_id": "turn_current",
            "reason": "更新用户偏好，编译故障仍未闭环。",
            "changes": [
                {
                    "action": "merge_memory",
                    "target_memory_id": "mem_preference",
                    "memory": {"summary": "信息不足时先询问。"},
                }
            ],
        }
        output = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "偏好可更新，但编译根因尚未定位。",
            "chain_state": {
                "chain_type": "compilation",
                "state": "failure_observed",
                "related_turn_ids": ["turn_current"],
                "missing_evidence": ["编译失败根因与修复后验证"],
            },
            "tool_requests": [],
            "changes": plan,
        }
        read_request = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "先读取目标偏好记忆，并检查未闭环技术链。",
            "tool_requests": [
                {
                    "tool": "read_memory_detail",
                    "memory_id": "mem_preference",
                    "reason": "确认已有偏好内容后再合并。",
                },
                {
                    "tool": "list_open_memory_chains",
                    "reason": "确认编译故障链状态。",
                    "limit": 10,
                },
            ],
            "changes": [],
        }
        context = self._minimal_context()
        context["memory_index"]["memories"] = [
            {"memory_id": "mem_preference", "status": "active"}
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "memory_lab.memory_react_agent.build_memory_agent_input",
            return_value=context,
        ), patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            return_value=plan,
        ), patch(
            "memory_lab.memory_react_agent.commit_memory_changes",
            return_value={
                "ok": True,
                "counts": {"applied": 1, "rejected": 0},
                "applied": [
                    {"index": 0, "action": "merge_memory", "memory_id": "mem_preference"}
                ],
            },
        ), patch(
            "memory_lab.memory_react_agent.list_open_memory_chains",
            return_value={
                "status": "complete",
                "tool": "list_open_memory_chains",
                "current_turn_id": "turn_current",
                "chains": [],
            },
        ), patch(
            "memory_lab.memory_react_agent.read_memory_detail",
            return_value={
                "status": "complete",
                "tool": "read_memory_detail",
                "memory": {"memory_id": "mem_preference", "memory_type": "user_preference"},
            },
        ), patch(
            "memory_lab.memory_react_agent.save_memory_chain_state",
            return_value={
                "status": "complete",
                "result": "created",
                "chain": {"chain_id": "chain_compile"},
            },
        ) as save_chain:
            result = run_memory_react_agent(
                source_database=Path("source.db"),
                turn_id="turn_current",
                pg_config_path=Path("unused.json"),
                dry_run=False,
                max_iterations=2,
                log_dir=Path(directory),
                client=SequenceClient([read_request, output]),
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(save_chain.call_args.kwargs["final_memory_ids"], [])

    def test_incomplete_runtime_chain_accepts_noop_with_state_marker(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "代码已经修改，但还没有用户重新烧录确认。",
            "chain_state": {
                "chain_type": "runtime",
                "state": "waiting_user_validation",
                "related_turn_ids": ["turn_previous", "turn_current"],
                "missing_evidence": ["用户重新烧录后的运行确认"],
            },
            "tool_requests": [],
            "changes": {
                "schema_version": "memory-changes-v1",
                "turn_id": "turn_current",
                "source_turn_ids": ["turn_previous", "turn_current"],
                "reason": "未闭环，不写正式技术记忆。",
                "changes": [{"action": "noop", "reason": "等待用户验证。", "evidence_refs": []}],
            },
        }
        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            side_effect=lambda changes, **_kwargs: changes,
        ):
            validated = validate_react_decision(
                decision,
                turn_id="turn_current",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": set(),
                    "pointer_ids": set(),
                    "observed_turn_ids": {"turn_previous"},
                    "open_chain_scan_completed": {"current_session"},
                },
                pg_config_path=Path("unused.json"),
            )
        self.assertEqual(validated["chain_state"]["state"], "waiting_user_validation")

    def test_technical_memory_write_requires_matching_chain_state(self) -> None:
        plan = {
            "schema_version": "memory-changes-v1",
            "turn_id": "turn_current",
            "reason": "运行故障已经闭环。",
            "changes": [
                {
                    "action": "create_memory",
                    "technical_closure": {
                        "chain_type": "runtime",
                        "state": "user_verified",
                        "verification_evidence_ids": ["ev_user"],
                    },
                }
            ],
        }
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "尝试写入技术记忆。",
            "tool_requests": [],
            "changes": plan,
        }
        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            return_value=plan,
        ):
            with self.assertRaisesRegex(Exception, "chain_state is required"):
                validate_react_decision(
                    decision,
                    turn_id="turn_current",
                    available_ids={
                        "topic_ids": set(),
                        "memory_ids": set(),
                        "pointer_ids": set(),
                    },
                    pg_config_path=Path("unused.json"),
                )

    def test_technical_memory_write_accepts_matching_closed_chain(self) -> None:
        plan = {
            "schema_version": "memory-changes-v1",
            "turn_id": "turn_current",
            "reason": "运行故障已经闭环。",
            "changes": [
                {
                    "action": "create_memory",
                    "technical_closure": {
                        "chain_type": "runtime",
                        "state": "user_verified",
                        "verification_evidence_ids": ["ev_user"],
                    },
                }
            ],
        }
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "用户重新烧录确认正常。",
            "chain_state": {
                "chain_type": "runtime",
                "state": "user_verified",
                "related_turn_ids": ["turn_current"],
                "missing_evidence": [],
            },
            "tool_requests": [],
            "changes": plan,
        }
        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            return_value=plan,
        ):
            validated = validate_react_decision(
                decision,
                turn_id="turn_current",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": set(),
                    "pointer_ids": set(),
                    "open_chain_scan_completed": {"current_session"},
                },
                pg_config_path=Path("unused.json"),
            )
        self.assertEqual(validated["chain_state"]["state"], "user_verified")

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

    def test_prompt_documents_bounded_history_turn_tools(self) -> None:
        self.assertIn("list_previous_session_turns", REACT_SYSTEM_PROMPT)
        self.assertIn("read_session_turn_evidence", REACT_SYSTEM_PROMPT)
        self.assertIn("只发现同会话更早 turn 的摘要", REACT_SYSTEM_PROMPT)
        self.assertIn("不要一次读取整个会话", REACT_SYSTEM_PROMPT)

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
        self.assertIn("原子化指“一条 memory 一个可复用核心结论”", REACT_SYSTEM_PROMPT)
        self.assertIn("同一故障链条中的症状、根因、修复、验证必须放在同一条 memory", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_verification_status_policy(self) -> None:
        self.assertIn("验证状态写法", REACT_SYSTEM_PROMPT)
        self.assertIn("用户反馈现象”只能证明现象存在", REACT_SYSTEM_PROMPT)
        self.assertIn("助手定位/推理”只能证明候选根因", REACT_SYSTEM_PROMPT)
        self.assertIn("用户确认修复后正常", REACT_SYSTEM_PROMPT)
        self.assertIn("不要因为描述详细就自动提高 confidence", REACT_SYSTEM_PROMPT)
        self.assertIn("不能创建或更新正式技术 memory", REACT_SYSTEM_PROMPT)
        self.assertIn("不要把旧验证状态投射到新现象上", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_allowed_memory_types(self) -> None:
        self.assertIn("memory_type 只能是", REACT_SYSTEM_PROMPT)
        self.assertIn("user_preference", REACT_SYSTEM_PROMPT)
        self.assertIn("compile_failure_experience", REACT_SYSTEM_PROMPT)
        self.assertIn("runtime_failure_experience", REACT_SYSTEM_PROMPT)
        self.assertIn("禁止发明近义类型", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_quality_admission_gates(self) -> None:
        self.assertIn("沉淀前必须先过三道门槛", REACT_SYSTEM_PROMPT)
        self.assertIn("长期价值门槛", REACT_SYSTEM_PROMPT)
        self.assertIn("实际编译范围造成的可重复失败、结果失真", REACT_SYSTEM_PROMPT)
        self.assertIn("不能仅因发生在当前环境而按项目参数过滤", REACT_SYSTEM_PROMPT)
        self.assertIn("类型必备证据门槛", REACT_SYSTEM_PROMPT)
        self.assertIn("当前项目参数过滤门槛", REACT_SYSTEM_PROMPT)
        self.assertIn("事实提取", REACT_SYSTEM_PROMPT)
        self.assertIn("机制抽象", REACT_SYSTEM_PROMPT)
        self.assertIn("复用判断", REACT_SYSTEM_PROMPT)
        self.assertIn("单次对象明确的临时命令仍然 noop", REACT_SYSTEM_PROMPT)
        self.assertIn("编译通过”本身不是长期经验", REACT_SYSTEM_PROMPT)
        self.assertIn("未达到 compile_verified 时只保留 turn 证据", REACT_SYSTEM_PROMPT)
        self.assertIn("达到 user_verified，才能写入正式 memory", REACT_SYSTEM_PROMPT)
        self.assertIn("输出 noop + chain_state", REACT_SYSTEM_PROMPT)
        self.assertIn("代码修好或编译通过都不能代替用户验证", REACT_SYSTEM_PROMPT)
        self.assertIn("阈值、IO、PWM、定时", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_exact_chain_state_protocol(self) -> None:
        self.assertIn("chain_state 使用严格枚举", REACT_SYSTEM_PROMPT)

    def test_repair_prompt_documents_complete_history_tool_shape(self) -> None:
        self.assertIn("decision=tool_requests 时必须同时输出 `changes: []`", REACT_REPAIR_PROMPT)
        self.assertIn("禁止用历史工具读取当前 turn", REACT_REPAIR_PROMPT)
        self.assertIn("不存在 raw/summary 层", REACT_REPAIR_PROMPT)
        self.assertIn('"changes":[]', REACT_REPAIR_PROMPT)
        self.assertIn("禁止重复请求 read_session_turn_evidence", REACT_REPAIR_PROMPT)

    def test_prompt_accepts_explicit_user_test_confirmation_without_literal_reflash(self) -> None:
        self.assertIn("我测试了，功能已实现/问题消失/现在正常", REACT_SYSTEM_PROMPT)
        self.assertIn("不要求机械地逐字出现", REACT_SYSTEM_PROMPT)
        self.assertIn("绝不能用一个问题的用户确认关闭另一个问题", REACT_SYSTEM_PROMPT)

    def test_rejects_repeating_already_observed_history_turn(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "重复读取历史。",
            "tool_requests": [
                {
                    "tool": "read_session_turn_evidence",
                    "turn_id": "turn_previous",
                    "reason": "再次读取。",
                }
            ],
            "changes": [],
        }
        available_ids = {
            "discovered_turn_ids": {"turn_previous"},
            "observed_turn_ids": {"turn_previous"},
        }
        with self.assertRaisesRegex(Exception, "already read"):
            validate_react_decision(
                decision,
                turn_id="turn_current",
                available_ids=available_ids,
            )
        self.assertIn("none，state=not_applicable", REACT_SYSTEM_PROMPT)
        self.assertIn("failure_observed、cause_identified、fix_applied、compile_verified", REACT_SYSTEM_PROMPT)
        self.assertIn("waiting_user_validation、user_verified、fix_rejected", REACT_SYSTEM_PROMPT)
        self.assertIn("needs_verification", REACT_SYSTEM_PROMPT)
        self.assertIn("noop 是记忆动作，不是 chain state", REACT_SYSTEM_PROMPT)
        self.assertIn("助手声称已修复", REACT_SYSTEM_PROMPT)

    def test_prompt_documents_multi_memory_granularity_gate(self) -> None:
        self.assertIn("单轮多条记忆自检", REACT_SYSTEM_PROMPT)
        self.assertIn("每条必须有不同的核心结论", REACT_SYSTEM_PROMPT)
        self.assertIn("同一根因的多个表现，合成一条", REACT_SYSTEM_PROMPT)
        self.assertIn("不要把“症状、原因、修复、验证”拆成多条 memory", REACT_SYSTEM_PROMPT)
        self.assertIn("如果同一轮要新增超过 2 条 memory", REACT_SYSTEM_PROMPT)
        self.assertIn("不同的未来检索入口和不同根因", REACT_SYSTEM_PROMPT)

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
                    "memory_type": "compile_failure_experience",
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

    def test_validate_rejects_search_with_legacy_memory_type(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "需要搜索旧记忆。",
            "tool_requests": [
                {
                    "tool": "search_related_memories",
                    "query": "Keil C51 编译错误",
                    "reason": "旧类型过滤不应继续使用。",
                    "memory_type": "compile_failure_pattern",
                }
            ],
            "changes": [],
        }

        with self.assertRaises(Exception) as raised:
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
        self.assertIn("memory_type must be one of", str(raised.exception))

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

    def test_validate_accepts_history_discovery_request(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "当前可能是故障链后续，需要先查看同会话摘要。",
            "tool_requests": [
                {
                    "tool": "list_previous_session_turns",
                    "reason": "查找相关现象、定位和修复 turn。",
                    "limit": 10,
                    "preview_chars": 320,
                }
            ],
            "changes": [],
        }

        validated = validate_react_decision(
            decision,
            turn_id="turn_current",
            available_ids={"topic_ids": set(), "memory_ids": set(), "pointer_ids": set()},
            pg_config_path=Path("unused.json"),
        )

        self.assertEqual(validated["tool_requests"][0]["tool"], "list_previous_session_turns")

    def test_validate_accepts_open_chain_discovery_request(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "当前可能是未闭环上板问题的确认 turn。",
            "tool_requests": [
                {
                    "tool": "list_open_memory_chains",
                    "reason": "先寻找同会话待用户验证的链。",
                    "limit": 10,
                }
            ],
            "changes": [],
        }
        validated = validate_react_decision(
            decision,
            turn_id="turn_current",
            available_ids={"topic_ids": set(), "memory_ids": set(), "pointer_ids": set()},
            pg_config_path=Path("unused.json"),
        )
        self.assertEqual(validated["tool_requests"][0]["tool"], "list_open_memory_chains")

    def test_open_chain_summary_discovers_turns_but_not_evidence(self) -> None:
        available_ids = {
            "evidence_ids": {"ev_current"},
            "pointer_ids": set(),
            "discovered_turn_ids": set(),
            "observed_turn_ids": set(),
            "historical_evidence_ids": set(),
        }
        _update_turn_access_ids(
            available_ids,
            [
                {
                    "status": "complete",
                    "result": {
                        "status": "complete",
                        "tool": "list_open_memory_chains",
                        "current_turn_id": "turn_current",
                        "chains": [
                            {
                                "chain_id": "chain_runtime_001",
                                "related_turn_ids": ["turn_previous", "turn_current"],
                                "related_turns": [
                                    {"turn_id": "turn_previous", "preview": "风扇不转"}
                                ],
                            }
                        ],
                    },
                }
            ],
        )
        self.assertEqual(available_ids["discovered_turn_ids"], {"turn_previous"})
        self.assertEqual(available_ids["observed_turn_ids"], set())
        self.assertEqual(available_ids["historical_evidence_ids"], set())
        self.assertEqual(available_ids["open_chain_scan_completed"], {"current_session"})
        self.assertEqual(available_ids["open_chain_ids"], {"chain_runtime_001"})
        self.assertEqual(
            available_ids["open_chain_required::chain_runtime_001"],
            {"turn_previous"},
        )

    def test_technical_final_requires_open_chain_scan(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "运行问题仍待用户验证。",
            "chain_state": {
                "chain_type": "runtime",
                "state": "waiting_user_validation",
                "related_turn_ids": ["turn_current"],
                "missing_evidence": ["用户重新烧录后的确认"],
            },
            "tool_requests": [],
            "changes": {
                "schema_version": "memory-changes-v1",
                "turn_id": "turn_current",
                "reason": "尚未闭环。",
                "changes": [{"action": "noop", "reason": "等待确认。", "evidence_refs": []}],
            },
        }
        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            side_effect=lambda changes, **_kwargs: changes,
        ):
            with self.assertRaisesRegex(Exception, "requires a completed list_open_memory_chains"):
                validate_react_decision(
                    decision,
                    turn_id="turn_current",
                    available_ids={"topic_ids": set(), "memory_ids": set(), "pointer_ids": set()},
                    pg_config_path=Path("unused.json"),
                )

    def test_existing_chain_id_must_be_observed_and_keep_all_turns(self) -> None:
        base = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "尝试关闭既有链。",
            "chain_state": {
                "chain_id": "chain_invented_001",
                "chain_type": "runtime",
                "state": "user_verified",
                "related_turn_ids": ["turn_current"],
                "missing_evidence": [],
            },
            "tool_requests": [],
            "changes": {
                "schema_version": "memory-changes-v1",
                "turn_id": "turn_current",
                "reason": "结构测试。",
                "changes": [{"action": "noop", "reason": "结构测试。", "evidence_refs": []}],
            },
        }
        available = {
            "topic_ids": set(),
            "memory_ids": set(),
            "pointer_ids": set(),
            "observed_turn_ids": {"turn_previous"},
            "open_chain_scan_completed": {"current_session"},
            "open_chain_ids": {"chain_runtime_001"},
            "open_chain_required::chain_runtime_001": {"turn_previous"},
        }
        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            side_effect=lambda changes, **_kwargs: changes,
        ):
            with self.assertRaisesRegex(Exception, "chain_id must come from"):
                validate_react_decision(
                    base,
                    turn_id="turn_current",
                    available_ids=available,
                    pg_config_path=Path("unused.json"),
                )
            base["chain_state"]["chain_id"] = "chain_runtime_001"
            with self.assertRaisesRegex(Exception, "cannot drop turns"):
                validate_react_decision(
                    base,
                    turn_id="turn_current",
                    available_ids=available,
                    pg_config_path=Path("unused.json"),
                )
            base["chain_state"]["related_turn_ids"] = ["turn_previous", "turn_current"]
            validated = validate_react_decision(
                base,
                turn_id="turn_current",
                available_ids=available,
                pg_config_path=Path("unused.json"),
            )
        self.assertEqual(validated["chain_state"]["chain_id"], "chain_runtime_001")

    def test_existing_chain_final_auto_reads_unobserved_chain_turns(self) -> None:
        requests = _auto_required_chain_reads(
            {
                "decision": "final_changes",
                "chain_state": {
                    "chain_id": "chain_runtime_001",
                    "chain_type": "runtime",
                    "state": "user_verified",
                    "related_turn_ids": ["turn_1", "turn_2", "turn_current"],
                    "missing_evidence": [],
                },
            },
            available_ids={
                "open_chain_ids": {"chain_runtime_001"},
                "open_chain_required::chain_runtime_001": {"turn_1", "turn_2"},
                "observed_turn_ids": {"turn_1"},
            },
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["tool"], "read_session_turn_evidence")
        self.assertEqual(requests[0]["turn_id"], "turn_2")

    def test_technical_final_auto_scans_open_chains_before_validation(self) -> None:
        requests = _auto_required_chain_reads(
            {
                "decision": "final_changes",
                "chain_state": {
                    "chain_type": "runtime",
                    "state": "waiting_user_validation",
                    "related_turn_ids": ["turn_current"],
                    "missing_evidence": ["用户重新烧录确认"],
                },
            },
            available_ids={
                "open_chain_scan_completed": set(),
                "open_chain_ids": set(),
                "observed_turn_ids": set(),
            },
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["tool"], "list_open_memory_chains")

    def test_nontechnical_final_does_not_auto_scan_open_chains(self) -> None:
        requests = _auto_required_chain_reads(
            {
                "decision": "final_changes",
                "chain_state": {
                    "chain_type": "none",
                    "state": "not_applicable",
                    "related_turn_ids": ["turn_current"],
                    "missing_evidence": [],
                },
            },
            available_ids={
                "open_chain_scan_completed": set(),
                "open_chain_ids": set(),
                "observed_turn_ids": set(),
            },
        )
        self.assertEqual(requests, [])

    def test_chain_auto_read_does_not_follow_invented_chain_id(self) -> None:
        requests = _auto_required_chain_reads(
            {
                "decision": "final_changes",
                "chain_state": {
                    "chain_id": "chain_invented_001",
                    "chain_type": "runtime",
                    "state": "user_verified",
                    "related_turn_ids": ["turn_current"],
                    "missing_evidence": [],
                },
            },
            available_ids={
                "open_chain_ids": {"chain_runtime_001"},
                "open_chain_required::chain_runtime_001": {"turn_1"},
                "observed_turn_ids": set(),
            },
        )
        self.assertEqual(requests, [])

    def test_validate_requires_discovery_before_history_detail(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "读取相关历史 turn。",
            "tool_requests": [
                {
                    "tool": "read_session_turn_evidence",
                    "turn_id": "turn_previous",
                    "reason": "读取故障现象证据。",
                }
            ],
            "changes": [],
        }

        with self.assertRaisesRegex(Exception, "must come from a completed"):
            validate_react_decision(
                decision,
                turn_id="turn_current",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": set(),
                    "pointer_ids": set(),
                    "discovered_turn_ids": set(),
                },
                pg_config_path=Path("unused.json"),
            )

    def test_validate_accepts_history_detail_after_discovery(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "读取已发现且相关的历史 turn。",
            "tool_requests": [
                {
                    "tool": "read_session_turn_evidence",
                    "turn_id": "turn_previous",
                    "reason": "读取故障现象和定位证据。",
                    "layers": ["core", "candidate"],
                    "max_chars": 1200,
                }
            ],
            "changes": [],
        }

        validated = validate_react_decision(
            decision,
            turn_id="turn_current",
            available_ids={
                "topic_ids": set(),
                "memory_ids": set(),
                "pointer_ids": set(),
                "discovered_turn_ids": {"turn_previous"},
            },
            pg_config_path=Path("unused.json"),
        )

        self.assertEqual(validated["tool_requests"][0]["turn_id"], "turn_previous")

    def test_validate_rejects_bad_history_detail_bounds_and_layers(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "tool_requests",
            "reason": "错误的历史读取参数。",
            "tool_requests": [
                {
                    "tool": "read_session_turn_evidence",
                    "turn_id": "turn_previous",
                    "reason": "边界验证。",
                    "layers": ["core", "core", "secret"],
                    "max_chars": 9000,
                }
            ],
            "changes": [],
        }

        with self.assertRaises(Exception) as raised:
            validate_react_decision(
                decision,
                turn_id="turn_current",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": set(),
                    "pointer_ids": set(),
                    "discovered_turn_ids": {"turn_previous"},
                },
                pg_config_path=Path("unused.json"),
            )
        message = str(raised.exception)
        self.assertIn("layers must be", message)
        self.assertIn("max_chars must be between 80 and 8000", message)

    def test_history_summary_does_not_authorize_evidence_until_detail_read(self) -> None:
        available_ids = {
            "evidence_ids": {"ev_current"},
            "pointer_ids": set(),
            "discovered_turn_ids": set(),
            "observed_turn_ids": set(),
            "historical_evidence_ids": set(),
        }
        summary_observation = {
            "status": "complete",
            "result": {
                "status": "complete",
                "tool": "list_previous_session_turns",
                "turns": [
                    {
                        "turn_id": "turn_previous",
                        "previews": [{"evidence_id": "ev_historical"}],
                    }
                ],
            },
        }

        _update_turn_access_ids(available_ids, [summary_observation])

        self.assertEqual(available_ids["discovered_turn_ids"], {"turn_previous"})
        self.assertNotIn("ev_historical", available_ids["evidence_ids"])
        self.assertEqual(available_ids["observed_turn_ids"], set())

        detail_observation = {
            "status": "complete",
            "result": {
                "status": "complete",
                "tool": "read_session_turn_evidence",
                "target_turn_id": "turn_previous",
                "evidence": {
                    "turn": {"turn_id": "turn_previous"},
                    "layers": {
                        "core": [{"evidence_id": "ev_historical"}],
                        "candidate": [],
                        "pointer": [
                            {
                                "evidence_id": "ptr_historical",
                                "raw_pointer_id": "raw_historical",
                            }
                        ],
                    },
                },
            },
        }

        _update_turn_access_ids(available_ids, [detail_observation])

        self.assertEqual(available_ids["observed_turn_ids"], {"turn_previous"})
        self.assertIn("ev_historical", available_ids["historical_evidence_ids"])
        self.assertIn("ptr_historical", available_ids["pointer_ids"])
        self.assertIn("raw_historical", available_ids["pointer_ids"])

    def test_react_runs_discover_then_read_history_before_final_decision(self) -> None:
        client = SequenceClient(
            [
                {
                    "schema_version": "memory-react-v1",
                    "decision": "tool_requests",
                    "reason": "先发现同会话历史。",
                    "tool_requests": [
                        {
                            "tool": "list_previous_session_turns",
                            "reason": "寻找故障链前序 turn。",
                            "limit": 5,
                        }
                    ],
                    "changes": [],
                },
                {
                    "schema_version": "memory-react-v1",
                    "decision": "tool_requests",
                    "reason": "读取已发现的相关 turn。",
                    "tool_requests": [
                        {
                            "tool": "read_session_turn_evidence",
                            "turn_id": "turn_previous",
                            "reason": "读取现象证据。",
                        }
                    ],
                    "changes": [],
                },
                {
                    "schema_version": "memory-react-v1",
                    "decision": "final_changes",
                    "reason": "多轮读取完成，但 P2 尚不提交跨 turn 证据。",
                    "tool_requests": [],
                    "changes": {
                        "schema_version": "memory-changes-v1",
                        "turn_id": "turn_current",
                        "reason": "P2 流程验收。",
                        "changes": [
                            {
                                "action": "noop",
                                "reason": "等待 P3 开放跨 turn 提交。",
                                "evidence_refs": [],
                            }
                        ],
                    },
                },
            ]
        )
        initial_context = {
            "status": "complete",
            "turn_id": "turn_current",
            "turn_evidence": {
                "turn": {"turn_id": "turn_current"},
                "layers": {
                    "core": [{"evidence_id": "ev_current"}],
                    "candidate": [],
                    "pointer": [],
                },
            },
            "topic_index": {"topics": []},
            "memory_index": {"topics": [], "memories": []},
            "filters": {"domain": "embedded", "project_key": "project_1"},
        }
        history_listing = {
            "status": "complete",
            "tool": "list_previous_session_turns",
            "turns": [{"turn_id": "turn_previous", "previews": []}],
            "count": 1,
        }
        history_detail = {
            "status": "complete",
            "tool": "read_session_turn_evidence",
            "target_turn_id": "turn_previous",
            "evidence": {
                "turn": {"turn_id": "turn_previous"},
                "layers": {
                    "core": [{"evidence_id": "ev_previous"}],
                    "candidate": [],
                    "pointer": [],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "memory_lab.memory_react_agent.build_memory_agent_input",
            return_value=initial_context,
        ), patch(
            "memory_lab.memory_react_agent.list_previous_session_turns",
            return_value=history_listing,
        ) as list_history, patch(
            "memory_lab.memory_react_agent.read_session_turn_evidence",
            return_value=history_detail,
        ) as read_history, patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            side_effect=lambda changes, **_kwargs: changes,
        ):
            result = run_memory_react_agent(
                source_database=Path("source.db"),
                turn_id="turn_current",
                pg_config_path=Path("unused.json"),
                dry_run=True,
                max_iterations=3,
                log_dir=Path(directory),
                client=client,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["observation_count"], 2)
        list_history.assert_called_once()
        read_history.assert_called_once()
        second_payload = client.calls[1]["user_payload"]
        self.assertEqual(
            second_payload["turn_access_state"]["discovered_turn_ids"],
            ["turn_previous"],
        )
        third_payload = client.calls[2]["user_payload"]
        self.assertEqual(
            third_payload["turn_access_state"]["observed_turn_ids"],
            ["turn_previous"],
        )
        self.assertEqual(
            third_payload["turn_access_state"]["authorized_historical_evidence_ids"],
            ["ev_previous"],
        )

    def test_final_changes_rejects_unobserved_source_turn(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "错误地直接引用只发现但未读取的 turn。",
            "tool_requests": [],
            "changes": {
                "schema_version": "memory-changes-v1",
                "turn_id": "turn_current",
                "source_turn_ids": ["turn_current", "turn_previous"],
                "reason": "不应通过。",
                "changes": [
                    {"action": "noop", "reason": "不应通过。", "evidence_refs": []}
                ],
            },
        }
        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            return_value=decision["changes"],
        ):
            with self.assertRaisesRegex(Exception, "were not read"):
                validate_react_decision(
                    decision,
                    turn_id="turn_current",
                    available_ids={
                        "topic_ids": set(),
                        "memory_ids": set(),
                        "pointer_ids": set(),
                        "discovered_turn_ids": {"turn_previous"},
                        "observed_turn_ids": set(),
                    },
                    pg_config_path=Path("unused.json"),
                )

    def test_final_changes_accepts_observed_source_turn(self) -> None:
        decision = {
            "schema_version": "memory-react-v1",
            "decision": "final_changes",
            "reason": "引用已经读取的历史 turn。",
            "tool_requests": [],
            "changes": {
                "schema_version": "memory-changes-v1",
                "turn_id": "turn_current",
                "source_turn_ids": ["turn_previous", "turn_current"],
                "reason": "合法跨 turn 证据。",
                "changes": [
                    {"action": "noop", "reason": "结构验收。", "evidence_refs": []}
                ],
            },
        }
        with patch(
            "memory_lab.memory_react_agent.validate_memory_changes",
            return_value=decision["changes"],
        ):
            validated = validate_react_decision(
                decision,
                turn_id="turn_current",
                available_ids={
                    "topic_ids": set(),
                    "memory_ids": set(),
                    "pointer_ids": set(),
                    "discovered_turn_ids": {"turn_previous"},
                    "observed_turn_ids": {"turn_previous"},
                },
                pg_config_path=Path("unused.json"),
            )

        self.assertEqual(
            validated["changes"]["source_turn_ids"],
            ["turn_previous", "turn_current"],
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
