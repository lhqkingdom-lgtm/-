from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.memory_changes import (
    MEMORY_CHANGES_SCHEMA_VERSION,
    MemoryChangesValidationError,
    commit_memory_changes,
    _digest_id,
    validate_memory_changes,
)


SNAPSHOT = {
    "turn_exists": True,
    "evidence_ids": ["ev_user", "ev_assistant"],
    "pointer_ids": ["ptr_user"],
    "memory_ids": ["mem_language"],
    "memory_topics": {"mem_language": "topic_language"},
    "conflict_memory_ids": [],
    "topic_ids": ["topic_language"],
    "topic_keys": ["general.reply_language"],
    "stable_keys": ["general.user_preference.reply_language"],
    "conflict_records": [
        {
            "conflict_id": "conflict_active",
            "memory_id": "mem_language",
            "topic_id": "topic_language",
            "status": "needs_review",
        },
        {
            "conflict_id": "conflict_other_memory",
            "memory_id": "mem_other",
            "topic_id": "topic_language",
            "status": "needs_review",
        },
        {
            "conflict_id": "conflict_resolved",
            "memory_id": "mem_language",
            "topic_id": "topic_language",
            "status": "resolved_by_update",
        },
    ],
}

RELATION_SNAPSHOT = {
    **SNAPSHOT,
    "memory_ids": ["mem_language", "mem_language_new"],
}

MERGE_SNAPSHOT = {
    **SNAPSHOT,
    "memory_ids": ["mem_target", "mem_source_a", "mem_source_b", "mem_source_c"],
    "conflict_records": [
        {
            "conflict_id": "conflict_source_a",
            "memory_id": "mem_source_a",
            "topic_id": "topic_language",
            "status": "needs_review",
        },
        {
            "conflict_id": "conflict_target",
            "memory_id": "mem_target",
            "topic_id": "topic_language",
            "status": "needs_review",
        },
    ],
}

CONFLICT_MEMORY_SNAPSHOT = {
    **SNAPSHOT,
    "memory_ids": [],
    "conflict_memory_ids": ["mem_language"],
}

NO_ACTIVE_CONFLICT_SNAPSHOT = {
    **SNAPSHOT,
    "conflict_records": [
        {
            "conflict_id": "conflict_resolved",
            "memory_id": "mem_language",
            "topic_id": "topic_language",
            "status": "resolved_by_update",
        },
    ],
}

MOVE_TOPIC_SNAPSHOT = {
    **SNAPSHOT,
    "memory_ids": ["mem_language"],
    "memory_topics": {"mem_language": "topic_language"},
    "topic_ids": ["topic_language", "topic_user_collaboration"],
}

MERGE_TOPICS_SNAPSHOT = {
    **SNAPSHOT,
    "memory_ids": ["mem_language", "mem_topic_source"],
    "memory_topics": {
        "mem_language": "topic_language",
        "mem_topic_source": "topic_source_duplicate",
    },
    "topic_ids": ["topic_language", "topic_source_duplicate", "topic_other_duplicate"],
}


def base_proposal(change: dict) -> dict:
    return {
        "schema_version": MEMORY_CHANGES_SCHEMA_VERSION,
        "turn_id": "turn_language_change",
        "reason": "测试一轮记忆工具计划。",
        "changes": [change],
    }


def evidence_refs() -> list[dict]:
    return [{"evidence_id": "ev_user", "reason": "用户明确表达。"}]


def assert_validation_errors(testcase: unittest.TestCase, proposal: dict, *needles: str) -> None:
    with patch(
        "memory_lab.memory_changes.load_pg_config",
        return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
    ), patch(
        "memory_lab.memory_changes._reference_snapshot",
        return_value=SNAPSHOT,
    ):
        with testcase.assertRaises(MemoryChangesValidationError) as raised:
            validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
    text = "\n".join(raised.exception.errors)
    for needle in needles:
        testcase.assertIn(needle, text)


def assert_valid(testcase: unittest.TestCase, proposal: dict) -> None:
    with patch(
        "memory_lab.memory_changes.load_pg_config",
        return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
    ), patch(
        "memory_lab.memory_changes._reference_snapshot",
        return_value=SNAPSHOT,
    ):
        normalized = validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
    testcase.assertEqual(normalized["turn_id"], proposal["turn_id"])


class MemoryChangesValidationTests(unittest.TestCase):
    def test_rejects_unknown_action(self) -> None:
        assert_validation_errors(
            self,
            base_proposal({"action": "rename_everything", "reason": "模型幻觉动作。"}),
            "changes[0].action is not allowed",
        )

    def test_rejects_evidence_from_other_turn(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "noop",
                    "reason": "测试错误证据。",
                    "evidence_refs": [{"evidence_id": "ev_other_turn"}],
                }
            ),
            "evidence_id does not belong to this turn",
        )

    def test_update_topic_accepts_title_summary_only(self) -> None:
        assert_valid(
            self,
            base_proposal(
                {
                    "action": "update_topic",
                    "reason": "memory 语义变化后修正主题展示。",
                    "target_topic_id": "topic_language",
                    "topic": {
                        "title": "回复语言偏好",
                        "summary": "记录用户当前有效回复语言偏好。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
        )

    def test_update_topic_rejects_topic_key_mutation(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "update_topic",
                    "reason": "模型试图越权改 topic_key。",
                    "target_topic_id": "topic_language",
                    "topic": {
                        "title": "回复语言偏好",
                        "summary": "记录用户当前有效回复语言偏好。",
                        "topic_key": "general.reply_language",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "topic has unsupported fields",
        )

    def test_update_topic_rejects_memory_patch(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "update_topic",
                    "reason": "模型试图用主题工具顺手修改记忆正文。",
                    "target_topic_id": "topic_language",
                    "topic": {
                        "title": "回复语言偏好",
                        "summary": "记录用户当前有效回复语言偏好。",
                    },
                    "memory": {
                        "summary": "不应被接受",
                        "content": "update_topic 不能修改 memory 内容。",
                        "confidence": 0.9,
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "has unsupported fields",
        )

    def test_update_topic_rejects_unknown_topic(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "update_topic",
                    "reason": "不存在的 topic。",
                    "target_topic_id": "topic_missing",
                    "topic": {"title": "回复语言偏好"},
                    "evidence_refs": evidence_refs(),
                }
            ),
            "target_topic_id does not exist or is not active",
        )

    def test_create_memory_rejects_existing_topic_key_to_avoid_implicit_topic_update(self) -> None:
        assert_validation_errors(
            self,
            {
                "schema_version": MEMORY_CHANGES_SCHEMA_VERSION,
                "turn_id": "turn_language_change",
                "reason": "新建 memory 时不允许顺手修改已有 topic。",
                "changes": [
                    {
                        "action": "create_memory",
                        "reason": "错误地带了已有 topic_key 的 topic 对象。",
                        "topic": {
                            "topic_key": "general.reply_language",
                            "title": "新的主题标题",
                            "summary": "不应通过 create_memory 修改已有主题。",
                            "topic_type": "user_preference",
                            "scope": {"level": "global"},
                        },
                        "memory": {
                            "stable_key": "general.user_preference.reply_language.new",
                            "memory_type": "user_preference",
                            "summary": "用户希望默认英文回复。",
                            "content": "用户希望默认英文回复。",
                            "scope": {"level": "global"},
                            "confidence": 0.9,
                        },
                        "evidence_refs": evidence_refs(),
                    }
                ],
            },
            "topic.topic_key already exists",
        )

    def test_create_memory_accepts_allowed_memory_type(self) -> None:
        assert_valid(
            self,
            base_proposal(
                {
                    "action": "create_memory",
                    "reason": "新增用户长期偏好。",
                    "topic_id": "topic_language",
                    "memory": {
                        "stable_key": "general.user_preference.reply_language.english",
                        "memory_type": "user_preference",
                        "summary": "用户希望默认英文回复。",
                        "content": "用户希望默认英文回复。",
                        "scope": {"level": "global"},
                        "confidence": 0.9,
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
        )

    def test_create_memory_rejects_legacy_memory_type(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "create_memory",
                    "reason": "旧类型不应继续写入。",
                    "topic_id": "topic_language",
                    "memory": {
                        "stable_key": "general.user_preference.reply_language.legacy_type",
                        "memory_type": "workflow_rule",
                        "summary": "用户希望默认英文回复。",
                        "content": "用户希望默认英文回复。",
                        "scope": {"level": "global"},
                        "confidence": 0.9,
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "memory.memory_type must be one of",
        )

    def test_record_conflict_decision_accepts_complete_payload(self) -> None:
        assert_valid(
            self,
            base_proposal(
                {
                    "action": "record_conflict_decision",
                    "reason": "记录用户偏好变更冲突。",
                    "target_memory_id": "mem_language",
                    "conflict": {
                        "conflict_type": "user_preference_changed",
                        "old_claim": "用户希望默认中文回复。",
                        "new_claim": "用户希望默认英文回复。",
                        "decision": "采用更新的明确用户偏好。",
                        "reason": "新证据是用户明确纠正。",
                        "status": "resolved_by_update",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
        )

    def test_record_conflict_decision_rejects_legacy_or_unknown_status(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "record_conflict_decision",
                    "reason": "记录用户偏好变更冲突。",
                    "target_memory_id": "mem_language",
                    "conflict": {
                        "conflict_type": "user_preference_changed",
                        "old_claim": "用户希望默认中文回复。",
                        "new_claim": "用户希望默认英文回复。",
                        "decision": "采用更新的明确用户偏好。",
                        "reason": "新证据是用户明确纠正。",
                        "status": "resolved",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "conflict.status must be one of",
        )

    def test_record_conflict_decision_accepts_conflict_memory(self) -> None:
        proposal = base_proposal(
            {
                "action": "record_conflict_decision",
                "reason": "记录已经标记为 conflict 的 memory 的冲突原因。",
                "target_memory_id": "mem_language",
                "conflict": {
                    "conflict_type": "user_preference_changed",
                    "old_claim": "用户希望默认中文回复。",
                    "new_claim": "用户希望默认英文回复。",
                    "decision": "需要继续复核后决定当前有效偏好。",
                    "reason": "该 memory 已经被标记为 conflict，但仍应允许记录冲突原因。",
                    "status": "needs_review",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=CONFLICT_MEMORY_SNAPSHOT,
        ):
            normalized = validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertEqual(normalized["changes"][0]["target_memory_id"], "mem_language")

    def test_update_memory_rejects_conflict_memory(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_memory",
                "reason": "update_memory 只能修改 active memory。",
                "target_memory_id": "mem_language",
                "memory": {
                    "summary": "不应更新 conflict memory。",
                    "content": "不应更新 conflict memory。",
                    "confidence": 0.5,
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=CONFLICT_MEMORY_SNAPSHOT,
        ):
            with self.assertRaises(MemoryChangesValidationError) as raised:
                validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertIn("target_memory_id does not exist or is not active", "\n".join(raised.exception.errors))

    def test_retire_memory_accepts_active_memory_with_conflict_disposition(self) -> None:
        assert_valid(
            self,
            base_proposal(
                {
                    "action": "retire_memory",
                    "reason": "用户明确表示这条旧记忆不再适用。",
                    "target_memory_id": "mem_language",
                    "conflict_disposition": {
                        "conflict_ids": ["conflict_active"],
                        "status": "discarded_as_stale",
                        "reason": "旧记忆退役后，该冲突记录只保留历史意义。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
        )

    def test_retire_memory_requires_conflict_disposition_when_active_conflict_exists(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "retire_memory",
                    "reason": "试图退役有 active 冲突的记忆。",
                    "target_memory_id": "mem_language",
                    "evidence_refs": evidence_refs(),
                }
            ),
            "conflict_disposition is required because target memory has active conflicts",
        )

    def test_retire_memory_allows_omitting_disposition_when_no_active_conflict_exists(self) -> None:
        proposal = base_proposal(
            {
                "action": "retire_memory",
                "reason": "退役没有 active 冲突的记忆。",
                "target_memory_id": "mem_language",
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=NO_ACTIVE_CONFLICT_SNAPSHOT,
        ):
            normalized = validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertEqual(normalized["changes"][0]["action"], "retire_memory")

    def test_retire_memory_rejects_conflict_memory_and_unsupported_memory_patch(self) -> None:
        proposal = base_proposal(
            {
                "action": "retire_memory",
                "reason": "retire_memory 只能退役 active memory，且不接受 memory patch。",
                "target_memory_id": "mem_language",
                "memory": {"summary": "越权字段"},
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=CONFLICT_MEMORY_SNAPSHOT,
        ):
            with self.assertRaises(MemoryChangesValidationError) as raised:
                validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        text = "\n".join(raised.exception.errors)
        self.assertIn("target_memory_id does not exist or is not active", text)
        self.assertIn("unsupported fields", text)

    def test_merge_existing_memories_accepts_multiple_active_sources_with_conflict_disposition(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_existing_memories",
                "reason": "多条同类记忆重复，合并到一个当前有效结论。",
                "target_memory_id": "mem_target",
                "source_memory_ids": ["mem_source_a", "mem_source_b"],
                "memory": {
                    "summary": "用户希望默认英文回复。",
                    "content": "合并后的完整记忆内容。",
                    "confidence": 0.91,
                },
                "conflict_disposition": {
                    "conflict_ids": ["conflict_source_a"],
                    "status": "resolved_by_merge",
                    "reason": "source 记忆被合并后，该冲突由合并结果解决。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_SNAPSHOT,
        ):
            normalized = validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertEqual(normalized["changes"][0]["action"], "merge_existing_memories")

    def test_merge_existing_memories_rejects_bad_sources_and_target_overlap(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_existing_memories",
                "reason": "错误合并。",
                "target_memory_id": "mem_target",
                "source_memory_ids": ["mem_target", "mem_missing", "mem_source_b", "mem_source_b"],
                "memory": {
                    "summary": "用户希望默认英文回复。",
                    "content": "合并后的完整记忆内容。",
                    "confidence": 0.91,
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_SNAPSHOT,
        ):
            with self.assertRaises(MemoryChangesValidationError) as raised:
                validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        text = "\n".join(raised.exception.errors)
        self.assertIn("must not equal target_memory_id", text)
        self.assertIn("does not exist or is not active", text)
        self.assertIn("is duplicated", text)

    def test_merge_existing_memories_requires_conflict_disposition_for_source_active_conflict(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_existing_memories",
                "reason": "source 有 active conflict 时不能静默合并。",
                "target_memory_id": "mem_target",
                "source_memory_ids": ["mem_source_a"],
                "memory": {
                    "summary": "用户希望默认英文回复。",
                    "content": "合并后的完整记忆内容。",
                    "confidence": 0.91,
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_SNAPSHOT,
        ):
            with self.assertRaises(MemoryChangesValidationError) as raised:
                validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertIn(
            "conflict_disposition is required because source memories have active conflicts",
            "\n".join(raised.exception.errors),
        )

    def test_update_memory_accepts_conflict_resolution_for_active_conflict(self) -> None:
        assert_valid(
            self,
            base_proposal(
                {
                    "action": "update_memory",
                    "reason": "用户明确纠正旧偏好。",
                    "target_memory_id": "mem_language",
                    "memory": {
                        "summary": "用户要求默认英文回复。",
                        "content": "用户要求默认英文回复。",
                        "confidence": 0.9,
                    },
                    "conflict_resolution": {
                        "conflict_ids": ["conflict_active"],
                        "status": "resolved_by_update",
                        "reason": "用户明确纠正，已用新内容更新 memory。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
        )

    def test_update_memory_conflict_resolution_requires_known_target_active_conflict(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "update_memory",
                    "reason": "错误关闭冲突。",
                    "target_memory_id": "mem_language",
                    "memory": {
                        "summary": "用户要求默认英文回复。",
                        "content": "用户要求默认英文回复。",
                        "confidence": 0.9,
                    },
                    "conflict_resolution": {
                        "conflict_ids": ["conflict_other_memory", "conflict_resolved", "conflict_missing"],
                        "status": "resolved_by_update",
                        "reason": "测试错误冲突引用。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "does not belong to target_memory_id",
            "is not an active conflict",
            "contains unknown conflict_id",
        )

    def test_update_memory_conflict_resolution_rejects_non_update_status(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "update_memory",
                    "reason": "错误关闭冲突。",
                    "target_memory_id": "mem_language",
                    "memory": {
                        "summary": "用户要求默认英文回复。",
                        "content": "用户要求默认英文回复。",
                        "confidence": 0.9,
                    },
                    "conflict_resolution": {
                        "conflict_ids": ["conflict_active"],
                        "status": "resolved_by_merge",
                        "reason": "update_memory 不能用 merge 状态关闭冲突。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "conflict_resolution.status must be resolved_by_update",
        )

    def test_conflict_resolution_is_not_allowed_on_other_actions(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "mark_conflict",
                    "reason": "mark_conflict 不负责关闭冲突。",
                    "target_memory_id": "mem_language",
                    "memory": {
                        "summary": "用户要求默认英文回复。",
                        "content": "用户要求默认英文回复。",
                        "confidence": 0.9,
                    },
                    "conflict_resolution": {
                        "conflict_ids": ["conflict_active"],
                        "status": "resolved_by_update",
                        "reason": "不允许。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "unsupported fields",
        )

    def test_record_memory_relation_accepts_complete_payload(self) -> None:
        proposal = base_proposal(
            {
                "action": "record_memory_relation",
                "reason": "记录旧偏好被新偏好替代。",
                "relation": {
                    "from_memory_id": "mem_language",
                    "to_memory_id": "mem_language_new",
                    "relation_type": "supersedes",
                    "reason": "用户后来明确改变回复语言偏好。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=RELATION_SNAPSHOT,
        ):
            normalized = validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertEqual(normalized["changes"][0]["relation"]["relation_type"], "supersedes")

    def test_record_memory_relation_accepts_prior_create_client_temp_id(self) -> None:
        proposal = {
            "schema_version": MEMORY_CHANGES_SCHEMA_VERSION,
            "turn_id": "turn_language_change",
            "reason": "同一计划内先创建新记忆再记录关系。",
            "changes": [
                {
                    "action": "create_memory",
                    "client_temp_id": "temp:new_language_preference",
                    "reason": "新增用户语言偏好。",
                    "topic_id": "topic_language",
                    "memory": {
                        "stable_key": "general.user_preference.reply_language.english",
                        "memory_type": "user_preference",
                        "summary": "用户希望默认英文回复。",
                        "content": "用户希望默认英文回复。",
                        "scope": {"level": "global"},
                        "confidence": 0.9,
                    },
                    "evidence_refs": evidence_refs(),
                },
                {
                    "action": "record_memory_relation",
                    "reason": "记录旧偏好被新偏好替代。",
                    "relation": {
                        "from_memory_id": "mem_language",
                        "to_memory_id": "temp:new_language_preference",
                        "relation_type": "supersedes",
                        "reason": "用户后来明确改变回复语言偏好。",
                    },
                    "evidence_refs": evidence_refs(),
                },
            ],
        }
        assert_valid(self, proposal)

    def test_record_memory_relation_rejects_future_client_temp_id(self) -> None:
        proposal = {
            "schema_version": MEMORY_CHANGES_SCHEMA_VERSION,
            "turn_id": "turn_language_change",
            "reason": "错误地先引用还未创建的临时记忆。",
            "changes": [
                {
                    "action": "record_memory_relation",
                    "reason": "关系写在创建之前。",
                    "relation": {
                        "from_memory_id": "mem_language",
                        "to_memory_id": "temp:new_language_preference",
                        "relation_type": "supersedes",
                        "reason": "用户后来明确改变回复语言偏好。",
                    },
                    "evidence_refs": evidence_refs(),
                },
                {
                    "action": "create_memory",
                    "client_temp_id": "temp:new_language_preference",
                    "reason": "新增用户语言偏好。",
                    "topic_id": "topic_language",
                    "memory": {
                        "stable_key": "general.user_preference.reply_language.english",
                        "memory_type": "user_preference",
                        "summary": "用户希望默认英文回复。",
                        "content": "用户希望默认英文回复。",
                        "scope": {"level": "global"},
                        "confidence": 0.9,
                    },
                    "evidence_refs": evidence_refs(),
                },
            ],
        }
        assert_validation_errors(
            self,
            proposal,
            "relation.to_memory_id does not exist or is not active",
        )

    def test_record_memory_relation_rejects_bad_targets(self) -> None:
        proposal = base_proposal(
            {
                "action": "record_memory_relation",
                "reason": "错误关系。",
                "relation": {
                    "from_memory_id": "mem_language",
                    "to_memory_id": "mem_missing",
                    "relation_type": "supersedes",
                    "reason": "目标不存在。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        assert_validation_errors(
            self,
            proposal,
            "relation.to_memory_id does not exist",
        )

    def test_record_memory_relation_rejects_self_relation(self) -> None:
        proposal = base_proposal(
            {
                "action": "record_memory_relation",
                "reason": "错误关系。",
                "relation": {
                    "from_memory_id": "mem_language",
                    "to_memory_id": "mem_language",
                    "relation_type": "supersedes",
                    "reason": "不能自己指向自己。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        assert_validation_errors(
            self,
            proposal,
            "from/to memory must be different",
        )

    def test_record_conflict_decision_requires_claims_and_decision(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "record_conflict_decision",
                    "reason": "缺字段。",
                    "target_memory_id": "mem_language",
                    "conflict": {
                        "conflict_type": "user_preference_changed",
                        "old_claim": "用户希望默认中文回复。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "conflict.new_claim must be a non-empty string",
            "conflict.decision must be a non-empty string",
            "conflict.reason must be a non-empty string",
        )

    def test_record_conflict_decision_rejects_unknown_memory(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "record_conflict_decision",
                    "reason": "不存在的 memory。",
                    "target_memory_id": "mem_missing",
                    "conflict": {
                        "conflict_type": "user_preference_changed",
                        "old_claim": "用户希望默认中文回复。",
                        "new_claim": "用户希望默认英文回复。",
                        "decision": "采用更新偏好。",
                        "reason": "用户明确纠正。",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
            "target_memory_id does not exist or is not active",
        )


class MemoryChangesCommitIdempotencyTests(unittest.TestCase):
    def test_create_memory_writes_initial_revision(self) -> None:
        proposal = base_proposal(
            {
                "action": "create_memory",
                "reason": "新增用户协作偏好。",
                "topic": {
                    "topic_key": "user_preference_answer_style",
                    "title": "回答风格偏好",
                    "summary": "记录用户对回答详略的偏好。",
                    "topic_type": "user_preference",
                    "scope": {"level": "global"},
                },
                "memory": {
                    "stable_key": "user.preference.answer_style.concise",
                    "memory_type": "user_preference",
                    "summary": "用户希望回答更简洁。",
                    "content": "用户希望回答更简洁，优先给结论。",
                    "scope": {"level": "global"},
                    "confidence": 0.8,
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))

        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("INSERT INTO memory_revisions", sql)
        self.assertIn("'create'", sql)
        self.assertIn("revision_number", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "created")

    def test_update_memory_appends_revision_and_supersedes_previous_current(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_memory",
                "reason": "更新已有偏好。",
                "target_memory_id": "mem_language",
                "memory": {
                    "summary": "用户要求默认英文回复。",
                    "content": "用户要求默认英文回复。",
                    "confidence": 0.9,
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=False,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))

        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("UPDATE memory_revisions", sql)
        self.assertIn("SET status = 'superseded'", sql)
        self.assertIn("INSERT INTO memory_revisions", sql)
        self.assertIn("'update_memory'", sql)
        self.assertNotIn("UPDATE memory_topics", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "updated")

    def test_update_memory_can_resolve_specified_conflict_in_same_transaction(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_memory",
                "reason": "用户明确纠正旧偏好。",
                "target_memory_id": "mem_language",
                "memory": {
                    "summary": "用户要求默认英文回复。",
                    "content": "用户要求默认英文回复。",
                    "confidence": 0.9,
                },
                "conflict_resolution": {
                    "conflict_ids": ["conflict_active"],
                    "status": "resolved_by_update",
                    "reason": "用户明确纠正，已用新内容更新 memory。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=False,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))

        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("UPDATE memories", sql)
        self.assertIn("INSERT INTO memory_revisions", sql)
        self.assertIn("UPDATE memory_conflict_records", sql)
        self.assertIn("status = 'resolved_by_update'", sql)
        self.assertIn("conflict_active", sql)
        self.assertNotIn("UPDATE memory_topics", sql)
        self.assertNotIn("INSERT INTO memory_relations", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["resolved_conflict_ids"], ["conflict_active"])

    def test_retire_memory_marks_memory_retired_and_disposes_conflict_without_topic_update(self) -> None:
        proposal = base_proposal(
            {
                "action": "retire_memory",
                "reason": "用户明确表示这条旧记忆不再适用。",
                "target_memory_id": "mem_language",
                "conflict_disposition": {
                    "conflict_ids": ["conflict_active"],
                    "status": "discarded_as_stale",
                    "reason": "旧记忆退役后，该冲突记录只保留历史意义。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=False,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))

        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("SET status = 'retired'", sql)
        self.assertIn("INSERT INTO memory_revisions", sql)
        self.assertIn("'retire_memory'", sql)
        self.assertIn("UPDATE memory_conflict_records", sql)
        self.assertIn("'discarded_as_stale'", sql)
        self.assertNotIn("UPDATE memory_topics", sql)
        self.assertNotIn("INSERT INTO memory_relations", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "retired")
        self.assertEqual(result["applied"][0]["status"], "retired")
        self.assertEqual(result["applied"][0]["disposed_conflict_ids"], ["conflict_active"])

    def test_retire_memory_duplicate_turn_is_skipped_not_rejected(self) -> None:
        proposal = base_proposal(
            {
                "action": "retire_memory",
                "reason": "同一 turn 重复退役。",
                "target_memory_id": "mem_language",
                "conflict_disposition": {
                    "conflict_ids": ["conflict_active"],
                    "status": "discarded_as_stale",
                    "reason": "旧记忆退役后，该冲突记录只保留历史意义。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=True,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_not_called()
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["skipped"][0]["result"], "skipped_duplicate_turn")

    def test_merge_existing_memories_updates_target_marks_sources_and_records_relations(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_existing_memories",
                "reason": "多条同类记忆重复，合并到目标记忆。",
                "target_memory_id": "mem_target",
                "source_memory_ids": ["mem_source_a", "mem_source_b"],
                "memory": {
                    "summary": "用户希望默认英文回复。",
                    "content": "合并后的完整记忆内容。",
                    "confidence": 0.91,
                },
                "conflict_disposition": {
                    "conflict_ids": ["conflict_source_a"],
                    "status": "resolved_by_merge",
                    "reason": "source 记忆被合并后，该冲突由合并结果解决。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=False,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))

        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("UPDATE memories", sql)
        self.assertIn("SET status = 'merged'", sql)
        self.assertIn("mem_source_a", sql)
        self.assertIn("mem_source_b", sql)
        self.assertIn("INSERT INTO memory_revisions", sql)
        self.assertIn("'merge_existing_memories'", sql)
        self.assertIn("INSERT INTO memory_relations", sql)
        self.assertIn("'merges'", sql)
        self.assertIn("UPDATE memory_conflict_records", sql)
        self.assertIn("'resolved_by_merge'", sql)
        self.assertNotIn("UPDATE memory_topics", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "merged")
        self.assertEqual(result["applied"][0]["source_memory_ids"], ["mem_source_a", "mem_source_b"])
        self.assertEqual(result["applied"][0]["disposed_conflict_ids"], ["conflict_source_a"])

    def test_merge_existing_memories_duplicate_turn_is_skipped_when_all_targets_seen(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_existing_memories",
                "reason": "同一 turn 重复合并。",
                "target_memory_id": "mem_target",
                "source_memory_ids": ["mem_source_a"],
                "memory": {
                    "summary": "用户希望默认英文回复。",
                    "content": "合并后的完整记忆内容。",
                    "confidence": 0.91,
                },
                "conflict_disposition": {
                    "conflict_ids": ["conflict_source_a"],
                    "status": "resolved_by_merge",
                    "reason": "source 记忆被合并后，该冲突由合并结果解决。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=True,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_not_called()
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["skipped"][0]["result"], "skipped_duplicate_turn")

    def test_update_topic_skips_when_title_and_summary_are_unchanged(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_topic",
                "reason": "主题已经是目标展示语义。",
                "target_topic_id": "topic_language",
                "topic": {
                    "title": "回复语言偏好",
                    "summary": "记录用户当前有效回复语言偏好。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._json_from_psql_object",
            return_value={
                "found": True,
                "topic_id": "topic_language",
                "title": "回复语言偏好",
                "summary": "记录用户当前有效回复语言偏好。",
            },
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_not_called()
        self.assertEqual(result["counts"]["applied"], 0)
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["skipped"][0]["result"], "skipped_no_change")

    def test_update_topic_updates_when_title_changes(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_topic",
                "reason": "修正旧主题标题。",
                "target_topic_id": "topic_language",
                "topic": {
                    "title": "回复语言偏好",
                    "summary": "记录用户当前有效回复语言偏好。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._json_from_psql_object",
            return_value={
                "found": True,
                "topic_id": "topic_language",
                "title": "默认使用中文回复",
                "summary": "记录用户旧回复语言偏好。",
            },
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("UPDATE memory_topics", sql)
        self.assertNotIn("UPDATE memories", sql)
        self.assertNotIn("INSERT INTO memories", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "updated")
        self.assertEqual(result["applied"][0]["target_id"], "topic_language")
        self.assertIn("message", result["applied"][0])
        self.assertIn("title", result["applied"][0]["changed_fields"])

    def test_move_memory_topic_accepts_active_memory_and_active_topics(self) -> None:
        proposal = base_proposal(
            {
                "action": "move_memory_topic",
                "reason": "memory 内容正确但主题归类错误，移动到用户协作偏好主题。",
                "target_memory_id": "mem_language",
                "from_topic_id": "topic_language",
                "to_topic_id": "topic_user_collaboration",
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MOVE_TOPIC_SNAPSHOT,
        ):
            normalized = validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertEqual(normalized["changes"][0]["action"], "move_memory_topic")

    def test_move_memory_topic_rejects_mixed_patches_and_bad_targets(self) -> None:
        assert_validation_errors(
            self,
            base_proposal(
                {
                    "action": "move_memory_topic",
                    "reason": "模型试图移动主题时顺手改 memory。",
                    "target_memory_id": "mem_language",
                    "from_topic_id": "topic_language",
                    "to_topic_id": "topic_missing",
                    "memory": {"summary": "不允许", "content": "不允许", "confidence": 0.9},
                    "evidence_refs": evidence_refs(),
                }
            ),
            "has unsupported fields",
            "to_topic_id does not exist or is not active",
        )

    def test_move_memory_topic_rejects_from_topic_mismatch_and_same_topic(self) -> None:
        proposal = base_proposal(
            {
                "action": "move_memory_topic",
                "reason": "from_topic 不匹配且 to/from 相同。",
                "target_memory_id": "mem_language",
                "from_topic_id": "topic_user_collaboration",
                "to_topic_id": "topic_user_collaboration",
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MOVE_TOPIC_SNAPSHOT,
        ):
            with self.assertRaises(MemoryChangesValidationError) as raised:
                validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        text = "\n".join(raised.exception.errors)
        self.assertIn("from_topic_id does not match current memory topic", text)
        self.assertIn("to_topic_id must be different from from_topic_id", text)

    def test_move_memory_topic_updates_only_topic_id_and_writes_revision(self) -> None:
        proposal = base_proposal(
            {
                "action": "move_memory_topic",
                "reason": "memory 内容正确但主题归类错误。",
                "target_memory_id": "mem_language",
                "from_topic_id": "topic_language",
                "to_topic_id": "topic_user_collaboration",
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MOVE_TOPIC_SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=False,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("UPDATE memories", sql)
        self.assertIn("topic_id = 'topic_user_collaboration'", sql)
        self.assertIn("'move_memory_topic'", sql)
        self.assertNotIn("UPDATE memory_topics", sql)
        self.assertNotIn("summary =", sql)
        self.assertNotIn("content =", sql)
        self.assertNotIn("confidence =", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "moved")
        self.assertEqual(result["applied"][0]["from_topic_id"], "topic_language")
        self.assertEqual(result["applied"][0]["to_topic_id"], "topic_user_collaboration")

    def test_merge_topics_accepts_active_target_and_sources(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_topics",
                "reason": "两个主题语义重复，合并减少召回噪声。",
                "target_topic_id": "topic_language",
                "source_topic_ids": ["topic_source_duplicate", "topic_other_duplicate"],
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_TOPICS_SNAPSHOT,
        ):
            normalized = validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertEqual(normalized["changes"][0]["action"], "merge_topics")

    def test_merge_topics_rejects_bad_sources_and_mixed_patches(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_topics",
                "reason": "错误主题合并计划。",
                "target_topic_id": "topic_language",
                "source_topic_ids": [
                    "topic_language",
                    "topic_source_duplicate",
                    "topic_source_duplicate",
                    "topic_missing",
                ],
                "topic": {"title": "不允许顺手改 target topic"},
                "memory": {"summary": "不允许", "content": "不允许", "confidence": 0.9},
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_TOPICS_SNAPSHOT,
        ):
            with self.assertRaises(MemoryChangesValidationError) as raised:
                validate_memory_changes(proposal, pg_config_path=Path("unused.json"))
        text = "\n".join(raised.exception.errors)
        self.assertIn("has unsupported fields", text)
        self.assertIn("must not equal target_topic_id", text)
        self.assertIn("is duplicated", text)
        self.assertIn("does not exist or is not active", text)

    def test_merge_topics_moves_source_active_memories_and_marks_sources_merged(self) -> None:
        proposal = base_proposal(
            {
                "action": "merge_topics",
                "reason": "重复主题合并到 target topic。",
                "target_topic_id": "topic_language",
                "source_topic_ids": ["topic_source_duplicate", "topic_other_duplicate"],
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=MERGE_TOPICS_SNAPSHOT,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("UPDATE memories", sql)
        self.assertIn("topic_id = 'topic_language'", sql)
        self.assertIn("UPDATE memory_topics", sql)
        self.assertIn("status = 'merged'", sql)
        self.assertIn("'merge_topics'", sql)
        self.assertNotIn("summary =", sql)
        self.assertNotIn("content =", sql)
        self.assertNotIn("confidence =", sql)
        self.assertNotIn("title =", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "merged")
        self.assertEqual(result["applied"][0]["target_topic_id"], "topic_language")
        self.assertEqual(
            result["applied"][0]["source_topic_ids"],
            ["topic_source_duplicate", "topic_other_duplicate"],
        )

    def test_record_conflict_decision_skips_identical_existing_record(self) -> None:
        change = {
            "action": "record_conflict_decision",
            "reason": "记录用户偏好变更冲突。",
            "target_memory_id": "mem_language",
            "conflict": {
                "conflict_type": "user_preference_changed",
                "old_claim": "用户希望默认中文回复。",
                "new_claim": "用户希望默认英文回复。",
                "decision": "采用更新的明确用户偏好。",
                "reason": "新证据是用户明确纠正。",
                "status": "resolved_by_update",
            },
            "evidence_refs": evidence_refs(),
        }
        proposal = base_proposal(change)
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._json_from_psql_object",
            return_value={
                "found": True,
                "conflict_id": "conflict_existing",
                "reason": "新证据是用户明确纠正。",
                "status": "resolved_by_update",
                "evidence_refs": evidence_refs(),
            },
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_not_called()
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["skipped"][0]["result"], "skipped_existing")
        self.assertIn("message", result["skipped"][0])

    def test_conflict_decision_id_ignores_wording_variations(self) -> None:
        first = _digest_id(
            "conflict",
            "local-default-user",
            "mem_language",
            "turn_language_change",
            "user_preference_changed",
        )
        second = _digest_id(
            "conflict",
            "local-default-user",
            "mem_language",
            "turn_language_change",
            "user_preference_changed",
        )
        self.assertEqual(first, second)

    def test_record_conflict_decision_creates_missing_record(self) -> None:
        proposal = base_proposal(
            {
                "action": "record_conflict_decision",
                "reason": "记录用户偏好变更冲突。",
                "target_memory_id": "mem_language",
                "conflict": {
                    "conflict_type": "user_preference_changed",
                    "old_claim": "用户希望默认中文回复。",
                    "new_claim": "用户希望默认英文回复。",
                    "decision": "采用更新的明确用户偏好。",
                    "reason": "新证据是用户明确纠正。",
                    "status": "resolved_by_update",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._json_from_psql_object",
            return_value={"found": False},
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("INSERT INTO memory_conflict_records", sql)
        self.assertNotIn("UPDATE memories", sql)
        self.assertNotIn("UPDATE memory_topics", sql)
        self.assertNotIn("INSERT INTO memory_relations", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "created")
        self.assertEqual(result["applied"][0]["target_id"], result["applied"][0]["conflict_id"])
        self.assertIn("message", result["applied"][0])

    def test_record_memory_relation_creates_missing_record(self) -> None:
        proposal = base_proposal(
            {
                "action": "record_memory_relation",
                "reason": "记录旧偏好被新偏好替代。",
                "relation": {
                    "from_memory_id": "mem_language",
                    "to_memory_id": "mem_language_new",
                    "relation_type": "supersedes",
                    "reason": "用户后来明确改变回复语言偏好。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=RELATION_SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._json_from_psql_object",
            return_value={"found": False},
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))

        run_psql.assert_called_once()
        sql = run_psql.call_args.args[1]
        self.assertIn("INSERT INTO memory_relations", sql)
        self.assertIn("supersedes", sql)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "created")

    def test_record_memory_relation_skips_identical_existing_record(self) -> None:
        proposal = base_proposal(
            {
                "action": "record_memory_relation",
                "reason": "记录旧偏好被新偏好替代。",
                "relation": {
                    "from_memory_id": "mem_language",
                    "to_memory_id": "mem_language_new",
                    "relation_type": "supersedes",
                    "reason": "用户后来明确改变回复语言偏好。",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=RELATION_SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._json_from_psql_object",
            return_value={
                "found": True,
                "relation_id": "rel_existing",
                "reason": "用户后来明确改变回复语言偏好。",
                "evidence_refs": evidence_refs(),
            },
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))

        run_psql.assert_not_called()
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["skipped"][0]["result"], "skipped_existing")

    def test_update_memory_duplicate_turn_is_skipped_not_rejected(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_memory",
                "reason": "同一 turn 重复更新。",
                "target_memory_id": "mem_language",
                "memory": {
                    "summary": "用户要求默认英文回复。",
                    "content": "用户要求默认英文回复。",
                    "confidence": 0.9,
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ), patch(
            "memory_lab.memory_changes._memory_already_processed_turn",
            return_value=True,
        ), patch("memory_lab.memory_changes._run_psql") as run_psql:
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        run_psql.assert_not_called()
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["skipped"][0]["result"], "skipped_duplicate_turn")

    def test_commit_response_contains_model_friendly_top_level_fields(self) -> None:
        proposal = base_proposal(
            {
                "action": "noop",
                "reason": "当前 turn 没有长期价值。",
                "evidence_refs": [],
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ):
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertTrue(result["ok"])
        self.assertIn("warnings", result)
        self.assertIn("suggested_next_actions", result)
        self.assertEqual(result["noop"][0]["result"], "noop")
        self.assertIn("message", result["noop"][0])

    def test_commit_response_reports_rejected_changes_with_warning(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_topic",
                "reason": "错误目标。",
                "target_topic_id": "topic_missing",
                "topic": {"title": "回复语言偏好"},
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ):
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["counts"]["rejected"], 1)
        self.assertEqual(result["warnings"][0]["type"], "PARTIAL_REJECTION")
        self.assertTrue(result["suggested_next_actions"])
        self.assertIn("TOPIC_NOT_FOUND", result["rejected"][0]["error_codes"])

    def test_commit_response_classifies_unsupported_field_errors(self) -> None:
        proposal = base_proposal(
            {
                "action": "update_topic",
                "reason": "模型试图改不允许字段。",
                "target_topic_id": "topic_language",
                "topic": {
                    "title": "回复语言偏好",
                    "topic_key": "general.reply_language",
                },
                "evidence_refs": evidence_refs(),
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ):
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertFalse(result["ok"])
        self.assertIn("UNSUPPORTED_FIELD", result["rejected"][0]["error_codes"])
        self.assertTrue(result["rejected"][0]["suggested_next_actions"])

    def test_commit_response_classifies_existing_topic_key(self) -> None:
        proposal = {
            "schema_version": MEMORY_CHANGES_SCHEMA_VERSION,
            "turn_id": "turn_language_change",
            "reason": "错误地用 create_memory 携带已有 topic_key。",
            "changes": [
                {
                    "action": "create_memory",
                    "reason": "已有 topic 下新增记忆时应使用 topic_id。",
                    "topic": {
                        "topic_key": "general.reply_language",
                        "title": "回复语言偏好",
                        "summary": "不应通过 create_memory 改已有 topic。",
                        "topic_type": "user_preference",
                        "scope": {"level": "global"},
                    },
                    "memory": {
                        "stable_key": "general.user_preference.reply_language.new",
                        "memory_type": "user_preference",
                        "summary": "用户希望默认英文回复。",
                        "content": "用户希望默认英文回复。",
                        "scope": {"level": "global"},
                        "confidence": 0.9,
                    },
                    "evidence_refs": evidence_refs(),
                }
            ],
        }
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ):
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertFalse(result["ok"])
        self.assertIn("TOPIC_KEY_EXISTS", result["rejected"][0]["error_codes"])
        self.assertIn("Use topic_id", result["rejected"][0]["suggested_next_actions"][0])

    def test_commit_response_classifies_wrong_turn_evidence(self) -> None:
        proposal = base_proposal(
            {
                "action": "noop",
                "reason": "错误证据。",
                "evidence_refs": [{"evidence_id": "ev_other_turn"}],
            }
        )
        with patch(
            "memory_lab.memory_changes.load_pg_config",
            return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
        ), patch(
            "memory_lab.memory_changes._reference_snapshot",
            return_value=SNAPSHOT,
        ):
            result = commit_memory_changes(proposal, pg_config_path=Path("unused.json"))
        self.assertFalse(result["ok"])
        self.assertIn("EVIDENCE_NOT_IN_TURN", result["rejected"][0]["error_codes"])
if __name__ == "__main__":
    unittest.main()
