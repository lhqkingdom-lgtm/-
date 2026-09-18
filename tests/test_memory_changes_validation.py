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
    "topic_ids": ["topic_language"],
    "stable_keys": ["general.user_preference.reply_language"],
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
                        "status": "resolved",
                    },
                    "evidence_refs": evidence_refs(),
                }
            ),
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
                    "memory_type": "collaboration_preference",
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
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "updated")

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
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "updated")
        self.assertEqual(result["applied"][0]["target_id"], "topic_language")
        self.assertIn("message", result["applied"][0])
        self.assertIn("title", result["applied"][0]["changed_fields"])

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
                "status": "resolved",
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
                "status": "resolved",
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
                    "status": "resolved",
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
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["applied"][0]["result"], "created")
        self.assertEqual(result["applied"][0]["target_id"], result["applied"][0]["conflict_id"])
        self.assertIn("message", result["applied"][0])

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
