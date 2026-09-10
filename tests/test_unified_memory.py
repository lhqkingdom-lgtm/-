from __future__ import annotations

import copy
import re
import unittest

from memory_lab.unified_memory import (
    CANDIDATE_SCHEMA_VERSION,
    MEMORY_TYPES,
    MemoryCandidateValidationError,
    MemoryEvidence,
    decide_memory_relation,
    scope_fingerprint_for,
    suggest_stable_key,
    validate_memory_candidate,
    validate_memory_candidate_batch,
)


class UnifiedMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user_old = MemoryEvidence(
            evidence_id="e_user_old",
            turn_id="turn_a",
            source_kind="user_message",
            occurred_at_ms=100,
            part_ordinal=0,
            text_offset=10,
            project_key="directory:project_a",
            session_id="session_a",
            explicit=True,
        )
        self.user_new = MemoryEvidence(
            evidence_id="e_user_new",
            turn_id="turn_b",
            source_kind="user_message",
            occurred_at_ms=200,
            part_ordinal=0,
            text_offset=5,
            project_key="directory:project_b",
            session_id="session_b",
            explicit=True,
        )
        self.tool = MemoryEvidence(
            evidence_id="e_tool",
            turn_id="turn_a",
            source_kind="tool_result",
            occurred_at_ms=150,
            project_key="directory:project_a",
            session_id="session_a",
            verified=True,
        )
        self.assistant = MemoryEvidence(
            evidence_id="e_assistant",
            turn_id="turn_a",
            source_kind="assistant_final",
            occurred_at_ms=140,
            project_key="directory:project_a",
            session_id="session_a",
        )
        self.catalog = [self.user_old, self.user_new, self.tool, self.assistant]

    @staticmethod
    def proposal(
        *,
        stable_key: str = "preference.response.structure",
        memory_type: str = "preference",
        statement: str = "回复时先给出结论，再说明必要步骤。",
        summary: str = "先结论后步骤",
        scope: dict | None = None,
        conditions: dict | None = None,
        trust_status: str = "declared",
        evidence_refs: list[str] | None = None,
        status: str = "active",
        **content: object,
    ) -> dict:
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "stable_key": stable_key,
            "type": memory_type,
            "summary": summary,
            "scope": scope or {"level": "global"},
            "conditions": {} if conditions is None else conditions,
            "trust_status": trust_status,
            "content": {"statement": statement, **content},
            "evidence_refs": evidence_refs or ["e_user_old"],
            "status": status,
        }

    def validate(self, proposal: dict, catalog: list[MemoryEvidence] | None = None) -> dict:
        return validate_memory_candidate(proposal, catalog or self.catalog)

    def test_all_memory_types_and_unverified_formal_candidate_are_supported(self) -> None:
        for memory_type in sorted(MEMORY_TYPES):
            with self.subTest(memory_type=memory_type):
                proposal = self.proposal(
                    stable_key=f"{memory_type}.durable_fact",
                    memory_type=memory_type,
                    trust_status="unverified",
                )
                candidate = self.validate(proposal)
                self.assertEqual(candidate["type"], memory_type)
                self.assertEqual(candidate["trust_status"], "unverified")
                self.assertEqual(candidate["status"], "active")
                self.assertTrue(candidate["candidate_id"].startswith("memcand_"))
                self.assertTrue(candidate["semantic_fingerprint"].startswith("memsem_"))

    def test_candidate_normalizes_safe_structured_content_without_a_tool_allowlist(self) -> None:
        proposal = self.proposal(
            stable_key="engineering.generic.cache_failure",
            memory_type="engineering",
            summary="构建缓存异常的可复用处理经验",
            statement="清理失效缓存后重新执行验证可恢复构建。",
            scope={"level": "project", "project_key": "directory:project_a"},
            conditions={"runtime": "A completely new tool 9.2", "mode": "debug"},
            trust_status="verified",
            evidence_refs=["e_tool", "e_assistant"],
            symptom="构建输出与当前源码不一致",
            solution={"operation": "刷新缓存", "verification": "测试通过"},
        )
        candidate = self.validate(proposal)
        self.assertEqual(candidate["conditions"]["runtime"], "A completely new tool 9.2")
        self.assertEqual(candidate["content"]["solution"]["verification"], "测试通过")
        self.assertEqual(candidate["trust_status"], "verified")

    def test_semantic_fingerprint_excludes_summary_scope_evidence_and_trust(self) -> None:
        first = self.validate(self.proposal(trust_status="declared"))
        second_proposal = self.proposal(
            summary="同一个偏好的另一种概要",
            scope={"level": "project", "project_key": "directory:project_b"},
            trust_status="unverified",
            evidence_refs=["e_user_new"],
        )
        second = self.validate(second_proposal)
        self.assertEqual(first["semantic_fingerprint"], second["semantic_fingerprint"])
        self.assertNotEqual(first["candidate_id"], second["candidate_id"])
        self.assertNotEqual(scope_fingerprint_for(first["scope"]), scope_fingerprint_for(second["scope"]))

    def test_scope_is_exactly_supported_by_cited_conversation_evidence(self) -> None:
        project = self.validate(
            self.proposal(
                scope={"level": "project", "project_key": "directory:project_a"}
            )
        )
        self.assertEqual(project["scope"]["project_key"], "directory:project_a")
        session = self.validate(
            self.proposal(scope={"level": "session", "session_id": "session_a"})
        )
        self.assertEqual(session["scope"], {"level": "session", "session_id": "session_a"})
        domain = self.validate(
            self.proposal(scope={"level": "domain", "domain": "Embedded"})
        )
        self.assertEqual(domain["scope"]["domain"], "embedded")
        cross_project = self.validate(
            self.proposal(
                scope={
                    "level": "cross_project",
                    "project_keys": ["directory:project_b", "directory:project_a"],
                },
                evidence_refs=["e_user_old", "e_user_new"],
            )
        )
        self.assertEqual(
            cross_project["scope"]["project_keys"],
            ["directory:project_a", "directory:project_b"],
        )

        with self.assertRaisesRegex(MemoryCandidateValidationError, "not supported") as raised:
            self.validate(
                self.proposal(
                    scope={"level": "project", "project_key": "directory:project_b"},
                    evidence_refs=["e_user_old"],
                )
            )
        self.assertEqual(raised.exception.code, "invalid_scope")

    def test_only_normalized_cp_turn_evidence_is_accepted(self) -> None:
        external = MemoryEvidence(
            evidence_id="external_file",
            turn_id="turn_external",
            source_kind="tool_result",
            occurred_at_ms=1,
            origin="project_file",
        )
        with self.assertRaises(MemoryCandidateValidationError) as raised:
            validate_memory_candidate(
                self.proposal(
                    trust_status="unverified", evidence_refs=["external_file"]
                ),
                [external],
            )
        self.assertEqual(raised.exception.code, "invalid_evidence_origin")

        with self.assertRaises(MemoryCandidateValidationError) as raised:
            self.validate(self.proposal(evidence_refs=["missing_evidence"]))
        self.assertEqual(raised.exception.code, "unknown_evidence")

    def test_secret_path_size_and_temporary_content_fail_with_stable_reasons(self) -> None:
        cases = (
            ("令牌 api_key=sk-very-secret-value", "sensitive_content"),
            ("API key is natural-language-secret", "sensitive_content"),
            ("访问令牌为 chinese-language-secret", "sensitive_content"),
            ("密码是 another-language-secret", "sensitive_content"),
            ("读取 D:\\private\\firmware.c 后处理", "unsafe_path"),
            ("读取D:\\private\\firmware.c后处理", "unsafe_path"),
            ("仅本次会话先使用这一规则", "temporary_content"),
            ("x" * 1_001, "text_too_long"),
        )
        for statement, expected in cases:
            with self.subTest(expected=expected), self.assertRaises(
                MemoryCandidateValidationError
            ) as raised:
                self.validate(self.proposal(statement=statement))
            self.assertEqual(raised.exception.code, expected)

        nested_secret = self.proposal(secret_note={"api_key": "hidden-value"})
        with self.assertRaises(MemoryCandidateValidationError) as raised:
            self.validate(nested_secret)
        self.assertEqual(raised.exception.code, "sensitive_content")

    def test_atomic_batch_keeps_good_items_and_reports_safe_rejection_reasons(self) -> None:
        first = self.proposal()
        second = self.proposal(
            stable_key="workflow.code.review",
            memory_type="workflow",
            summary="代码审核流程",
            statement="提交前先完成代码审核。",
        )
        multiplexed = self.proposal(
            stable_key="other.many_claims",
            memory_type="other",
            trust_status="unverified",
            claims=["第一条", "第二条"],
        )
        result = validate_memory_candidate_batch(
            [first, second, multiplexed, copy.deepcopy(first)], self.catalog
        )
        self.assertEqual(len(result["accepted"]), 2)
        self.assertEqual(
            [item["reason"] for item in result["rejected"]],
            ["non_atomic_candidate", "duplicate_atomic_candidate"],
        )
        self.assertNotIn("第一条", str(result["rejected"]))

    def test_trust_status_cannot_be_invented(self) -> None:
        with self.assertRaises(MemoryCandidateValidationError) as raised:
            self.validate(self.proposal(trust_status="verified"))
        self.assertEqual(raised.exception.code, "invalid_trust_status")
        with self.assertRaises(MemoryCandidateValidationError):
            self.validate(
                self.proposal(
                    trust_status="observed", evidence_refs=["e_assistant"]
                )
            )
        with self.assertRaises(MemoryCandidateValidationError):
            self.validate(
                self.proposal(
                    trust_status="declared", evidence_refs=["e_assistant"]
                )
            )

    def test_relation_actions_cover_create_merge_scope_and_condition_variants(self) -> None:
        existing = self.validate(self.proposal())
        create = decide_memory_relation(None, existing, self.catalog)
        self.assertEqual(create["action"], "create")

        same = self.validate(
            self.proposal(summary="新的概要", evidence_refs=["e_user_new"])
        )
        merge = decide_memory_relation(existing, same, self.catalog)
        self.assertEqual(merge["action"], "merge_evidence")

        scoped = self.validate(
            self.proposal(
                scope={"level": "project", "project_key": "directory:project_b"},
                evidence_refs=["e_user_new"],
            )
        )
        variant = decide_memory_relation(existing, scoped, self.catalog)
        self.assertEqual(variant["action"], "add_scope_variant")
        self.assertEqual(variant["relation"], "scope_variant")

        conditional = self.validate(
            self.proposal(conditions={"language": "English"})
        )
        condition_variant = decide_memory_relation(existing, conditional, self.catalog)
        self.assertEqual(condition_variant["action"], "add_scope_variant")
        self.assertEqual(condition_variant["relation"], "condition_variant")

    def test_latest_explicit_user_contract_revises_even_within_same_message(self) -> None:
        later_clause = MemoryEvidence(
            evidence_id="e_later_clause",
            turn_id="turn_a",
            source_kind="user_message",
            occurred_at_ms=100,
            part_ordinal=0,
            text_offset=90,
            project_key="directory:project_a",
            session_id="session_a",
            explicit=True,
        )
        catalog = [self.user_old, later_clause]
        existing = validate_memory_candidate(self.proposal(), catalog)
        incoming = validate_memory_candidate(
            self.proposal(
                statement="回复时先说明步骤，最后再给出结论。",
                summary="先步骤后结论",
                evidence_refs=["e_later_clause"],
            ),
            catalog,
        )
        decision = decide_memory_relation(existing, incoming, catalog)
        self.assertEqual(decision["action"], "revise")
        self.assertTrue(decision["latest_wins"])
        self.assertEqual(decision["winner"], "incoming")

        reverse = decide_memory_relation(incoming, existing, catalog)
        self.assertEqual(reverse["action"], "revise")
        self.assertEqual(reverse["winner"], "existing")

    def test_engineering_and_other_technical_claims_never_use_latest_wins(self) -> None:
        existing = self.validate(
            self.proposal(
                stable_key="engineering.compiler.optimization",
                memory_type="engineering",
                statement="开启该优化能够通过验证。",
                summary="优化配置有效",
                trust_status="unverified",
            )
        )
        incoming = self.validate(
            self.proposal(
                stable_key="engineering.compiler.optimization",
                memory_type="engineering",
                statement="开启该优化会导致验证失败。",
                summary="优化配置无效",
                trust_status="unverified",
                evidence_refs=["e_user_new"],
            )
        )
        decision = decide_memory_relation(existing, incoming, self.catalog)
        self.assertEqual(decision["action"], "open_conflict")
        self.assertFalse(decision["latest_wins"])

    def test_suggested_stable_keys_are_safe_and_deterministic(self) -> None:
        english = suggest_stable_key("engineering", "Keil missing header include path")
        chinese = suggest_stable_key("project", "产品的串口通信约束")
        self.assertEqual(
            english,
            suggest_stable_key("engineering", "Keil missing header include path"),
        )
        self.assertEqual(
            chinese,
            suggest_stable_key("project", "产品的串口通信约束"),
        )
        pattern = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
        self.assertRegex(english, pattern)
        self.assertRegex(chinese, pattern)


if __name__ == "__main__":
    unittest.main()
