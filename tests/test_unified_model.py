from __future__ import annotations

import copy
import json
import unittest

from memory_lab.llm_client import JsonClientTransportError
from memory_lab.unified_model import (
    MODEL_OUTPUT_SCHEMA_VERSION,
    MAX_EXISTING_MEMORY_PAYLOAD_CHARS,
    REVIEW_OUTPUT_SCHEMA_VERSION,
    DeepSeekUnifiedMemoryPlanner,
    DeepSeekUnifiedMemoryReviewer,
    UnifiedOutputError,
    UnifiedTransportError,
)


def turn_bundle(
    *,
    suffix: str = "a",
    user_text: str = "以后无论在哪个项目，回复时先给结论。",
    assistant_text: str = "已记录这个长期回复偏好。",
    project_key: str | None = None,
    session_id: str | None = None,
    occurred_at_ms: int = 1000,
    tools: list[dict] | None = None,
) -> dict:
    project_key = project_key or f"project-{suffix}"
    session_id = session_id or f"session-{suffix}"
    return {
        "schema_version": "memory-lab-turn-evidence-v1",
        "bundle_id": f"bundle-{suffix}",
        "bundle_hash": f"hash-{suffix}",
        "turn": {
            "turn_id": f"turn-{suffix}",
            "state": "completed",
            "occurred_at_ms": occurred_at_ms,
            "source_signature": f"signature-{suffix}",
            "user_message_id": f"user-{suffix}",
            "terminal_assistant_id": f"assistant-{suffix}",
        },
        "workspace": {
            "project_key": project_key,
            "display_name": f"project {suffix}",
            "path_alias": f"<WORKSPACE:{suffix}>",
            "path_fingerprint": f"fingerprint-{suffix}",
        },
        "session": {
            "session_id": session_id,
            "parent_session_id": None,
            "root_session_id": session_id,
            "branch_depth": 0,
            "lineage": [],
        },
        "user": {
            "message_id": f"user-{suffix}",
            "text_parts": [
                {
                    "ordinal": 1,
                    "source_part_id": f"user-part-{suffix}",
                    "text": user_text,
                    "truncated": False,
                }
            ],
        },
        "user_answers": [],
        "assistant": {
            "message_id": f"assistant-{suffix}",
            "text_parts": [
                {
                    "ordinal": 1,
                    "source_part_id": f"assistant-part-{suffix}",
                    "text": assistant_text,
                    "truncated": False,
                }
            ],
        },
        "tools": tools or [],
        "safety": {"hidden_reasoning_included": False},
    }


def tool_part(suffix: str = "a") -> dict:
    return {
        "ordinal": 1,
        "message_ordinal": 2,
        "part_ordinal": 2,
        "assistant_part_ordinal": 2,
        "source_message_id": f"assistant-{suffix}",
        "source_part_id": f"tool-part-{suffix}",
        "name": "build",
        "status": "completed",
        "input": {
            "present": True,
            "text": '{"target":"demo"}',
            "truncated": False,
            "sanitized_hash": "input-hash",
        },
        "output": {
            "present": True,
            "text": "build succeeded; 0 errors",
            "truncated": False,
            "sanitized_hash": "output-hash",
        },
        "error": {
            "present": False,
            "text": None,
            "truncated": False,
            "sanitized_hash": "error-hash",
        },
    }


def user_answer_part(suffix: str = "a") -> dict:
    return {
        "ordinal": 1001,
        "message_ordinal": 1,
        "source_message_id": f"assistant-{suffix}",
        "source_part_id": f"question-part-{suffix}:answer:1",
        "occurred_at_ms": 1010,
        "text": (
            'selected_answers=["基于现有工程二次开发"]\n'
            'question_header="工程方式"\n'
            'question="希望以现有工程为基础还是新建工程？"'
        ),
        "truncated": False,
        "sanitized_hash": "answer-hash",
    }


def question_tool_part(suffix: str = "a") -> dict:
    return {
        "ordinal": 2,
        "message_ordinal": 1,
        "part_ordinal": 2,
        "assistant_part_ordinal": 2,
        "source_message_id": f"assistant-{suffix}",
        "source_part_id": f"question-part-{suffix}",
        "name": "question",
        "status": "completed",
        "input": {
            "present": True,
            "text": '{"questions":[{"question":"希望以现有工程为基础还是新建工程？"}]}',
            "truncated": False,
            "sanitized_hash": "question-input-hash",
        },
        "output": {
            "present": True,
            "text": "User selected 基于现有工程二次开发",
            "truncated": False,
            "sanitized_hash": "question-output-hash",
        },
        "error": {
            "present": False,
            "text": None,
            "truncated": False,
            "sanitized_hash": "question-error-hash",
        },
    }


class FakeClient:
    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.responder(kwargs["user_payload"])


def empty_output(request: dict) -> dict:
    return {
        "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "request_id": request["request_id"],
        "candidates": [],
    }


def candidate_output(
    request: dict,
    *,
    observation_index: int = 0,
    stable_key: str = "preference.response.structure",
    memory_type: str = "preference",
    scope: dict | None = None,
    trust_status: str = "declared",
    status: str = "active",
    action_kind: str = "create_or_match",
    related_candidate_ref=None,
    related_memory_ref=None,
    summary: str = "回复先给结论",
) -> dict:
    observation = request["turns"][0]["observations"][observation_index]
    text = observation["text"]
    quote = text
    return {
        "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "request_id": request["request_id"],
        "candidates": [
            {
                "candidate_ref": "candidate_1",
                "stable_key": stable_key,
                "type": memory_type,
                "summary": summary,
                "scope": scope
                or {
                    "level": "global",
                    "project_refs": [],
                    "session_refs": [],
                    "domain": None,
                },
                "conditions": {},
                "trust_status": trust_status,
                "content": {"statement": "回复时先给结论，再说明步骤。"},
                "status": status,
                "evidence": [
                    {
                        "evidence_ref": observation["evidence_ref"],
                        "quote": quote,
                    }
                ],
                "action_hint": {
                    "kind": action_kind,
                    "related_candidate_ref": related_candidate_ref,
                    "related_memory_ref": related_memory_ref,
                    "reason": "原文明确表达可复用规则",
                },
            }
        ],
    }


class UnifiedModelTests(unittest.TestCase):
    def test_planner_returns_generic_validated_candidate_and_never_commits(self) -> None:
        client = FakeClient(candidate_output)
        plan = DeepSeekUnifiedMemoryPlanner(client).plan(turn_bundle())

        self.assertEqual(plan["candidate_count"], 1)
        self.assertFalse(plan["commit_allowed"])
        memory = plan["candidates"][0]["memory"]
        self.assertEqual(memory["schema_version"], "unified-memory-candidate-v1")
        self.assertEqual(memory["scope"], {"level": "global"})
        self.assertEqual(memory["trust_status"], "declared")
        self.assertTrue(memory["candidate_id"].startswith("memcand_"))
        self.assertEqual(len(plan["evidence_catalog"]), 1)
        self.assertEqual(plan["evidence_catalog"][0]["turn_id"], "turn-a")
        self.assertEqual(plan["evidence_catalog"][0]["occurred_at_ms"], 1000)

    def test_user_answer_is_visible_as_explicit_user_evidence(self) -> None:
        bundle = turn_bundle(
            user_text="我要开发 MJ60B 温控风扇控制器。",
            assistant_text="已完成需求对齐。",
        )
        bundle["user_answers"] = [user_answer_part()]

        def answer_candidate(request: dict) -> dict:
            answer_index = next(
                index
                for index, item in enumerate(request["turns"][0]["observations"])
                if item["source_type"] == "user_answer"
            )
            return candidate_output(
                request,
                observation_index=answer_index,
                stable_key="project.development_mode",
                memory_type="project",
                scope={
                    "level": "project",
                    "project_refs": ["project_1"],
                    "session_refs": [],
                    "domain": None,
                },
                trust_status="declared",
                summary="当前项目基于现有工程二次开发",
            )

        client = FakeClient(answer_candidate)
        plan = DeepSeekUnifiedMemoryPlanner(client).plan(bundle)

        observations = client.calls[0]["user_payload"]["turns"][0]["observations"]
        self.assertIn("user_answer", {item["source_type"] for item in observations})
        memory = plan["candidates"][0]["memory"]
        self.assertEqual(memory["trust_status"], "declared")
        self.assertEqual(plan["evidence_catalog"][0]["source_kind"], "user_answer")
        self.assertTrue(plan["evidence_catalog"][0]["explicit"])

    def test_question_tool_payload_is_omitted_when_user_answer_exists(self) -> None:
        bundle = turn_bundle(
            user_text="我要开发 MJ60B 温控风扇控制器。",
            assistant_text="已完成需求对齐。",
            tools=[question_tool_part()],
        )
        bundle["user_answers"] = [user_answer_part()]

        client = FakeClient(empty_output)
        DeepSeekUnifiedMemoryPlanner(client).plan(bundle)

        observations = client.calls[0]["user_payload"]["turns"][0]["observations"]
        self.assertIn("user_answer", {item["source_type"] for item in observations})
        self.assertNotIn("tool_result", {item["source_type"] for item in observations})
        self.assertNotIn("User selected 基于现有工程二次开发", json.dumps(observations, ensure_ascii=False))

    def test_request_redacts_secrets_paths_and_excludes_thinking(self) -> None:
        bundle = turn_bundle(
            user_text=(
                r"本轮处理 C:\private\firmware.c，api_key=abcdef123456。"
                "API key is english-secret-value，访问令牌为 chinese-secret-value。"
            ),
            assistant_text="Thinking: hidden chain of thought",
            tools=[tool_part()],
        )
        client = FakeClient(empty_output)
        plan = DeepSeekUnifiedMemoryPlanner(client).plan(bundle)

        payload = client.calls[0]["user_payload"]
        wire = str(payload)
        self.assertNotIn("abcdef123456", wire)
        self.assertNotIn("english-secret-value", wire)
        self.assertNotIn("chinese-secret-value", wire)
        self.assertNotIn(r"C:\private", wire)
        self.assertNotIn("hidden chain of thought", wire)
        self.assertIn("secret", payload["safety"]["redactions_applied"])
        self.assertTrue(payload["safety"]["hidden_reasoning_excluded"])
        self.assertFalse(payload["safety"]["project_files_scanned"])
        self.assertEqual(plan["candidate_count"], 0)

    def test_all_observable_tool_data_is_present_without_tool_whitelist(self) -> None:
        bundle = turn_bundle(tools=[tool_part()])
        client = FakeClient(empty_output)
        DeepSeekUnifiedMemoryPlanner(client).plan(bundle)
        observations = client.calls[0]["user_payload"]["turns"][0]["observations"]
        tool = next(item for item in observations if item["source_type"] == "tool_result")
        self.assertIn("tool: build", tool["text"])
        self.assertIn("input:", tool["text"])
        self.assertIn("output:", tool["text"])
        self.assertIn("status: completed", tool["text"])

    def test_existing_memory_summaries_are_bounded_and_reuse_stable_key(self) -> None:
        existing = [
            {
                "stable_key": "preference.response.structure",
                "type": "preference",
                "scope": {"level": "global"},
                "summary": "先给结论",
                "statement": "回复时先给结论。",
            }
        ]

        def responder(request):
            self.assertEqual(request["existing_memories"][0]["memory_ref"], "memory_1")
            return candidate_output(
                request,
                action_kind="merge_evidence",
                related_memory_ref="memory_1",
            )

        plan = DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(
            turn_bundle(), existing_memories=existing
        )
        hint = plan["candidates"][0]["action_hint"]
        self.assertEqual(hint["kind"], "merge_evidence")
        self.assertTrue(hint["related_existing_memory"])
        self.assertEqual(hint["related_candidate_id"], "memory_1")

    def test_existing_scope_conditions_are_preserved_redacted_and_payload_bounded(self) -> None:
        existing = [
            {
                "stable_key": "workflow.build.toolchain",
                "type": "workflow",
                "scope": {
                    "level": "project",
                    "project_key": "project-a",
                    "conditions": {
                        "chip": "TG032Plus",
                        "config": r"C:\private\project\firmware.uvproj",
                    },
                },
                "summary": "摘" * 1_000,
                "statement": "详" * 4_000,
            }
        ]
        existing.extend(
            {
                "stable_key": f"knowledge.topic_{index}",
                "type": "knowledge",
                "scope": {"level": "global"},
                "summary": "摘" * 1_000,
                "statement": "详" * 4_000,
            }
            for index in range(1, 200)
        )
        client = FakeClient(empty_output)
        DeepSeekUnifiedMemoryPlanner(client).plan(
            turn_bundle(), existing_memories=existing
        )

        projection = client.calls[0]["user_payload"]["existing_memories"]
        self.assertEqual(projection[0]["scope"]["conditions"]["chip"], "TG032Plus")
        self.assertIn("[LOCAL_PATH]", projection[0]["scope"]["conditions"]["config"])
        self.assertNotIn(r"C:\private", str(projection))
        self.assertEqual(len(projection[0]["summary"]), 320)
        self.assertEqual(projection[-1]["statement"], "")
        encoded = json.dumps(
            projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.assertLessEqual(len(encoded), MAX_EXISTING_MEMORY_PAYLOAD_CHARS)

    def test_writer_contract_exposes_only_commit_supported_scope_and_status(self) -> None:
        client = FakeClient(empty_output)
        DeepSeekUnifiedMemoryPlanner(client).plan(turn_bundle())
        request = client.calls[0]["user_payload"]
        self.assertEqual(request["policy"]["scope_levels"], ["domain", "global", "project"])
        self.assertEqual(request["policy"]["memory_statuses"], ["active"])
        self.assertEqual(
            request["policy"]["trust_statuses"],
            ["contested", "declared", "observed", "unverified", "verified"],
        )
        candidate_schema = request["output_json_schema"]["candidates"][0]
        self.assertEqual(candidate_schema["scope"]["level"], "global|domain|project")
        self.assertEqual(candidate_schema["status"], "active")
        self.assertEqual(
            set(candidate_schema["evidence"][0]), {"evidence_ref", "quote"}
        )

    def test_unsupported_scope_and_non_active_status_fail_at_model_boundary(self) -> None:
        for level in ("session", "cross_project"):
            with self.subTest(scope=level):
                def scoped(request, level=level):
                    result = candidate_output(request)
                    result["candidates"][0]["scope"]["level"] = level
                    return result

                with self.assertRaisesRegex(UnifiedOutputError, "level is unsupported"):
                    DeepSeekUnifiedMemoryPlanner(FakeClient(scoped)).plan(turn_bundle())
        for status in ("superseded", "conflict"):
            with self.subTest(status=status):
                def stateful(request, status=status):
                    return candidate_output(request, status=status)

                with self.assertRaisesRegex(UnifiedOutputError, "status must be active"):
                    DeepSeekUnifiedMemoryPlanner(FakeClient(stateful)).plan(turn_bundle())

    def test_same_message_user_correction_keeps_latest_and_all_audit_evidence(self) -> None:
        text = (
            "以后无论在哪个项目，回复先详细说明步骤。"
            "以后无论在哪个项目，回复先给结论。"
        )

        def responder(request):
            observation = request["turns"][0]["observations"][0]
            clauses = [
                "以后无论在哪个项目，回复先详细说明步骤。",
                "以后无论在哪个项目，回复先给结论。",
            ]
            candidates = []
            for index, (quote, statement) in enumerate(
                zip(clauses, ("回复先详细说明步骤。", "回复先给结论。"), strict=True),
                start=1,
            ):
                start = observation["text"].index(quote)
                item = candidate_output(request)["candidates"][0]
                item["candidate_ref"] = f"candidate_{index}"
                item["summary"] = statement
                item["content"] = {"statement": statement}
                item["evidence"] = [
                    {
                        "evidence_ref": observation["evidence_ref"],
                        "quote": quote,
                    }
                ]
                candidates.append(item)
            return {
                "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
                "request_id": request["request_id"],
                # Deliberately reverse model output order: source offsets, not
                # list position, must decide which explicit requirement wins.
                "candidates": list(reversed(candidates)),
            }

        plan = DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(
            turn_bundle(user_text=text)
        )
        self.assertEqual(plan["candidate_count"], 1)
        candidate = plan["candidates"][0]
        self.assertEqual(candidate["memory"]["content"]["statement"], "回复先给结论。")
        self.assertEqual(len(candidate["memory"]["evidence_refs"]), 2)
        self.assertEqual(len(candidate["evidence_quotes"]), 2)
        self.assertEqual(
            [item["quote"] for item in candidate["evidence_quotes"]],
            [
                "以后无论在哪个项目，回复先详细说明步骤。",
                "以后无论在哪个项目，回复先给结论。",
            ],
        )

    def test_same_identity_technical_candidates_never_use_latest_wins(self) -> None:
        text = "以后无论在哪个项目，构建结果可能成功，也可能失败。"

        def responder(request):
            first = candidate_output(
                request,
                stable_key="engineering.build.outcome",
                memory_type="engineering",
                trust_status="unverified",
            )["candidates"][0]
            first["content"] = {"statement": "构建成功。"}
            second = copy.deepcopy(first)
            second["candidate_ref"] = "candidate_2"
            second["content"] = {"statement": "构建失败。"}
            return {
                "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
                "request_id": request["request_id"],
                "candidates": [first, second],
            }

        with self.assertRaisesRegex(UnifiedOutputError, "cannot be resolved by source time"):
            DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(
                turn_bundle(user_text=text)
            )

    def test_same_key_with_different_scope_conditions_is_not_collapsed(self) -> None:
        def responder(request):
            first = candidate_output(request)["candidates"][0]
            first["conditions"] = {"toolchain": "keil"}
            second = copy.deepcopy(first)
            second["candidate_ref"] = "candidate_2"
            second["conditions"] = {"toolchain": "gcc"}
            second["content"] = {"statement": "使用 GCC 时也先给结论。"}
            return {
                "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
                "request_id": request["request_id"],
                "candidates": [first, second],
            }

        plan = DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(turn_bundle())
        self.assertEqual(plan["candidate_count"], 2)
        self.assertEqual(
            [item["memory"]["conditions"] for item in plan["candidates"]],
            [{"toolchain": "keil"}, {"toolchain": "gcc"}],
        )

    def test_forged_evidence_reference_rejects_whole_batch(self) -> None:
        def responder(request):
            result = candidate_output(request)
            result["candidates"][0]["evidence"][0]["evidence_ref"] = "evidence_forged"
            return result

        with self.assertRaisesRegex(UnifiedOutputError, "unknown or duplicated"):
            DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(turn_bundle())

    def test_evidence_quote_must_be_an_exact_source_substring(self) -> None:
        def responder(request):
            result = candidate_output(request)
            result["candidates"][0]["evidence"][0]["quote"] = "模型编造的原文"
            return result

        with self.assertRaisesRegex(UnifiedOutputError, "exact source substring"):
            DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(turn_bundle())

    def test_offsets_are_derived_locally_from_a_unicode_quote(self) -> None:
        prefix = "这是前缀。"
        quote = "以后无论在哪个项目，回复我时先给结论。"
        bundle = turn_bundle(user_text=prefix + quote)

        def responder(request):
            result = candidate_output(request)
            result["candidates"][0]["evidence"][0]["quote"] = quote
            return result

        plan = DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(bundle)
        evidence = plan["candidates"][0]["evidence_quotes"][0]
        self.assertEqual(evidence["start_offset"], len(prefix))
        self.assertEqual(evidence["end_offset"], len(prefix) + len(quote))
        self.assertEqual(evidence["quote"], quote)

    def test_ambiguous_short_quote_is_rejected(self) -> None:
        bundle = turn_bundle(user_text="以后默认先给结论，以后默认先给结论。")

        def responder(request):
            result = candidate_output(request)
            result["candidates"][0]["evidence"][0]["quote"] = "以后默认先给结论"
            return result

        with self.assertRaisesRegex(UnifiedOutputError, "ambiguous"):
            DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(bundle)

    def test_legacy_model_offsets_are_rejected(self) -> None:
        def responder(request):
            result = candidate_output(request)
            result["candidates"][0]["evidence"][0]["start_offset"] = 0
            result["candidates"][0]["evidence"][0]["end_offset"] = 1
            return result

        with self.assertRaisesRegex(UnifiedOutputError, "unsupported fields"):
            DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(turn_bundle())

    def test_global_scope_requires_explicit_global_user_words(self) -> None:
        bundle = turn_bundle(user_text="回复时先给结论。")
        plan = DeepSeekUnifiedMemoryPlanner(FakeClient(candidate_output)).plan(bundle)
        self.assertEqual(
            plan["candidates"][0]["memory"]["scope"],
            {"level": "project", "project_key": "project-a"},
        )

    def test_model_extra_fields_and_transport_failure_fail_closed(self) -> None:
        def extra(request):
            result = candidate_output(request)
            result["unexpected"] = True
            return result

        with self.assertRaisesRegex(UnifiedOutputError, "unsupported fields"):
            DeepSeekUnifiedMemoryPlanner(FakeClient(extra)).plan(turn_bundle())

        class FailureClient:
            def complete_json(self, **kwargs):
                raise JsonClientTransportError("provider body must never escape")

        with self.assertRaises(UnifiedTransportError):
            DeepSeekUnifiedMemoryPlanner(FailureClient()).plan(turn_bundle())

    def test_writer_retries_one_invalid_structure_without_partial_output(self) -> None:
        class OnceInvalidClient:
            def __init__(self):
                self.calls = 0
                self.payloads = []

            def complete_json(self, **kwargs):
                self.calls += 1
                self.payloads.append(kwargs["user_payload"])
                result = candidate_output(kwargs["user_payload"])
                if self.calls == 1:
                    result["unexpected"] = True
                return result

        client = OnceInvalidClient()
        plan = DeepSeekUnifiedMemoryPlanner(client).plan(turn_bundle())
        self.assertEqual(client.calls, 2)
        self.assertEqual(plan["candidate_count"], 1)
        self.assertNotIn("validation_feedback", client.payloads[0])
        feedback = client.payloads[1]["validation_feedback"]
        self.assertEqual(feedback["attempt"], 2)
        self.assertEqual(feedback["previous_output"], "rejected_by_local_validator")
        self.assertEqual(feedback["error_code"], "schema_validation_failed")
        self.assertEqual(
            client.payloads[0]["request_id"], client.payloads[1]["request_id"]
        )

    def test_writer_downgrades_evidence_bound_trust_without_weakening_it(self) -> None:
        class TrustRepairClient:
            def __init__(self):
                self.calls = []

            def complete_json(self, **kwargs):
                request = kwargs["user_payload"]
                self.calls.append(request)
                assistant_index = next(
                    index
                    for index, item in enumerate(request["turns"][0]["observations"])
                    if item["source_type"] == "assistant_final"
                )
                return candidate_output(
                    request,
                    observation_index=assistant_index,
                    stable_key="project.response.summary",
                    memory_type="project",
                    scope={
                        "level": "project",
                        "project_refs": ["project_1"],
                        "session_refs": [],
                        "domain": None,
                    },
                    trust_status=(
                        "declared"
                        if "validation_feedback" not in request
                        else "unverified"
                    ),
                )

        client = TrustRepairClient()
        plan = DeepSeekUnifiedMemoryPlanner(client).plan(turn_bundle())
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            plan["candidates"][0]["memory"]["trust_status"], "unverified"
        )

    def test_verified_claim_cannot_be_created_from_unverified_tool_status(self) -> None:
        def responder(request):
            tool_index = next(
                index
                for index, item in enumerate(request["turns"][0]["observations"])
                if item["source_type"] == "tool_result"
            )
            return candidate_output(
                request,
                observation_index=tool_index,
                stable_key="engineering.build.result",
                memory_type="engineering",
                scope={
                    "level": "project",
                    "project_refs": ["project_1"],
                    "session_refs": [],
                    "domain": None,
                },
                trust_status="verified",
            )

        with self.assertRaisesRegex(UnifiedOutputError, "invalid_trust_status"):
            DeepSeekUnifiedMemoryPlanner(FakeClient(responder)).plan(
                turn_bundle(tools=[tool_part()])
            )

    def test_same_input_and_output_produce_stable_plan_ids(self) -> None:
        bundle = turn_bundle()
        first = DeepSeekUnifiedMemoryPlanner(FakeClient(candidate_output)).plan(bundle)
        second = DeepSeekUnifiedMemoryPlanner(FakeClient(candidate_output)).plan(bundle)
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertEqual(first["plan_id"], second["plan_id"])
        self.assertEqual(first["candidates"], second["candidates"])


class UnifiedReviewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = turn_bundle()
        self.writer_plan = DeepSeekUnifiedMemoryPlanner(
            FakeClient(candidate_output)
        ).plan(self.bundle)

    def decision(self, outcome: str, revised_output=None):
        def responder(request):
            return {
                "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
                "review_id": request["review_id"],
                "writer_plan_id": request["writer_plan_id"],
                "outcome": outcome,
                "reason": f"review {outcome}",
                "revised_output": revised_output(request) if callable(revised_output) else revised_output,
            }

        return responder

    def test_independent_reviewer_accept_enables_only_reviewed_plan(self) -> None:
        client = FakeClient(self.decision("accept"))
        result = DeepSeekUnifiedMemoryReviewer(client).review(
            self.writer_plan, self.bundle
        )
        self.assertEqual(result["outcome"], "accept")
        self.assertTrue(result["commit_allowed"])
        self.assertEqual(result["reviewed_plan"], self.writer_plan)
        self.assertEqual(result["review_rounds"], 1)
        self.assertEqual(len(client.calls), 1)

    def test_reviewer_retries_one_invalid_structure(self) -> None:
        class OnceInvalidClient:
            def __init__(self, decision):
                self.calls = 0
                self.decision = decision

            def complete_json(self, **kwargs):
                self.calls += 1
                result = self.decision(kwargs["user_payload"])
                if self.calls == 1:
                    result["unexpected"] = True
                return result

        client = OnceInvalidClient(self.decision("accept"))
        result = DeepSeekUnifiedMemoryReviewer(client).review(
            self.writer_plan, self.bundle
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["outcome"], "accept")

    def test_reviewer_reject_returns_no_plan_and_zero_commit(self) -> None:
        result = DeepSeekUnifiedMemoryReviewer(
            FakeClient(self.decision("reject"))
        ).review(self.writer_plan, self.bundle)
        self.assertEqual(result["outcome"], "reject")
        self.assertIsNone(result["reviewed_plan"])
        self.assertFalse(result["commit_allowed"])

    def test_reviewer_may_revise_once_against_same_evidence(self) -> None:
        def revised(request):
            extraction = request["extraction_request"]
            return candidate_output(extraction, summary="默认先给结论")

        result = DeepSeekUnifiedMemoryReviewer(
            FakeClient(self.decision("revise", revised))
        ).review(self.writer_plan, self.bundle)
        self.assertEqual(result["outcome"], "revise")
        self.assertTrue(result["commit_allowed"])
        self.assertEqual(
            result["reviewed_plan"]["candidates"][0]["memory"]["summary"],
            "默认先给结论",
        )
        self.assertEqual(result["review_rounds"], 1)

    def test_reviewer_forged_revision_fails_closed(self) -> None:
        def revised(request):
            extraction = request["extraction_request"]
            output = candidate_output(extraction)
            output["candidates"][0]["evidence"][0]["evidence_ref"] = "evidence_forged"
            return output

        with self.assertRaises(UnifiedOutputError):
            DeepSeekUnifiedMemoryReviewer(
                FakeClient(self.decision("revise", revised))
            ).review(self.writer_plan, self.bundle)

    def test_tampered_writer_plan_is_blocked_before_reviewer_call(self) -> None:
        tampered = copy.deepcopy(self.writer_plan)
        tampered["evidence_spans"][next(iter(tampered["evidence_spans"]))][
            "quote"
        ] = "伪造"
        client = FakeClient(self.decision("accept"))
        with self.assertRaisesRegex(UnifiedOutputError, "evidence spans"):
            DeepSeekUnifiedMemoryReviewer(client).review(tampered, self.bundle)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
