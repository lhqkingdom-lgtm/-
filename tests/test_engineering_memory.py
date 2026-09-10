from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.engineering_memory import (
    CANDIDATE_SCHEMA_VERSION,
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EngineeringCandidateValidationError,
    assess_cross_project_promotion,
    build_engineering_candidate_template,
    decide_engineering_relation,
    engineering_candidate_recall_eligible,
    load_engineering_evidence_bundles,
    summarize_engineering_candidates,
    validate_engineering_candidate,
)
from memory_lab.episodes import EXTRACTOR_VERSION
from memory_lab.store import open_store


class EngineeringMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "memory_lab.db"
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO source(
                    source_id, source_kind, source_path, created_at,
                    baseline_cutoff_ms
                ) VALUES ('source_1', 'sqlite', 'D:\\PRIVATE\\opencode.db',
                          '2026-09-10T00:00:00Z', 0)
                """
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at,
                    completed_at, settle_ms, status
                ) VALUES ('turn_run', 'source_1', 'test',
                          '2026-09-10T00:00:00Z',
                          '2026-09-10T00:00:01Z', 0, 'completed')
                """
            )
            connection.execute(
                """
                INSERT INTO episode_extraction_run(
                    run_id, source_id, extractor_version, started_at,
                    completed_at, status
                ) VALUES ('episode_run', 'source_1', ?,
                          '2026-09-10T00:00:01Z',
                          '2026-09-10T00:00:02Z', 'completed')
                """,
                (EXTRACTOR_VERSION,),
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _add_episode(
        self,
        name: str,
        *,
        project_key: str,
        outcome: str,
        tool_name: str = "keil_build",
        state_status: str = "completed",
        metadata: dict | None = None,
        extractor_version: str = EXTRACTOR_VERSION,
        has_error: bool = False,
        started_at: int | None = 10,
        ended_at: int | None = 20,
    ) -> None:
        event_id = "event_" + name
        turn_id = "turn_" + name
        episode_id = "episode_" + name
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO event(
                    event_id, source_id, session_id, message_id, part_id,
                    event_type, directory, timestamp, content, content_hash,
                    source_kind, ingested_at, raw_json
                ) VALUES (?, 'source_1', ?, ?, ?, 'user_text',
                          'D:\\TOP_SECRET\\firmware', 1,
                          'USER_TEXT_MUST_NOT_BE_IN_BUNDLE', 'hash', 'sqlite',
                          '2026-09-10T00:00:00Z',
                          '{"private":"RAW_EVENT_SECRET"}')
                """,
                (event_id, "session_" + name, "user_" + name, "part_" + name),
            )
            connection.execute(
                """
                INSERT INTO conversation_turn(
                    turn_id, source_id, session_id, user_message_id,
                    primary_event_id, user_timestamp, current_state,
                    first_detected_at, last_detected_at,
                    last_detection_run_id, last_structural_signature,
                    assistant_count, part_count
                ) VALUES (?, 'source_1', ?, ?, ?, 1, 'completed',
                          '2026-09-10T00:00:00Z',
                          '2026-09-10T00:00:01Z', 'turn_run', ?, 1, 1)
                """,
                (
                    turn_id,
                    "session_" + name,
                    "user_" + name,
                    event_id,
                    (name[0] if name else "a") * 64,
                ),
            )
            connection.execute(
                """
                INSERT INTO trace_episode(
                    episode_id, idempotency_key, extraction_run_id,
                    extractor_version, source_id, turn_id, primary_event_id,
                    project_key, source_signature, trace_signature, turn_state,
                    evidence_class, domain_verification_outcome,
                    semantic_status, assistant_count, tool_event_count,
                    positive_event_count, negative_event_count,
                    unverified_event_count, recovered_tool_count, created_at
                ) VALUES (?, ?, 'episode_run', ?, 'source_1', ?, ?, ?, ?, ?,
                          'completed', 'unverified', ?, 'not_evaluated', 1, 1,
                          0, 0, 1, 0, '2026-09-10T00:00:02Z')
                """,
                (
                    episode_id,
                    "key_" + name,
                    extractor_version,
                    turn_id,
                    event_id,
                    project_key,
                    "s" * 64,
                    (name[-1] if name else "b") * 64,
                    outcome,
                ),
            )
            connection.execute(
                """
                INSERT INTO trace_episode_event(
                    episode_event_id, episode_id, ordinal,
                    source_assistant_message_id, source_part_id, tool_name,
                    state_status, started_at_ms, ended_at_ms,
                    has_error_payload, metadata_json,
                    input_hash, input_length, output_hash, output_length,
                    error_hash, error_length, evidence_verdict,
                    event_signature, created_at
                ) VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?,
                          'INPUT_HASH_MUST_NOT_LEAVE', 777,
                          'OUTPUT_HASH_MUST_NOT_LEAVE', 888,
                          'ERROR_HASH_MUST_NOT_LEAVE', 999,
                          'unverified', ?, '2026-09-10T00:00:02Z')
                """,
                (
                    "episode_event_" + name,
                    episode_id,
                    "assistant_" + name,
                    "source_part_" + name,
                    tool_name,
                    state_status,
                    started_at,
                    ended_at,
                    int(has_error),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    "e" * 64,
                ),
            )

    @staticmethod
    def _success_metadata(**extra: object) -> dict:
        return {
            "success": True,
            "errorCount": 0,
            "warningCount": 0,
            **extra,
        }

    @staticmethod
    def _failure_metadata(**extra: object) -> dict:
        return {
            "success": False,
            "errorCount": 2,
            "warningCount": 0,
            "failureKind": "source",
            **extra,
        }

    def _proposal(
        self,
        bundles: list[dict],
        *,
        kind: str = "successful_resolution",
        status: str = "verified",
        scope_level: str | None = None,
        symptom: str = "编译阶段出现可复现错误",
        solution: str = "调整兼容配置后重新进行编译验证",
    ) -> dict:
        events = [event for bundle in bundles for event in bundle["trace"]["events"]]
        all_refs = [event["evidence_id"] for event in events]
        verification_refs = sorted(
            {
                ref
                for bundle in bundles
                for ref in bundle["verification"]["evidence_refs"]
            }
        )
        verification_outcomes = {bundle["verification"]["outcome"] for bundle in bundles}
        verification_outcome = (
            next(iter(verification_outcomes))
            if len(verification_outcomes) == 1
            else "ambiguous"
        )
        project_keys = sorted({bundle["project_key"] for bundle in bundles})
        fingerprint = bundles[0]["environment"]["fingerprint"]
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "source_bundle_ids": [bundle["bundle_id"] for bundle in bundles],
            "stable_key": "engineering.embedded.compatible_build",
            "experience_kind": kind,
            "symptom": {"summary": symptom, "evidence_refs": all_refs[:1]},
            "environment": {
                "summary": "Keil 工具链下的嵌入式工程",
                "fingerprint": fingerprint,
                "toolchains": sorted(
                    {
                        toolchain
                        for bundle in bundles
                        for toolchain in bundle["environment"]["toolchains"]
                    }
                ),
                "evidence_refs": all_refs,
            },
            "attempts": [
                {
                    "summary": "执行结构化工程验证",
                    "outcome": "succeeded" if status == "verified" else "unknown",
                    "evidence_refs": [event["evidence_id"]],
                }
                for event in events
            ],
            "cause": {
                "summary": "环境配置与工程预期不一致",
                "certainty": "inferred",
                "evidence_refs": all_refs[:1],
            },
            "solution": {
                "summary": solution,
                "certainty": "inferred",
                "evidence_refs": all_refs[:1],
            },
            "verification": {
                "outcome": verification_outcome,
                "evidence_refs": verification_refs,
            },
            "scope": {
                "level": scope_level or ("cross_project" if len(project_keys) >= 2 else "project"),
                "project_keys": project_keys,
                "environment_fingerprint": fingerprint,
            },
            "status": status,
        }

    def test_bundle_reads_only_safe_v3_projection_and_reapplies_allowlist(self) -> None:
        self._add_episode(
            "success",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(
                token="METADATA_TOKEN_SECRET",
                path="D:\\PRIVATE\\firmware.uvprojx",
                command="C51.EXE PRIVATE_COMMAND",
            ),
        )
        self._add_episode(
            "legacy",
            project_key="directory:legacy",
            outcome="verified_success",
            extractor_version="cp-trace-episode-v2",
            metadata=self._success_metadata(),
        )

        bundles = load_engineering_evidence_bundles(self.database)

        self.assertEqual(len(bundles), 1)
        bundle = bundles[0]
        self.assertEqual(bundle["schema_version"], EVIDENCE_BUNDLE_SCHEMA_VERSION)
        self.assertEqual(bundle["project_key"], "directory:project-a")
        self.assertEqual(bundle["verification"]["outcome"], "verified_success")
        self.assertTrue(bundle["verification"]["structured"])
        self.assertTrue(bundle["verification"]["trusted"])
        self.assertEqual(bundle["environment"]["toolchains"], ["keil"])
        self.assertEqual(
            bundle["trace"]["events"][0]["facts"],
            {"errorCount": 0, "success": True, "warningCount": 0},
        )
        serialized = json.dumps(bundle, ensure_ascii=False, sort_keys=True)
        for forbidden in (
            "USER_TEXT_MUST_NOT_BE_IN_BUNDLE",
            "RAW_EVENT_SECRET",
            "TOP_SECRET",
            "METADATA_TOKEN_SECRET",
            "PRIVATE_COMMAND",
            "firmware.uvprojx",
            "INPUT_HASH_MUST_NOT_LEAVE",
            "OUTPUT_HASH_MUST_NOT_LEAVE",
            "ERROR_HASH_MUST_NOT_LEAVE",
            "source_part_success",
            "assistant_success",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(all(value is False for value in bundle["privacy"].values()))

    def test_stored_verification_mismatch_fails_closed(self) -> None:
        self._add_episode(
            "mismatch",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._failure_metadata(),
        )
        bundle = load_engineering_evidence_bundles(self.database)[0]
        self.assertEqual(bundle["verification"]["outcome"], "unverified")
        self.assertFalse(bundle["verification"]["trusted"])
        self.assertEqual(bundle["verification"]["evidence_refs"], [])
        self.assertIn("does_not_match", bundle["verification"]["reason"])

    def test_template_keeps_semantic_fields_unset(self) -> None:
        self._add_episode(
            "success",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        bundle = load_engineering_evidence_bundles(self.database)[0]
        template = build_engineering_candidate_template(bundle)
        self.assertIsNone(template["stable_key"])
        self.assertIsNone(template["symptom"]["summary"])
        self.assertIsNone(template["cause"])
        self.assertIsNone(template["solution"])
        self.assertEqual(template["status"], "unverified")
        self.assertEqual(template["verification"]["outcome"], "verified_success")

    def test_verified_success_candidate_requires_structured_exact_evidence(self) -> None:
        self._add_episode(
            "success",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        bundles = load_engineering_evidence_bundles(self.database)
        candidate = validate_engineering_candidate(self._proposal(bundles), bundles)
        self.assertEqual(candidate["status"], "verified")
        self.assertEqual(candidate["verification"]["outcome"], "verified_success")
        self.assertTrue(candidate["verification"]["structured"])
        self.assertTrue(candidate["candidate_id"].startswith("engcand_"))
        self.assertTrue(candidate["semantic_fingerprint"].startswith("engsem_"))
        self.assertTrue(engineering_candidate_recall_eligible(candidate))

        bad = self._proposal(bundles)
        bad["verification"]["evidence_refs"] = []
        with self.assertRaisesRegex(
            EngineeringCandidateValidationError, "exactly the structured"
        ):
            validate_engineering_candidate(bad, bundles)

    def test_unverified_episode_cannot_claim_verified(self) -> None:
        self._add_episode(
            "unverified",
            project_key="directory:project-a",
            outcome="unverified",
            tool_name="edit",
            metadata={"success": True},
        )
        bundles = load_engineering_evidence_bundles(self.database)
        proposal = self._proposal(bundles, status="verified")
        with self.assertRaisesRegex(EngineeringCandidateValidationError, "status must be unverified"):
            validate_engineering_candidate(proposal, bundles)

        proposal = self._proposal(bundles, status="unverified")
        candidate = validate_engineering_candidate(proposal, bundles)
        self.assertEqual(candidate["status"], "unverified")
        self.assertFalse(candidate["verification"]["structured"])
        self.assertFalse(engineering_candidate_recall_eligible(candidate))

    def test_verified_failure_can_only_verify_failure_mode(self) -> None:
        self._add_episode(
            "failure",
            project_key="directory:project-a",
            outcome="verified_failure",
            metadata=self._failure_metadata(),
        )
        bundles = load_engineering_evidence_bundles(self.database)
        good = self._proposal(bundles, kind="failure_mode")
        good["verification"]["outcome"] = "verified_failure"
        candidate = validate_engineering_candidate(good, bundles)
        self.assertEqual(candidate["status"], "verified")
        self.assertEqual(candidate["experience_kind"], "failure_mode")

        bad = self._proposal(bundles, kind="successful_resolution")
        bad["verification"]["outcome"] = "verified_failure"
        with self.assertRaisesRegex(
            EngineeringCandidateValidationError, "requires structured success"
        ):
            validate_engineering_candidate(bad, bundles)

    def test_free_form_candidate_rejects_paths_commands_secrets_and_payloads(self) -> None:
        self._add_episode(
            "success",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        bundles = load_engineering_evidence_bundles(self.database)
        forbidden = (
            "读取 D:\\private\\source.c 后修复",
            "运行 powershell.exe -File build.ps1",
            "执行 C51.EXE firmware.uvproj",
            "令牌 api_key=VERY_PRIVATE",
            '{"command":"build"}',
            '结果包含 {"command":"build"} 的原始载荷',
        )
        for value in forbidden:
            proposal = self._proposal(bundles, solution=value)
            with self.subTest(value=value), self.assertRaisesRegex(
                EngineeringCandidateValidationError, "path, command, secret, or raw payload"
            ):
                validate_engineering_candidate(proposal, bundles)

    def test_different_environment_is_variant_and_never_latest_wins(self) -> None:
        self._add_episode(
            "one",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        bundles = load_engineering_evidence_bundles(self.database)
        first = validate_engineering_candidate(self._proposal(bundles), bundles)
        second = copy.deepcopy(first)
        second["candidate_id"] = "engcand_other"
        second["environment"]["fingerprint"] = "env_other"
        second["scope"]["environment_fingerprint"] = "env_other"
        relation = decide_engineering_relation(first, second)
        self.assertEqual(relation["relation"], "different_environment")
        self.assertEqual(relation["action"], "add_environment_variant")
        self.assertFalse(relation["latest_wins"])

    def test_same_environment_different_semantics_opens_conflict(self) -> None:
        self._add_episode(
            "one",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        bundles = load_engineering_evidence_bundles(self.database)
        first = validate_engineering_candidate(self._proposal(bundles), bundles)
        proposal = self._proposal(bundles, solution="保持原配置并调整工程约束后验证")
        second = validate_engineering_candidate(proposal, bundles)
        relation = decide_engineering_relation(first, second)
        self.assertEqual(relation["relation"], "conflict")
        self.assertEqual(relation["action"], "open_engineering_conflict")
        self.assertFalse(relation["latest_wins"])

    def test_two_verified_projects_are_required_for_promotion(self) -> None:
        self._add_episode(
            "one",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        first_bundle = load_engineering_evidence_bundles(self.database)
        first = validate_engineering_candidate(self._proposal(first_bundle), first_bundle)
        one_project = assess_cross_project_promotion([first])
        self.assertFalse(one_project["eligible"])
        self.assertEqual(one_project["reason"], "requires_two_distinct_projects")

        self._add_episode(
            "two",
            project_key="directory:project-b",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        bundles = load_engineering_evidence_bundles(self.database)
        by_project = {bundle["project_key"]: bundle for bundle in bundles}
        second_bundle = [by_project["directory:project-b"]]
        second = validate_engineering_candidate(self._proposal(second_bundle), second_bundle)
        decision = assess_cross_project_promotion([first, second])
        self.assertTrue(decision["eligible"])
        self.assertEqual(
            decision["project_keys"],
            ["directory:project-a", "directory:project-b"],
        )
        self.assertFalse(decision["latest_wins"])

    def test_cross_project_candidate_validates_only_with_two_verified_projects(self) -> None:
        self._add_episode(
            "one",
            project_key="directory:project-a",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        self._add_episode(
            "two",
            project_key="directory:project-b",
            outcome="verified_success",
            metadata=self._success_metadata(),
        )
        bundles = load_engineering_evidence_bundles(self.database)
        candidate = validate_engineering_candidate(self._proposal(bundles), bundles)
        self.assertEqual(candidate["scope"]["level"], "cross_project")
        self.assertEqual(len(candidate["scope"]["project_keys"]), 2)

        one_bundle = bundles[:1]
        bad = self._proposal(one_bundle, scope_level="cross_project")
        with self.assertRaisesRegex(EngineeringCandidateValidationError, "at least two projects"):
            validate_engineering_candidate(bad, one_bundle)

    def test_summary_reports_current_realistic_boundary_without_formal_write(self) -> None:
        self._add_episode(
            "unverified",
            project_key="directory:project-a",
            outcome="unverified",
            tool_name="edit",
            metadata={},
        )
        report = summarize_engineering_candidates(self.database)
        self.assertEqual(report["bundle_count"], 1)
        self.assertEqual(report["verification_counts"], {"unverified": 1})
        self.assertFalse(report["formal_memory_written"])
        self.assertIn("cannot establish", report["semantic_boundary"])

    def test_episode_filter_rejects_missing_and_non_v3_identifiers(self) -> None:
        self._add_episode(
            "legacy",
            project_key="directory:legacy",
            outcome="unverified",
            extractor_version="cp-trace-episode-v2",
            tool_name="edit",
            metadata={},
        )
        with self.assertRaisesRegex(ValueError, "unknown or non-v3"):
            load_engineering_evidence_bundles(
                self.database, episode_ids=["episode_legacy"]
            )


if __name__ == "__main__":
    unittest.main()
