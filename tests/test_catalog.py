from __future__ import annotations

import json
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.catalog import (
    CatalogValidationError,
    _consistent_read,
    build_preview_catalog,
    recall_from_catalog,
    run_catalog,
)
from memory_lab.consolidation import latest_admission_snapshot
from memory_lab.shadow import formal_state_snapshot
from memory_lab.source import open_read_only
from memory_lab.store import open_store


def _row(
    candidate: str,
    quote: str,
    *,
    should_store: bool,
    key: str | None = None,
    memory_type: str | None = None,
    scope: dict | None = None,
    value: str | None = None,
    statement: str | None = None,
    relation: str = "new",
    disposition: str = "eligible",
    triggers: tuple[str, ...] = (),
    project_key: str = "project-a",
    occurred_at: int = 1,
) -> dict:
    result = {
        "should_store": should_store,
        "type": memory_type if should_store else None,
        "key": key if should_store else None,
        "scope": scope if should_store else None,
        "value": (
            {"normalized": value, "statement": statement or quote}
            if should_store
            else None
        ),
        "relation": {
            "kind": relation if should_store else "unrelated",
            "target_kind": None,
            "target_id": None,
        },
        "confidence": 0.98,
        "reason": "test",
        "evidence": [
            {
                "event_id": f"event-{candidate}",
                "part_id": f"part-{candidate}",
                "quote": quote,
                "start_offset": 0,
                "end_offset": len(quote),
            }
        ],
    }
    return {
        "candidate_key": candidate,
        "status": "valid",
        "execution_mode": "judge_call",
        "result_json": json.dumps(result, ensure_ascii=False),
        "exact_quote": quote,
        "start_offset": 0,
        "end_offset": len(quote),
        "activation_triggers_json": json.dumps(list(triggers), ensure_ascii=False),
        "disposition": disposition,
        "event_id": f"event-{candidate}",
        "session_id": f"session-{candidate}",
        "message_id": f"message-{candidate}",
        "part_id": f"part-{candidate}",
        "occurred_at": occurred_at,
        "project_key": project_key,
    }


class PreviewCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _row(
                "global-confirm-1",
                "生成正式代码前先向我确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "global"},
                value="required",
                triggers=("生成正式代码", "确认"),
            ),
            _row(
                "global-confirm-2",
                "正式代码之前需要先问我。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "global"},
                value="required",
                relation="equivalent",
                triggers=("生成正式代码", "确认"),
                occurred_at=2,
            ),
            _row(
                "project-exception",
                "仅在当前项目中，生成正式代码不需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "project", "project_key": "project-a"},
                value="not_required",
                relation="different_scope",
                triggers=("生成正式代码", "确认"),
                occurred_at=3,
            ),
            _row(
                "language",
                "以后默认使用中文回复。",
                should_store=True,
                key="preference.response.language",
                memory_type="response_preference",
                scope={"level": "global"},
                value="zh",
                triggers=("回复", "语言"),
                occurred_at=4,
            ),
            _row(
                "temporary",
                "这一次只解释方案。",
                should_store=False,
                disposition="temporary",
                occurred_at=5,
            ),
        ]

    def test_catalog_merges_equivalent_evidence_and_keeps_scope_variant(self) -> None:
        catalog = build_preview_catalog(self.rows)

        self.assertTrue(catalog["preview_only"])
        self.assertEqual(catalog["positive_judgments"], 4)
        self.assertEqual(catalog["unique_memory_count"], 3)
        self.assertEqual(catalog["filtered_counts"], {"temporary": 1})
        self.assertEqual(catalog["scope_variant_group_count"], 1)
        global_rule = next(
            memory
            for memory in catalog["memories"]
            if memory["stable_key"] == "workflow.code_generation.confirmation"
            and memory["scope"]["level"] == "global"
        )
        self.assertEqual(global_rule["evidence_count"], 2)
        self.assertEqual(
            global_rule["source_candidate_keys"],
            ["global-confirm-1", "global-confirm-2"],
        )

    def test_opposite_values_in_the_same_scope_fail_closed_as_conflict(self) -> None:
        rows = self.rows[:1] + [
            _row(
                "global-opposite",
                "以后生成正式代码不需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "global"},
                value="not_required",
                relation="conflict",
            )
        ]

        catalog = build_preview_catalog(rows)

        self.assertEqual(catalog["unique_memory_count"], 0)
        self.assertEqual(catalog["same_scope_conflict_count"], 1)
        self.assertEqual(
            {item["value"] for item in catalog["conflicts"][0]["variants"]},
            {"required", "not_required"},
        )
        relevant = recall_from_catalog(catalog, "请生成正式代码")
        unrelated = recall_from_catalog(catalog, "查看天气")
        self.assertEqual(len(relevant["unresolved_conflicts"]), 1)
        self.assertEqual(unrelated["unresolved_conflicts"], [])

    def test_project_scope_overrides_global_rule_during_recall(self) -> None:
        catalog = build_preview_catalog(self.rows)

        result = recall_from_catalog(
            catalog,
            "请在当前项目生成正式代码，实现一个新功能。",
            project_key="project-a",
        )

        confirmation = [
            item
            for item in result["selected"]
            if item["stable_key"] == "workflow.code_generation.confirmation"
        ]
        self.assertEqual(len(confirmation), 1)
        self.assertEqual(confirmation[0]["normalized_value"], "not_required")
        self.assertEqual(confirmation[0]["scope"]["level"], "project")
        self.assertEqual(len(result["suppressed_by_scope"]), 1)
        self.assertEqual(
            result["suppressed_by_scope"][0]["normalized_value"], "required"
        )

    def test_other_project_uses_global_rule(self) -> None:
        catalog = build_preview_catalog(self.rows)

        result = recall_from_catalog(
            catalog,
            "请生成正式代码。",
            project_key="project-b",
        )

        confirmation = next(
            item
            for item in result["selected"]
            if item["stable_key"] == "workflow.code_generation.confirmation"
        )
        self.assertEqual(confirmation["normalized_value"], "required")
        self.assertEqual(confirmation["scope"]["level"], "global")
        self.assertEqual(result["suppressed_by_scope"], [])

    def test_unrelated_query_returns_only_always_applicable_response_contract(self) -> None:
        catalog = build_preview_catalog(self.rows)

        result = recall_from_catalog(catalog, "查看明天的天气", project_key="project-a")

        self.assertEqual(
            [item["stable_key"] for item in result["selected"]],
            ["preference.response.language"],
        )
        self.assertEqual(result["suppressed_by_scope"], [])

    def test_domain_is_inferred_for_embedded_workflow(self) -> None:
        rows = self.rows + [
            _row(
                "embedded-compile",
                "所有单片机项目修改代码后必须编译检查。",
                should_store=True,
                key="workflow.embedded.code_change.compile_check",
                memory_type="workflow_rule",
                scope={"level": "domain", "domain": "embedded"},
                value="required",
                triggers=("单片机", "修改代码", "编译检查"),
            )
        ]
        catalog = build_preview_catalog(rows)

        result = recall_from_catalog(catalog, "修改单片机代码并验证", project_key="project-b")

        self.assertEqual(result["context"]["domain"], "embedded")
        self.assertIn(
            "workflow.embedded.code_change.compile_check",
            {item["stable_key"] for item in result["selected"]},
        )
        self.assertEqual(result["suppressed_by_scope"], [])

    def test_recall_rejects_empty_query_and_invalid_limit(self) -> None:
        catalog = build_preview_catalog(self.rows)
        with self.assertRaises(ValueError):
            recall_from_catalog(catalog, "")
        with self.assertRaises(ValueError):
            recall_from_catalog(catalog, "x" * 4001)
        with self.assertRaises(ValueError):
            recall_from_catalog(catalog, "query", limit=0)

    def test_locally_blocked_sensitive_quote_is_redacted_from_catalog(self) -> None:
        secret = "api_key=synthetic-secret-value"
        blocked = {
            "candidate_key": "sensitive",
            "status": "blocked",
            "execution_mode": "local_block",
            "result_json": None,
            "exact_quote": secret,
            "disposition": "sensitive_blocked",
        }

        catalog = build_preview_catalog([blocked])

        self.assertEqual(catalog["filtered_counts"], {"sensitive_local_block": 1})
        self.assertNotIn(secret, json.dumps(catalog, ensure_ascii=False))
        self.assertEqual(
            catalog["filtered"][0]["quote"], "[REDACTED sensitive candidate]"
        )

    def test_conditional_scope_is_not_widened_without_typed_context(self) -> None:
        rows = [
            _row(
                "conditional",
                "在 Windows 环境生成代码前需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "global", "conditions": {"os": "windows"}},
                value="required",
                triggers=("生成正式代码", "确认"),
            )
        ]
        catalog = build_preview_catalog(rows)

        result = recall_from_catalog(catalog, "请生成正式代码", project_key="project-a")

        self.assertEqual(result["selected"], [])
        self.assertEqual(
            result["excluded_by_context"][0]["reason"],
            "conditional_scope_requires_typed_context",
        )

    def test_project_conflict_blocks_global_fallback(self) -> None:
        rows = self.rows[:1] + [
            _row(
                "project-required",
                "仅在当前项目中，生成正式代码前需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "project", "project_key": "project-a"},
                value="required",
                triggers=("生成正式代码", "确认"),
            ),
            _row(
                "project-not-required",
                "仅在当前项目中，生成正式代码前不需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "project", "project_key": "project-a"},
                value="not_required",
                triggers=("生成正式代码", "确认"),
            ),
        ]
        catalog = build_preview_catalog(rows)

        result = recall_from_catalog(
            catalog,
            "请生成正式代码。",
            project_key="project-a",
        )

        self.assertNotIn(
            "workflow.code_generation.confirmation",
            {item["stable_key"] for item in result["selected"]},
        )
        self.assertEqual(len(result["unresolved_conflicts"]), 1)
        self.assertEqual(result["unresolved_conflicts"][0]["scope"]["level"], "project")

    def test_conditional_project_variant_blocks_global_fallback(self) -> None:
        rows = self.rows[:1] + [
            _row(
                "project-windows",
                "当前项目在 Windows 环境生成代码前不需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={
                    "level": "project",
                    "project_key": "project-a",
                    "conditions": {"os": "windows"},
                },
                value="not_required",
                triggers=("生成正式代码", "确认"),
            )
        ]
        catalog = build_preview_catalog(rows)

        result = recall_from_catalog(
            catalog,
            "请生成正式代码。",
            project_key="project-a",
        )

        self.assertEqual(result["selected"], [])
        self.assertEqual(len(result["unresolved_context"]), 1)
        self.assertEqual(
            result["unresolved_context"][0]["blockers"][0]["reason"],
            "conditional_scope_requires_typed_context",
        )

    def test_unknown_project_context_blocks_global_when_project_variant_exists(self) -> None:
        catalog = build_preview_catalog(self.rows)

        result = recall_from_catalog(catalog, "请生成正式代码。")

        self.assertNotIn(
            "workflow.code_generation.confirmation",
            {item["stable_key"] for item in result["selected"]},
        )
        self.assertFalse(result["context"]["project_context_known"])
        self.assertEqual(
            result["unresolved_context"][0]["blockers"][0]["reason"],
            "project_context_unknown",
        )

    def test_generic_confirmation_word_does_not_recall_code_workflow(self) -> None:
        catalog = build_preview_catalog(self.rows)

        result = recall_from_catalog(
            catalog,
            "确认天气是否晴朗。",
            project_key="project-a",
        )

        self.assertNotIn(
            "workflow.code_generation.confirmation",
            {item["stable_key"] for item in result["selected"]},
        )
        self.assertEqual(result["unresolved_conflicts"], [])

    def test_conditional_conflict_remains_an_unknown_blocker(self) -> None:
        conditional_scope = {
            "level": "project",
            "project_key": "project-a",
            "conditions": {"os": "windows"},
        }
        rows = self.rows[:1] + [
            _row(
                "conditional-required",
                "Windows 下生成正式代码前需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope=conditional_scope,
                value="required",
                triggers=("生成正式代码", "确认"),
            ),
            _row(
                "conditional-not-required",
                "Windows 下生成正式代码前不需要确认。",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope=conditional_scope,
                value="not_required",
                triggers=("生成正式代码", "确认"),
            ),
        ]
        catalog = build_preview_catalog(rows)

        result = recall_from_catalog(
            catalog,
            "请生成正式代码。",
            project_key="project-a",
        )

        self.assertEqual(result["selected"], [])
        self.assertEqual(result["unresolved_conflicts"], [])
        self.assertEqual(result["unresolved_context"][0]["blockers"][0]["kind"], "conflict")

    def test_known_inapplicable_variant_cannot_make_global_rule_relevant(self) -> None:
        rows = [
            _row(
                "global-alpha",
                "Use this workflow only for alpha tasks.",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "global"},
                value="required",
                statement="Alpha-only confirmation workflow.",
                triggers=("alpha only",),
            ),
            _row(
                "project-zebra",
                "Deploy zebra without confirmation in project A.",
                should_store=True,
                key="workflow.code_generation.confirmation",
                memory_type="workflow_rule",
                scope={"level": "project", "project_key": "project-a"},
                value="not_required",
                statement="Project A zebra deployment skips confirmation.",
                triggers=("deploy zebra",),
            ),
        ]
        catalog = build_preview_catalog(rows)

        result = recall_from_catalog(
            catalog,
            "deploy zebra",
            project_key="project-b",
        )

        self.assertEqual(result["selected"], [])
        self.assertEqual(result["unresolved_context"], [])

    def test_explicit_domain_is_trimmed_before_scope_matching(self) -> None:
        rows = [
            _row(
                "embedded",
                "嵌入式项目修改代码后必须编译检查。",
                should_store=True,
                key="workflow.embedded.code_change.compile_check",
                memory_type="workflow_rule",
                scope={"level": "domain", "domain": "embedded"},
                value="required",
                triggers=("修改代码", "编译检查"),
            )
        ]
        catalog = build_preview_catalog(rows)

        result = recall_from_catalog(
            catalog,
            "修改代码并进行编译检查。",
            project_key="project-a",
            domain=" embedded ",
        )

        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["context"]["domain"], "embedded")


class CatalogReadOnlyIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "memory_lab.db"
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO source(source_id, source_kind, source_path, created_at, baseline_cutoff_ms)
                VALUES ('source', 'sqlite', 'source.db', '2026-01-01T00:00:00Z', 1)
                """
            )
            connection.execute(
                """
                INSERT INTO project_context(
                    project_key, directory, display_name, identity_source, created_at
                ) VALUES ('project-a', 'D:\\project-a', 'project-a', 'test',
                          '2026-01-01T00:00:00Z')
                """
            )
            connection.execute(
                """
                INSERT INTO session_context(
                    source_id, session_id, project_key, directory, context_source,
                    first_seen_at, last_seen_at
                ) VALUES ('source', 'session', 'project-a', 'D:\\project-a', 'test',
                          '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
            quote = "以后生成正式代码前先向我确认。"
            connection.execute(
                """
                INSERT INTO event(
                    event_id, source_id, session_id, message_id, part_id, event_type,
                    timestamp, content, content_hash, source_kind, eligibility_status,
                    ingested_at, raw_json
                ) VALUES ('event', 'source', 'session', 'message', 'part', 'user_text',
                          1, ?, 'hash', 'codepulse_sqlite', 'eligible',
                          '2026-01-01T00:00:00Z', '{}')
                """,
                (quote,),
            )
            connection.execute(
                """
                INSERT INTO event_annotation(
                    event_id, semantic_content, project_key, annotation_source, created_at
                ) VALUES ('event', ?, 'project-a', 'test', '2026-01-01T00:00:00Z')
                """,
                (quote,),
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at, completed_at,
                    settle_ms, status
                ) VALUES ('turn-run', 'source', 'test', '2026-01-01T00:00:00Z',
                          '2026-01-01T00:00:01Z', 0, 'complete')
                """
            )
            connection.execute(
                """
                INSERT INTO conversation_turn(
                    turn_id, source_id, session_id, user_message_id, primary_event_id,
                    user_timestamp, current_state, first_detected_at, last_detected_at,
                    last_detection_run_id, last_structural_signature
                ) VALUES ('turn', 'source', 'session', 'message', 'event', 1, 'completed',
                          '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z',
                          'turn-run', 'signature')
                """
            )
            connection.execute(
                "INSERT INTO turn_event(turn_id, event_id) VALUES ('turn', 'event')"
            )
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, completed_at, status,
                    decision_count, counts_json
                ) VALUES ('admission', 'test', '2026-01-01T00:00:02Z',
                          '2026-01-01T00:00:03Z', 'complete', 1, '{"eligible":1}')
                """
            )
            connection.execute(
                """
                INSERT INTO admission_decision(
                    decision_id, run_id, candidate_key, event_id, ordinal,
                    exact_quote, start_offset, end_offset, family, memory_type,
                    stable_key_suggestion, scope_json, activation_triggers_json,
                    disposition, signals_json, instruction_coverage_json,
                    reasons_json, decided_at
                ) VALUES ('decision', 'admission', 'candidate', 'event', 1, ?, 0, ?,
                          'user_contract', 'workflow_rule',
                          'workflow.code_generation.confirmation', '{"level":"global"}',
                          '["生成正式代码","确认"]', 'eligible', '{}',
                          '{"status":"none"}', '[]', '2026-01-01T00:00:03Z')
                """,
                (quote, len(quote)),
            )
            _admission, _rows, snapshot_hash = latest_admission_snapshot(connection)
            self.assertIsNotNone(snapshot_hash)
            state = json.dumps(formal_state_snapshot(connection), sort_keys=True)
            connection.execute(
                """
                INSERT INTO llm_shadow_run(
                    shadow_run_id, shadow_version, admission_run_id,
                    admission_snapshot_hash, provider, model, prompt_version,
                    input_schema_version, output_schema_version, started_at, completed_at,
                    status, candidate_count, success_count, error_count, replay_count,
                    counts_json, formal_state_before_json, formal_state_after_json
                ) VALUES ('shadow', 'test', 'admission', ?, 'mock', 'mock', 'prompt',
                          'input', 'output', '2026-01-01T00:00:04Z',
                          '2026-01-01T00:00:05Z', 'complete', 1, 1, 0, 0,
                          '{"valid":1}', ?, ?)
                """,
                (snapshot_hash, state, state),
            )
            result = {
                "should_store": True,
                "type": "workflow_rule",
                "key": "workflow.code_generation.confirmation",
                "scope": {"level": "global"},
                "value": {
                    "normalized": "required",
                    "statement": "生成正式代码前先确认。",
                },
                "relation": {"kind": "new", "target_kind": None, "target_id": None},
                "confidence": 0.99,
                "reason": "test",
                "evidence": [
                    {
                        "event_id": "event",
                        "part_id": "part",
                        "quote": quote,
                        "start_offset": 0,
                        "end_offset": len(quote),
                    }
                ],
            }
            connection.execute(
                """
                INSERT INTO llm_judgment(
                    judgment_id, shadow_run_id, admission_run_id, decision_id,
                    candidate_key, request_hash, model, prompt_version,
                    input_schema_version, output_schema_version, status, execution_mode,
                    result_json, should_store, memory_type, stable_key, scope_json,
                    semantic_value, relation, confidence, created_at
                ) VALUES ('judgment', 'shadow', 'admission', 'decision', 'candidate',
                          'request', 'mock', 'prompt', 'input', 'output', 'valid',
                          'judge_call', ?, 1, 'workflow_rule',
                          'workflow.code_generation.confirmation', '{"level":"global"}',
                          'required', 'new', 0.99, '2026-01-01T00:00:05Z')
                """,
                (json.dumps(result, ensure_ascii=False),),
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _digest(self) -> str:
        return hashlib.sha256(self.database.read_bytes()).hexdigest()

    def test_catalog_reads_exact_current_shadow_without_changing_database(self) -> None:
        before = self._digest()

        result = run_catalog(self.database, "shadow")

        self.assertEqual(result["unique_memory_count"], 1)
        self.assertTrue(result["formal_state_unchanged"])
        self.assertEqual(self._digest(), before)

    def test_newer_incomplete_admission_fails_closed(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, status,
                    decision_count, counts_json
                ) VALUES ('newer', 'test', '2026-01-01T00:00:06Z', 'running', 0, '{}')
                """
            )

        with self.assertRaisesRegex(CatalogValidationError, "newer admission"):
            run_catalog(self.database, "shadow")

    def test_formal_state_change_after_shadow_fails_closed(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO memory(
                    memory_id, stable_key, scope_fingerprint, scope_json,
                    family, memory_type, status, created_at, updated_at
                ) VALUES ('memory', 'key', 'scope', '{"level":"global"}',
                          'user_contract', 'workflow_rule', 'active',
                          '2026-01-01T00:00:06Z', '2026-01-01T00:00:06Z')
                """
            )

        with self.assertRaisesRegex(CatalogValidationError, "formal memory state changed"):
            run_catalog(self.database, "shadow")

    def test_default_does_not_fall_back_past_newer_running_shadow(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            state = json.dumps(formal_state_snapshot(connection), sort_keys=True)
            _admission, _rows, snapshot_hash = latest_admission_snapshot(connection)
            connection.execute(
                """
                INSERT INTO llm_shadow_run(
                    shadow_run_id, shadow_version, admission_run_id,
                    admission_snapshot_hash, provider, model, prompt_version,
                    input_schema_version, output_schema_version, started_at,
                    status, candidate_count, success_count, error_count, replay_count,
                    counts_json, formal_state_before_json
                ) VALUES ('newer-shadow', 'test', 'admission', ?, 'mock', 'mock',
                          'prompt', 'input', 'output', '2026-01-01T00:00:06Z',
                          'running', 1, 0, 0, 0, '{}', ?)
                """,
                (snapshot_hash, state),
            )

        with self.assertRaisesRegex(CatalogValidationError, "not complete"):
            run_catalog(self.database)
        self.assertEqual(run_catalog(self.database, "shadow")["unique_memory_count"], 1)

    def test_replayed_shadow_is_not_a_catalog_source(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                "UPDATE llm_shadow_run SET replay_count = 1 WHERE shadow_run_id = 'shadow'"
            )

        with self.assertRaisesRegex(CatalogValidationError, "fresh model judgments"):
            run_catalog(self.database, "shadow")

    def test_complete_run_with_local_block_is_a_valid_redacted_catalog_source(self) -> None:
        with closing(open_store(self.database)) as connection, connection:
            _admission, _rows, snapshot_hash = latest_admission_snapshot(connection)
            state = json.dumps(formal_state_snapshot(connection), sort_keys=True)
            connection.execute(
                """
                INSERT INTO llm_shadow_run(
                    shadow_run_id, shadow_version, admission_run_id,
                    admission_snapshot_hash, provider, model, prompt_version,
                    input_schema_version, output_schema_version, started_at, completed_at,
                    status, candidate_count, success_count, error_count, replay_count,
                    counts_json, formal_state_before_json, formal_state_after_json
                ) VALUES ('blocked-shadow', 'test', 'admission', ?, 'mock', 'mock',
                          'prompt', 'input', 'output', '2026-01-01T00:00:06Z',
                          '2026-01-01T00:00:07Z', 'complete', 1, 0, 0, 0,
                          '{"blocked":1}', ?, ?)
                """
                ,
                (snapshot_hash, state, state),
            )
            connection.execute(
                """
                INSERT INTO llm_judgment(
                    judgment_id, shadow_run_id, admission_run_id, decision_id,
                    candidate_key, request_hash, model, prompt_version,
                    input_schema_version, output_schema_version, status, execution_mode,
                    error_code, error_message, created_at
                ) VALUES ('blocked-judgment', 'blocked-shadow', 'admission', 'decision',
                          'candidate', 'blocked-request', 'mock', 'prompt', 'input',
                          'output', 'blocked', 'local_block', 'sensitive_data',
                          'blocked locally', '2026-01-01T00:00:07Z')
                """
            )

        result = run_catalog(self.database, "blocked-shadow")

        self.assertEqual(result["unique_memory_count"], 0)
        self.assertEqual(result["filtered_counts"], {"sensitive_local_block": 1})
        self.assertNotIn(
            "以后生成正式代码前先向我确认。",
            json.dumps(result, ensure_ascii=False),
        )

    def test_consistent_read_detects_a_concurrent_commit(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.commit()

        with closing(open_read_only(self.database)) as reader:
            def operation() -> None:
                reader.execute("SELECT COUNT(*) FROM source").fetchone()
                with closing(sqlite3.connect(self.database)) as writer:
                    writer.execute(
                        "UPDATE source SET source_path = 'changed.db' WHERE source_id = 'source'"
                    )
                    writer.commit()

            with self.assertRaisesRegex(CatalogValidationError, "changed during preview"):
                _consistent_read(reader, operation)


if __name__ == "__main__":
    unittest.main()
