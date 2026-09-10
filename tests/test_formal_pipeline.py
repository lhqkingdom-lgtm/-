from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.consolidation import rollback_memory, semantic_value_for
from memory_lab.formal_pipeline import (
    REVIEW_DECISION_SCHEMA_VERSION,
    judgment_to_update_plan,
    run_auto_memory_pipeline,
)
from memory_lab.store import open_store
from tests.test_consolidation import ConsolidationTests as _ConsolidationFixture

_add_snapshot = _ConsolidationFixture.add_snapshot
_table_counts = _ConsolidationFixture.table_counts
_confirmation = _ConsolidationFixture.confirmation
del _ConsolidationFixture


class DeterministicWriter:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    @staticmethod
    def _identity(quote: str) -> tuple[str, str]:
        if "中文" in quote or "英文" in quote:
            return "preference.response.language", "response_preference"
        if "结论" in quote and ("步骤" in quote or "细节" in quote):
            # Real R1 DeepSeek normalized this broader admission label to the
            # response-specific formal type.
            return "preference.response.structure", "response_preference"
        if "单片机" in quote and "编译" in quote:
            return "workflow.embedded.code_change.compile_check", "workflow_rule"
        return "workflow.code_generation.confirmation", "workflow_rule"

    @staticmethod
    def _scope(request: dict) -> dict:
        limit = request["hard_signals"]["scope_limit"]
        level = limit["allowed_levels"][0]
        result = {"level": level}
        if level == "domain":
            result["domain"] = limit["domain"]
        elif level == "project":
            result["project_key"] = limit["project_key"]
        return result

    def write(self, request, *, family: str):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("writer unavailable")
        request = dict(request)
        quote = request["claim"]["exact_quote"]
        key, memory_type = self._identity(quote)
        value = semantic_value_for(key, quote)
        self.assert_value(value)
        scope = self._scope(request)
        same_key = [
            ("memory", item["memory_id"], item)
            for item in request["current_memories"]
            if item["stable_key"] == key
        ] + [
            ("proposal", item["proposal_id"], item)
            for item in request["pending_proposals"]
            if item["stable_key"] == key
        ]
        same_scope = [item for item in same_key if item[2]["scope"] == scope]
        if same_scope:
            target_kind, target_id, target = same_scope[0]
            target_value = target.get("value", target.get("semantic_value"))
            if isinstance(target_value, dict):
                target_value = target_value["normalized"]
            if target_value == value:
                relation = "equivalent"
            else:
                relation = (
                    "correction"
                    if request["hard_signals"]["correction"]
                    else "conflict"
                )
        elif same_key:
            target_kind, target_id, _target = same_key[0]
            relation = "different_scope"
        else:
            target_kind = target_id = None
            relation = "new"
        statement = {
            ("preference.response.language", "zh"): "默认使用中文回复。",
            ("preference.response.language", "en"): "默认使用英文回复。",
            (
                "preference.response.structure",
                "conclusion_then_details",
            ): "回复时先给结论，再说明步骤或细节。",
            (
                "workflow.code_generation.confirmation",
                "required",
            ): "生成正式代码前必须先征得用户确认。",
            (
                "workflow.code_generation.confirmation",
                "not_required",
            ): "生成正式代码前不要求先征得用户确认。",
            (
                "workflow.embedded.code_change.compile_check",
                "required",
            ): "单片机项目修改代码后必须进行编译检查。",
        }[(key, value)]
        result = {
            "should_store": True,
            "type": memory_type,
            "key": key,
            "scope": scope,
            "value": {"normalized": value, "statement": statement},
            "relation": {
                "kind": relation,
                "target_kind": target_kind,
                "target_id": target_id,
            },
            "evidence": [
                {
                    "event_id": request["claim"]["event_id"],
                    "part_id": request["claim"]["part_id"],
                    "quote": quote,
                    "start_offset": request["claim"]["start_offset"],
                    "end_offset": request["claim"]["end_offset"],
                }
            ],
            "confidence": 0.99,
            "reason": "durable exact user contract",
        }
        return judgment_to_update_plan(result, request, family=family)

    @staticmethod
    def assert_value(value: str | None) -> None:
        if value is None:
            raise AssertionError("fixture quote has no canonical semantic value")


class AcceptReviewer:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        revise: bool = False,
        forge_evidence: bool = False,
        callback=None,
    ) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.revise = revise
        self.forge_evidence = forge_evidence
        self.callback = callback

    def review(self, request):
        self.calls += 1
        if self.callback:
            self.callback(self.calls)
        if self.fail_on_call == self.calls:
            raise RuntimeError("reviewer unavailable")
        revised = None
        outcome = "accept"
        if self.revise:
            outcome = "revise"
            revised = copy.deepcopy(request["update_plan"])
            revised["reason"] = revised["reason"] + "; independently checked"
            if self.forge_evidence:
                revised["evidence"][0]["start_offset"] += 1
        return {
            "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
            "request_id": request["request_id"],
            "reviewed_plan_hash": request["update_plan_hash"],
            "outcome": outcome,
            "revised_plan": revised,
            "confidence": 0.98,
            "reason": "independent evidence and scope check",
        }


class WrongWorkflowTypeWriter(DeterministicWriter):
    def write(self, request, *, family: str):
        plan = super().write(request, family=family)
        plan["memory_type"] = "project_rule"
        return plan


class NoMemoryWriter:
    def __init__(self) -> None:
        self.calls = 0

    def write(self, request, *, family: str):
        self.calls += 1
        claim = request["claim"]
        result = {
            "should_store": False,
            "type": None,
            "key": None,
            "scope": None,
            "value": None,
            "relation": {"kind": "unrelated", "target_kind": None, "target_id": None},
            "evidence": [
                {
                    "event_id": claim["event_id"],
                    "part_id": claim["part_id"],
                    "quote": claim["exact_quote"],
                    "start_offset": claim["start_offset"],
                    "end_offset": claim["end_offset"],
                }
            ],
            "confidence": 0.98,
            "reason": "durable wording but no reusable supported user contract",
        }
        return judgment_to_update_plan(result, request, family=family)


class RejectReviewer(AcceptReviewer):
    def review(self, request):
        self.calls += 1
        return {
            "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
            "request_id": request["request_id"],
            "reviewed_plan_hash": request["update_plan_hash"],
            "outcome": "reject",
            "revised_plan": None,
            "confidence": 0.99,
            "reason": "no reusable memory should be committed",
        }


class UnboundedSemanticKeyWriter(DeterministicWriter):
    def write(self, request, *, family: str):
        plan = super().write(request, family=family)
        plan["stable_key"] = "preference.response.unbounded_dynamic_rule"
        plan["value"] = {
            "normalized": "arbitrary",
            "statement": "An arbitrary model-defined rule.",
        }
        return plan


class FormalPipelineTests(unittest.TestCase):
    # Reuse only the mature database fixture builder, not its test assertions.
    add_snapshot = _add_snapshot
    table_counts = _table_counts
    confirmation = staticmethod(_confirmation)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "memory_lab.db"
        self.run_counter = 0
        self.item_counter = 0
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO source(source_id, source_kind, source_path, created_at)
                VALUES ('cp', 'sqlite', 'source.db', '2026-01-01T00:00:00.000+00:00')
                """
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at,
                    completed_at, settle_ms, status
                ) VALUES (
                    'turn_run', 'cp', 'test', '2026-01-01T00:00:00.000+00:00',
                    '2026-01-01T00:00:01.000+00:00', 0, 'complete'
                )
                """
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_pipeline(self, writer, reviewer, *, refresh: bool = False):
        return run_auto_memory_pipeline(
            self.database,
            writer,
            reviewer,
            provider="test",
            model="test-model",
            refresh=refresh,
        )

    def test_f1_two_atomic_plans_are_independently_reviewed_and_committed(self) -> None:
        self.add_snapshot(
            [
                {
                    "quote": "以后回复时先给结论，再说明处理步骤。",
                    "stable_key": "preference.response.structure",
                    "memory_type": "preference",
                    "scope": {"level": "global"},
                },
                {
                    "quote": "以后在所有单片机项目中，修改代码后都必须进行编译检查。",
                    "stable_key": "workflow.embedded.code_change.compile_check",
                    "scope": {"level": "domain", "domain": "embedded"},
                },
            ]
        )
        writer = DeterministicWriter()
        reviewer = AcceptReviewer()
        report = self.run_pipeline(writer, reviewer)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["counts"]["actions"], {"created": 2})
        self.assertEqual((writer.calls, reviewer.calls), (2, 2))
        self.assertEqual(self.table_counts(), (2, 2, 2, 0))
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_writer_result").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_reviewer_result").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM candidate_review").fetchone()[0], 2)
            memory_types = connection.execute(
                "SELECT stable_key,memory_type FROM memory ORDER BY stable_key"
            ).fetchall()
            payload_types = [
                json.loads(row[0])["memory_type"]
                for row in connection.execute(
                    "SELECT application_payload_json FROM memory_proposal ORDER BY stable_key"
                )
            ]
        self.assertEqual(
            memory_types,
            [
                ("preference.response.structure", "response_preference"),
                ("workflow.embedded.code_change.compile_check", "workflow_rule"),
            ],
        )
        self.assertEqual(payload_types, ["response_preference", "workflow_rule"])

    def test_workflow_type_policy_rejects_project_rule_without_widening(self) -> None:
        self.add_snapshot([self.confirmation()])
        report = self.run_pipeline(WrongWorkflowTypeWriter(), AcceptReviewer())
        self.assertEqual(report["status"], "failed")
        self.assertIn("memory_type", report["error_message"])
        self.assertTrue(report["formal_state_unchanged"])
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))

    def test_semantic_review_classifies_and_merges_cross_project_synonym(self) -> None:
        self.add_snapshot(
            [
                {
                    "quote": "以后回复时先给结论，再说明处理步骤。",
                    "stable_key": "preference.response.structure",
                    "memory_type": "preference",
                    "scope": {"level": "global"},
                    "project_key": "project_a",
                }
            ]
        )
        self.run_pipeline(DeterministicWriter(), AcceptReviewer())

        synonym = "以后回答我的问题时，先告诉我最终结论，然后再展开具体步骤。"
        self.add_snapshot(
            [
                {
                    "quote": synonym,
                    "stable_key": None,
                    "memory_type": None,
                    "scope": {"level": "global"},
                    "project_key": "project_b",
                    "disposition": "needs_semantic_review",
                    "candidate_key": "semantic-synonym",
                }
            ]
        )
        writer = DeterministicWriter()
        reviewer = AcceptReviewer()
        report = self.run_pipeline(writer, reviewer)

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["counts"]["actions"], {"merged_evidence": 1})
        self.assertEqual((writer.calls, reviewer.calls), (1, 1))
        self.assertEqual(self.table_counts(), (1, 1, 2, 0))
        with closing(sqlite3.connect(self.database)) as connection:
            projects = connection.execute(
                "SELECT project_key FROM evidence ORDER BY occurred_at"
            ).fetchall()
        self.assertEqual(projects, [("project_a",), ("project_b",)])

        replay = self.run_pipeline(writer, reviewer)
        self.assertEqual(replay["actions"], [])
        self.assertEqual((writer.calls, reviewer.calls), (1, 1))
        self.assertEqual(self.table_counts(), (1, 1, 2, 0))

    def test_unknown_semantic_claim_reaches_reviewer_and_is_rejected(self) -> None:
        self.add_snapshot(
            [
                {
                    "quote": "以后请始终记住一些并不能复用的模糊事情。",
                    "stable_key": None,
                    "memory_type": None,
                    "scope": {"level": "global"},
                    "disposition": "needs_semantic_review",
                }
            ]
        )
        writer = NoMemoryWriter()
        reviewer = RejectReviewer()
        report = self.run_pipeline(writer, reviewer)

        self.assertEqual(report["status"], "complete_with_rejections")
        self.assertEqual(report["counts"]["review"], {"reject": 1})
        self.assertEqual((writer.calls, reviewer.calls), (1, 1))
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))

    def test_semantic_review_cannot_create_a_dynamic_stable_key(self) -> None:
        self.add_snapshot(
            [
                {
                    "quote": "以后回答我的问题时，先告诉我最终结论，然后再展开具体步骤。",
                    "stable_key": None,
                    "memory_type": None,
                    "scope": {"level": "global"},
                    "disposition": "needs_semantic_review",
                }
            ]
        )
        report = self.run_pipeline(UnboundedSemanticKeyWriter(), AcceptReviewer())
        self.assertEqual(report["status"], "failed")
        self.assertIn("bounded", report["error_message"])
        self.assertTrue(report["formal_state_unchanged"])
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))

    def test_semantic_review_preserves_hard_vetoes_before_model_calls(self) -> None:
        for signals_patch, coverage in (
            ({"temporary": ["本轮"]}, {"status": "none", "scope": None}),
            ({"sensitive": True}, {"status": "none", "scope": None}),
            ({}, {"status": "full", "scope": None}),
        ):
            other_db = Path(self.temporary_directory.name) / (
                f"veto-{len(list(Path(self.temporary_directory.name).glob('veto-*.db')))}.db"
            )
            self.database = other_db
            self.run_counter = self.item_counter = 0
            with closing(open_store(self.database)) as connection, connection:
                connection.execute(
                    "INSERT INTO source VALUES ('cp','sqlite','source.db','x',NULL,NULL)"
                )
                connection.execute(
                    """
                    INSERT INTO turn_detection_run(
                        run_id,source_id,detector_version,started_at,completed_at,
                        settle_ms,status
                    ) VALUES ('turn_run','cp','test','x','x',0,'complete')
                    """
                )
            self.add_snapshot(
                [
                    {
                        "quote": "以后回答我的问题时，先告诉我最终结论，然后再展开具体步骤。",
                        "stable_key": None,
                        "memory_type": None,
                        "scope": {"level": "global"},
                        "disposition": "needs_semantic_review",
                    }
                ]
            )
            with closing(open_store(self.database)) as connection, connection:
                row = connection.execute(
                    "SELECT decision_id,signals_json FROM admission_decision"
                ).fetchone()
                signals = json.loads(row["signals_json"])
                signals.update(signals_patch)
                connection.execute(
                    """
                    UPDATE admission_decision
                    SET signals_json=?, instruction_coverage_json=?
                    WHERE decision_id=?
                    """,
                    (json.dumps(signals), json.dumps(coverage), row["decision_id"]),
                )
            writer = DeterministicWriter()
            reviewer = AcceptReviewer()
            report = self.run_pipeline(writer, reviewer)
            self.assertEqual(report["status"], "complete_with_rejections")
            self.assertEqual((writer.calls, reviewer.calls), (0, 0))
            self.assertEqual(self.table_counts(), (0, 0, 0, 0))

    def test_repeated_snapshot_is_pre_model_noop_and_adds_no_business_rows(self) -> None:
        self.add_snapshot([self.confirmation(candidate_key="same-candidate")])
        writer = DeterministicWriter()
        reviewer = AcceptReviewer()
        first = self.run_pipeline(writer, reviewer)
        self.assertEqual(first["counts"]["actions"], {"created": 1})
        before_counts = self.table_counts()
        second = self.run_pipeline(writer, reviewer)
        self.assertEqual(second["actions"], [])
        self.assertFalse(second["formal_state_changed"])
        self.assertEqual(self.table_counts(), before_counts)
        self.assertEqual((writer.calls, reviewer.calls), (1, 1))
        self.assertEqual(second["audit"][0]["outcome"], "noop")

    def test_writer_or_reviewer_failure_keeps_formal_tables_empty(self) -> None:
        self.add_snapshot(
            [
                self.confirmation(candidate_key="first"),
                {
                    "quote": "以后默认使用中文回复。",
                    "stable_key": "preference.response.language",
                    "memory_type": "response_preference",
                    "scope": {"level": "global"},
                    "candidate_key": "second",
                },
            ]
        )
        writer_failure = self.run_pipeline(
            DeterministicWriter(fail_on_call=2), AcceptReviewer()
        )
        self.assertEqual(writer_failure["status"], "failed")
        self.assertTrue(writer_failure["formal_state_unchanged"])
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))

        other_db = Path(self.temporary_directory.name) / "reviewer.db"
        self.database = other_db
        self.run_counter = self.item_counter = 0
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                "INSERT INTO source VALUES ('cp','sqlite','source.db','x',NULL,NULL)"
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id,source_id,detector_version,started_at,completed_at,
                    settle_ms,status
                ) VALUES ('turn_run','cp','test','x','x',0,'complete')
                """
            )
        self.add_snapshot([self.confirmation()])
        review_failure = self.run_pipeline(
            DeterministicWriter(), AcceptReviewer(fail_on_call=1)
        )
        self.assertEqual(review_failure["status"], "failed")
        self.assertTrue(review_failure["formal_state_unchanged"])
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))

    def test_reviewer_may_revise_once_but_forged_evidence_fails_closed(self) -> None:
        self.add_snapshot([self.confirmation()])
        revised = self.run_pipeline(
            DeterministicWriter(), AcceptReviewer(revise=True)
        )
        self.assertEqual(revised["status"], "complete")
        self.assertEqual(revised["counts"]["review"], {"revise": 1})
        self.assertEqual(self.table_counts(), (1, 1, 1, 0))

        forged_db = Path(self.temporary_directory.name) / "forged.db"
        self.database = forged_db
        self.run_counter = self.item_counter = 0
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                "INSERT INTO source VALUES ('cp','sqlite','source.db','x',NULL,NULL)"
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id,source_id,detector_version,started_at,completed_at,
                    settle_ms,status
                ) VALUES ('turn_run','cp','test','x','x',0,'complete')
                """
            )
        self.add_snapshot([self.confirmation()])
        forged = self.run_pipeline(
            DeterministicWriter(),
            AcceptReviewer(revise=True, forge_evidence=True),
        )
        self.assertEqual(forged["status"], "failed")
        self.assertTrue(forged["formal_state_unchanged"])
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))

    def test_same_message_later_user_contract_wins_without_open_conflict(self) -> None:
        first = "以后生成正式代码前先向我确认。"
        second = "以后生成正式代码不用先问我。"
        source = first + "\n" + second
        common = {
            "event_id": "same-event",
            "session_id": "same-session",
            "message_id": "same-message",
            "part_id": "same-part",
            "project_key": "same-project",
            "event_timestamp": 100,
            "evidence_source_text": source,
            "stable_key": "workflow.code_generation.confirmation",
            "scope": {"level": "global"},
        }
        self.add_snapshot(
            [
                {
                    **common,
                    "quote": first,
                    "candidate_key": "first",
                    "ordinal": 1,
                    "start_offset": 0,
                    "end_offset": len(first),
                },
                {
                    **common,
                    "quote": second,
                    "candidate_key": "second",
                    "ordinal": 2,
                    "start_offset": len(first) + 1,
                    "end_offset": len(source),
                },
            ]
        )
        report = self.run_pipeline(DeterministicWriter(), AcceptReviewer())
        self.assertEqual(report["counts"]["actions"], {"created": 1, "revised": 1})
        self.assertEqual(self.table_counts(), (1, 2, 2, 0))
        with closing(sqlite3.connect(self.database)) as connection:
            revisions = connection.execute(
                "SELECT semantic_value,status FROM memory_revision ORDER BY revision_number"
            ).fetchall()
            reason = connection.execute(
                "SELECT reason FROM consolidation_action WHERE action='revised'"
            ).fetchone()[0]
        self.assertEqual(revisions, [("required", "superseded"), ("not_required", "current")])
        self.assertEqual(reason, "latest_durable_user_contract_by_source_order")

    def test_admission_evidence_change_during_review_fails_batch_cas(self) -> None:
        self.add_snapshot([self.confirmation()])

        def mutate_source(_call: int) -> None:
            with closing(open_store(self.database)) as connection, connection:
                row = connection.execute("SELECT * FROM admission_decision").fetchone()
                quote = row["exact_quote"] + " "
                connection.execute(
                    "UPDATE admission_decision SET exact_quote=?, end_offset=? WHERE decision_id=?",
                    (quote, len(quote), row["decision_id"]),
                )
                connection.execute(
                    "UPDATE event_annotation SET semantic_content=? WHERE event_id=?",
                    (quote, row["event_id"]),
                )

        report = self.run_pipeline(
            DeterministicWriter(), AcceptReviewer(callback=mutate_source)
        )
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["formal_state_unchanged"])
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))

    def test_unverified_engineering_candidate_is_blocked_before_both_models(self) -> None:
        self.add_snapshot(
            [
                {
                    "quote": "以后在所有单片机项目中，修改代码后都必须进行编译检查。",
                    "stable_key": "workflow.embedded.code_change.compile_check",
                    "scope": {"level": "domain", "domain": "embedded"},
                    "family": "engineering_memory",
                }
            ]
        )
        writer = DeterministicWriter()
        reviewer = AcceptReviewer()
        report = self.run_pipeline(writer, reviewer)
        self.assertEqual(report["status"], "complete_with_rejections")
        self.assertEqual((writer.calls, reviewer.calls), (0, 0))
        self.assertEqual(self.table_counts(), (0, 0, 0, 0))
        self.assertEqual(report["audit"][0]["writer"], "local_block")

    def test_rollback_appends_a_new_revision_and_obeys_state_version_cas(self) -> None:
        self.add_snapshot([self.confirmation(candidate_key="required")])
        self.run_pipeline(DeterministicWriter(), AcceptReviewer())
        self.add_snapshot(
            [
                self.confirmation(
                    "not_required",
                    candidate_key="not-required",
                    event_timestamp=200,
                )
            ]
        )
        self.run_pipeline(DeterministicWriter(), AcceptReviewer())
        with closing(sqlite3.connect(self.database)) as connection:
            memory_id, state_version = connection.execute(
                "SELECT memory_id,state_version FROM memory"
            ).fetchone()
            target_revision = connection.execute(
                "SELECT revision_id FROM memory_revision WHERE revision_number=1"
            ).fetchone()[0]
        with self.assertRaisesRegex(ValueError, "state_version"):
            rollback_memory(
                self.database,
                memory_id,
                target_revision,
                "stale request",
                expected_state_version=state_version - 1,
            )
        result = rollback_memory(
            self.database,
            memory_id,
            target_revision,
            "restore confirmed earlier contract",
            expected_state_version=state_version,
        )
        self.assertEqual(result["revision_number"], 3)
        self.assertEqual(result["semantic_value"], "required")
        with closing(sqlite3.connect(self.database)) as connection:
            current = connection.execute(
                """
                SELECT r.semantic_value,r.revision_number,m.state_version
                FROM memory AS m JOIN memory_revision AS r
                  ON r.revision_id=m.current_revision_id
                """
            ).fetchone()
            audit = connection.execute("SELECT COUNT(*) FROM memory_rollback").fetchone()[0]
        self.assertEqual(current, ("required", 3, state_version + 1))
        self.assertEqual(audit, 1)


if __name__ == "__main__":
    unittest.main()
