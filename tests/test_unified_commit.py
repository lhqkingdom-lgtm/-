from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from memory_lab.consolidation import rollback_memory
from memory_lab.shadow import formal_state_snapshot
from memory_lab.store import open_store
from memory_lab.unified_commit import (
    UnifiedCommitCASMismatch,
    UnifiedCommitValidationError,
    commit_prepared_unified_batch,
    commit_reviewed_unified_plan,
    prepare_unified_commit,
)
from memory_lab.unified_memory import (
    CANDIDATE_SCHEMA_VERSION,
    MemoryEvidence,
    validate_memory_candidate,
)


class UnifiedCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "memory_lab.db"
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO source(source_id, source_kind, source_path, created_at)
                VALUES ('source_1', 'sqlite', 'redacted-source.db', '2026-09-10T00:00:00Z')
                """
            )
            connection.execute(
                """
                INSERT INTO turn_detection_run(
                    run_id, source_id, detector_version, started_at, completed_at,
                    settle_ms, status
                ) VALUES ('turn_run', 'source_1', 'test', '2026-09-10T00:00:00Z',
                          '2026-09-10T00:00:01Z', 0, 'complete')
                """
            )
            for project in ("a", "b"):
                connection.execute(
                    """
                    INSERT INTO project_context(
                        project_key, directory, display_name, identity_source, created_at
                    ) VALUES (?, ?, ?, 'test', '2026-09-10T00:00:00Z')
                    """,
                    (
                        f"directory:project_{project}",
                        f"<WORKSPACE_{project.upper()}>",
                        f"project-{project}",
                    ),
                )
            self._add_turn(connection, "one", 100, "a")
            self._add_turn(connection, "two", 200, "a")
            self._add_turn(connection, "three", 300, "a")
            self._add_turn(connection, "four", 400, "b")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _add_turn(
        connection: sqlite3.Connection, name: str, timestamp: int, project: str
    ) -> None:
        session_id = f"session_{name}"
        event_id = f"event_{name}"
        message_id = f"message_{name}"
        part_id = f"part_{name}"
        project_key = f"directory:project_{project}"
        connection.execute(
            """
            INSERT INTO session_context(
                source_id, session_id, project_key, directory, context_source,
                first_seen_at, last_seen_at
            ) VALUES ('source_1', ?, ?, ?, 'test',
                      '2026-09-10T00:00:00Z', '2026-09-10T00:00:01Z')
            """,
            (session_id, project_key, f"<WORKSPACE_{project.upper()}>")
        )
        connection.execute(
            """
            INSERT INTO event(
                event_id, source_id, session_id, message_id, part_id, event_type,
                directory, timestamp, content, content_hash, source_kind,
                eligibility_status, ingested_at, raw_json
            ) VALUES (?, 'source_1', ?, ?, ?, 'user_text', ?, ?, ?, ?, 'sqlite',
                      'eligible', '2026-09-10T00:00:00Z', '{}')
            """,
            (
                event_id,
                session_id,
                message_id,
                part_id,
                f"<WORKSPACE_{project.upper()}>",
                timestamp,
                f"source text {name}",
                f"hash_{name}",
            ),
        )
        connection.execute(
            """
            INSERT INTO conversation_turn(
                turn_id, source_id, session_id, user_message_id, primary_event_id,
                user_timestamp, current_state, first_detected_at, last_detected_at,
                last_detection_run_id, last_structural_signature, assistant_count,
                part_count
            ) VALUES (?, 'source_1', ?, ?, ?, ?, 'completed',
                      '2026-09-10T00:00:00Z', '2026-09-10T00:00:01Z',
                      'turn_run', ?, 1, 2)
            """,
            (
                f"turn_{name}",
                session_id,
                message_id,
                event_id,
                timestamp,
                (name[0] * 64),
            ),
        )

    @staticmethod
    def evidence(
        name: str,
        timestamp: int,
        project: str = "a",
        *,
        source_kind: str = "user_message",
        explicit: bool = True,
        verified: bool = False,
    ) -> tuple[MemoryEvidence, dict]:
        evidence = MemoryEvidence(
            evidence_id=f"memory_evidence_{name}",
            turn_id=f"turn_{name}",
            source_kind=source_kind,
            occurred_at_ms=timestamp,
            part_ordinal=1,
            text_offset=0,
            project_key=f"directory:project_{project}",
            session_id=f"session_{name}",
            explicit=explicit,
            verified=verified,
        )
        quote = f"durable evidence {name}"
        span = {
            "source_evidence_ref": f"source_ref_{name}",
            "quote": quote,
            "start_offset": 0,
            "end_offset": len(quote),
            "turn_id": evidence.turn_id,
            "source_kind": source_kind,
            "occurred_at_ms": timestamp,
            "project_key": evidence.project_key,
            "session_id": evidence.session_id,
            "source_message_id": f"source_message_{name}",
            "source_part_id": f"source_part_{name}",
            "part_ordinal": 1,
        }
        return evidence, span

    @staticmethod
    def proposal(
        evidence: MemoryEvidence,
        *,
        stable_key: str = "preference.response.detail_level",
        memory_type: str = "preference",
        statement: str = "回复时保留必要的实现细节。",
        summary: str = "回复保留实现细节",
        scope: dict | None = None,
        conditions: dict | None = None,
        trust_status: str = "declared",
    ) -> dict:
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "stable_key": stable_key,
            "type": memory_type,
            "summary": summary,
            "scope": scope or {"level": "global"},
            "conditions": conditions or {},
            "trust_status": trust_status,
            "content": {"statement": statement},
            "evidence_refs": [evidence.evidence_id],
            "status": "active",
        }

    def review_result(
        self,
        specs: list[tuple[dict, MemoryEvidence, dict]],
        *,
        identity: str,
        outcome: str = "accept",
    ) -> dict:
        catalog = [evidence for _, evidence, _ in specs]
        candidates = []
        spans = {}
        for proposal, evidence, span in specs:
            candidate = validate_memory_candidate(proposal, catalog)
            candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "memory": candidate,
                    "evidence_quotes": [
                        {
                            "evidence_id": evidence.evidence_id,
                            "evidence_ref": span["source_evidence_ref"],
                            "quote": span["quote"],
                            "start_offset": span["start_offset"],
                            "end_offset": span["end_offset"],
                        }
                    ],
                    "action_hint": {
                        "kind": "create_or_match",
                        "related_candidate_id": None,
                        "related_existing_memory": False,
                        "reason": "advisory only",
                        "advisory_only": True,
                    },
                }
            )
            spans[evidence.evidence_id] = span
        plan_id = f"unified_plan_{identity}"
        plan = {
            "schema_version": "memory-lab-unified-plan-v1",
            "plan_id": plan_id,
            "request_id": f"unified_request_{identity}",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "evidence_catalog": [asdict(item) for item in catalog],
            "evidence_spans": spans,
            "commit_allowed": False,
        }
        return {
            "schema_version": "memory-lab-unified-reviewed-plan-v1",
            "review_id": f"unified_review_{identity}",
            "writer_plan_id": plan_id,
            "outcome": outcome,
            "reason": "independent semantic review passed",
            "reviewed_plan": plan if outcome != "reject" else None,
            "commit_allowed": outcome != "reject",
            "review_rounds": 1,
        }

    def multi_evidence_review_result(
        self,
        proposal: dict,
        evidence_and_spans: list[tuple[MemoryEvidence, dict]],
        *,
        identity: str,
    ) -> dict:
        catalog = [evidence for evidence, _ in evidence_and_spans]
        candidate = validate_memory_candidate(proposal, catalog)
        plan_id = f"unified_plan_{identity}"
        plan = {
            "schema_version": "memory-lab-unified-plan-v1",
            "plan_id": plan_id,
            "request_id": f"unified_request_{identity}",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "memory": candidate,
                    "evidence_quotes": [
                        {
                            "evidence_id": evidence.evidence_id,
                            "evidence_ref": span["source_evidence_ref"],
                            "quote": span["quote"],
                            "start_offset": span["start_offset"],
                            "end_offset": span["end_offset"],
                        }
                        for evidence, span in evidence_and_spans
                    ],
                    "action_hint": {
                        "kind": "create_or_match",
                        "related_candidate_id": None,
                        "related_existing_memory": False,
                        "reason": "advisory only",
                        "advisory_only": True,
                    },
                }
            ],
            "evidence_catalog": [asdict(item) for item in catalog],
            "evidence_spans": {
                evidence.evidence_id: span for evidence, span in evidence_and_spans
            },
            "commit_allowed": False,
        }
        return {
            "schema_version": "memory-lab-unified-reviewed-plan-v1",
            "review_id": f"unified_review_{identity}",
            "writer_plan_id": plan_id,
            "outcome": "accept",
            "reason": "independent semantic review passed",
            "reviewed_plan": plan,
            "commit_allowed": True,
            "review_rounds": 1,
        }

    def counts(self) -> dict[str, int]:
        with closing(open_store(self.database)) as connection:
            return {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "memory",
                    "memory_revision",
                    "evidence",
                    "conflict",
                    "memory_proposal",
                    "candidate_review",
                )
            }

    def test_dynamic_create_uses_reviewed_statement_and_source_trace(self) -> None:
        evidence, span = self.evidence("one", 100)
        result = commit_reviewed_unified_plan(
            self.database,
            self.review_result([(self.proposal(evidence), evidence, span)], identity="create"),
        )
        self.assertEqual(result["counts"], {"created": 1})
        with closing(open_store(self.database)) as connection:
            memory = connection.execute("SELECT * FROM memory").fetchone()
            revision = connection.execute("SELECT * FROM memory_revision").fetchone()
            stored_evidence = connection.execute("SELECT * FROM evidence").fetchone()
            payload = connection.execute(
                "SELECT application_payload_json FROM memory_proposal"
            ).fetchone()[0]
        self.assertEqual(memory["stable_key"], "preference.response.detail_level")
        self.assertEqual(revision["statement"], "回复时保留必要的实现细节。")
        self.assertNotIn("memsem_", revision["statement"])
        self.assertEqual(stored_evidence["event_id"], "event_one")
        self.assertEqual(stored_evidence["source_id"], "source_1")
        self.assertEqual(stored_evidence["message_id"], "source_message_one")
        self.assertEqual(stored_evidence["part_id"], "source_part_one")
        self.assertIn('"statement":"回复时保留必要的实现细节。"', payload)

    def test_all_reviewed_evidence_spans_are_formal_and_replay_is_idempotent(self) -> None:
        first, first_span = self.evidence("one", 100)
        second, second_span = self.evidence("two", 200)
        proposal = self.proposal(first)
        proposal["evidence_refs"] = [first.evidence_id, second.evidence_id]
        reviewed = self.multi_evidence_review_result(
            proposal,
            [(first, first_span), (second, second_span)],
            identity="multi-evidence",
        )

        committed = commit_reviewed_unified_plan(self.database, reviewed)
        with closing(open_store(self.database)) as connection:
            rows = connection.execute(
                "SELECT * FROM evidence ORDER BY occurred_at, candidate_key"
            ).fetchall()
        self.assertEqual(committed["counts"], {"created": 1})
        self.assertEqual(committed["actions"][0]["details"]["secondary_evidence_persisted"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [(row["event_id"], row["message_id"], row["part_id"], row["exact_quote"]) for row in rows],
            [
                ("event_one", "source_message_one", "source_part_one", "durable evidence one"),
                ("event_two", "source_message_two", "source_part_two", "durable evidence two"),
            ],
        )
        self.assertEqual(len({row["candidate_key"] for row in rows}), 2)
        self.assertEqual(len({row["decision_id"] for row in rows}), 1)
        self.assertEqual(len({row["proposal_id"] for row in rows}), 1)
        self.assertEqual(len({row["review_id"] for row in rows}), 1)
        self.assertEqual({row["evidence_role"] for row in rows}, {"supporting"})

        replay = commit_reviewed_unified_plan(self.database, reviewed)
        self.assertEqual(replay["status"], "replayed")
        with closing(open_store(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], 2)

        semantic_replay = self.multi_evidence_review_result(
            proposal,
            [(first, first_span), (second, second_span)],
            identity="multi-evidence-new-review",
        )
        noop = commit_reviewed_unified_plan(self.database, semantic_replay)
        self.assertEqual(noop["counts"], {"noop": 1})
        with closing(open_store(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], 2)

    def test_equivalent_candidate_merges_evidence_idempotently(self) -> None:
        first, first_span = self.evidence("one", 100)
        second, second_span = self.evidence("two", 200)
        commit_reviewed_unified_plan(
            self.database,
            self.review_result([(self.proposal(first), first, first_span)], identity="merge1"),
        )
        result = commit_reviewed_unified_plan(
            self.database,
            self.review_result([(self.proposal(second), second, second_span)], identity="merge2"),
        )
        self.assertEqual(result["counts"], {"merged_evidence": 1})
        counts = self.counts()
        self.assertEqual(counts["memory"], 1)
        self.assertEqual(counts["memory_revision"], 1)
        self.assertEqual(counts["evidence"], 2)

        replay = commit_reviewed_unified_plan(
            self.database,
            self.review_result([(self.proposal(second), second, second_span)], identity="merge2r"),
        )
        self.assertEqual(replay["counts"], {"noop": 1})
        self.assertEqual(self.counts()["evidence"], 2)

    def test_project_scope_and_conditions_create_a_scope_variant(self) -> None:
        first, first_span = self.evidence("one", 100)
        project, project_span = self.evidence("four", 400, "b")
        commit_reviewed_unified_plan(
            self.database,
            self.review_result([(self.proposal(first), first, first_span)], identity="scope1"),
        )
        project_proposal = self.proposal(
            project,
            scope={"level": "project", "project_key": "directory:project_b"},
            conditions={"runtime": "v9"},
        )
        result = commit_reviewed_unified_plan(
            self.database,
            self.review_result([(project_proposal, project, project_span)], identity="scope2"),
        )
        self.assertEqual(result["counts"], {"created_scope_variant": 1})
        with closing(open_store(self.database)) as connection:
            scopes = [
                row[0]
                for row in connection.execute("SELECT scope_json FROM memory ORDER BY scope_json")
            ]
        self.assertEqual(len(scopes), 2)
        self.assertTrue(any('"conditions":{"runtime":"v9"}' in item for item in scopes))

    def test_newer_explicit_user_contract_revises_and_rollback_still_works(self) -> None:
        first, first_span = self.evidence("one", 100)
        later, later_span = self.evidence("three", 300)
        commit_reviewed_unified_plan(
            self.database,
            self.review_result([(self.proposal(first), first, first_span)], identity="revise1"),
        )
        changed = self.proposal(
            later,
            statement="回复时默认只保留简短结论。",
            summary="回复默认简短",
        )
        result = commit_reviewed_unified_plan(
            self.database,
            self.review_result([(changed, later, later_span)], identity="revise2"),
        )
        self.assertEqual(result["counts"], {"revised": 1})
        with closing(open_store(self.database)) as connection:
            memory = connection.execute("SELECT * FROM memory").fetchone()
            revisions = connection.execute(
                "SELECT * FROM memory_revision ORDER BY revision_number"
            ).fetchall()
        self.assertEqual([row["status"] for row in revisions], ["superseded", "current"])
        self.assertEqual(revisions[1]["statement"], "回复时默认只保留简短结论。")
        rollback = rollback_memory(
            self.database,
            memory["memory_id"],
            revisions[0]["revision_id"],
            "test rollback",
            expected_state_version=memory["state_version"],
        )
        self.assertEqual(rollback["memory_id"], memory["memory_id"])
        self.assertEqual(rollback["target_revision_id"], revisions[0]["revision_id"])
        self.assertEqual(rollback["state_version_after"], memory["state_version"] + 1)
        with closing(open_store(self.database)) as connection:
            current = connection.execute(
                """
                SELECT r.statement FROM memory AS m
                JOIN memory_revision AS r ON r.revision_id = m.current_revision_id
                """
            ).fetchone()[0]
        self.assertEqual(current, "回复时保留必要的实现细节。")

    def test_same_reviewed_batch_replays_original_result_without_writes(self) -> None:
        evidence, span = self.evidence("one", 100)
        reviewed = self.review_result(
            [(self.proposal(evidence), evidence, span)], identity="crash-retry"
        )
        first = commit_reviewed_unified_plan(self.database, reviewed)
        counts_before = self.counts()
        with closing(open_store(self.database)) as connection:
            audit_before = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("admission_run", "admission_decision", "consolidation_run", "consolidation_action")
            }

        replay = commit_reviewed_unified_plan(self.database, reviewed)

        self.assertEqual(first["status"], "complete")
        self.assertEqual(replay["status"], "replayed")
        self.assertEqual(replay["reason"], "reviewed_batch_already_committed")
        self.assertEqual(replay["counts"], first["counts"])
        self.assertEqual(replay["actions"], first["actions"])
        self.assertEqual(replay["formal_state_before"], first["formal_state_before"])
        self.assertEqual(replay["formal_state_after"], first["formal_state_after"])
        self.assertFalse(replay["formal_state_changed_since_commit"])
        self.assertEqual(self.counts(), counts_before)
        with closing(open_store(self.database)) as connection:
            audit_after = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("admission_run", "admission_decision", "consolidation_run", "consolidation_action")
            }
        self.assertEqual(audit_after, audit_before)

    def test_engineering_semantic_change_opens_conflict_not_latest_wins(self) -> None:
        first, first_span = self.evidence(
            "one", 100, source_kind="assistant_final", explicit=False
        )
        later, later_span = self.evidence(
            "three", 300, source_kind="assistant_final", explicit=False
        )
        base = self.proposal(
            first,
            stable_key="engineering.cache.rebuild",
            memory_type="engineering",
            statement="重建缓存后构建成功。",
            summary="重建缓存可恢复构建",
            trust_status="unverified",
        )
        opposite = self.proposal(
            later,
            stable_key="engineering.cache.rebuild",
            memory_type="engineering",
            statement="重建缓存后构建仍然失败。",
            summary="重建缓存无法恢复构建",
            trust_status="unverified",
        )
        commit_reviewed_unified_plan(
            self.database,
            self.review_result([(base, first, first_span)], identity="conflict1"),
        )
        result = commit_reviewed_unified_plan(
            self.database,
            self.review_result([(opposite, later, later_span)], identity="conflict2"),
        )
        self.assertEqual(result["counts"], {"opened_conflict": 1})
        counts = self.counts()
        self.assertEqual(counts["memory_revision"], 1)
        self.assertEqual(counts["conflict"], 1)
        with closing(open_store(self.database)) as connection:
            conflict = connection.execute("SELECT * FROM conflict").fetchone()
        self.assertEqual(conflict["incoming_statement"], "重建缓存后构建仍然失败。")

    def test_unsupported_scope_and_reviewer_rejection_write_nothing(self) -> None:
        evidence, span = self.evidence("one", 100)
        session_proposal = self.proposal(
            evidence, scope={"level": "session", "session_id": "session_one"}
        )
        before = self.counts()
        with self.assertRaises(UnifiedCommitValidationError):
            commit_reviewed_unified_plan(
                self.database,
                self.review_result(
                    [(session_proposal, evidence, span)], identity="unsupported"
                ),
            )
        self.assertEqual(self.counts(), before)

        rejected = self.review_result([], identity="rejected", outcome="reject")
        result = commit_reviewed_unified_plan(self.database, rejected)
        self.assertEqual(result["status"], "reviewer_rejected")
        self.assertEqual(self.counts(), before)

    def test_formal_cas_change_rejects_whole_prepared_batch(self) -> None:
        first, first_span = self.evidence("one", 100)
        prepared = prepare_unified_commit(
            self.database,
            self.review_result([(self.proposal(first), first, first_span)], identity="stale"),
        )
        other, other_span = self.evidence("two", 200)
        other_proposal = self.proposal(
            other,
            stable_key="project.protocol.choice",
            memory_type="project",
            statement="该项目使用稳定的通信协议。",
            summary="项目通信协议",
            trust_status="unverified",
        )
        commit_reviewed_unified_plan(
            self.database,
            self.review_result([(other_proposal, other, other_span)], identity="other"),
        )
        before = self.counts()
        with self.assertRaises(UnifiedCommitCASMismatch):
            commit_prepared_unified_batch(self.database, prepared)
        self.assertEqual(self.counts(), before)


if __name__ == "__main__":
    unittest.main()
