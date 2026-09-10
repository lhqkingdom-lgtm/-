from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from contextlib import closing
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from memory_lab.admission import _suggest_identity
from memory_lab.cli import main as cli_main
from memory_lab.consolidation import (
    canonicalize_scope,
    record_candidate_review,
    record_proposal_review,
    run_consolidation,
    scope_fingerprint,
    semantic_value_for,
)
from memory_lab.store import open_store


class ConsolidationTests(unittest.TestCase):
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
                    run_id, source_id, detector_version, started_at, completed_at,
                    settle_ms, status
                ) VALUES (
                    'turn_run', 'cp', 'test', '2026-01-01T00:00:00.000+00:00',
                    '2026-01-01T00:00:01.000+00:00', 0, 'complete'
                )
                """
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_snapshot(
        self, items: list[dict], *, status: str = "complete"
    ) -> tuple[str, list[dict]]:
        """Insert one admission run whose decisions are the complete snapshot."""
        self.run_counter += 1
        run_id = f"admission_{self.run_counter:03d}"
        instant = f"2026-01-01T00:00:{self.run_counter:02d}.000+00:00"
        inserted: list[dict] = []
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, completed_at, status,
                    decision_count, counts_json
                ) VALUES (?, 'test', ?, ?, ?, ?, '{}')
                """,
                (
                    run_id,
                    instant,
                    instant if status in {"complete", "failed"} else None,
                    status,
                    len(items),
                ),
            )
            for item in items:
                self.item_counter += 1
                suffix = f"{self.item_counter:03d}"
                quote = item["quote"]
                evidence_source_text = item.get("evidence_source_text", quote)
                candidate_key = item.get("candidate_key", f"candidate_{suffix}")
                event_id = item.get("event_id", f"event_{suffix}")
                session_id = item.get("session_id", f"session_{suffix}")
                project_key = item.get("project_key", f"project_{suffix}")
                decision_id = f"decision_{self.run_counter:03d}_{suffix}"
                message_id = item.get("message_id", f"message_{suffix}")
                part_id = item.get("part_id", f"part_{suffix}")
                directory = item.get("directory", rf"D:\{project_key}")
                event_timestamp = item.get("event_timestamp", self.item_counter * 1000)
                ordinal = item.get("ordinal", 1)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO project_context(
                        project_key, directory, display_name, identity_source, created_at
                    ) VALUES (?, ?, ?, 'test', ?)
                    """,
                    (project_key, directory, project_key, instant),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_context(
                        source_id, session_id, project_key, directory, context_source,
                        first_seen_at, last_seen_at
                    ) VALUES ('cp', ?, ?, ?, 'test', ?, ?)
                    """,
                    (session_id, project_key, directory, instant, instant),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO event(
                        event_id, source_id, session_id, message_id, part_id,
                        event_type, directory, project_id, timestamp, content,
                        content_hash, source_kind, eligibility_status, ingested_at,
                        raw_json
                    ) VALUES (?, 'cp', ?, ?, ?, 'user_text', ?, ?, ?, ?, ?,
                              'sqlite', 'eligible', ?, '{}')
                    """,
                    (
                        event_id,
                        session_id,
                        message_id,
                        part_id,
                        directory,
                        item.get("source_project_id", f"source-{project_key}"),
                        event_timestamp,
                        evidence_source_text,
                        hashlib.sha256(evidence_source_text.encode("utf-8")).hexdigest(),
                        instant,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO event_annotation(
                        event_id, semantic_content, project_key,
                        annotation_source, created_at
                    ) VALUES (?, ?, ?, 'test', ?)
                    """,
                    (event_id, evidence_source_text, project_key, instant),
                )
                turn_id = "turn_" + hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:20]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO conversation_turn(
                        turn_id, source_id, session_id, user_message_id,
                        primary_event_id, user_timestamp, current_state,
                        first_detected_at, last_detected_at,
                        last_detection_run_id, last_structural_signature
                    ) VALUES (?, 'cp', ?, ?, ?, ?, 'completed', ?, ?, 'turn_run', ?)
                    """,
                    (
                        turn_id,
                        session_id,
                        message_id,
                        event_id,
                        event_timestamp,
                        instant,
                        instant,
                        "sig_" + event_id,
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO turn_event(turn_id, event_id) VALUES (?, ?)",
                    (turn_id, event_id),
                )
                signals = {
                    "long_term": ["以后"],
                    "temporary": [],
                    "correction": ["更正"] if item.get("correction") else [],
                }
                connection.execute(
                    """
                    INSERT INTO admission_decision(
                        decision_id, run_id, candidate_key, event_id, ordinal,
                        exact_quote, start_offset, end_offset, family, memory_type,
                        stable_key_suggestion, scope_json, activation_triggers_json,
                        disposition, signals_json, instruction_coverage_json,
                        reasons_json, decided_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '[]', ?)
                    """,
                    (
                        decision_id,
                        run_id,
                        candidate_key,
                        event_id,
                        ordinal,
                        quote,
                        item.get("start_offset", 0),
                        item.get("end_offset", len(quote)),
                        item.get("family", "user_contract"),
                        item.get("memory_type", "workflow_rule"),
                        item.get("stable_key"),
                        json.dumps(item.get("scope", {"level": "global"}), ensure_ascii=False),
                        json.dumps(item.get("triggers", ["trigger"]), ensure_ascii=False),
                        item.get("disposition", "eligible"),
                        json.dumps(signals, ensure_ascii=False),
                        instant,
                    ),
                )
                inserted.append(
                    {
                        "candidate_key": candidate_key,
                        "decision_id": decision_id,
                        "event_id": event_id,
                        "project_key": project_key,
                    }
                )
        return run_id, inserted

    def propose(self) -> tuple[dict, dict[str, str]]:
        report = run_consolidation(self.database)
        proposals = {
            action["candidate_key"]: action["proposal_id"]
            for action in report["actions"]
            if action["proposal_id"] is not None
        }
        return report, proposals

    def approve(self, *proposal_ids: str, reviewer: str = "test-owner") -> None:
        for proposal_id in proposal_ids:
            record_proposal_review(
                self.database, proposal_id, "approve", reviewer, "verified"
            )

    def activate(self, item: dict) -> tuple[dict, str]:
        self.add_snapshot([item])
        proposed, proposals = self.propose()
        self.assertEqual(proposed["counts"], {"proposed": 1})
        proposal_id = next(iter(proposals.values()))
        self.approve(proposal_id)
        return run_consolidation(self.database), proposal_id

    def table_counts(self) -> tuple[int, int, int, int]:
        with closing(sqlite3.connect(self.database)) as connection:
            return tuple(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("memory", "memory_revision", "evidence", "conflict")
            )

    @staticmethod
    def confirmation(value: str = "required", **overrides) -> dict:
        quote = (
            "以后生成正式代码前先向我确认。"
            if value == "required"
            else "以后生成正式代码不用先问我。"
        )
        return {
            "quote": quote,
            "stable_key": "workflow.code_generation.confirmation",
            "scope": {"level": "global"},
            **overrides,
        }

    def test_exact_proposal_review_and_full_audit_evidence(self) -> None:
        quote = "以后生成正式代码前先向我确认。"
        item = self.confirmation(
            quote=quote,
            session_id="session-a",
            project_key="project-a",
            source_project_id="raw-a",
            ordinal=3,
            start_offset=2,
            end_offset=2 + len(quote),
            evidence_source_text="xx" + quote,
            event_timestamp=12345,
        )
        report, proposal_id = self.activate(item)
        self.assertEqual(report["counts"], {"created": 1})
        self.assertEqual(self.table_counts(), (1, 1, 1, 0))
        with closing(sqlite3.connect(self.database)) as connection:
            connection.row_factory = sqlite3.Row
            revision = connection.execute("SELECT * FROM memory_revision").fetchone()
            evidence = connection.execute("SELECT * FROM evidence").fetchone()
            action = connection.execute(
                "SELECT * FROM consolidation_action WHERE action = 'created'"
            ).fetchone()
            memory = connection.execute("SELECT * FROM memory").fetchone()
        self.assertEqual(memory["family"], "user_contract")
        self.assertEqual(json.loads(memory["scope_json"]), {"level": "global"})
        self.assertEqual(revision["source_proposal_id"], proposal_id)
        self.assertIsNotNone(revision["review_id"])
        self.assertEqual(evidence["proposal_id"], proposal_id)
        self.assertEqual(evidence["reviewer"], "test-owner")
        self.assertEqual(
            (evidence["session_id"], evidence["project_key"], evidence["source_project_id"]),
            ("session-a", "project-a", "raw-a"),
        )
        self.assertEqual(
            (evidence["ordinal"], evidence["start_offset"], evidence["end_offset"], evidence["occurred_at"]),
            (3, 2, 2 + len(quote), 12345),
        )
        self.assertEqual(action["proposal_id"], proposal_id)
        self.assertEqual(action["reviewer"], "test-owner")

    def test_same_scope_same_value_merges_cross_project_evidence_idempotently(self) -> None:
        self.activate(self.confirmation(session_id="s-a", project_key="p-a"))
        self.add_snapshot(
            [
                self.confirmation(
                    quote="我再确认一次，生成正式代码之前需要先问我。",
                    session_id="s-b",
                    project_key="p-b",
                )
            ]
        )
        _, proposals = self.propose()
        self.approve(*proposals.values())
        merged = run_consolidation(self.database)
        self.assertEqual(merged["counts"], {"merged_evidence": 1})
        self.assertEqual(self.table_counts(), (1, 1, 2, 0))
        repeated = run_consolidation(self.database)
        self.assertEqual(repeated["counts"], {"noop": 1})
        self.assertEqual(self.table_counts(), (1, 1, 2, 0))

    def test_two_create_proposals_use_cas_and_second_requires_new_review(self) -> None:
        _, rows = self.add_snapshot(
            [
                self.confirmation(session_id="s-a", project_key="p-a", event_timestamp=10),
                self.confirmation(
                    quote="以后正式代码生成之前需要确认。",
                    session_id="s-b",
                    project_key="p-b",
                    event_timestamp=20,
                ),
            ]
        )
        first, proposals = self.propose()
        self.assertEqual(first["counts"], {"proposed": 2})
        self.approve(*proposals.values())
        applied = run_consolidation(self.database)
        self.assertEqual(applied["counts"], {"created": 1, "stale": 1})
        stale = next(action for action in applied["actions"] if action["action"] == "stale")
        replacement = stale["details"]["replacement_proposal_id"]
        self.assertIsNotNone(replacement)
        self.approve(replacement)
        final = run_consolidation(self.database)
        self.assertEqual(final["counts"], {"merged_evidence": 1, "noop": 1})
        self.assertEqual(self.table_counts(), (1, 1, 2, 0))
        self.assertEqual({row["candidate_key"] for row in rows}, set(proposals))

    def test_stale_approved_correction_cannot_overwrite_new_revision(self) -> None:
        self.activate(self.confirmation())
        correction = self.confirmation(
            "not_required",
            quote="更正：以后生成正式代码不用再问我。",
            disposition="needs_correction_review",
            correction=True,
        )
        self.add_snapshot([correction])
        _, proposals = self.propose()
        old_proposal = next(iter(proposals.values()))
        self.approve(old_proposal)

        with closing(open_store(self.database)) as connection, connection:
            memory = connection.execute("SELECT * FROM memory").fetchone()
            old_revision = memory["current_revision_id"]
            decision_id = connection.execute(
                "SELECT decision_id FROM memory_proposal WHERE proposal_id = ?", (old_proposal,)
            ).fetchone()[0]
            connection.execute(
                "UPDATE memory_revision SET status = 'superseded' WHERE revision_id = ?",
                (old_revision,),
            )
            connection.execute(
                """
                INSERT INTO memory_revision(
                    revision_id, memory_id, revision_number, semantic_value, statement,
                    activation_triggers_json, status, supersedes_revision_id,
                    source_decision_id, created_at
                ) VALUES ('external_revision', ?, 2, 'required', 'newer rule', '[]',
                          'current', ?, ?, '2026-01-02T00:00:00.000+00:00')
                """,
                (memory["memory_id"], old_revision, decision_id),
            )
            connection.execute(
                "UPDATE memory SET current_revision_id = 'external_revision' WHERE memory_id = ?",
                (memory["memory_id"],),
            )

        report = run_consolidation(self.database)
        action = report["actions"][0]
        self.assertEqual((action["action"], action["reason"]), ("stale", "proposal_baseline_changed"))
        self.assertEqual(action["details"]["stale_reason"], "current_revision_changed")
        with closing(sqlite3.connect(self.database)) as connection:
            current = connection.execute("SELECT current_revision_id FROM memory").fetchone()[0]
            values = connection.execute(
                "SELECT semantic_value FROM memory_revision ORDER BY revision_number"
            ).fetchall()
        self.assertEqual(current, "external_revision")
        self.assertEqual(values, [("required",), ("required",)])

    def test_open_conflict_set_change_stales_correction_approval(self) -> None:
        self.activate(self.confirmation())
        correction = self.confirmation(
            "not_required",
            quote="更正：以后生成正式代码不用再问我。",
            disposition="needs_correction_review",
            correction=True,
        )
        self.add_snapshot([correction])
        _, proposals = self.propose()
        proposal_id = next(iter(proposals.values()))
        self.approve(proposal_id)
        with closing(open_store(self.database)) as connection, connection:
            memory = connection.execute("SELECT * FROM memory").fetchone()
            decision = connection.execute(
                "SELECT decision_id FROM memory_proposal WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO conflict(
                    conflict_id, conflict_fingerprint, memory_id, current_revision_id,
                    incoming_candidate_key, incoming_decision_id,
                    incoming_semantic_value, incoming_statement, status, created_at
                ) VALUES ('external_conflict', 'external_fingerprint', ?, ?,
                          'external_candidate', ?, 'not_required', 'external',
                          'open', '2026-01-02T00:00:00.000+00:00')
                """,
                (memory["memory_id"], memory["current_revision_id"], decision),
            )
        report = run_consolidation(self.database)
        self.assertEqual(report["actions"][0]["action"], "stale")
        self.assertEqual(report["actions"][0]["details"]["stale_reason"], "open_conflict_set_changed")
        self.assertEqual(self.table_counts(), (1, 1, 1, 1))

    def test_any_prior_memory_mutation_stales_later_approved_proposal(self) -> None:
        self.activate(self.confirmation())
        self.add_snapshot(
            [
                self.confirmation(
                    quote="我再次确认，正式代码生成之前要先询问我。",
                    event_timestamp=10,
                ),
                self.confirmation(
                    "not_required",
                    quote="更正：以后生成正式代码不用再问我。",
                    disposition="needs_correction_review",
                    correction=True,
                    event_timestamp=20,
                ),
            ]
        )
        _, proposals = self.propose()
        self.approve(*proposals.values())
        report = run_consolidation(self.database)
        self.assertEqual(report["counts"], {"merged_evidence": 1, "stale": 1})
        stale = next(action for action in report["actions"] if action["action"] == "stale")
        self.assertEqual(stale["details"]["stale_reason"], "memory_state_version_changed")
        with closing(sqlite3.connect(self.database)) as connection:
            current_value = connection.execute(
                """
                SELECT r.semantic_value FROM memory AS m
                JOIN memory_revision AS r ON r.revision_id = m.current_revision_id
                """
            ).fetchone()[0]
        self.assertEqual(current_value, "required")

    def test_conflict_resolved_then_new_counterexample_opens_new_row(self) -> None:
        self.activate(self.confirmation())
        self.activate(self.confirmation("not_required"))
        self.assertEqual(self.table_counts(), (1, 1, 2, 1))

        self.activate(
            self.confirmation(
                quote="更正：最终仍需在正式代码生成前确认。",
                disposition="needs_correction_review",
                correction=True,
            )
        )
        self.activate(
            self.confirmation(
                "not_required", quote="后续生成正式代码无需确认。"
            )
        )
        self.assertEqual(self.table_counts(), (1, 1, 4, 2))
        with closing(sqlite3.connect(self.database)) as connection:
            statuses = dict(
                connection.execute("SELECT status, COUNT(*) FROM conflict GROUP BY status")
            )
            resolution_audit = connection.execute(
                """
                SELECT resolution_proposal_id, resolution_review_id,
                       resolution_reviewer
                FROM conflict WHERE status = 'resolved'
                """
            ).fetchone()
        self.assertEqual(statuses, {"open": 1, "resolved": 1})
        self.assertTrue(all(resolution_audit))

    def test_explicit_correction_revises_and_same_value_correction_does_not(self) -> None:
        self.activate(self.confirmation())
        revised, _ = self.activate(
            self.confirmation(
                "not_required",
                quote="更正：从现在起生成正式代码不用再问我。",
                disposition="needs_correction_review",
                correction=True,
            )
        )
        self.assertEqual(revised["counts"], {"revised": 1})
        confirmed, _ = self.activate(
            self.confirmation(
                "not_required",
                quote="更正：最终仍然不需要在生成正式代码前问我。",
                disposition="needs_correction_review",
                correction=True,
            )
        )
        self.assertEqual(confirmed["counts"], {"confirmed_revision": 1})
        self.assertEqual(self.table_counts(), (1, 2, 3, 0))
        with closing(sqlite3.connect(self.database)) as connection:
            revisions = connection.execute(
                "SELECT revision_number, semantic_value, status FROM memory_revision ORDER BY revision_number"
            ).fetchall()
        self.assertEqual(
            revisions,
            [(1, "required", "superseded"), (2, "not_required", "current")],
        )

    def test_different_scope_is_audited_as_scope_variant(self) -> None:
        self.activate(self.confirmation())
        report, _ = self.activate(
            self.confirmation(
                "not_required",
                quote="当前项目以后生成正式代码不用确认。",
                project_key="special-project",
                scope={
                    "level": "project",
                    "project_key": " special-project ",
                    "directory": r"D:\special-project",
                    "version": " V2 ",
                },
            )
        )
        self.assertEqual(report["counts"], {"created_scope_variant": 1})
        self.assertEqual(report["state"]["memories"], 2)
        self.assertEqual(report["state"]["open_conflicts"], 0)

    def test_latest_complete_snapshot_is_exclusive_and_empty_snapshot_clears_work(self) -> None:
        run_a, rows_a = self.add_snapshot([self.confirmation(candidate_key="candidate-a")])
        _, proposals_a = self.propose()
        self.approve(*proposals_a.values())
        run_b, rows_b = self.add_snapshot(
            [
                {
                    "quote": "以后默认使用中文回复。",
                    "stable_key": "preference.response.language",
                    "scope": {"level": "global"},
                    "candidate_key": "candidate-b",
                }
            ]
        )
        report_b = run_consolidation(self.database)
        self.assertEqual(report_b["admission_run_id"], run_b)
        self.assertEqual([a["candidate_key"] for a in report_b["actions"]], ["candidate-b"])
        self.assertEqual(report_b["state"]["memories"], 0)
        self.assertNotEqual(run_a, run_b)
        self.assertEqual(rows_a[0]["candidate_key"], "candidate-a")
        self.assertEqual(rows_b[0]["candidate_key"], "candidate-b")

        empty_run, _ = self.add_snapshot([])
        empty = run_consolidation(self.database)
        self.assertEqual(empty["admission_run_id"], empty_run)
        self.assertEqual(empty["candidate_count"], 0)
        self.assertEqual(empty["actions"], [])
        self.assertEqual(empty["state"]["memories"], 0)

    def test_newer_running_and_failed_admission_runs_do_not_replace_complete_snapshot(self) -> None:
        complete_run, _ = self.add_snapshot([self.confirmation(candidate_key="kept")])
        self.add_snapshot(
            [self.confirmation(candidate_key="running")], status="running"
        )
        self.add_snapshot(
            [self.confirmation(candidate_key="failed")], status="failed"
        )
        report = run_consolidation(self.database)
        self.assertEqual(report["admission_run_id"], complete_run)
        self.assertEqual([a["candidate_key"] for a in report["actions"]], ["kept"])
        with closing(sqlite3.connect(self.database)) as connection:
            recorded = connection.execute(
                "SELECT admission_run_id, admission_snapshot_hash FROM consolidation_run WHERE run_id = ?",
                (report["run_id"],),
            ).fetchone()
        self.assertEqual(recorded[0], complete_run)
        self.assertTrue(recorded[1].startswith("snapshot_"))

    def test_apply_order_is_event_timestamp_event_id_then_ordinal(self) -> None:
        items = [
            self.confirmation(candidate_key="hash-first", event_id="z-event", event_timestamp=200),
            self.confirmation(candidate_key="hash-second", event_id="b-event", event_timestamp=100, ordinal=2),
            self.confirmation(candidate_key="hash-third", event_id="b-event", event_timestamp=100, ordinal=1),
            self.confirmation(candidate_key="hash-fourth", event_id="a-event", event_timestamp=100, ordinal=9),
        ]
        self.add_snapshot(items)
        report = run_consolidation(self.database)
        self.assertEqual(
            [action["candidate_key"] for action in report["actions"]],
            ["hash-fourth", "hash-third", "hash-second", "hash-first"],
        )

    def test_pending_proposal_uses_frozen_semantics_when_parser_changes(self) -> None:
        self.add_snapshot(
            [
                {
                    "quote": "以后统一使用英文回复。",
                    "stable_key": "preference.response.language",
                    "scope": {"level": "global"},
                }
            ]
        )
        _, proposals = self.propose()
        proposal_id = next(iter(proposals.values()))
        self.approve(proposal_id)
        with patch("memory_lab.consolidation.semantic_value_for", return_value="zh"):
            applied = run_consolidation(self.database)
        self.assertEqual(applied["counts"], {"created": 1})
        with closing(sqlite3.connect(self.database)) as connection:
            value = connection.execute("SELECT semantic_value FROM memory_revision").fetchone()[0]
        self.assertEqual(value, "en")

    def test_source_mutation_after_review_stales_exact_proposal(self) -> None:
        self.add_snapshot([self.confirmation()])
        _, proposals = self.propose()
        proposal_id = next(iter(proposals.values()))
        self.approve(proposal_id)
        replacement_quote = "以后生成正式代码之前必须先确认。"
        with closing(open_store(self.database)) as connection, connection:
            decision = connection.execute(
                """
                SELECT p.decision_id, d.event_id
                FROM memory_proposal AS p
                JOIN admission_decision AS d ON d.decision_id = p.decision_id
                WHERE p.proposal_id = ?
                """,
                (proposal_id,),
            ).fetchone()
            connection.execute(
                """
                UPDATE admission_decision
                SET exact_quote = ?, start_offset = 0, end_offset = ?,
                    family = 'tampered-family', activation_triggers_json = '["tampered"]'
                WHERE decision_id = ?
                """,
                (replacement_quote, len(replacement_quote), decision["decision_id"]),
            )
            connection.execute(
                "UPDATE event_annotation SET semantic_content = ? WHERE event_id = ?",
                (replacement_quote, decision["event_id"]),
            )
        with self.assertRaisesRegex(ValueError, "admission snapshot is stale"):
            record_proposal_review(self.database, proposal_id, "approve", "other-owner")
        report = run_consolidation(self.database)
        self.assertEqual(report["counts"], {"stale": 1})
        self.assertEqual(
            report["actions"][0]["reason"], "proposal_admission_snapshot_changed"
        )
        self.assertIsNotNone(report["actions"][0]["details"]["replacement_proposal_id"])
        self.assertEqual(report["state"]["memories"], 0)

    def test_same_candidate_in_new_snapshot_is_immediate_noop_without_new_proposal(self) -> None:
        candidate = "durable-candidate"
        event = "durable-event"
        session = "durable-session"
        project = "durable-project"
        self.activate(
            self.confirmation(
                candidate_key=candidate,
                event_id=event,
                session_id=session,
                project_key=project,
            )
        )
        with closing(sqlite3.connect(self.database)) as connection:
            before = connection.execute("SELECT COUNT(*) FROM memory_proposal").fetchone()[0]
        self.add_snapshot(
            [
                self.confirmation(
                    candidate_key=candidate,
                    event_id=event,
                    session_id=session,
                    project_key=project,
                )
            ]
        )
        report = run_consolidation(self.database)
        self.assertEqual(report["counts"], {"noop": 1})
        with closing(sqlite3.connect(self.database)) as connection:
            after = connection.execute("SELECT COUNT(*) FROM memory_proposal").fetchone()[0]
        self.assertEqual(after, before)
        self.assertEqual(self.table_counts(), (1, 1, 1, 0))

    def test_proposals_are_database_immutable_and_candidate_key_review_is_disabled(self) -> None:
        self.add_snapshot([self.confirmation()])
        _, proposals = self.propose()
        proposal_id = next(iter(proposals.values()))
        with closing(open_store(self.database)) as connection, self.assertRaises(sqlite3.IntegrityError):
            with connection:
                connection.execute(
                    "UPDATE memory_proposal SET semantic_value = 'not_required' WHERE proposal_id = ?",
                    (proposal_id,),
                )
        with closing(open_store(self.database)) as connection, self.assertRaises(sqlite3.IntegrityError):
            with connection:
                connection.execute(
                    "DELETE FROM memory_proposal WHERE proposal_id = ?", (proposal_id,)
                )
        with self.assertRaisesRegex(ValueError, "candidate-key review is disabled"):
            record_candidate_review(self.database, "candidate", "approve", "owner")

        review = record_proposal_review(
            self.database, proposal_id, "approve", "append-only-owner"
        )
        with closing(open_store(self.database)) as connection, self.assertRaises(sqlite3.IntegrityError):
            with connection:
                connection.execute(
                    "UPDATE candidate_review SET outcome = 'reject' WHERE review_id = ?",
                    (review["review_id"],),
                )

    def test_nonactivatable_dispositions_never_get_proposals(self) -> None:
        inactive = [
            "covered_by_cp_instructions",
            "temporary",
            "not_memory",
            "needs_scope_review",
            "needs_semantic_review",
            "needs_conflict_review",
        ]
        self.add_snapshot(
            [
                {
                    "quote": "以后默认使用中文回复。",
                    "stable_key": "preference.response.language",
                    "scope": {"level": "global"},
                    "disposition": disposition,
                }
                for disposition in inactive
            ]
        )
        report = run_consolidation(self.database)
        self.assertEqual(report["counts"], {"skipped": len(inactive)})
        self.assertEqual(report["state"]["proposals"], 0)
        self.assertEqual(report["state"]["memories"], 0)

    def test_scope_canonicalization_conditions_and_fail_closed_fields(self) -> None:
        self.assertEqual(
            canonicalize_scope(
                {
                    "level": " DOMAIN ",
                    "domain": " 嵌入式系统 ",
                    "directory": r"D:\ignored",
                    "override_policy": "specific",
                    "conditions": {"Toolchain": " Keil C51 ", "chip": " TG032PLUS "},
                    "version": " Ｖ２ ",
                }
            ),
            {
                "level": "domain",
                "domain": "embedded",
                "conditions": {
                    "chip": "tg032plus",
                    "toolchain": "keil c51",
                    "version": "v2",
                },
            },
        )
        base = scope_fingerprint({"level": "global", "version": "v1"})
        self.assertNotEqual(base, scope_fingerprint({"level": "global", "version": "v2"}))
        self.assertNotEqual(base, scope_fingerprint({"level": "global", "chip": "tg032"}))
        self.assertIsNone(canonicalize_scope({"level": "global", "mystery": "x"}))
        self.assertIsNone(canonicalize_scope({"level": "global", "time": "today"}))
        self.assertIsNone(canonicalize_scope({"level": "global", "time": "当前版本"}))
        self.assertIsNone(
            canonicalize_scope({"level": "global", "conditions": {"unknown": "x"}})
        )

    def test_project_scope_must_match_event_project_and_known_context(self) -> None:
        self.add_snapshot(
            [
                self.confirmation(
                    project_key="observed-project",
                    scope={"level": "project", "project_key": "other-project"},
                )
            ]
        )
        report = run_consolidation(self.database)
        self.assertEqual(report["actions"][0]["reason"], "scope_not_actionable")
        self.assertEqual(report["state"]["proposals"], 0)

    def test_semantic_edge_cases_and_english_stable_key(self) -> None:
        self.assertIsNone(
            semantic_value_for("preference.response.language", "回复不要使用中文。")
        )
        self.assertEqual(
            semantic_value_for(
                "preference.response.language", "不要使用中文，统一使用英文。"
            ),
            "en",
        )
        self.assertEqual(
            semantic_value_for(
                "preference.response.language", "不要英文，统一中文。"
            ),
            "zh",
        )
        self.assertEqual(
            semantic_value_for(
                "workflow.code_generation.confirmation", "正式代码前不要问我。"
            ),
            "not_required",
        )
        self.assertEqual(
            semantic_value_for(
                "workflow.code_generation.confirmation", "禁止未经确认就生成正式代码。"
            ),
            "required",
        )
        self.assertEqual(
            semantic_value_for(
                "preference.response.language", "以后统一使用英文回复。"
            ),
            "en",
        )
        self.assertEqual(
            _suggest_identity("以后统一使用英文回复。")[0],
            "preference.response.language",
        )
        for negative_language in (
            "以后不要继续使用中文回复。",
            "以后无需再使用英文回复。",
            "以后不建议使用中文回复。",
        ):
            self.assertIsNone(
                semantic_value_for("preference.response.language", negative_language)
            )
        self.assertIsNone(
            semantic_value_for(
                "preference.response.language", "以后统一使用中文，也统一使用英文回复。"
            )
        )
        self.assertEqual(
            semantic_value_for(
                "workflow.code_generation.confirmation", "不要未经确认就生成正式代码。"
            ),
            "required",
        )
        self.assertEqual(
            semantic_value_for(
                "workflow.code_generation.confirmation", "未经确认不要生成正式代码。"
            ),
            "required",
        )
        self.assertIsNone(
            semantic_value_for(
                "preference.response.structure", "不要先给结论，再说明步骤。"
            )
        )
        self.assertEqual(
            semantic_value_for(
                "workflow.embedded.code_change.compile_check", "禁止未经编译检查就提交。"
            ),
            "required",
        )
        self.assertEqual(
            semantic_value_for(
                "workflow.code_generation.confirmation", "以后生成正式代码前禁止跳过确认。"
            ),
            "required",
        )
        self.assertEqual(
            semantic_value_for(
                "workflow.embedded.code_change.compile_check",
                "以后单片机修改代码后禁止跳过编译检查。",
            ),
            "required",
        )
        self.assertEqual(
            semantic_value_for(
                "toolchain.tg032plus.compiler",
                "更正：TG032Plus 不再使用 Keil C51，改用 IAR。",
            ),
            "iar",
        )

    def test_partial_unique_index_allows_only_one_current_revision(self) -> None:
        self.activate(self.confirmation())
        with closing(open_store(self.database)) as connection, self.assertRaises(sqlite3.IntegrityError):
            with connection:
                memory = connection.execute("SELECT * FROM memory").fetchone()
                decision = connection.execute("SELECT decision_id FROM admission_decision LIMIT 1").fetchone()[0]
                connection.execute(
                    """
                    INSERT INTO memory_revision(
                        revision_id, memory_id, revision_number, semantic_value,
                        statement, activation_triggers_json, status,
                        source_decision_id, created_at
                    ) VALUES ('illegal_current', ?, 2, 'required', 'duplicate', '[]',
                              'current', ?, '2026-01-02T00:00:00.000+00:00')
                    """,
                    (memory["memory_id"], decision),
                )

    def test_prior_u2_schema_is_migrated_without_rewriting_rows(self) -> None:
        legacy = Path(self.temporary_directory.name) / "legacy-u2.db"
        with closing(sqlite3.connect(legacy)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE candidate_review (
                    review_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL,
                    candidate_key TEXT NOT NULL, outcome TEXT NOT NULL,
                    reviewer TEXT NOT NULL, note TEXT, reviewed_at TEXT NOT NULL
                );
                CREATE TABLE memory (
                    memory_id TEXT PRIMARY KEY, stable_key TEXT NOT NULL,
                    scope_fingerprint TEXT NOT NULL, scope_json TEXT NOT NULL,
                    memory_type TEXT NOT NULL, status TEXT NOT NULL,
                    current_revision_id TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, UNIQUE(stable_key, scope_fingerprint)
                );
                CREATE TABLE memory_revision (
                    revision_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL,
                    revision_number INTEGER NOT NULL, semantic_value TEXT NOT NULL,
                    statement TEXT NOT NULL, activation_triggers_json TEXT NOT NULL,
                    status TEXT NOT NULL, supersedes_revision_id TEXT,
                    source_decision_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(memory_id, revision_number)
                );
                CREATE TABLE evidence (
                    evidence_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL,
                    revision_id TEXT, candidate_key TEXT NOT NULL UNIQUE,
                    decision_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    source_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL, part_id TEXT NOT NULL,
                    project_key TEXT NOT NULL, source_project_id TEXT,
                    exact_quote TEXT NOT NULL, evidence_role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE conflict (
                    conflict_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL,
                    current_revision_id TEXT NOT NULL,
                    incoming_candidate_key TEXT NOT NULL UNIQUE,
                    incoming_decision_id TEXT NOT NULL,
                    incoming_semantic_value TEXT NOT NULL,
                    incoming_statement TEXT NOT NULL, status TEXT NOT NULL,
                    resolution_revision_id TEXT, resolution_reason TEXT,
                    created_at TEXT NOT NULL, resolved_at TEXT
                );
                CREATE TABLE consolidation_run (
                    run_id TEXT PRIMARY KEY, consolidator_version TEXT NOT NULL,
                    started_at TEXT NOT NULL, completed_at TEXT, status TEXT NOT NULL,
                    candidate_count INTEGER NOT NULL DEFAULT 0,
                    counts_json TEXT NOT NULL DEFAULT '{}', error TEXT
                );
                CREATE TABLE consolidation_action (
                    action_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    candidate_key TEXT NOT NULL, decision_id TEXT NOT NULL,
                    action TEXT NOT NULL, memory_id TEXT, before_revision_id TEXT,
                    after_revision_id TEXT, conflict_id TEXT, reason TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    UNIQUE(run_id, candidate_key)
                );
                INSERT INTO candidate_review VALUES (
                    'legacy-review', 'legacy-decision', 'legacy-candidate',
                    'approve', 'legacy-owner', NULL,
                    '2026-01-01T00:00:00.000+00:00'
                );
                """
            )
        with closing(open_store(legacy)) as connection:
            expected = {
                "candidate_review": {"proposal_id"},
                "memory": {"family", "state_version"},
                "memory_proposal": {"expected_memory_state_version"},
                "memory_revision": {"source_proposal_id", "review_id"},
                "evidence": {
                    "conflict_id", "proposal_id", "review_id", "reviewer",
                    "ordinal", "start_offset", "end_offset", "occurred_at",
                },
                "conflict": {
                    "conflict_fingerprint", "incoming_proposal_id", "review_id", "reviewer",
                    "resolution_proposal_id", "resolution_review_id", "resolution_reviewer",
                },
                "consolidation_run": {"admission_run_id", "admission_snapshot_hash"},
                "consolidation_action": {"proposal_id", "review_id", "reviewer"},
            }
            for table, columns in expected.items():
                actual = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                self.assertTrue(columns <= actual, (table, columns - actual))
            legacy_review = connection.execute(
                "SELECT review_id, proposal_id, reviewer FROM candidate_review"
            ).fetchone()
        self.assertEqual(tuple(legacy_review), ("legacy-review", None, "legacy-owner"))

    def test_malformed_complete_snapshot_fails_closed_without_partial_candidates(self) -> None:
        run_id, rows = self.add_snapshot([self.confirmation()])
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                "UPDATE admission_run SET completed_at = NULL WHERE run_id = ?", (run_id,)
            )
        missing_completion = run_consolidation(self.database)
        self.assertEqual(missing_completion["status"], "failed")
        self.assertEqual(missing_completion["error"], "invalid_complete_admission_snapshot")
        self.assertEqual(missing_completion["state"]["memories"], 0)

        run_id, rows = self.add_snapshot([self.confirmation()])
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                "DELETE FROM session_context WHERE session_id = ?",
                (rows[0]["project_key"].replace("project", "session"),),
            )
        partial = run_consolidation(self.database)
        self.assertEqual(partial["status"], "failed")
        self.assertEqual(partial["error"], "invalid_complete_admission_snapshot")
        self.assertEqual(partial["state"]["memories"], 0)

        run_id, _ = self.add_snapshot([self.confirmation()])
        with closing(open_store(self.database)) as connection, connection:
            connection.execute(
                "UPDATE admission_run SET decision_count = 2 WHERE run_id = ?", (run_id,)
            )
        bad_count = run_consolidation(self.database)
        self.assertEqual(bad_count["status"], "failed")
        self.assertEqual(bad_count["error"], "invalid_complete_admission_snapshot")

    def test_cli_reports_failed_status_and_refuses_typo_database_creation(self) -> None:
        missing = Path(self.temporary_directory.name) / "typo.db"
        output = StringIO()
        with redirect_stdout(output):
            code = cli_main(["--lab-db", str(missing), "consolidate"])
        self.assertEqual(code, 1)
        self.assertFalse(missing.exists())
        self.assertFalse(json.loads(output.getvalue())["ok"])

        empty = Path(self.temporary_directory.name) / "empty.db"
        with closing(open_store(empty)):
            pass
        output = StringIO()
        with redirect_stdout(output):
            code = cli_main(["--lab-db", str(empty), "consolidate"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "failed")

    def test_cli_refuses_non_lab_sqlite_without_modifying_it(self) -> None:
        arbitrary = Path(self.temporary_directory.name) / "arbitrary.db"
        with closing(sqlite3.connect(arbitrary)) as connection, connection:
            connection.execute("CREATE TABLE keep_me(value TEXT)")
        output = StringIO()
        with redirect_stdout(output):
            code = cli_main(["--lab-db", str(arbitrary), "consolidate"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        with closing(sqlite3.connect(arbitrary)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertEqual(tables, {"keep_me"})


if __name__ == "__main__":
    unittest.main()
