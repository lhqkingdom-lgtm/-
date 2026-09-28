from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.pg_store import _upsert_sql, list_open_memory_chains, save_memory_chain_state


CONFIG = {"user": {"user_id": "user_1"}, "postgres": {}}


class PgMemoryChainStateTests(unittest.TestCase):
    def _context(self, *, turns=None, chains=None):
        return {
            "current_found": True,
            "session_id": "session_1",
            "project_key": "project_1",
            "turns": turns
            or [
                {"turn_id": "turn_1", "memory_chains": []},
                {"turn_id": "turn_2", "memory_chains": []},
            ],
            "conversation_chains": chains or [],
        }

    def test_creates_open_chain_on_earliest_related_turn(self) -> None:
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object",
            side_effect=[self._context(), {"updated": True}],
        ) as query:
            result = save_memory_chain_state(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_2",
                chain_state={
                    "chain_type": "runtime",
                    "state": "waiting_user_validation",
                    "related_turn_ids": ["turn_1", "turn_2"],
                    "missing_evidence": ["用户重新烧录后的确认"],
                },
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["result"], "created")
        self.assertEqual(result["chain"]["anchor_turn_id"], "turn_1")
        self.assertEqual(result["chain"]["status"], "open")
        self.assertEqual(query.call_count, 2)
        self.assertIn("jsonb_set", query.call_args_list[1].args[1])

    def test_same_state_is_idempotent(self) -> None:
        existing = {
            "schema_version": "memory-chain-state-v1",
            "chain_id": "chain_existing",
            "chain_type": "runtime",
            "state": "waiting_user_validation",
            "anchor_turn_id": "turn_1",
            "related_turn_ids": ["turn_1", "turn_2"],
            "missing_evidence": ["用户重新烧录后的确认"],
            "status": "open",
            "updated_by_turn_id": "turn_2",
            "final_memory_ids": [],
        }
        context = self._context(
            turns=[
                {"turn_id": "turn_1", "memory_chains": [existing]},
                {"turn_id": "turn_2", "memory_chains": []},
            ],
            chains=[{"stored_on_turn_id": "turn_1", "chain": existing}],
        )
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=context
        ) as query:
            result = save_memory_chain_state(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_2",
                chain_state={
                    "chain_id": "chain_existing",
                    "chain_type": "runtime",
                    "state": "waiting_user_validation",
                    "related_turn_ids": ["turn_1", "turn_2"],
                    "missing_evidence": ["用户重新烧录后的确认"],
                },
            )
        self.assertEqual(result["result"], "skipped_existing")
        query.assert_called_once()

    def test_rejects_related_turn_outside_conversation(self) -> None:
        context = self._context(
            turns=[{"turn_id": "turn_2", "memory_chains": []}]
        )
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=context
        ) as query:
            result = save_memory_chain_state(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_2",
                chain_state={
                    "chain_type": "runtime",
                    "state": "cause_identified",
                    "related_turn_ids": ["turn_other", "turn_2"],
                    "missing_evidence": ["修复和用户验证"],
                },
            )
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["error_code"], "related_turn_outside_conversation")
        query.assert_called_once()

    def test_rejects_regression_of_existing_chain(self) -> None:
        existing = {
            "chain_id": "chain_existing",
            "chain_type": "compilation",
            "state": "fix_applied",
            "anchor_turn_id": "turn_1",
            "related_turn_ids": ["turn_1", "turn_2"],
            "missing_evidence": ["重新编译结果"],
            "status": "open",
            "updated_by_turn_id": "turn_2",
            "final_memory_ids": [],
        }
        context = self._context(
            turns=[
                {"turn_id": "turn_1", "memory_chains": [existing]},
                {"turn_id": "turn_2", "memory_chains": []},
                {"turn_id": "turn_3", "memory_chains": []},
            ],
            chains=[{"stored_on_turn_id": "turn_1", "chain": existing}],
        )
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=context
        ):
            result = save_memory_chain_state(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_3",
                chain_state={
                    "chain_id": "chain_existing",
                    "chain_type": "compilation",
                    "state": "cause_identified",
                    "related_turn_ids": ["turn_1", "turn_2", "turn_3"],
                    "missing_evidence": ["修复和重新编译"],
                },
            )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "invalid_chain_transition")

    def test_rejects_unknown_explicit_chain_id(self) -> None:
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object",
            return_value=self._context(),
        ):
            result = save_memory_chain_state(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_2",
                chain_state={
                    "chain_id": "chain_unknown_001",
                    "chain_type": "runtime",
                    "state": "waiting_user_validation",
                    "related_turn_ids": ["turn_1", "turn_2"],
                    "missing_evidence": ["用户确认"],
                },
            )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "unknown_chain_id")

    def test_overlap_requires_explicit_chain_id(self) -> None:
        existing = {
            "chain_id": "chain_existing",
            "chain_type": "runtime",
            "state": "waiting_user_validation",
            "anchor_turn_id": "turn_1",
            "related_turn_ids": ["turn_1"],
            "missing_evidence": ["用户确认"],
            "status": "open",
            "updated_by_turn_id": "turn_1",
            "final_memory_ids": [],
        }
        context = self._context(
            turns=[
                {"turn_id": "turn_1", "memory_chains": [existing]},
                {"turn_id": "turn_2", "memory_chains": []},
            ],
            chains=[{"stored_on_turn_id": "turn_1", "chain": existing}],
        )
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=context
        ):
            result = save_memory_chain_state(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_2",
                chain_state={
                    "chain_type": "runtime",
                    "state": "waiting_user_validation",
                    "related_turn_ids": ["turn_1", "turn_2"],
                    "missing_evidence": ["用户确认"],
                },
            )
        self.assertEqual(result["error_code"], "chain_id_required_for_existing_chain")

    def test_detects_optimistic_concurrency_conflict(self) -> None:
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object",
            side_effect=[self._context(), {"updated": False}],
        ):
            result = save_memory_chain_state(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_2",
                chain_state={
                    "chain_type": "compilation",
                    "state": "failure_observed",
                    "related_turn_ids": ["turn_1", "turn_2"],
                    "missing_evidence": ["根因、修复和编译验证"],
                },
            )
        self.assertEqual(result["status"], "retry")
        self.assertEqual(result["error_code"], "concurrent_chain_update")

    def test_turn_reingest_sql_preserves_memory_chains(self) -> None:
        pool = {
            "turn": {"turn_id": "turn_1", "state": "completed", "source_signature": "sig"},
            "workspace": {"project_key": "project_1"},
            "session": {"session_id": "session_1"},
            "rule_phase": {"primary_phase": "compilation"},
            "stats": {},
            "core_evidence": [],
            "candidate_evidence": [],
            "raw_pointers": [],
        }
        sql = _upsert_sql(
            user_id="user_1",
            source_database=Path("source.db"),
            source_identifier="source_1",
            pool=pool,
        )
        self.assertIn("turns.metadata ? 'memory_chains'", sql)
        self.assertIn("jsonb_build_object('memory_chains'", sql)
        self.assertIn("turns.user_id = EXCLUDED.user_id", sql)

    def test_turn_reingest_sql_does_not_preserve_chains_across_users(self) -> None:
        pool = {
            "turn": {"turn_id": "turn_1", "state": "completed", "source_signature": "sig"},
            "workspace": {"project_key": "project_1"},
            "session": {"session_id": "session_1"},
            "rule_phase": {"primary_phase": "runtime"},
            "stats": {},
            "core_evidence": [],
            "candidate_evidence": [],
            "raw_pointers": [],
        }
        sql = _upsert_sql(
            user_id="isolated_eval_user",
            source_database=Path("source.db"),
            source_identifier="source_1",
            pool=pool,
        )
        preservation_case = sql.split("metadata = EXCLUDED.metadata ||", 1)[1]
        self.assertIn("WHEN turns.user_id = EXCLUDED.user_id", preservation_case)
        self.assertIn("ELSE '{}'::jsonb", preservation_case)

    def test_lists_open_chains_without_exposing_evidence_ids(self) -> None:
        payload = {
            "current_found": True,
            "chains": [
                {
                    "chain_id": "chain_1",
                    "chain_type": "runtime",
                    "state": "waiting_user_validation",
                    "status": "open",
                    "anchor_turn_id": "turn_1",
                    "related_turn_ids": ["turn_1", "turn_2"],
                    "missing_evidence": ["用户确认"],
                    "related_turns": [
                        {"turn_id": "turn_1", "preview": "风扇不转"},
                        {"turn_id": "turn_2", "preview": "已修复 ADC 读取"},
                    ],
                }
            ],
        }
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=payload
        ) as query:
            result = list_open_memory_chains(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_3",
                limit=50,
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["limit"], 20)
        self.assertEqual(result["chains"][0]["related_turn_ids"], ["turn_1", "turn_2"])
        self.assertNotIn("evidence_id", str(result))
        sql = query.call_args.args[1]
        self.assertIn("t.session_id = a.session_id", sql)
        self.assertIn("COALESCE(t.project_key, '')", sql)


if __name__ == "__main__":
    unittest.main()
