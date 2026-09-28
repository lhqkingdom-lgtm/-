from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.pg_store import (
    get_turn_evidence,
    list_previous_session_turns,
    read_session_turn_evidence,
)


CONFIG = {
    "user": {"user_id": "user_a"},
    "postgres": {"password": "unused"},
}


class PgSessionTurnToolsTests(unittest.TestCase):
    def test_list_previous_turns_is_bounded_to_same_conversation(self) -> None:
        payload = {
            "anchor_found": True,
            "turns": [
                {
                    "turn_id": "turn_previous",
                    "phase_hint": "runtime_diagnosis",
                    "evidence_count": 4,
                    "evidence_kinds": ["runtime_symptom", "diagnosis_or_solution"],
                    "previews": [{"evidence_id": "ev_old", "text": "风扇不转"}],
                }
            ],
        }
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=payload
        ) as query_json:
            result = list_previous_session_turns(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_current",
                limit=999,
                preview_chars=9999,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["limits"], {"limit": 20, "preview_chars": 800})
        self.assertIn("必须继续调用 read_session_turn_evidence", result["guidance"])
        sql = query_json.call_args.args[1]
        self.assertIn("t.user_id = a.user_id", sql)
        self.assertIn("t.source_id = a.source_id", sql)
        self.assertIn("t.session_id = a.session_id", sql)
        self.assertIn("COALESCE(t.project_key, '') = COALESCE(a.project_key, '')", sql)
        self.assertIn("t.turn_id <> a.turn_id", sql)
        self.assertIn(") < (", sql)
        self.assertIn("LIMIT 20", sql)
        self.assertIn("LIMIT 6", sql)

    def test_list_previous_turns_rejects_invalid_bounds_before_query(self) -> None:
        result = list_previous_session_turns(
            pg_config_path=Path("unused.json"),
            current_turn_id="turn_current",
            limit="many",  # type: ignore[arg-type]
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "invalid_bounds")

    def test_list_previous_turns_reports_missing_anchor(self) -> None:
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object",
            return_value={"anchor_found": False, "turns": []},
        ):
            result = list_previous_session_turns(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_missing",
            )

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["error_code"], "current_turn_not_found")
        self.assertEqual(result["turns"], [])

    def test_read_history_rejects_current_turn_as_target_without_query(self) -> None:
        result = read_session_turn_evidence(
            pg_config_path=Path("unused.json"),
            current_turn_id="turn_current",
            target_turn_id="turn_current",
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "target_is_current_turn")

    def test_read_history_denies_other_session_source_project_or_future(self) -> None:
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object",
            return_value={
                "anchor_found": True,
                "target_found": True,
                "authorized": False,
            },
        ) as query_json, patch("memory_lab.pg_store.get_turn_evidence") as get_evidence:
            result = read_session_turn_evidence(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_current",
                target_turn_id="turn_untrusted",
            )

        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["error_code"], "target_outside_current_conversation")
        get_evidence.assert_not_called()
        sql = query_json.call_args.args[1]
        self.assertIn("t.source_id = a.source_id", sql)
        self.assertIn("t.session_id = a.session_id", sql)
        self.assertIn("COALESCE(t.project_key, '') = COALESCE(a.project_key, '')", sql)
        self.assertIn(") < (", sql)

    def test_read_history_reports_missing_target(self) -> None:
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object",
            return_value={
                "anchor_found": True,
                "target_found": False,
                "authorized": False,
            },
        ):
            result = read_session_turn_evidence(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_current",
                target_turn_id="turn_missing",
            )

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["error_code"], "target_turn_not_found")

    def test_read_history_returns_evidence_only_after_authorization(self) -> None:
        detail = {
            "status": "complete",
            "tool": "get_turn_evidence",
            "turn_id": "turn_previous",
            "found": True,
            "evidence": {
                "turn": {
                    "turn_id": "turn_previous",
                    "session_id": "session_1",
                    "source_id": "source_1",
                    "project_key": "project_1",
                },
                "layers": {
                    "core": [{"evidence_id": "ev_old"}],
                    "candidate": [],
                    "pointer": [],
                },
            },
        }
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object",
            return_value={
                "anchor_found": True,
                "target_found": True,
                "authorized": True,
            },
        ), patch("memory_lab.pg_store.get_turn_evidence", return_value=detail) as get_evidence:
            result = read_session_turn_evidence(
                pg_config_path=Path("unused.json"),
                current_turn_id="turn_current",
                target_turn_id="turn_previous",
                layers=["core"],
                max_chars=600,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["target_turn_id"], "turn_previous")
        self.assertEqual(result["evidence"]["layers"]["core"][0]["evidence_id"], "ev_old")
        get_evidence.assert_called_once_with(
            pg_config_path=Path("unused.json"),
            turn_id="turn_previous",
            layers=["core"],
            max_chars=600,
        )

    def test_get_turn_evidence_exposes_session_id(self) -> None:
        fake = {
            "found": True,
            "turn": {"turn_id": "turn_1", "session_id": "session_1"},
            "requested_layers": ["core"],
            "layers": {"core": []},
        }
        with patch("memory_lab.pg_store.load_pg_config", return_value=CONFIG), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=fake
        ) as query_json:
            result = get_turn_evidence(
                pg_config_path=Path("unused.json"),
                turn_id="turn_1",
                layers=["core"],
            )

        self.assertEqual(result["evidence"]["turn"]["session_id"], "session_1")
        self.assertIn("'session_id', t.session_id", query_json.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
