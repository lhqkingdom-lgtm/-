from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.pg_store import (
    _search_terms,
    get_current_turn_evidence_summary,
    search_related_memories,
)


class PgMemorySearchTests(unittest.TestCase):
    def test_search_terms_extracts_mixed_english_and_chinese_terms(self) -> None:
        terms = _search_terms("Keil C51 风扇阈值频繁启停")

        self.assertIn("Keil", terms)
        self.assertIn("C51", terms)
        self.assertIn("风扇阈值频繁启停", terms)
        self.assertIn("风扇", terms)
        self.assertIn("启停", terms)
        self.assertLessEqual(len(terms), 12)

    def test_empty_query_returns_model_friendly_rejection(self) -> None:
        result = search_related_memories(pg_config_path=Path("unused.json"), query="  ")

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "empty_query")
        self.assertEqual(result["matches"], [])
        self.assertIn("重新搜索", result["guidance"])

    def test_search_clamps_limit_and_returns_candidates(self) -> None:
        config = {
            "user": {"user_id": "local-default-user"},
            "postgres": {"password": "unused"},
        }
        fake_matches = [
            {
                "memory_id": "mem_stub",
                "topic_id": "topic_stub",
                "summary": "硬件输入桩函数固定返回常量会导致下游状态机无响应。",
                "matched_fields": ["memory.summary"],
                "matched_terms": ["桩函数"],
                "score": 9,
            }
        ]
        with patch("memory_lab.pg_store.load_pg_config", return_value=config), patch(
            "memory_lab.pg_store._json_from_psql_array", return_value=fake_matches
        ) as query_json:
            result = search_related_memories(
                pg_config_path=Path("unused.json"),
                query="桩函数 固定返回",
                domain="embedded",
                limit=999,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["filters"]["limit"], 20)
        self.assertEqual(result["filters"]["min_term_matches"], 1)
        self.assertEqual(result["matches"][0]["memory_id"], "mem_stub")
        self.assertIn("修改前必须 read_memory_detail", result["guidance"])
        sql = query_json.call_args.args[1]
        self.assertIn("mt.title", sql)
        self.assertIn("mt.summary", sql)
        self.assertIn("m.summary", sql)
        self.assertIn("m.content", sql)
        self.assertIn("trust_level", sql)
        self.assertIn("source_rank", sql)
        self.assertIn("verification_status", sql)
        self.assertIn("next_steps", sql)
        self.assertIn("LIMIT 20", sql)

    def test_no_matches_returns_search_guidance(self) -> None:
        config = {
            "user": {"user_id": "local-default-user"},
            "postgres": {"password": "unused"},
        }
        with patch("memory_lab.pg_store.load_pg_config", return_value=config), patch(
            "memory_lab.pg_store._json_from_psql_array", return_value=[]
        ):
            result = search_related_memories(
                pg_config_path=Path("unused.json"),
                query="不存在的模式",
                limit=3,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["matches"], [])
        self.assertIn("未找到候选", result["guidance"])

    def test_current_turn_evidence_summary_compacts_layers_and_ids(self) -> None:
        fake_detail = {
            "status": "complete",
            "tool": "get_turn_evidence",
            "turn_id": "turn_1",
            "found": True,
            "evidence": {
                "turn": {"turn_id": "turn_1", "phase_hint": "runtime_debug"},
                "layers": {
                    "core": [
                        {
                            "evidence_id": "ev_core_1",
                            "layer": "core",
                            "source_type": "user",
                            "kind": "message",
                            "text": "用户反馈风扇不转",
                            "raw_pointer_id": None,
                            "ordinal": 1,
                            "truncated": False,
                        },
                        {
                            "evidence_id": "ev_core_2",
                            "layer": "core",
                            "source_type": "assistant",
                            "kind": "answer",
                            "text": "定位 adc_sample 固定返回",
                            "raw_pointer_id": None,
                            "ordinal": 2,
                            "truncated": False,
                        },
                    ],
                    "candidate": [],
                    "pointer": [
                        {
                            "evidence_id": "ptr_1",
                            "layer": "pointer",
                            "source_type": "tool",
                            "kind": "read",
                            "text": "read completed",
                            "raw_pointer_id": "ptr_1",
                            "ordinal": 3,
                            "truncated": True,
                        }
                    ],
                },
            },
        }
        with patch("memory_lab.pg_store.get_turn_evidence", return_value=fake_detail):
            result = get_current_turn_evidence_summary(
                pg_config_path=Path("unused.json"),
                turn_id="turn_1",
                max_items_per_layer=1,
                max_chars=80,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counts"]["core"], 2)
        self.assertEqual(len(result["layers"]["core"]), 1)
        self.assertIn("ev_core_1", result["evidence_ids"])
        self.assertIn("ptr_1", result["pointer_ids"])
        self.assertIn("只能引用 evidence_ids", result["guidance"])


if __name__ == "__main__":
    unittest.main()
