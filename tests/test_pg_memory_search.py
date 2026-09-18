from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.pg_store import _search_terms, search_related_memories


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


if __name__ == "__main__":
    unittest.main()
