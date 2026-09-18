from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.pg_store import (
    MEMORY_RELATION_TYPES,
    TRUST_LEVELS,
    _memory_versioning_migration_sql,
    migrate_memory_versioning,
    trace_memory_relations,
)


class PgMemoryVersioningTests(unittest.TestCase):
    def test_migration_sql_adds_minimal_versioning_schema(self) -> None:
        sql = _memory_versioning_migration_sql(user_id="local-default-user")

        self.assertIn("ADD COLUMN IF NOT EXISTS trust_level", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS memory_revisions", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS memory_relations", sql)
        self.assertIn("revision_number", sql)
        self.assertIn("relation_type", sql)
        self.assertIn("supersedes", sql)
        self.assertIn("merges", sql)
        self.assertIn("conflicts_with", sql)
        self.assertIn("WHERE NOT EXISTS", sql)
        self.assertIn("revision_number = 1", sql)

    def test_migrate_memory_versioning_returns_report_and_enums(self) -> None:
        config = {
            "user": {"user_id": "local-default-user"},
            "postgres": {"password": "unused"},
        }
        report = {
            "memory_count": 2,
            "revision_count": 2,
            "initial_revision_count": 2,
            "relation_count": 0,
            "trust_level_counts": {"inferred": 2},
        }
        with patch("memory_lab.pg_store.load_pg_config", return_value=config), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=report
        ) as query_json:
            result = migrate_memory_versioning(pg_config_path=Path("unused.json"))

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["tool"], "migrate_memory_versioning")
        self.assertEqual(result["trust_levels"], list(TRUST_LEVELS))
        self.assertEqual(result["relation_types"], list(MEMORY_RELATION_TYPES))
        self.assertEqual(result["report"]["initial_revision_count"], 2)
        sql = query_json.call_args.args[1]
        self.assertIn("BEGIN;", sql)
        self.assertIn("COMMIT;", sql)
        self.assertIn("memory_revisions", sql)
        self.assertIn("memory_relations", sql)

    def test_trace_memory_relations_returns_history_and_links(self) -> None:
        config = {
            "user": {"user_id": "local-default-user"},
            "postgres": {"password": "unused"},
        }
        detail = {
            "found": True,
            "memory": {"memory_id": "mem_new", "summary": "新记忆"},
            "revisions": [{"revision_id": "rev_2", "revision_number": 2}],
            "backward": [
                {
                    "relation_id": "rel_1",
                    "relation_type": "supersedes",
                    "from_memory": {"memory_id": "mem_old", "summary": "旧记忆"},
                }
            ],
            "forward": [],
        }
        with patch("memory_lab.pg_store.load_pg_config", return_value=config), patch(
            "memory_lab.pg_store._json_from_psql_object", return_value=detail
        ) as query_json:
            result = trace_memory_relations(
                pg_config_path=Path("unused.json"),
                memory_id="mem_new",
                direction="backward",
                max_depth=99,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["max_depth"], 5)
        self.assertEqual(result["detail"]["backward"][0]["relation_type"], "supersedes")
        self.assertIn("backward", result["guidance"])
        sql = query_json.call_args.args[1]
        self.assertIn("memory_revisions", sql)
        self.assertIn("memory_relations", sql)
        self.assertIn("WHERE true", sql)

    def test_trace_memory_relations_rejects_bad_direction(self) -> None:
        result = trace_memory_relations(
            pg_config_path=Path("unused.json"),
            memory_id="mem_new",
            direction="sideways",
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "invalid_direction")


if __name__ == "__main__":
    unittest.main()
