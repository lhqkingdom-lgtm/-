from __future__ import annotations

import sqlite3
import unittest

from memory_lab.topics import (
    TOPIC_PLAN_VERSION,
    apply_topic_plan,
    ensure_topic_assignments,
    formal_semantic_hash,
    topic_catalog,
)


def fixture() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE memory (
          memory_id TEXT PRIMARY KEY, stable_key TEXT, family TEXT,
          scope_fingerprint TEXT, scope_json TEXT, memory_type TEXT, status TEXT,
          current_revision_id TEXT, state_version INTEGER, created_at TEXT, updated_at TEXT
        );
        CREATE TABLE memory_revision (
          revision_id TEXT PRIMARY KEY, memory_id TEXT, revision_number INTEGER,
          semantic_value TEXT, statement TEXT, activation_triggers_json TEXT,
          status TEXT, supersedes_revision_id TEXT, source_decision_id TEXT,
          source_proposal_id TEXT, review_id TEXT, created_at TEXT
        );
        CREATE TABLE evidence (evidence_id TEXT PRIMARY KEY, memory_id TEXT);
        CREATE TABLE conflict (conflict_id TEXT PRIMARY KEY, memory_id TEXT);
        """
    )
    for index, key in enumerate(
        ("preference.response.language", "preference.response.structure"), start=1
    ):
        connection.execute(
            "INSERT INTO memory VALUES (?, ?, 'user_contract', ?, '{}', 'response_preference', 'active', ?, 1, 't', 't')",
            (f"m{index}", key, f"s{index}", f"r{index}"),
        )
        connection.execute(
            "INSERT INTO memory_revision VALUES (?, ?, 1, 'v', 'statement', '[]', 'current', NULL, 'd', NULL, NULL, 't')",
            (f"r{index}", f"m{index}"),
        )
    return connection


class TopicMaintenanceTests(unittest.TestCase):
    def test_default_assignment_has_one_owner_per_stable_key(self) -> None:
        connection = fixture()
        result = ensure_topic_assignments(connection)
        self.assertEqual(result["memberships_created"], 2)
        catalog = topic_catalog(connection)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0]["stable_keys"]), 2)

    def test_dynamic_topic_uses_reviewed_memory_summary(self) -> None:
        connection = fixture()
        connection.execute(
            "INSERT INTO memory VALUES ('m3', 'knowledge.flash.alignment', "
            "'knowledge_memory', 's3', '{}', 'knowledge', 'active', 'r3', 1, 't', 'u')"
        )
        connection.execute(
            "INSERT INTO memory_revision VALUES ('r3', 'm3', 1, 'v', 'statement', "
            "?, 'current', NULL, 'd', NULL, NULL, 'u')",
            ('["Flash 写入需要按页边界对齐", "knowledge", "flash"]',),
        )

        ensure_topic_assignments(connection)
        topic = next(
            item
            for item in topic_catalog(connection)
            if "knowledge.flash.alignment" in item["stable_keys"]
        )

        self.assertEqual(topic["description"], "Flash 写入需要按页边界对齐")

    def test_split_changes_only_topic_membership(self) -> None:
        connection = fixture()
        ensure_topic_assignments(connection)
        baseline = formal_semantic_hash(connection)
        result = apply_topic_plan(
            connection,
            {
                "schema_version": TOPIC_PLAN_VERSION,
                "baseline_memory_hash": baseline,
                "actions": [
                    {
                        "kind": "split_topic",
                        "source_slug": "user_response_preferences",
                        "targets": [
                            {
                                "slug": "response_language",
                                "name": "回复语言",
                                "description": "回复语言偏好",
                                "family": "user_contract",
                                "stable_keys": ["preference.response.language"],
                            },
                            {
                                "slug": "response_structure",
                                "name": "回复结构",
                                "description": "回复结构偏好",
                                "family": "user_contract",
                                "stable_keys": ["preference.response.structure"],
                            },
                        ],
                    }
                ],
            },
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(formal_semantic_hash(connection), baseline)
        self.assertEqual(len(topic_catalog(connection)), 2)
        self.assertEqual(
            apply_topic_plan(
                connection,
                {
                    "schema_version": TOPIC_PLAN_VERSION,
                    "baseline_memory_hash": baseline,
                    "actions": [
                        {
                            "kind": "split_topic",
                            "source_slug": "user_response_preferences",
                            "targets": [
                                {
                                    "slug": "response_language",
                                    "name": "回复语言",
                                    "description": "回复语言偏好",
                                    "family": "user_contract",
                                    "stable_keys": ["preference.response.language"],
                                },
                                {
                                    "slug": "response_structure",
                                    "name": "回复结构",
                                    "description": "回复结构偏好",
                                    "family": "user_contract",
                                    "stable_keys": ["preference.response.structure"],
                                },
                            ],
                        }
                    ],
                },
            )["status"],
            "noop",
        )


if __name__ == "__main__":
    unittest.main()
