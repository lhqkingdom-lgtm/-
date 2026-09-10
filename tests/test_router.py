from __future__ import annotations

import unittest

from memory_lab.router import TopicRouterError, route_topics, sanitize_routing_text


class Selector:
    def __init__(self, selected):
        self.selected = selected

    def select(self, request):
        return {"schema_version": "memory-topic-router-v1", "selected": self.selected}


TOPICS = [
    {
        "topic_id": "real-a",
        "name": "回复偏好",
        "description": "回复结构",
        "family": "user_contract",
        "updated_at": "2026-09-10T00:00:00Z",
        "stable_keys": ["preference.response.structure"],
        "always": True,
    },
    {
        "topic_id": "real-b",
        "name": "嵌入式流程",
        "description": "编译检查",
        "family": "user_contract",
        "updated_at": "2026-09-10T00:00:00Z",
        "stable_keys": ["workflow.embedded.code_change.compile_check"],
        "always": False,
    },
]


class RouterTests(unittest.TestCase):
    def test_aliases_are_validated_and_always_topic_is_added(self):
        result = route_topics(
            "修改单片机代码",
            TOPICS,
            Selector([{"topic_ref": "topic_1", "score": 0.9, "reason": "编译相关"}]),
            purpose="recall",
        )
        self.assertEqual([item["topic_id"] for item in result["selected"]], ["real-a", "real-b"])

    def test_fabricated_topic_is_rejected(self):
        with self.assertRaises(TopicRouterError):
            route_topics(
                "任务",
                TOPICS,
                Selector([{"topic_ref": "topic_9", "score": 1, "reason": "x"}]),
                purpose="update",
            )

    def test_secrets_and_paths_are_redacted_before_routing(self):
        text, redacted = sanitize_routing_text(r"处理 C:\\private\\a.c，token=abc1234567890")
        self.assertTrue(redacted)
        self.assertNotIn("abc1234567890", text)
        self.assertNotIn("private", text)


if __name__ == "__main__":
    unittest.main()
