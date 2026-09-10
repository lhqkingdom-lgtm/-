from __future__ import annotations

import unittest

from memory_lab.model_recall import _router_topics


class ModelRecallAdapterTests(unittest.TestCase):
    def test_response_preferences_are_marked_always_on(self):
        topics = _router_topics(
            {
                "topics": [
                    {
                        "topic_id": "t",
                        "name": "回复偏好",
                        "summary": "如何回复",
                        "type": "user",
                        "updated_at": "now",
                        "stable_keys": ["preference.response.structure"],
                    }
                ]
            }
        )
        self.assertTrue(topics[0]["always"])


if __name__ == "__main__":
    unittest.main()
