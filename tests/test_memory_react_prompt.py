from __future__ import annotations

import unittest

from memory_lab.memory_react_agent import REACT_SYSTEM_PROMPT


class MemoryReactPromptTests(unittest.TestCase):
    def test_prompt_documents_conflict_maintenance_triplet(self) -> None:
        self.assertIn("组合动作规则", REACT_SYSTEM_PROMPT)
        self.assertIn("update_memory：更新当前有效结论", REACT_SYSTEM_PROMPT)
        self.assertIn("update_topic：如果 topic 标题/概要仍是旧语义", REACT_SYSTEM_PROMPT)
        self.assertIn("record_conflict_decision：记录旧说法", REACT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
