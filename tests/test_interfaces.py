from __future__ import annotations

import unittest

from memory_lab.interfaces import RecallRequest, SourceCursor, UpdatePlan


class InterfaceContractTests(unittest.TestCase):
    def test_contract_values_are_immutable(self) -> None:
        cursor = SourceCursor("cursor-1")
        plan = UpdatePlan("plan-1", "snapshot-1", ())
        with self.assertRaises(Exception):
            cursor.value = "changed"  # type: ignore[misc]
        self.assertEqual(plan.snapshot_hash, "snapshot-1")

    def test_recall_defaults_are_bounded(self) -> None:
        request = RecallRequest("实现一个功能")
        self.assertEqual(request.limit, 5)
        self.assertEqual(dict(request.conditions), {})


if __name__ == "__main__":
    unittest.main()
