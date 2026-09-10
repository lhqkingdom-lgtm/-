from __future__ import annotations

import unittest

from memory_lab.cli import build_parser


class NewCliSurfaceTests(unittest.TestCase):
    def test_formal_and_cycle_commands_are_exposed(self):
        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["auto-memory", "--refresh"]).command,
            "auto-memory",
        )
        recall = parser.parse_args(
            [
                "formal-recall",
                "--query",
                "生成代码",
                "--model-router",
                "--condition",
                "toolchain=Keil C51",
            ]
        )
        self.assertTrue(recall.model_router)
        self.assertEqual(recall.condition, ["toolchain=Keil C51"])
        self.assertEqual(parser.parse_args(["cycle"]).command, "cycle")
        self.assertEqual(parser.parse_args(["watch"]).command, "watch")
        generic = parser.parse_args(
            ["session-memory", "--batch-size", "8", "--settle-ms", "0"]
        )
        self.assertEqual(generic.command, "session-memory")
        self.assertEqual(generic.batch_size, 8)
        self.assertEqual(generic.settle_ms, 0)
        rebuilt = parser.parse_args(
            ["change-report", "--run-id", "automemory_example"]
        )
        self.assertEqual(rebuilt.command, "change-report")
        self.assertEqual(rebuilt.run_id, "automemory_example")


if __name__ == "__main__":
    unittest.main()
