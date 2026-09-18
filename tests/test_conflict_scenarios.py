from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory_lab.conflict_scenarios import (
    REQUIRED_DIMENSIONS,
    REQUIRED_SCENARIO_IDS,
    SCENARIOS,
    build_conflict_scenarios_report,
)


class ConflictScenariosTests(unittest.TestCase):
    def test_required_scenarios_and_dimensions_are_covered(self) -> None:
        report = build_conflict_scenarios_report()

        self.assertEqual(report["status"], "complete")
        self.assertEqual(set(report["covered_scenarios"]), REQUIRED_SCENARIO_IDS)
        self.assertTrue(REQUIRED_DIMENSIONS.issubset(set(report["covered_dimensions"])))
        self.assertEqual(report["missing_scenarios"], [])
        self.assertEqual(report["missing_dimensions"], [])

    def test_each_scenario_has_actions_and_negative_assertions(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.scenario_id):
                self.assertTrue(scenario.expected_actions)
                self.assertTrue(scenario.evidence_required)
                self.assertTrue(scenario.expected_revision_assertions)
                self.assertTrue(scenario.expected_relation_assertions)
                self.assertTrue(scenario.negative_assertions)
                self.assertTrue(scenario.covered_dimensions)

    def test_report_writes_reviewable_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_conflict_scenarios_report(output_dir=Path(directory))
            path = Path(report["markdown_path"])

            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("长期记忆矛盾处理测试集", text)
            self.assertIn("用户偏好变化", text)
            self.assertIn("验证结果覆盖猜测", text)
            self.assertIn("当前项目参数不沉淀", text)


if __name__ == "__main__":
    unittest.main()
