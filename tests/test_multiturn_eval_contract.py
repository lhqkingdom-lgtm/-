import json
import tempfile
import unittest
from pathlib import Path

from memory_lab.multiturn_eval_contract import (
    MultiturnEvalContractError,
    validate_multiturn_evalset,
)


ROOT = Path(__file__).resolve().parents[1]
EVALSET = ROOT / "evalsets" / "memory_multiturn_v1"


class MultiturnEvalContractTests(unittest.TestCase):
    def test_repository_evalset_passes_static_contract(self) -> None:
        report = validate_multiturn_evalset(EVALSET)
        self.assertTrue(report["ok"])
        self.assertEqual(report["scenario_count"], 10)
        self.assertEqual(report["release_gate_scenario_count"], 9)
        self.assertEqual(report["known_gap_scenario_count"], 1)
        self.assertEqual(report["unique_turn_count"], 29)

    def test_rejects_premature_runtime_memory_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(Path(directory))
            path = target / "scenarios.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["scenarios"][0]["steps"][0]["expected_chain"]["formal_memory_allowed"] = True
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(MultiturnEvalContractError, "user_verified"):
                validate_multiturn_evalset(target)

    def test_rejects_future_history_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(Path(directory))
            path = target / "scenarios.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            scenario = doc["scenarios"][0]
            scenario["steps"][0]["required_history_turn_ids"] = [scenario["turn_ids"][1]]
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(MultiturnEvalContractError, "未来或场景外"):
                validate_multiturn_evalset(target)

    def test_allows_preference_write_while_compilation_chain_is_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(Path(directory))
            path = target / "scenarios.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            step = doc["scenarios"][5]["steps"][2]
            step["expected_chain"]["formal_memory_types_allowed"] = ["user_preference"]
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            self.assertTrue(validate_multiturn_evalset(target)["ok"])

    def test_rejects_compile_memory_write_while_compilation_chain_is_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(Path(directory))
            path = target / "scenarios.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            step = doc["scenarios"][5]["steps"][2]
            step["expected_chain"]["formal_memory_types_allowed"] = [
                "user_preference",
                "compile_failure_experience",
            ]
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(MultiturnEvalContractError, "compile_verified"):
                validate_multiturn_evalset(target)

    def test_accepts_close_old_and_open_new_chain_operations_contract(self) -> None:
        report = validate_multiturn_evalset(EVALSET)
        self.assertTrue(report["ok"])
        final_step = json.loads(
            (EVALSET / "scenarios.json").read_text(encoding="utf-8")
        )["scenarios"][-1]["steps"][-1]
        self.assertNotIn("expected_chain", final_step)
        self.assertEqual(len(final_step["expected_chain_operations"]), 2)

    def test_rejects_open_new_chain_linked_to_formal_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(Path(directory))
            path = target / "scenarios.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            new_operation = doc["scenarios"][-1]["steps"][-1][
                "expected_chain_operations"
            ][1]
            new_operation["linked_formal_memory"] = True
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(
                MultiturnEvalContractError,
                "只有允许正式记忆的闭环操作",
            ):
                validate_multiturn_evalset(target)

    def test_rejects_both_single_and_multi_chain_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(Path(directory))
            path = target / "scenarios.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            final_step = doc["scenarios"][-1]["steps"][-1]
            final_step["expected_chain"] = {
                "chain_type": "runtime",
                "allowed_states": ["user_verified"],
                "formal_memory_allowed": True,
            }
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(MultiturnEvalContractError, "互斥"):
                validate_multiturn_evalset(target)

    @staticmethod
    def _copy(parent: Path) -> Path:
        target = parent / "evalset"
        (target / "fixtures").mkdir(parents=True)
        for name in ("manifest.json", "scenarios.json", "rubric.json", "cases.jsonl"):
            (target / name).write_bytes((EVALSET / name).read_bytes())
        (target / "fixtures" / "initial_states.json").write_bytes(
            (EVALSET / "fixtures" / "initial_states.json").read_bytes()
        )
        return target


if __name__ == "__main__":
    unittest.main()
