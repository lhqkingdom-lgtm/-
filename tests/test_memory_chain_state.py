from __future__ import annotations

import unittest

from memory_lab.memory_chain_state import (
    MemoryChainStateValidationError,
    validate_chain_state,
    validate_chain_transition,
)


class MemoryChainStateTests(unittest.TestCase):
    def test_chain_id_is_optional_for_new_chain_and_validated_when_present(self) -> None:
        result = validate_chain_state(
            {
                "chain_type": "runtime",
                "state": "symptom_observed",
                "related_turn_ids": ["turn_current"],
                "missing_evidence": ["根因、修复和验证"],
            },
            current_turn_id="turn_current",
        )
        self.assertNotIn("chain_id", result)
        with self.assertRaisesRegex(MemoryChainStateValidationError, "existing chain_ id"):
            validate_chain_state(
                {
                    "chain_id": "made-up",
                    "chain_type": "runtime",
                    "state": "symptom_observed",
                    "related_turn_ids": ["turn_current"],
                    "missing_evidence": ["根因、修复和验证"],
                },
                current_turn_id="turn_current",
            )

    def test_open_compile_chain_requires_missing_evidence(self) -> None:
        with self.assertRaisesRegex(MemoryChainStateValidationError, "missing_evidence"):
            validate_chain_state(
                {
                    "chain_type": "compilation",
                    "state": "fix_applied",
                    "related_turn_ids": ["turn_current"],
                    "missing_evidence": [],
                },
                current_turn_id="turn_current",
            )

    def test_compile_verified_is_closed(self) -> None:
        result = validate_chain_state(
            {
                "chain_type": "compilation",
                "state": "compile_verified",
                "related_turn_ids": ["turn_previous", "turn_current"],
                "missing_evidence": [],
            },
            current_turn_id="turn_current",
            observed_turn_ids={"turn_previous"},
        )
        self.assertEqual(result["status"], "closed")

    def test_runtime_waiting_for_user_is_open(self) -> None:
        result = validate_chain_state(
            {
                "chain_type": "runtime",
                "state": "waiting_user_validation",
                "related_turn_ids": ["turn_previous", "turn_current"],
                "missing_evidence": ["用户重新烧录后的运行确认"],
            },
            current_turn_id="turn_current",
            observed_turn_ids={"turn_previous"},
        )
        self.assertEqual(result["status"], "open")

    def test_closed_chain_rejects_missing_evidence(self) -> None:
        with self.assertRaisesRegex(MemoryChainStateValidationError, "must not contain"):
            validate_chain_state(
                {
                    "chain_type": "runtime",
                    "state": "user_verified",
                    "related_turn_ids": ["turn_current"],
                    "missing_evidence": ["仍待确认"],
                },
                current_turn_id="turn_current",
            )

    def test_related_turn_requires_read_authorization_and_current_turn(self) -> None:
        with self.assertRaises(MemoryChainStateValidationError) as raised:
            validate_chain_state(
                {
                    "chain_type": "runtime",
                    "state": "symptom_observed",
                    "related_turn_ids": ["turn_unread"],
                    "missing_evidence": ["根因、修复和验证"],
                },
                current_turn_id="turn_current",
            )
        message = str(raised.exception)
        self.assertIn("was not read", message)
        self.assertIn("must include current turn_id", message)

    def test_transition_can_skip_forward_but_not_regress(self) -> None:
        validate_chain_transition(
            {
                "chain_type": "compilation",
                "state": "failure_observed",
                "related_turn_ids": ["turn_1"],
            },
            {
                "chain_type": "compilation",
                "state": "compile_verified",
                "related_turn_ids": ["turn_1", "turn_2"],
            },
        )
        with self.assertRaisesRegex(MemoryChainStateValidationError, "cannot move backward"):
            validate_chain_transition(
                {
                    "chain_type": "compilation",
                    "state": "fix_applied",
                    "related_turn_ids": ["turn_1", "turn_2"],
                },
                {
                    "chain_type": "compilation",
                    "state": "cause_identified",
                    "related_turn_ids": ["turn_1", "turn_2", "turn_3"],
                },
            )

    def test_fix_rejected_can_start_another_repair_attempt(self) -> None:
        validate_chain_transition(
            {
                "chain_type": "runtime",
                "state": "fix_rejected",
                "related_turn_ids": ["turn_1", "turn_2"],
            },
            {
                "chain_type": "runtime",
                "state": "cause_identified",
                "related_turn_ids": ["turn_1", "turn_2", "turn_3"],
            },
        )

    def test_transition_cannot_drop_old_turns(self) -> None:
        with self.assertRaisesRegex(MemoryChainStateValidationError, "cannot drop"):
            validate_chain_transition(
                {
                    "chain_type": "runtime",
                    "state": "cause_identified",
                    "related_turn_ids": ["turn_1", "turn_2"],
                },
                {
                    "chain_type": "runtime",
                    "state": "fix_applied",
                    "related_turn_ids": ["turn_2", "turn_3"],
                },
            )


if __name__ == "__main__":
    unittest.main()
