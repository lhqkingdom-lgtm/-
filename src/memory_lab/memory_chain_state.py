"""Validation helpers for persistent multi-turn technical memory chains.

Incomplete compile/runtime chains live on ``turns.metadata``. They are not
formal memories and therefore are not returned by the memory recall path.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


MEMORY_CHAIN_SCHEMA_VERSION = "memory-chain-state-v1"

CHAIN_STATES = {
    "none": ("not_applicable",),
    "compilation": (
        "failure_observed",
        "cause_identified",
        "fix_applied",
        "compile_verified",
    ),
    "runtime": (
        "symptom_observed",
        "cause_identified",
        "fix_applied",
        "waiting_user_validation",
        "user_verified",
        "fix_rejected",
    ),
}

CLOSED_STATES = {
    ("compilation", "compile_verified"),
    ("runtime", "user_verified"),
}


class MemoryChainStateValidationError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = [str(error) for error in errors]
        super().__init__("; ".join(self.errors))


def validate_chain_state(
    value: Any,
    *,
    current_turn_id: str,
    observed_turn_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate the model-facing chain state and return a normalized copy."""

    errors: list[str] = []
    if not isinstance(value, Mapping):
        raise MemoryChainStateValidationError(["chain_state must be an object"])

    required = {"chain_type", "state", "related_turn_ids", "missing_evidence"}
    allowed = required | {"chain_id"}
    extra = set(value) - allowed
    missing = required - set(value)
    if extra:
        errors.append(f"chain_state has unsupported fields: {sorted(extra)}")
    if missing:
        errors.append(f"chain_state missing fields: {sorted(missing)}")

    chain_type = str(value.get("chain_type") or "")
    state = str(value.get("state") or "")
    chain_id = value.get("chain_id")
    if chain_id is not None:
        chain_id = str(chain_id or "")
        if not chain_id.startswith("chain_") or len(chain_id) < 12:
            errors.append("chain_state.chain_id must be an existing chain_ id")
    if chain_type not in CHAIN_STATES:
        errors.append("chain_state.chain_type must be none, compilation or runtime")
    elif state not in CHAIN_STATES[chain_type]:
        errors.append(f"chain_state.state is invalid for chain_type {chain_type}")

    related = value.get("related_turn_ids")
    normalized_related: list[str] = []
    if not isinstance(related, list) or not related:
        errors.append("chain_state.related_turn_ids must be a non-empty array")
    elif len(related) > 12:
        errors.append("chain_state.related_turn_ids exceeds max 12")
    else:
        authorized = {str(current_turn_id), *(str(item) for item in observed_turn_ids)}
        seen: set[str] = set()
        for index, item in enumerate(related):
            turn_id = str(item or "")
            if not turn_id:
                errors.append(
                    f"chain_state.related_turn_ids[{index}] must be a non-empty string"
                )
                continue
            if turn_id in seen:
                errors.append(f"chain_state.related_turn_ids[{index}] is duplicated")
                continue
            seen.add(turn_id)
            normalized_related.append(turn_id)
            if turn_id not in authorized:
                errors.append(
                    f"chain_state.related_turn_ids[{index}] was not read by read_session_turn_evidence"
                )
        if str(current_turn_id) not in seen:
            errors.append("chain_state.related_turn_ids must include current turn_id")

    missing_evidence = value.get("missing_evidence")
    normalized_missing: list[str] = []
    if not isinstance(missing_evidence, list) or len(missing_evidence) > 10:
        errors.append("chain_state.missing_evidence must be an array with at most 10 items")
    else:
        if any(not isinstance(item, str) or not item.strip() for item in missing_evidence):
            errors.append("chain_state.missing_evidence items must be non-empty strings")
        normalized_missing = [str(item).strip() for item in missing_evidence if str(item).strip()]

    is_closed = (chain_type, state) in CLOSED_STATES
    if chain_type in {"compilation", "runtime"}:
        if is_closed and normalized_missing:
            errors.append("closed chain_state must not contain missing_evidence")
        if not is_closed and not normalized_missing:
            errors.append("open technical chain_state must describe missing_evidence")
    elif chain_type == "none" and normalized_missing:
        errors.append("chain_state with chain_type none must not contain missing_evidence")

    if errors:
        raise MemoryChainStateValidationError(errors)
    normalized = {
        "chain_type": chain_type,
        "state": state,
        "related_turn_ids": normalized_related,
        "missing_evidence": normalized_missing,
        "status": "closed" if is_closed else ("ignored" if chain_type == "none" else "open"),
    }
    if chain_id:
        normalized["chain_id"] = chain_id
    return normalized


def validate_chain_transition(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Reject regressions while allowing evidence-backed state skips and retries."""

    errors: list[str] = []
    previous_type = str(previous.get("chain_type") or "")
    current_type = str(current.get("chain_type") or "")
    previous_state = str(previous.get("state") or "")
    current_state = str(current.get("state") or "")
    if previous.get("chain_id") and current.get("chain_id"):
        if str(previous.get("chain_id")) != str(current.get("chain_id")):
            errors.append("chain_id cannot change for an existing chain")
    if previous_type != current_type:
        errors.append("chain_type cannot change for an existing chain")
    elif previous_type == "compilation":
        order = list(CHAIN_STATES["compilation"])
        if previous_state in order and current_state in order:
            if order.index(current_state) < order.index(previous_state):
                errors.append("compilation chain cannot move backward")
    elif previous_type == "runtime":
        if previous_state == "user_verified" and current_state != "user_verified":
            errors.append("closed runtime chain cannot be reopened; create a new chain")
        elif previous_state != "fix_rejected":
            order = [
                "symptom_observed",
                "cause_identified",
                "fix_applied",
                "waiting_user_validation",
                "user_verified",
            ]
            if previous_state in order and current_state in order:
                if order.index(current_state) < order.index(previous_state):
                    errors.append("runtime chain cannot move backward")
        # fix_rejected starts another repair attempt within the same phenomenon.
    previous_turns = {str(item) for item in previous.get("related_turn_ids") or []}
    current_turns = {str(item) for item in current.get("related_turn_ids") or []}
    if not previous_turns.issubset(current_turns):
        errors.append("related_turn_ids cannot drop turns from an existing chain")
    if errors:
        raise MemoryChainStateValidationError(errors)


__all__ = [
    "CHAIN_STATES",
    "CLOSED_STATES",
    "MEMORY_CHAIN_SCHEMA_VERSION",
    "MemoryChainStateValidationError",
    "validate_chain_state",
    "validate_chain_transition",
]
