"""AI writer/reviewer orchestration with an atomic formal-memory commit.

The two model roles are stateless and independent.  They can only emit strict
JSON; deterministic code validates evidence, scope, identity, ordering and CAS
before reusing the existing immutable proposal/revision machinery.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .config import DEFAULT_CONFIG, load_llm_config
from .consolidation import (
    CONSOLIDATOR_VERSION,
    _analyze_decision,
    _apply_proposal,
    _insert_or_get_proposal,
    _latest_admission_snapshot,
    _proposal_payload,
    _state,
    canonicalize_scope,
    semantic_value_for,
)
from .judge import (
    ApiJudge,
    JudgeError,
    JudgeValidationError,
    MemoryJudge,
    PROMPT_VERSION as JUDGE_PROMPT_VERSION,
    _assert_wire_text_safe,
    _wire_strings,
    request_hash as judge_request_hash,
    validate_judge_request,
    validate_judge_result,
)
from .llm_client import OpenAICompatibleJsonClient
from .safety import require_memory_lab_database
from .shadow import _request_for_row, formal_state_snapshot
from .store import now_iso, open_store


PIPELINE_VERSION = "auto-memory-pipeline-v2"
UPDATE_PLAN_SCHEMA_VERSION = "memory-update-plan-v1"
REVIEW_INPUT_SCHEMA_VERSION = "memory-review-input-v1"
REVIEW_DECISION_SCHEMA_VERSION = "memory-review-decision-v1"
WRITER_PROMPT_VERSION = "memory-writer-v1+" + JUDGE_PROMPT_VERSION
REVIEWER_PROMPT_VERSION = "memory-reviewer-v1"

UPDATE_OPERATIONS = {
    "reject",
    "create_entry",
    "merge_evidence",
    "add_scope_variant",
    "revise_entry",
    "resolve_user_contract_by_time",
    "open_engineering_conflict",
}
REVIEW_OUTCOMES = {"accept", "revise", "reject"}
FORMAL_SUCCESS_ACTIONS = {
    "created",
    "created_scope_variant",
    "merged_evidence",
    "confirmed_revision",
    "revised",
    "opened_conflict",
    "merged_conflict_evidence",
    "noop",
}

# Type is not a model-defined free-form label.  The stable-key family owns a
# small compatibility set and one canonical value for formal persistence.
_MEMORY_TYPE_POLICY: tuple[tuple[str, str, frozenset[str]], ...] = (
    (
        "preference.response.",
        "response_preference",
        frozenset({"preference", "response_preference"}),
    ),
    ("workflow.", "workflow_rule", frozenset({"workflow_rule"})),
    ("toolchain.", "project_rule", frozenset({"project_rule"})),
)

# Rule admission is intentionally conservative.  A durable user contract that
# it cannot name may be semantically classified by Writer + Reviewer, but only
# into this already-supported ontology.  Models cannot create new key families,
# values, types, or widen scope through this bridge.
_SEMANTIC_USER_CONTRACT_POLICY: dict[str, dict[str, frozenset[str]]] = {
    "preference.response.language": {
        "values": frozenset({"zh", "en"}),
        "scope_levels": frozenset({"global", "domain", "project"}),
    },
    "preference.response.structure": {
        "values": frozenset({"conclusion_then_details", "details_then_conclusion"}),
        "scope_levels": frozenset({"global", "domain", "project"}),
    },
    "workflow.code_generation.confirmation": {
        "values": frozenset({"required", "not_required"}),
        "scope_levels": frozenset({"global", "domain", "project"}),
    },
    "workflow.embedded.code_change.compile_check": {
        "values": frozenset({"required", "not_required"}),
        "scope_levels": frozenset({"domain", "project"}),
    },
    "toolchain.tg032plus.compiler": {
        "values": frozenset({"keil_c51", "iar", "cmake", "makefile"}),
        "scope_levels": frozenset({"project"}),
    },
}


class FormalPipelineError(RuntimeError):
    """The automatic pipeline failed closed before committing formal memory."""


class UpdatePlanValidationError(FormalPipelineError):
    """A Writer plan violated the frozen structured contract."""


class ReviewDecisionValidationError(FormalPipelineError):
    """A Reviewer decision violated the frozen structured contract."""


class FormalCommitError(FormalPipelineError):
    """The formal state or evidence changed before the atomic commit."""


class PlanWriter(Protocol):
    def write(self, request: Mapping[str, Any], *, family: str) -> dict[str, Any]:
        """Return one strict UpdatePlan for one atomic evidence claim."""


class PlanReviewer(Protocol):
    def review(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return accept, one revised plan, or reject."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_family(family: str) -> str:
    if family == "engineering_experience":
        return "engineering_memory"
    return family


def _canonical_memory_type(
    stable_key: str, admission_type: object, writer_type: object
) -> str:
    for prefix, canonical, allowed in _MEMORY_TYPE_POLICY:
        if stable_key.startswith(prefix):
            if admission_type not in allowed:
                raise FormalCommitError(
                    "admission memory_type is not allowed for the stable-key family"
                )
            if writer_type not in allowed:
                raise FormalCommitError(
                    "Writer memory_type is not allowed for the stable-key family"
                )
            return canonical
    raise FormalCommitError("stable key has no formal memory_type policy")


def _hash(value: Any, *, prefix: str) -> str:
    return prefix + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _strict_object(
    value: Any,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UpdatePlanValidationError(f"{name} must be an object")
    result = dict(value)
    missing = required - result.keys()
    extra = result.keys() - required - (optional or set())
    if missing:
        raise UpdatePlanValidationError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise UpdatePlanValidationError(f"{name} has unsupported fields: {sorted(extra)}")
    return result


def _operation_for_judgment(result: Mapping[str, Any], family: str) -> str:
    family = _canonical_family(family)
    if not result["should_store"]:
        return "reject"
    relation = result["relation"]["kind"]
    if relation == "new":
        return "create_entry"
    if relation in {"equivalent", "extension"}:
        return "merge_evidence"
    if relation == "different_scope":
        return "add_scope_variant"
    if relation == "correction":
        return "revise_entry"
    if relation == "conflict":
        return (
            "resolve_user_contract_by_time"
            if family == "user_contract"
            else "open_engineering_conflict"
        )
    raise UpdatePlanValidationError(f"stored result has unsupported relation: {relation}")


def judgment_to_update_plan(
    result: Mapping[str, Any], request: Mapping[str, Any], *, family: str
) -> dict[str, Any]:
    normalized = validate_judge_result(result, request)
    plan = {
        "schema_version": UPDATE_PLAN_SCHEMA_VERSION,
        "request_id": validate_judge_request(request)["request_id"],
        "operation": _operation_for_judgment(normalized, family),
        "should_store": normalized["should_store"],
        "memory_type": normalized["type"],
        "stable_key": normalized["key"],
        "scope": normalized["scope"],
        "value": normalized["value"],
        "relation": normalized["relation"],
        "evidence": normalized["evidence"],
        "confidence": normalized["confidence"],
        "reason": normalized["reason"],
    }
    return validate_update_plan(plan, request, family=family)


def validate_update_plan(
    plan: Mapping[str, Any], request: Mapping[str, Any], *, family: str
) -> dict[str, Any]:
    family = _canonical_family(family)
    if family not in {"user_contract", "engineering_memory"}:
        raise UpdatePlanValidationError("unsupported memory family")
    obj = _strict_object(
        plan,
        name="update_plan",
        required={
            "schema_version",
            "request_id",
            "operation",
            "should_store",
            "memory_type",
            "stable_key",
            "scope",
            "value",
            "relation",
            "evidence",
            "confidence",
            "reason",
        },
    )
    if obj["schema_version"] != UPDATE_PLAN_SCHEMA_VERSION:
        raise UpdatePlanValidationError("unsupported update plan schema_version")
    request_obj = validate_judge_request(request)
    if obj["request_id"] != request_obj["request_id"]:
        raise UpdatePlanValidationError("update plan request_id does not match")
    if obj["operation"] not in UPDATE_OPERATIONS:
        raise UpdatePlanValidationError("unsupported update plan operation")
    judgment = {
        "should_store": obj["should_store"],
        "type": obj["memory_type"],
        "key": obj["stable_key"],
        "scope": obj["scope"],
        "value": obj["value"],
        "relation": obj["relation"],
        "evidence": obj["evidence"],
        "confidence": obj["confidence"],
        "reason": obj["reason"],
    }
    try:
        normalized = validate_judge_result(judgment, request_obj)
    except JudgeError as error:
        raise UpdatePlanValidationError(str(error)) from error
    expected_operation = _operation_for_judgment(normalized, family)
    if obj["operation"] != expected_operation:
        raise UpdatePlanValidationError(
            f"operation {obj['operation']} does not match relation-derived {expected_operation}"
        )
    normalized_plan = {
        "schema_version": UPDATE_PLAN_SCHEMA_VERSION,
        "request_id": request_obj["request_id"],
        "operation": expected_operation,
        "should_store": normalized["should_store"],
        "memory_type": normalized["type"],
        "stable_key": normalized["key"],
        "scope": normalized["scope"],
        "value": normalized["value"],
        "relation": normalized["relation"],
        "evidence": normalized["evidence"],
        "confidence": normalized["confidence"],
        "reason": normalized["reason"],
    }
    for wire_text in _wire_strings(normalized_plan):
        try:
            _assert_wire_text_safe(wire_text)
        except JudgeValidationError as error:
            raise UpdatePlanValidationError(str(error)) from error
    return json.loads(_json(normalized_plan))


def update_plan_hash(plan: Mapping[str, Any]) -> str:
    return _hash(plan, prefix="update_plan_")


def build_review_request(
    writer_request: Mapping[str, Any],
    update_plan: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    family = _canonical_family(family)
    request_obj = validate_judge_request(writer_request)
    plan = validate_update_plan(update_plan, request_obj, family=family)
    return {
        "schema_version": REVIEW_INPUT_SCHEMA_VERSION,
        "request_id": "review_" + request_obj["request_id"],
        "family": family,
        "writer_request": request_obj,
        "update_plan": plan,
        "update_plan_hash": update_plan_hash(plan),
    }


def validate_review_request(request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        obj = _strict_object(
            request,
            name="review_request",
            required={
                "schema_version",
                "request_id",
                "family",
                "writer_request",
                "update_plan",
                "update_plan_hash",
            },
        )
        if obj["schema_version"] != REVIEW_INPUT_SCHEMA_VERSION:
            raise ReviewDecisionValidationError("unsupported review input schema_version")
        if not isinstance(obj["request_id"], str) or not obj["request_id"]:
            raise ReviewDecisionValidationError("review request_id must not be empty")
        writer_request = validate_judge_request(obj["writer_request"])
        plan = validate_update_plan(
            obj["update_plan"], writer_request, family=obj["family"]
        )
        expected_hash = update_plan_hash(plan)
        if obj["update_plan_hash"] != expected_hash:
            raise ReviewDecisionValidationError("review request plan hash does not match")
        return {
            "schema_version": REVIEW_INPUT_SCHEMA_VERSION,
            "request_id": obj["request_id"],
            "family": obj["family"],
            "writer_request": writer_request,
            "update_plan": plan,
            "update_plan_hash": expected_hash,
        }
    except ReviewDecisionValidationError:
        raise
    except (JudgeError, UpdatePlanValidationError) as error:
        raise ReviewDecisionValidationError(str(error)) from error


def validate_review_decision(
    decision: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    request_obj = validate_review_request(request)
    try:
        obj = _strict_object(
            decision,
            name="review_decision",
            required={
                "schema_version",
                "request_id",
                "reviewed_plan_hash",
                "outcome",
                "revised_plan",
                "confidence",
                "reason",
            },
        )
    except UpdatePlanValidationError as error:
        raise ReviewDecisionValidationError(str(error)) from error
    if obj["schema_version"] != REVIEW_DECISION_SCHEMA_VERSION:
        raise ReviewDecisionValidationError("unsupported review decision schema_version")
    if obj["request_id"] != request_obj["request_id"]:
        raise ReviewDecisionValidationError("review decision request_id does not match")
    if obj["reviewed_plan_hash"] != request_obj["update_plan_hash"]:
        raise ReviewDecisionValidationError("Reviewer did not bind the exact Writer plan")
    if obj["outcome"] not in REVIEW_OUTCOMES:
        raise ReviewDecisionValidationError("unsupported review outcome")
    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ReviewDecisionValidationError("review confidence must be between 0 and 1")
    if not isinstance(obj["reason"], str) or not obj["reason"].strip() or len(obj["reason"]) > 300:
        raise ReviewDecisionValidationError("review reason must contain 1..300 characters")
    revised_plan = obj["revised_plan"]
    if obj["outcome"] == "revise":
        try:
            revised_plan = validate_update_plan(
                revised_plan,
                request_obj["writer_request"],
                family=request_obj["family"],
            )
        except (JudgeError, UpdatePlanValidationError) as error:
            raise ReviewDecisionValidationError(
                f"Reviewer revised_plan is invalid: {error}"
            ) from error
        if update_plan_hash(revised_plan) == request_obj["update_plan_hash"]:
            raise ReviewDecisionValidationError(
                "revise outcome must materially change the Writer plan"
            )
    elif revised_plan is not None:
        raise ReviewDecisionValidationError(
            "accept/reject outcome requires revised_plan=null"
        )
    normalized = {
        "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
        "request_id": request_obj["request_id"],
        "reviewed_plan_hash": request_obj["update_plan_hash"],
        "outcome": obj["outcome"],
        "revised_plan": revised_plan,
        "confidence": float(confidence),
        "reason": obj["reason"].strip(),
    }
    for wire_text in _wire_strings(normalized):
        try:
            _assert_wire_text_safe(wire_text)
        except JudgeValidationError as error:
            raise ReviewDecisionValidationError(str(error)) from error
    return json.loads(_json(normalized))


@dataclass(frozen=True)
class JudgePlanWriter:
    """Treat the proven judge contract as the semantic Writer boundary."""

    judge: MemoryJudge

    def write(self, request: Mapping[str, Any], *, family: str) -> dict[str, Any]:
        request_obj = validate_judge_request(request)
        result = self.judge.judge(request_obj)
        return judgment_to_update_plan(result, request_obj, family=family)


REVIEWER_SYSTEM_PROMPT = """You are the independent memory-review Agent.
Review one Writer plan against only the supplied atomic quote, hard vetoes,
scope limits, current-memory targets and deterministic policy.  Do not invent
evidence, IDs, project scope, facts or hidden reasoning.  Return JSON only.

Use accept when the exact plan is safe and correct.  Use reject when no memory
should be committed.  Use revise only once, supplying one complete replacement
plan that obeys the same schema and exact evidence span.  For same-scope durable
user contracts with opposite values, the newer database-ordered requirement is
resolved by resolve_user_contract_by_time; that policy never applies to
engineering experience.  Give only a short audit reason."""


@dataclass(frozen=True)
class JsonPlanReviewer:
    client: OpenAICompatibleJsonClient

    @staticmethod
    def _public_request(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        request_obj = validate_review_request(request)
        writer_public, reverse = ApiJudge._provider_projection(
            request_obj["writer_request"]
        )
        actual_to_alias = {actual: alias for alias, actual in reverse.items()}
        plan = json.loads(_json(request_obj["update_plan"]))
        plan["request_id"] = "request_0"
        for evidence in plan["evidence"]:
            for field in ("event_id", "part_id"):
                evidence[field] = actual_to_alias.get(evidence[field], evidence[field])
        relation = plan["relation"]
        if relation.get("target_id") is not None:
            relation["target_id"] = actual_to_alias.get(
                relation["target_id"], relation["target_id"]
            )
        if isinstance(plan.get("scope"), dict) and plan["scope"].get("level") == "project":
            plan["scope"]["project_key"] = actual_to_alias.get(
                plan["scope"]["project_key"], plan["scope"]["project_key"]
            )
        public = {
            "schema_version": REVIEW_INPUT_SCHEMA_VERSION,
            "request_id": "review_0",
            "review_token": request_obj["update_plan_hash"],
            "family": request_obj["family"],
            "writer_request": writer_public,
            "update_plan": plan,
            "required_output": {
                "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
                "request_id": "review_0",
                "reviewed_plan_hash": request_obj["update_plan_hash"],
                "outcome": "accept|revise|reject",
                "revised_plan": "complete update_plan object only for revise; otherwise null",
                "confidence": "number 0..1",
                "reason": "short audit reason",
            },
        }
        return public, reverse

    @staticmethod
    def _restore_decision(
        decision: Mapping[str, Any],
        request: Mapping[str, Any],
        reverse: Mapping[str, str],
    ) -> dict[str, Any]:
        restored = json.loads(_json(decision))
        restored["request_id"] = request["request_id"]
        plan = restored.get("revised_plan")
        if isinstance(plan, dict):
            plan["request_id"] = request["writer_request"]["request_id"]
            for evidence in plan.get("evidence", []):
                if isinstance(evidence, dict):
                    for field in ("event_id", "part_id"):
                        if evidence.get(field) in reverse:
                            evidence[field] = reverse[evidence[field]]
            relation = plan.get("relation")
            if isinstance(relation, dict) and relation.get("target_id") in reverse:
                relation["target_id"] = reverse[relation["target_id"]]
            scope = plan.get("scope")
            if isinstance(scope, dict) and scope.get("project_key") in reverse:
                scope["project_key"] = reverse[scope["project_key"]]
        return restored

    def review(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_obj = validate_review_request(request)
        public, reverse = self._public_request(request_obj)
        for text in _wire_strings(public):
            _assert_wire_text_safe(text)
        raw = self.client.complete_json(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_payload=public,
            max_tokens=2200,
        )
        restored = self._restore_decision(raw, request_obj, reverse)
        return validate_review_decision(restored, request_obj)


def _safe_error(error: Exception) -> tuple[str, str]:
    return type(error).__name__[:80], str(error)[:500]


def _latest_attempt_is_snapshot(
    connection: sqlite3.Connection, admission_run_id: str
) -> bool:
    latest = connection.execute(
        """
        SELECT run_id, status, completed_at FROM admission_run
        ORDER BY started_at DESC, rowid DESC LIMIT 1
        """
    ).fetchone()
    return bool(
        latest
        and latest["run_id"] == admission_run_id
        and latest["status"] == "complete"
        and latest["completed_at"] is not None
    )


def _insert_writer_result(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    admission_run_id: str,
    row: sqlite3.Row,
    digest: str,
    provider: str,
    model: str,
    status: str,
    execution_mode: str,
    plan: Mapping[str, Any] | None = None,
    error: Exception | None = None,
    replayed_from: str | None = None,
) -> sqlite3.Row:
    result_id = "writer_" + uuid4().hex
    error_code, error_message = _safe_error(error) if error else (None, None)
    plan_hash = update_plan_hash(plan) if plan else None
    connection.execute(
        """
        INSERT INTO memory_writer_result(
            writer_result_id, run_id, admission_run_id, decision_id,
            candidate_key, request_hash, plan_hash, provider, model,
            prompt_version, schema_version, status, execution_mode,
            plan_json, error_code, error_message,
            replayed_from_writer_result_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result_id,
            run_id,
            admission_run_id,
            row["decision_id"],
            row["candidate_key"],
            digest,
            plan_hash,
            provider,
            model,
            WRITER_PROMPT_VERSION,
            UPDATE_PLAN_SCHEMA_VERSION,
            status,
            execution_mode,
            _json(plan) if plan else None,
            error_code,
            error_message,
            replayed_from,
            now_iso(),
        ),
    )
    return connection.execute(
        "SELECT * FROM memory_writer_result WHERE writer_result_id = ?", (result_id,)
    ).fetchone()


def _insert_reviewer_result(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    writer_result: sqlite3.Row,
    admission_run_id: str,
    row: sqlite3.Row,
    digest: str,
    reviewed_plan_hash: str,
    provider: str,
    model: str,
    status: str,
    execution_mode: str,
    decision: Mapping[str, Any] | None = None,
    final_plan: Mapping[str, Any] | None = None,
    error: Exception | None = None,
    replayed_from: str | None = None,
) -> sqlite3.Row:
    result_id = "reviewer_" + uuid4().hex
    error_code, error_message = _safe_error(error) if error else (None, None)
    connection.execute(
        """
        INSERT INTO memory_reviewer_result(
            reviewer_result_id, run_id, writer_result_id, admission_run_id,
            decision_id, candidate_key, request_hash, reviewed_plan_hash,
            final_plan_hash, provider, model, prompt_version, schema_version,
            status, execution_mode, outcome, decision_json, final_plan_json,
            error_code, error_message, replayed_from_reviewer_result_id,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result_id,
            run_id,
            writer_result["writer_result_id"],
            admission_run_id,
            row["decision_id"],
            row["candidate_key"],
            digest,
            reviewed_plan_hash,
            update_plan_hash(final_plan) if final_plan else None,
            provider,
            model,
            REVIEWER_PROMPT_VERSION,
            REVIEW_DECISION_SCHEMA_VERSION,
            status,
            execution_mode,
            decision.get("outcome") if decision else None,
            _json(decision) if decision else None,
            _json(final_plan) if final_plan else None,
            error_code,
            error_message,
            replayed_from,
            now_iso(),
        ),
    )
    return connection.execute(
        "SELECT * FROM memory_reviewer_result WHERE reviewer_result_id = ?",
        (result_id,),
    ).fetchone()


def _cached_writer(
    connection: sqlite3.Connection, digest: str, model: str
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM memory_writer_result
        WHERE request_hash = ? AND model = ? AND prompt_version = ?
          AND schema_version = ? AND status IN ('valid', 'replayed')
          AND plan_json IS NOT NULL
        ORDER BY created_at DESC, rowid DESC LIMIT 1
        """,
        (digest, model, WRITER_PROMPT_VERSION, UPDATE_PLAN_SCHEMA_VERSION),
    ).fetchone()


def _cached_reviewer(
    connection: sqlite3.Connection, digest: str, model: str
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM memory_reviewer_result
        WHERE request_hash = ? AND model = ? AND prompt_version = ?
          AND schema_version = ? AND status IN ('valid', 'replayed')
          AND decision_json IS NOT NULL
        ORDER BY created_at DESC, rowid DESC LIMIT 1
        """,
        (digest, model, REVIEWER_PROMPT_VERSION, REVIEW_DECISION_SCHEMA_VERSION),
    ).fetchone()


def _compile_plan(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    snapshot_hash: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_review = row["disposition"] == "needs_semantic_review"
    reviewed_identity = None
    if semantic_review:
        policy = _SEMANTIC_USER_CONTRACT_POLICY.get(plan["stable_key"])
        if row["family"] != "user_contract" or policy is None:
            raise FormalCommitError(
                "semantic review may only select a bounded user-contract stable key"
            )
        if plan["value"]["normalized"] not in policy["values"]:
            raise FormalCommitError(
                "semantic review value is outside the bounded stable-key ontology"
            )
        if plan["scope"]["level"] not in policy["scope_levels"]:
            raise FormalCommitError(
                "semantic review scope is not allowed for the selected stable key"
            )
        # With no rule identity the Writer request contains no identity target.
        # The model classifies semantics; current-state relation is recomputed
        # below under the transaction instead of trusting a guessed target.
        if plan["operation"] != "create_entry" or plan["relation"] != {
            "kind": "new",
            "target_kind": None,
            "target_id": None,
        }:
            raise FormalCommitError(
                "semantic review must leave relation ownership to deterministic code"
            )
        reviewed_identity = {
            "stable_key": plan["stable_key"],
            "semantic_value": plan["value"]["normalized"],
            "memory_type": plan["memory_type"],
        }

    spec, error = _analyze_decision(
        connection,
        row,
        snapshot_hash,
        reviewed_identity=reviewed_identity,
    )
    if spec is None:
        raise FormalCommitError(f"reviewed plan is no longer actionable: {error}")
    if plan["stable_key"] != spec["stable_key"]:
        raise FormalCommitError("Writer stable_key disagrees with exact local evidence")
    if plan["value"]["normalized"] != spec["semantic_value"]:
        raise FormalCommitError("Writer value disagrees with exact local evidence")
    canonical_plan_scope = canonicalize_scope(
        plan["scope"], event_project_key=row["project_key"], connection=connection
    )
    if canonical_plan_scope is None or _json(canonical_plan_scope) != spec["canonical_scope_json"]:
        raise FormalCommitError("Writer scope disagrees with canonical local scope")
    admission_type = plan["memory_type"] if semantic_review else row["memory_type"]
    canonical_memory_type = _canonical_memory_type(
        spec["stable_key"], admission_type, plan["memory_type"]
    )
    payload = json.loads(spec["application_payload_json"])
    if not isinstance(payload, dict):
        raise FormalCommitError("proposal application payload is not an object")
    payload["memory_type"] = canonical_memory_type
    spec = dict(spec)
    spec["application_payload_json"] = _json(payload)

    relation = spec["relation"]
    expected_operation = {
        "create": "create_entry",
        "create_scope_variant": "add_scope_variant",
        "merge": "merge_evidence",
        "confirm": "merge_evidence",
        "revise": "revise_entry",
        "conflict": "open_engineering_conflict",
    }.get(relation)
    if relation == "conflict" and row["family"] == "user_contract":
        # Only a reviewed, durable, same-scope user contract reaches this point.
        # Source ordering is frozen by latest_admission_snapshot; applying plans
        # in that order means the later requirement deterministically wins.
        spec = dict(spec)
        spec["relation"] = "revise"
        expected_operation = "resolve_user_contract_by_time"
    if expected_operation is None or (
        not semantic_review and plan["operation"] != expected_operation
    ):
        raise FormalCommitError(
            f"Writer operation {plan['operation']} disagrees with current relation {relation}"
        )
    return spec


def _semantic_review_is_model_eligible(row: sqlite3.Row) -> bool:
    """Permit only durable, non-vetoed user contracts into semantic review."""

    if row["disposition"] != "needs_semantic_review" or row["family"] != "user_contract":
        return False
    if row["stable_key_suggestion"] is not None or row["memory_type"] is not None:
        return False
    try:
        signals = json.loads(row["signals_json"])
        coverage = json.loads(row["instruction_coverage_json"])
        raw_scope = json.loads(row["scope_json"])
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(signals, dict) or not signals.get("long_term"):
        return False
    if signals.get("temporary") or signals.get("sensitive"):
        return False
    if not isinstance(coverage, dict) or coverage.get("status") == "full":
        return False
    canonical_scope = canonicalize_scope(raw_scope, event_project_key=row["project_key"])
    return canonical_scope is not None


def _existing_evidence_state(
    connection: sqlite3.Connection, row: sqlite3.Row
) -> str | None:
    """Return same/reinterpreted for an already committed candidate key."""

    evidence = connection.execute(
        """
        SELECT m.stable_key, m.scope_json, e.event_id, e.part_id,
               e.project_key, e.exact_quote, e.ordinal, e.start_offset, e.end_offset,
               COALESCE(r.semantic_value, c.incoming_semantic_value) AS semantic_value
        FROM evidence AS e
        JOIN memory AS m ON m.memory_id = e.memory_id
        LEFT JOIN memory_revision AS r ON r.revision_id = e.revision_id
        LEFT JOIN conflict AS c ON c.conflict_id = e.conflict_id
        WHERE e.candidate_key = ?
        """,
        (row["candidate_key"],),
    ).fetchone()
    if evidence is None:
        return None
    if row["disposition"] == "needs_semantic_review":
        # A committed candidate has already passed both model roles and the
        # bounded semantic bridge.  Replays are no-ops only when the immutable
        # source coordinates and quote still match; changed evidence fails.
        expected = (
            row["event_id"],
            row["part_id"],
            row["project_key"],
            row["exact_quote"],
            row["ordinal"],
            row["start_offset"],
            row["end_offset"],
        )
        actual = tuple(
            evidence[field]
            for field in (
                "event_id",
                "part_id",
                "project_key",
                "exact_quote",
                "ordinal",
                "start_offset",
                "end_offset",
            )
        )
        return "same" if actual == expected else "reinterpreted"
    try:
        raw_scope = json.loads(row["scope_json"])
    except (TypeError, json.JSONDecodeError):
        return "reinterpreted"
    canonical_scope = canonicalize_scope(
        raw_scope, event_project_key=row["project_key"], connection=connection
    )
    semantic_value = semantic_value_for(
        str(row["stable_key_suggestion"] or ""), row["exact_quote"]
    )
    if (
        canonical_scope is not None
        and evidence["stable_key"] == row["stable_key_suggestion"]
        and evidence["scope_json"] == _json(canonical_scope)
        and evidence["semantic_value"] == semantic_value
    ):
        return "same"
    return "reinterpreted"


def _engineering_candidate_is_verified(row: sqlite3.Row) -> bool:
    if _canonical_family(row["family"]) != "engineering_memory":
        return True
    try:
        signals = json.loads(row["signals_json"])
    except (TypeError, json.JSONDecodeError):
        return False
    status = (
        signals.get("verification_status")
        or signals.get("domain_verification_outcome")
        or signals.get("verification")
    )
    contested = bool(
        signals.get("contested")
        or signals.get("unresolved_conflict")
        or signals.get("verification_conflict")
    )
    return status == "verified" and not contested


def _fail_run(
    connection: sqlite3.Connection,
    run_id: str,
    error: Exception,
    before: Mapping[str, Any],
) -> dict[str, Any]:
    code, message = _safe_error(error)
    after = formal_state_snapshot(connection)
    connection.execute(
        """
        UPDATE auto_memory_run
        SET completed_at = ?, status = 'failed', formal_state_after_json = ?,
            error_code = ?, error_message = ?
        WHERE run_id = ?
        """,
        (now_iso(), _json(after), code, message, run_id),
    )
    connection.commit()
    return {
        "run_id": run_id,
        "status": "failed",
        "error_code": code,
        "error_message": message,
        "formal_state_before": dict(before),
        "formal_state_after": after,
        "formal_state_unchanged": dict(before) == after,
    }


def run_auto_memory_pipeline(
    lab_database: Path,
    writer: PlanWriter,
    reviewer: PlanReviewer,
    *,
    provider: str,
    model: str,
    refresh: bool = False,
) -> dict[str, Any]:
    """Run independent Writer/Reviewer calls, then commit the batch atomically."""

    require_memory_lab_database(lab_database)
    run_id = "automemory_" + uuid4().hex
    with closing(open_store(lab_database)) as connection:
        admission, rows, snapshot_hash = _latest_admission_snapshot(connection)
        before = formal_state_snapshot(connection)
        if admission is None or snapshot_hash is None:
            raise FormalPipelineError("no structurally complete admission snapshot exists")
        if not _latest_attempt_is_snapshot(connection, admission["run_id"]):
            raise FormalPipelineError(
                "latest admission attempt is not the complete snapshot; rerun admission"
            )
        connection.execute(
            """
            INSERT INTO auto_memory_run(
                run_id, pipeline_version, admission_run_id,
                admission_snapshot_hash, provider, model,
                writer_prompt_version, reviewer_prompt_version,
                writer_schema_version, reviewer_schema_version,
                started_at, status, candidate_count, formal_state_before_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                run_id,
                PIPELINE_VERSION,
                admission["run_id"],
                snapshot_hash,
                provider,
                model,
                WRITER_PROMPT_VERSION,
                REVIEWER_PROMPT_VERSION,
                UPDATE_PLAN_SCHEMA_VERSION,
                REVIEW_DECISION_SCHEMA_VERSION,
                now_iso(),
                len(rows),
                _json(before),
            ),
        )
        connection.commit()

        reviewed: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        prior_rule_targets: list[dict[str, Any]] = []
        replay_count = 0
        for row in rows:
            existing_evidence = _existing_evidence_state(connection, row)
            if existing_evidence == "reinterpreted":
                error = FormalCommitError(
                    "candidate_key already belongs to different formal evidence semantics"
                )
                return {
                    **_fail_run(connection, run_id, error, before),
                    "admission_run_id": admission["run_id"],
                    "candidate_count": len(rows),
                    "audit": audit,
                }
            if existing_evidence == "same":
                digest = "existing_evidence_" + hashlib.sha256(
                    row["candidate_key"].encode("utf-8")
                ).hexdigest()
                writer_row = _insert_writer_result(
                    connection,
                    run_id=run_id,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=digest,
                    provider=provider,
                    model=model,
                    status="valid",
                    execution_mode="local_block",
                )
                reviewer_row = _insert_reviewer_result(
                    connection,
                    run_id=run_id,
                    writer_result=writer_row,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=digest,
                    reviewed_plan_hash=digest,
                    provider=provider,
                    model=model,
                    status="valid",
                    execution_mode="local_block",
                    decision={
                        "outcome": "accept",
                        "reason": "candidate_already_consolidated",
                    },
                )
                connection.commit()
                reviewed.append(
                    {
                        "row": row,
                        "outcome": "accept",
                        "final_plan": None,
                        "reviewer_result": reviewer_row,
                    }
                )
                audit.append(
                    {
                        "candidate_key": row["candidate_key"],
                        "writer": "local_block",
                        "reviewer": "local_block",
                        "outcome": "noop",
                    }
                )
                continue

            local_disposition = None
            model_eligible_dispositions = {"eligible", "needs_correction_review"}
            if _semantic_review_is_model_eligible(row):
                model_eligible_dispositions.add("needs_semantic_review")
            if row["disposition"] not in model_eligible_dispositions:
                local_disposition = row["disposition"]
            elif not _engineering_candidate_is_verified(row):
                local_disposition = "engineering_not_verified_or_contested"
            if local_disposition is not None:
                digest = "local_block_" + hashlib.sha256(
                    f"{row['decision_id']}\0{local_disposition}".encode("utf-8")
                ).hexdigest()
                writer_row = _insert_writer_result(
                    connection,
                    run_id=run_id,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=digest,
                    provider=provider,
                    model=model,
                    status="valid",
                    execution_mode="local_block",
                )
                decision = {
                    "outcome": "reject",
                    "reason": "local_hard_filter:" + local_disposition,
                }
                reviewer_row = _insert_reviewer_result(
                    connection,
                    run_id=run_id,
                    writer_result=writer_row,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=digest,
                    reviewed_plan_hash=digest,
                    provider=provider,
                    model=model,
                    status="valid",
                    execution_mode="local_block",
                    decision=decision,
                )
                connection.commit()
                reviewed.append(
                    {
                        "row": row,
                        "outcome": "reject",
                        "final_plan": None,
                        "reviewer_result": reviewer_row,
                    }
                )
                audit.append(
                    {
                        "candidate_key": row["candidate_key"],
                        "writer": "local_block",
                        "reviewer": "local_block",
                        "outcome": "reject",
                    }
                )
                continue

            writer_row: sqlite3.Row | None = None
            plan: dict[str, Any] | None = None
            final_plan: dict[str, Any] | None = None
            digest = "invalid_" + row["decision_id"]
            review_digest = "invalid_" + row["decision_id"]
            try:
                writer_request = _request_for_row(connection, row, prior_rule_targets)
                digest = judge_request_hash(writer_request)
                cached_writer = None if refresh else _cached_writer(connection, digest, model)
                if cached_writer is not None:
                    plan = validate_update_plan(
                        json.loads(cached_writer["plan_json"]),
                        writer_request,
                        family=row["family"],
                    )
                    writer_row = _insert_writer_result(
                        connection,
                        run_id=run_id,
                        admission_run_id=admission["run_id"],
                        row=row,
                        digest=digest,
                        provider=provider,
                        model=model,
                        status="replayed",
                        execution_mode="cache_replay",
                        plan=plan,
                        replayed_from=cached_writer["writer_result_id"],
                    )
                    replay_count += 1
                    writer_mode = "cache_replay"
                else:
                    plan = validate_update_plan(
                        writer.write(writer_request, family=row["family"]),
                        writer_request,
                        family=row["family"],
                    )
                    writer_row = _insert_writer_result(
                        connection,
                        run_id=run_id,
                        admission_run_id=admission["run_id"],
                        row=row,
                        digest=digest,
                        provider=provider,
                        model=model,
                        status="valid",
                        execution_mode="writer_call",
                        plan=plan,
                    )
                    writer_mode = "writer_call"
                connection.commit()

                review_request = build_review_request(
                    writer_request, plan, family=row["family"]
                )
                review_digest = _hash(review_request, prefix="review_req_")
                cached_review = (
                    None if refresh else _cached_reviewer(connection, review_digest, model)
                )
                if cached_review is not None:
                    review_decision = validate_review_decision(
                        json.loads(cached_review["decision_json"]), review_request
                    )
                    review_mode = "cache_replay"
                    replay_count += 1
                    replayed_from = cached_review["reviewer_result_id"]
                else:
                    review_decision = validate_review_decision(
                        reviewer.review(review_request), review_request
                    )
                    review_mode = "reviewer_call"
                    replayed_from = None
                final_plan = (
                    plan
                    if review_decision["outcome"] == "accept"
                    else review_decision["revised_plan"]
                    if review_decision["outcome"] == "revise"
                    else None
                )
                reviewer_row = _insert_reviewer_result(
                    connection,
                    run_id=run_id,
                    writer_result=writer_row,
                    admission_run_id=admission["run_id"],
                    row=row,
                    digest=review_digest,
                    reviewed_plan_hash=review_request["update_plan_hash"],
                    provider=provider,
                    model=model,
                    status="replayed" if cached_review is not None else "valid",
                    execution_mode=review_mode,
                    decision=review_decision,
                    final_plan=final_plan,
                    replayed_from=replayed_from,
                )
                connection.commit()
                reviewed.append(
                    {
                        "row": row,
                        "outcome": review_decision["outcome"],
                        "final_plan": final_plan,
                        "reviewer_result": reviewer_row,
                    }
                )
                audit.append(
                    {
                        "candidate_key": row["candidate_key"],
                        "writer": writer_mode,
                        "reviewer": review_mode,
                        "outcome": review_decision["outcome"],
                        "operation": final_plan["operation"] if final_plan else None,
                    }
                )
            except Exception as error:
                if writer_row is None:
                    _insert_writer_result(
                        connection,
                        run_id=run_id,
                        admission_run_id=admission["run_id"],
                        row=row,
                        digest=digest,
                        provider=provider,
                        model=model,
                        status="invalid",
                        execution_mode="writer_call",
                        error=error,
                    )
                else:
                    _insert_reviewer_result(
                        connection,
                        run_id=run_id,
                        writer_result=writer_row,
                        admission_run_id=admission["run_id"],
                        row=row,
                        digest=review_digest,
                        reviewed_plan_hash=(
                            update_plan_hash(plan) if plan is not None else "invalid"
                        ),
                        provider=provider,
                        model=model,
                        status="invalid",
                        execution_mode="reviewer_call",
                        error=error,
                    )
                connection.commit()
                return {
                    **_fail_run(connection, run_id, error, before),
                    "admission_run_id": admission["run_id"],
                    "candidate_count": len(rows),
                    "audit": audit,
                }
            if final_plan is not None and final_plan["should_store"]:
                prior_rule_targets.append(
                    {
                        "proposal_id": "reviewed_target_" + row["decision_id"],
                        "stable_key": final_plan["stable_key"],
                        "scope": final_plan["scope"],
                        "value": final_plan["value"]["normalized"],
                        "statement": final_plan["value"]["statement"],
                    }
                )

        # All model decisions are complete before the first formal-memory write.
        # BEGIN IMMEDIATE plus full-row fingerprints creates one batch CAS point.
        try:
            connection.execute("BEGIN IMMEDIATE")
            current_admission, current_rows, current_hash = _latest_admission_snapshot(
                connection
            )
            if (
                current_admission is None
                or current_admission["run_id"] != admission["run_id"]
                or current_hash != snapshot_hash
                or [item["decision_id"] for item in current_rows]
                != [item["decision_id"] for item in rows]
            ):
                raise FormalCommitError("admission evidence changed during model review")
            if formal_state_snapshot(connection) != before:
                raise FormalCommitError("formal memory CAS changed during model review")

            consolidation_run_id = "consolidation_" + uuid4().hex
            connection.execute(
                """
                INSERT INTO consolidation_run(
                    run_id, consolidator_version, admission_run_id,
                    admission_snapshot_hash, started_at, status, candidate_count
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    consolidation_run_id,
                    CONSOLIDATOR_VERSION + "+auto-v2",
                    admission["run_id"],
                    snapshot_hash,
                    now_iso(),
                    len(rows),
                ),
            )
            actions: list[dict[str, Any]] = []
            for item in reviewed:
                plan = item["final_plan"]
                if item["outcome"] == "reject" or plan is None or not plan["should_store"]:
                    continue
                row = item["row"]
                spec = _compile_plan(connection, row, snapshot_hash, plan)
                proposal, _created = _insert_or_get_proposal(connection, spec)
                review_id = "review_" + uuid4().hex
                reviewer_name = f"ai-reviewer:{provider}:{model}"[:240]
                connection.execute(
                    """
                    INSERT INTO candidate_review(
                        review_id, proposal_id, decision_id, candidate_key,
                        outcome, reviewer, note, reviewed_at
                    ) VALUES (?, ?, ?, ?, 'approve', ?, ?, ?)
                    """,
                    (
                        review_id,
                        proposal["proposal_id"],
                        proposal["decision_id"],
                        proposal["candidate_key"],
                        reviewer_name,
                        _json(
                            {
                                "auto_memory_run_id": run_id,
                                "reviewer_result_id": item["reviewer_result"][
                                    "reviewer_result_id"
                                ],
                                "outcome": item["outcome"],
                                "final_plan_hash": update_plan_hash(plan),
                            }
                        ),
                        now_iso(),
                    ),
                )
                review_row = connection.execute(
                    "SELECT * FROM candidate_review WHERE review_id = ?", (review_id,)
                ).fetchone()
                action = _apply_proposal(
                    connection, consolidation_run_id, row, proposal, review_row
                )
                if action["action"] not in FORMAL_SUCCESS_ACTIONS:
                    raise FormalCommitError(
                        f"deterministic apply failed: {action['action']}:{action['reason']}"
                    )
                actions.append(action)

            action_counts = dict(sorted(Counter(item["action"] for item in actions).items()))
            connection.execute(
                """
                UPDATE consolidation_run
                SET completed_at = ?, status = 'complete', counts_json = ?, error = NULL
                WHERE run_id = ?
                """,
                (now_iso(), _json(action_counts), consolidation_run_id),
            )
            after = formal_state_snapshot(connection)
            review_counts = Counter(item["outcome"] for item in reviewed)
            run_status = (
                "complete_with_rejections" if review_counts.get("reject") else "complete"
            )
            counts = {
                "review": dict(sorted(review_counts.items())),
                "actions": action_counts,
            }
            connection.execute(
                """
                UPDATE auto_memory_run
                SET completed_at = ?, status = ?, accepted_count = ?,
                    revised_count = ?, rejected_count = ?, replay_count = ?,
                    counts_json = ?, formal_state_after_json = ?,
                    error_code = NULL, error_message = NULL
                WHERE run_id = ?
                """,
                (
                    now_iso(),
                    run_status,
                    review_counts.get("accept", 0),
                    review_counts.get("revise", 0),
                    review_counts.get("reject", 0),
                    replay_count,
                    _json(counts),
                    _json(after),
                    run_id,
                ),
            )
            connection.commit()
        except Exception as error:
            connection.rollback()
            return {
                **_fail_run(connection, run_id, error, before),
                "admission_run_id": admission["run_id"],
                "candidate_count": len(rows),
                "audit": audit,
            }

    return {
        "run_id": run_id,
        "status": run_status,
        "admission_run_id": admission["run_id"],
        "admission_snapshot_hash": snapshot_hash,
        "candidate_count": len(rows),
        "counts": counts,
        "actions": actions,
        "audit": audit,
        "formal_state_before": before,
        "formal_state_after": after,
        "formal_state_changed": before != after,
        "state": _state_snapshot(lab_database),
    }


def _state_snapshot(lab_database: Path) -> dict[str, Any]:
    with closing(open_store(lab_database)) as connection:
        return _state(connection)


def run_deepseek_auto_memory(
    lab_database: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    refresh: bool = False,
) -> dict[str, Any]:
    config = load_llm_config(config_path)
    if config.provider != "deepseek":
        raise ValueError("the configured provider is not deepseek")
    return run_auto_memory_pipeline(
        lab_database,
        JudgePlanWriter(ApiJudge.from_config(config_path)),
        JsonPlanReviewer(OpenAICompatibleJsonClient.from_config(config_path)),
        provider=config.provider,
        model=config.model,
        refresh=refresh,
    )
