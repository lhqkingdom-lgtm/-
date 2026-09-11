"""Atomic bridge from reviewed unified candidates to the formal memory store.

The bridge intentionally reuses the existing immutable ``memory_proposal`` /
``candidate_review`` audit chain and the existing ``memory``, revision,
evidence, conflict, consolidation-action and rollback machinery.  It does not
create a second authoritative memory store.

Review happens before this module is called.  The review result is validated
and frozen together with a complete formal-state fingerprint.  Commit then
uses ``BEGIN IMMEDIATE`` and rejects the whole batch if that fingerprint has
changed.  Model action hints are ignored; every relation is recomputed from
the normalized candidate and current formal rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .consolidation import (
    CONSOLIDATOR_VERSION,
    SCOPE_SCHEMA_VERSION,
    _apply_proposal,
    _hash,
    _insert_or_get_proposal,
    _memory_for_identity,
    _open_conflicts_hash,
    _state,
)
from .safety import require_memory_lab_database
from .shadow import formal_state_snapshot
from .source import open_read_only
from .store import now_iso, open_store
from .prepare import project_identity
from .unified_memory import (
    CANDIDATE_SCHEMA_VERSION,
    POLICY_VERSION,
    MemoryCandidateValidationError,
    MemoryEvidence,
    normalize_evidence_catalog,
    validate_memory_candidate,
)


PREPARED_SCHEMA_VERSION = "memory-lab-unified-commit-prepared-v1"
REVIEWED_PLAN_SCHEMA_VERSION = "memory-lab-unified-reviewed-plan-v1"
NORMALIZED_PLAN_SCHEMA_VERSION = "memory-lab-unified-plan-v1"
COMMIT_VERSION = "unified-formal-commit-v1"
REPLAY_METADATA_SCHEMA_VERSION = "memory-lab-unified-commit-replay-v1"

_COMMIT_SCOPE_LEVELS = frozenset({"global", "domain", "project"})
_USER_CONTRACT_TYPES = frozenset({"user", "preference", "workflow"})
_EXPLICIT_USER_SOURCE_KINDS = frozenset({"user_message", "user_answer"})
_SUCCESS_ACTIONS = frozenset(
    {
        "created",
        "created_scope_variant",
        "merged_evidence",
        "revised",
        "opened_conflict",
        "merged_conflict_evidence",
        "noop",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,511}$")


class UnifiedCommitError(RuntimeError):
    """Base error for the reviewed-candidate formal bridge."""


class UnifiedCommitValidationError(UnifiedCommitError):
    """The review result or prepared batch is malformed or unsupported."""


class UnifiedCommitCASMismatch(UnifiedCommitError):
    """Formal memory changed after the reviewed batch was prepared."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any, prefix: str) -> str:
    return prefix + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _strict_object(
    value: Any,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UnifiedCommitValidationError(f"{name} must be an object")
    result = dict(value)
    missing = required - result.keys()
    extra = result.keys() - required - (optional or set())
    if missing:
        raise UnifiedCommitValidationError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise UnifiedCommitValidationError(
            f"{name} has unsupported fields: {sorted(extra)}"
        )
    return result


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise UnifiedCommitValidationError(f"{name} is invalid")
    return value


def _reviewed_plan(review_result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    review = _strict_object(
        review_result,
        "review_result",
        {
            "schema_version",
            "review_id",
            "writer_plan_id",
            "outcome",
            "reason",
            "reviewed_plan",
            "commit_allowed",
            "review_rounds",
        },
    )
    if review["schema_version"] != REVIEWED_PLAN_SCHEMA_VERSION:
        raise UnifiedCommitValidationError("review_result schema is unsupported")
    _safe_id(review["review_id"], "review_result.review_id")
    _safe_id(review["writer_plan_id"], "review_result.writer_plan_id")
    if review["outcome"] not in {"accept", "revise"} or review["commit_allowed"] is not True:
        raise UnifiedCommitValidationError("review result is not approved for commit")
    if review["review_rounds"] != 1:
        raise UnifiedCommitValidationError("review must contain exactly one independent round")
    if not isinstance(review["reason"], str) or not review["reason"].strip() or len(review["reason"]) > 500:
        raise UnifiedCommitValidationError("review reason is invalid")
    plan = _strict_object(
        review["reviewed_plan"],
        "reviewed_plan",
        {
            "schema_version",
            "plan_id",
            "request_id",
            "candidate_count",
            "candidates",
            "evidence_catalog",
            "evidence_spans",
            "commit_allowed",
        },
    )
    if plan["schema_version"] != NORMALIZED_PLAN_SCHEMA_VERSION:
        raise UnifiedCommitValidationError("reviewed plan schema is unsupported")
    _safe_id(plan["plan_id"], "reviewed_plan.plan_id")
    _safe_id(plan["request_id"], "reviewed_plan.request_id")
    if plan["commit_allowed"] is not False:
        raise UnifiedCommitValidationError("semantic plan must not authorize its own commit")
    if review["outcome"] == "accept" and plan["plan_id"] != review["writer_plan_id"]:
        raise UnifiedCommitValidationError("accepted review is not bound to its writer plan")
    if (
        isinstance(plan["candidate_count"], bool)
        or not isinstance(plan["candidate_count"], int)
        or not isinstance(plan["candidates"], list)
        or plan["candidate_count"] != len(plan["candidates"])
        or len(plan["candidates"]) > 64
    ):
        raise UnifiedCommitValidationError("reviewed plan candidate count is invalid")
    return review, plan


def _canonical_scope(candidate: Mapping[str, Any]) -> dict[str, Any]:
    scope = dict(candidate["scope"])
    level = scope.get("level")
    if level not in _COMMIT_SCOPE_LEVELS:
        raise UnifiedCommitValidationError(
            f"scope {level!r} is not supported by the formal-store MVP"
        )
    expected = {
        "global": {"level"},
        "domain": {"level", "domain"},
        "project": {"level", "project_key"},
    }[level]
    if set(scope) != expected:
        raise UnifiedCommitValidationError("candidate scope is not canonical")
    if candidate["conditions"]:
        scope["conditions"] = dict(candidate["conditions"])
    return json.loads(_json(scope))


def _family(memory_type: str) -> str:
    if memory_type in _USER_CONTRACT_TYPES:
        return "user_contract"
    return {
        "engineering": "engineering_memory",
        "project": "project_memory",
        "reference": "reference_memory",
        "knowledge": "knowledge_memory",
        "other": "other_memory",
    }[memory_type]


def _activation_triggers(candidate: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for item in [candidate["summary"], *candidate["stable_key"].split(".")]:
        normalized = str(item).strip()
        if len(normalized) >= 2 and normalized not in result:
            result.append(normalized[:200])
    return result[:12]


def _catalog_and_spans(
    plan: Mapping[str, Any],
) -> tuple[dict[str, MemoryEvidence], dict[str, dict[str, Any]]]:
    raw_catalog = plan["evidence_catalog"]
    if not isinstance(raw_catalog, list):
        raise UnifiedCommitValidationError("reviewed evidence_catalog must be an array")
    try:
        catalog = normalize_evidence_catalog(raw_catalog)
    except MemoryCandidateValidationError as error:
        raise UnifiedCommitValidationError(
            f"reviewed evidence catalog is invalid: {error.code}"
        ) from error
    raw_spans = plan["evidence_spans"]
    if not isinstance(raw_spans, Mapping):
        raise UnifiedCommitValidationError("reviewed evidence_spans must be an object")
    spans: dict[str, dict[str, Any]] = {}
    required = {
        "source_evidence_ref",
        "quote",
        "start_offset",
        "end_offset",
        "turn_id",
        "source_kind",
        "occurred_at_ms",
        "project_key",
        "session_id",
        "source_message_id",
        "source_part_id",
        "part_ordinal",
    }
    for raw_id, raw_span in raw_spans.items():
        evidence_id = _safe_id(raw_id, "evidence_spans key")
        if evidence_id not in catalog:
            raise UnifiedCommitValidationError("evidence span has no catalog record")
        span = _strict_object(raw_span, f"evidence_spans.{evidence_id}", required)
        evidence = catalog[evidence_id]
        for field in ("source_evidence_ref", "turn_id", "source_message_id", "source_part_id"):
            _safe_id(span[field], f"evidence_spans.{evidence_id}.{field}")
        if span["source_kind"] != evidence.source_kind:
            raise UnifiedCommitValidationError("evidence span source kind changed")
        if span["turn_id"] != evidence.turn_id:
            raise UnifiedCommitValidationError("evidence span turn changed")
        if span["occurred_at_ms"] != evidence.occurred_at_ms:
            raise UnifiedCommitValidationError("evidence span timestamp changed")
        if span["project_key"] != evidence.project_key or span["session_id"] != evidence.session_id:
            raise UnifiedCommitValidationError("evidence span scope changed")
        if span["part_ordinal"] != evidence.part_ordinal:
            raise UnifiedCommitValidationError("evidence span part order changed")
        quote = span["quote"]
        start = span["start_offset"]
        end = span["end_offset"]
        if (
            not isinstance(quote, str)
            or not quote.strip()
            or len(quote) > 4_000
            or "[REDACTED_SECRET]" in quote
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or evidence.text_offset != start
        ):
            raise UnifiedCommitValidationError("evidence span is invalid")
        spans[evidence_id] = json.loads(_json(span))
    if set(spans) != set(catalog):
        raise UnifiedCommitValidationError("evidence catalog and span set differ")
    return catalog, spans


def _validated_items(
    plan: Mapping[str, Any],
    catalog: Mapping[str, MemoryEvidence],
    spans: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    catalog_values = list(catalog.values())
    for index, raw_item in enumerate(plan["candidates"]):
        item = _strict_object(
            raw_item,
            f"candidates[{index}]",
            {"candidate_id", "memory", "evidence_quotes", "action_hint"},
        )
        reviewed_memory = item["memory"]
        if not isinstance(reviewed_memory, Mapping):
            raise UnifiedCommitValidationError("reviewed candidate memory must be an object")
        proposal_fields = {
            key: reviewed_memory[key]
            for key in (
                "schema_version",
                "stable_key",
                "type",
                "summary",
                "scope",
                "conditions",
                "trust_status",
                "content",
                "evidence_refs",
                "status",
            )
            if key in reviewed_memory
        }
        try:
            candidate = validate_memory_candidate(proposal_fields, catalog_values)
        except MemoryCandidateValidationError as error:
            raise UnifiedCommitValidationError(
                f"candidate {index} failed deterministic validation: {error.code}"
            ) from error
        if item["candidate_id"] != candidate["candidate_id"] or dict(reviewed_memory) != candidate:
            raise UnifiedCommitValidationError("candidate identity changed after review")
        if candidate["schema_version"] != CANDIDATE_SCHEMA_VERSION or candidate["policy_version"] != POLICY_VERSION:
            raise UnifiedCommitValidationError("candidate policy version is unsupported")
        if candidate["status"] != "active":
            raise UnifiedCommitValidationError("only active candidates can enter formal memory")
        canonical_scope = _canonical_scope(candidate)
        fingerprint = _hash(canonical_scope, prefix="scope_")
        identity = (candidate["stable_key"], fingerprint)
        if identity in identities:
            raise UnifiedCommitValidationError(
                "one reviewed batch cannot contain duplicate stable-key scope identities"
            )
        identities.add(identity)
        if candidate["type"] in _USER_CONTRACT_TYPES and not any(
            catalog[reference].source_kind in _EXPLICIT_USER_SOURCE_KINDS
            and catalog[reference].explicit
            for reference in candidate["evidence_refs"]
        ):
            raise UnifiedCommitValidationError(
                "a user contract requires explicit user evidence"
            )
        quotes = item["evidence_quotes"]
        if not isinstance(quotes, list) or not quotes:
            raise UnifiedCommitValidationError("candidate evidence_quotes is invalid")
        quoted_ids: set[str] = set()
        for raw_quote in quotes:
            quote = _strict_object(
                raw_quote,
                "candidate.evidence_quote",
                {"evidence_id", "evidence_ref", "quote", "start_offset", "end_offset"},
            )
            evidence_id = quote["evidence_id"]
            if evidence_id in quoted_ids or evidence_id not in candidate["evidence_refs"]:
                raise UnifiedCommitValidationError("candidate evidence quote identity is invalid")
            quoted_ids.add(evidence_id)
            frozen = spans.get(evidence_id)
            if frozen is None or any(
                quote[field] != frozen[field]
                for field in ("source_evidence_ref", "quote", "start_offset", "end_offset")
                if field != "source_evidence_ref"
            ):
                # evidence_ref is the source_evidence_ref under its model-facing name.
                if frozen is None or quote.get("evidence_ref") != frozen.get("source_evidence_ref") or any(
                    quote[field] != frozen[field]
                    for field in ("quote", "start_offset", "end_offset")
                ):
                    raise UnifiedCommitValidationError("candidate evidence quote changed after review")
        if quoted_ids != set(candidate["evidence_refs"]):
            raise UnifiedCommitValidationError("candidate evidence quote set is incomplete")
        result.append(
            {
                "candidate": candidate,
                "canonical_scope": canonical_scope,
                "scope_fingerprint": fingerprint,
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["candidate"]["stable_key"],
            {"global": 0, "domain": 1, "project": 2}[item["canonical_scope"]["level"]],
            _json(item["canonical_scope"]),
        ),
    )


def _primary_evidence(
    candidate: Mapping[str, Any], catalog: Mapping[str, MemoryEvidence]
) -> MemoryEvidence:
    evidence = [catalog[reference] for reference in candidate["evidence_refs"]]
    if candidate["type"] in _USER_CONTRACT_TYPES:
        explicit = [
            item
            for item in evidence
            if item.source_kind in _EXPLICIT_USER_SOURCE_KINDS and item.explicit
        ]
        if explicit:
            evidence = explicit
    else:
        verified = [item for item in evidence if item.verified]
        if verified:
            evidence = verified
    return max(evidence, key=lambda item: item.source_order)


def _source_row(connection: sqlite3.Connection, evidence: MemoryEvidence) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT t.turn_id, t.current_state, t.primary_event_id,
               e.source_id, e.session_id, e.message_id AS primary_message_id,
               e.part_id AS primary_part_id, e.project_id AS source_project_id,
               e.directory AS source_directory,
               COALESCE(a.project_key, sc.project_key) AS project_key
        FROM conversation_turn AS t
        JOIN event AS e ON e.event_id = t.primary_event_id
        LEFT JOIN event_annotation AS a ON a.event_id = e.event_id
        LEFT JOIN session_context AS sc
          ON sc.source_id = e.source_id AND sc.session_id = e.session_id
        WHERE t.turn_id = ?
        """,
        (evidence.turn_id,),
    ).fetchone()
    if row is None or row["current_state"] not in {
        "completed",
        "failed",
        "interrupted",
    }:
        raise UnifiedCommitValidationError(
            "candidate evidence does not belong to a stable Lab conversation turn"
        )
    result = dict(row)
    if result["project_key"] is None and isinstance(result["source_directory"], str):
        # New completed turns may not have passed through prepare_dataset yet.
        # Match turn_evidence's deterministic directory identity without reading
        # or scanning any project file.
        result["project_key"] = project_identity(result["source_directory"])[0]
    if result["session_id"] != evidence.session_id or result["project_key"] != evidence.project_key:
        raise UnifiedCommitValidationError("candidate evidence no longer matches Lab scope")
    return result


def _existing_source_order(
    connection: sqlite3.Connection, memory_id: str, revision_id: str
) -> tuple[int, int, int, str] | None:
    row = connection.execute(
        """
        SELECT occurred_at, ordinal, start_offset, evidence_id
        FROM evidence
        WHERE memory_id = ? AND revision_id = ?
        ORDER BY COALESCE(occurred_at, -1) DESC,
                 COALESCE(ordinal, -1) DESC,
                 COALESCE(start_offset, -1) DESC,
                 evidence_id DESC
        LIMIT 1
        """,
        (memory_id, revision_id),
    ).fetchone()
    if row is None or row["occurred_at"] is None:
        return None
    return (
        int(row["occurred_at"]),
        int(row["ordinal"] or 0),
        int(row["start_offset"] or 0),
        str(row["evidence_id"]),
    )


def _relation(
    connection: sqlite3.Connection,
    candidate: Mapping[str, Any],
    fingerprint: str,
    primary: MemoryEvidence,
    virtual_scopes: Mapping[str, set[str]],
) -> tuple[str, sqlite3.Row | None]:
    memory = _memory_for_identity(connection, candidate["stable_key"], fingerprint)
    if memory is None:
        return (
            "create_scope_variant"
            if virtual_scopes.get(candidate["stable_key"])
            else "create",
            None,
        )
    if memory["status"] != "active" or memory["current_revision_id"] is None:
        raise UnifiedCommitValidationError("existing formal memory is not active")
    if memory["current_semantic_value"] == candidate["semantic_fingerprint"]:
        return "merge", memory
    current_family = str(memory["family"] or memory["memory_type"])
    incoming_family = _family(candidate["type"])
    if current_family != incoming_family and not (
        current_family == "user_contract" and incoming_family == "user_contract"
    ):
        return "conflict", memory
    if (
        candidate["type"] in _USER_CONTRACT_TYPES
        and primary.source_kind in _EXPLICIT_USER_SOURCE_KINDS
        and primary.explicit
    ):
        old_order = _existing_source_order(
            connection, memory["memory_id"], memory["current_revision_id"]
        )
        if old_order is not None and primary.source_order > old_order:
            return "revise", memory
    return "conflict", memory


def _formal_state_hash(snapshot: Mapping[str, Any]) -> str:
    return _sha(snapshot, "formal_")


def prepare_unified_commit(
    lab_database: Path,
    review_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an approved Reviewer result and freeze a read-only commit CAS."""

    require_memory_lab_database(Path(lab_database))
    review, plan = _reviewed_plan(review_result)
    catalog, spans = _catalog_and_spans(plan)
    items = _validated_items(plan, catalog, spans)
    source_review_hash = _sha(review, "reviewed_")
    with closing(open_read_only(Path(lab_database))) as connection:
        before = formal_state_snapshot(connection)
        known_scopes: dict[str, set[str]] = {}
        for row in connection.execute(
            "SELECT stable_key, scope_fingerprint FROM memory ORDER BY stable_key, scope_fingerprint"
        ):
            known_scopes.setdefault(str(row["stable_key"]), set()).add(
                str(row["scope_fingerprint"])
            )
        prepared_items: list[dict[str, Any]] = []
        for item in items:
            candidate = item["candidate"]
            primary = _primary_evidence(candidate, catalog)
            evidence_sources = {
                evidence_id: _source_row(connection, catalog[evidence_id])
                for evidence_id in candidate["evidence_refs"]
            }
            source = evidence_sources[primary.evidence_id]
            relation, memory = _relation(
                connection,
                candidate,
                item["scope_fingerprint"],
                primary,
                known_scopes,
            )
            known_scopes.setdefault(candidate["stable_key"], set()).add(
                item["scope_fingerprint"]
            )
            prepared_items.append(
                {
                    **item,
                    "primary_evidence_id": primary.evidence_id,
                    "primary_source": source,
                    "evidence_sources": evidence_sources,
                    "relation": relation,
                    "expected": {
                        "memory_id": memory["memory_id"] if memory else None,
                        "current_revision_id": memory["current_revision_id"] if memory else None,
                        "current_semantic_value": memory["current_semantic_value"] if memory else None,
                        "memory_state_version": memory["state_version"] if memory else None,
                        "open_conflicts_hash": _open_conflicts_hash(
                            connection, memory["memory_id"] if memory else None
                        ),
                    },
                }
            )
    basis = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "source_review_hash": source_review_hash,
        "review_id": review["review_id"],
        "writer_plan_id": review["writer_plan_id"],
        "review_outcome": review["outcome"],
        "reviewer": "deepseek-independent-reviewer",
        "review_reason": review["reason"].strip(),
        "formal_state_before": before,
        "formal_state_hash": _formal_state_hash(before),
        "evidence_catalog": [
            {
                "evidence_id": item.evidence_id,
                "turn_id": item.turn_id,
                "source_kind": item.source_kind,
                "occurred_at_ms": item.occurred_at_ms,
                "part_ordinal": item.part_ordinal,
                "text_offset": item.text_offset,
                "project_key": item.project_key,
                "session_id": item.session_id,
                "explicit": item.explicit,
                "verified": item.verified,
                "origin": item.origin,
            }
            for item in sorted(catalog.values(), key=lambda value: value.evidence_id)
        ],
        "evidence_spans": {key: spans[key] for key in sorted(spans)},
        "items": prepared_items,
    }
    return {**basis, "prepared_batch_hash": _sha(basis, "prepared_")}


def _validate_prepared(prepared: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_review_hash",
        "review_id",
        "writer_plan_id",
        "review_outcome",
        "reviewer",
        "review_reason",
        "formal_state_before",
        "formal_state_hash",
        "evidence_catalog",
        "evidence_spans",
        "items",
        "prepared_batch_hash",
    }
    obj = _strict_object(prepared, "prepared_batch", required)
    if obj["schema_version"] != PREPARED_SCHEMA_VERSION:
        raise UnifiedCommitValidationError("prepared batch schema is unsupported")
    basis = {key: obj[key] for key in required - {"prepared_batch_hash"}}
    if obj["prepared_batch_hash"] != _sha(basis, "prepared_"):
        raise UnifiedCommitValidationError("prepared batch hash is invalid")
    if obj["formal_state_hash"] != _formal_state_hash(obj["formal_state_before"]):
        raise UnifiedCommitValidationError("prepared formal-state hash is invalid")
    if not isinstance(obj["items"], list):
        raise UnifiedCommitValidationError("prepared items must be an array")
    return obj


def _proposal_spec(
    connection: sqlite3.Connection,
    item: Mapping[str, Any],
    catalog: Mapping[str, MemoryEvidence],
    spans: Mapping[str, Mapping[str, Any]],
    *,
    admission_run_id: str,
    snapshot_hash: str,
    decision_id: str,
) -> dict[str, Any]:
    candidate = item["candidate"]
    primary = catalog[item["primary_evidence_id"]]
    span = spans[primary.evidence_id]
    source = item["primary_source"]
    payload = {
        "family": _family(candidate["type"]),
        "memory_type": candidate["type"],
        "activation_triggers": _activation_triggers(candidate),
        "statement": candidate["content"]["statement"],
        "trust_status": candidate["trust_status"],
        "unified_candidate": candidate,
        "all_evidence_refs": candidate["evidence_refs"],
        "event_id": source["primary_event_id"],
        "source_id": source["source_id"],
        "session_id": span["session_id"],
        "message_id": span["source_message_id"],
        "part_id": span["source_part_id"],
        "project_key": span["project_key"],
        "source_project_id": source["source_project_id"],
        "exact_quote": span["quote"],
        "ordinal": max(1, int(span["part_ordinal"])),
        "start_offset": span["start_offset"],
        "end_offset": span["end_offset"],
        "occurred_at": span["occurred_at_ms"],
    }
    expected = item["expected"]
    return {
        "admission_run_id": admission_run_id,
        "admission_snapshot_hash": snapshot_hash,
        "decision_id": decision_id,
        "candidate_key": candidate["candidate_id"],
        "stable_key": candidate["stable_key"],
        "semantic_value": candidate["semantic_fingerprint"],
        "raw_scope_json": _json(item["canonical_scope"]),
        "canonical_scope_json": _json(item["canonical_scope"]),
        "scope_schema_version": SCOPE_SCHEMA_VERSION + "+unified-v1",
        "scope_fingerprint": item["scope_fingerprint"],
        "relation": item["relation"],
        "disposition": "unified_reviewer_approved",
        "is_correction": int(item["relation"] == "revise"),
        "expected_memory_id": expected["memory_id"],
        "expected_current_revision_id": expected["current_revision_id"],
        "expected_current_semantic_value": expected["current_semantic_value"],
        "expected_memory_state_version": expected["memory_state_version"],
        "expected_open_conflicts_hash": expected["open_conflicts_hash"],
        "application_payload_json": _json(payload),
    }


def _run_ids(prepared: Mapping[str, Any]) -> tuple[str, str]:
    """Return retry-stable audit run IDs for one independently reviewed plan.

    A prepared hash includes the formal-state CAS and therefore changes after a
    successful commit.  The reviewed result hash does not.  Keying the audit
    run by the latter lets a cycle retry after report/publish failure without
    applying the same reviewed batch a second time.
    """

    suffix = str(prepared["source_review_hash"]).removeprefix("reviewed_")[:32]
    return "unified_admission_" + suffix, "unified_commit_" + suffix


def _json_object(raw: Any, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise UnifiedCommitValidationError(f"stored {name} is invalid") from error
    if not isinstance(value, dict):
        raise UnifiedCommitValidationError(f"stored {name} is not an object")
    return value


def _stored_actions(
    connection: sqlite3.Connection, consolidation_run_id: str
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT candidate_key, decision_id, proposal_id, review_id, reviewer,
               action, reason, memory_id, before_revision_id,
               after_revision_id, conflict_id, details_json
        FROM consolidation_action
        WHERE run_id = ?
        ORDER BY created_at, action_id
        """,
        (consolidation_run_id,),
    ).fetchall()
    return [
        {
            "candidate_key": row["candidate_key"],
            "decision_id": row["decision_id"],
            "proposal_id": row["proposal_id"],
            "review_id": row["review_id"],
            "reviewer": row["reviewer"],
            "action": row["action"],
            "reason": row["reason"],
            "memory_id": row["memory_id"],
            "before_revision_id": row["before_revision_id"],
            "after_revision_id": row["after_revision_id"],
            "conflict_id": row["conflict_id"],
            "details": _json_object(row["details_json"], "action details"),
        }
        for row in rows
    ]


def _secondary_candidate_key(candidate_id: str, evidence_id: str) -> str:
    """Derive a stable unique evidence identity without inventing a decision."""

    suffix = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:24]
    return f"{candidate_id}:e:{suffix}"


def _persist_reviewed_evidence_set(
    connection: sqlite3.Connection,
    item: Mapping[str, Any],
    catalog: Mapping[str, MemoryEvidence],
    spans: Mapping[str, Mapping[str, Any]],
) -> int:
    """Persist every non-primary reviewed evidence span beside the primary.

    ``_apply_proposal`` writes the primary evidence and determines its exact
    revision/conflict role.  Secondary spans copy only that formal placement;
    each keeps its own immutable turn/source/message/part/quote coordinates.
    """

    candidate = item["candidate"]
    primary_id = item["primary_evidence_id"]
    primary = connection.execute(
        "SELECT * FROM evidence WHERE candidate_key = ?",
        (candidate["candidate_id"],),
    ).fetchone()
    if primary is None:
        raise UnifiedCommitError("formal primary evidence is missing after apply")
    inserted = 0
    for evidence_id in candidate["evidence_refs"]:
        if evidence_id == primary_id:
            continue
        evidence = catalog[evidence_id]
        span = spans[evidence_id]
        source = item["evidence_sources"][evidence_id]
        candidate_key = _secondary_candidate_key(candidate["candidate_id"], evidence_id)
        expected = {
            "memory_id": primary["memory_id"],
            "revision_id": primary["revision_id"],
            "conflict_id": primary["conflict_id"],
            "candidate_key": candidate_key,
            # A semantic replay may arrive through a fresh audit run.  Keep the
            # full evidence set bound to the same audit chain as its already
            # persisted primary instead of rewriting history to the new noop.
            "decision_id": primary["decision_id"],
            "proposal_id": primary["proposal_id"],
            "review_id": primary["review_id"],
            "reviewer": primary["reviewer"],
            "event_id": source["primary_event_id"],
            "source_id": source["source_id"],
            "session_id": span["session_id"],
            "message_id": span["source_message_id"],
            "part_id": span["source_part_id"],
            "project_key": span["project_key"],
            "source_project_id": source["source_project_id"],
            "exact_quote": span["quote"],
            "ordinal": max(1, int(span["part_ordinal"])),
            "start_offset": span["start_offset"],
            "end_offset": span["end_offset"],
            "occurred_at": span["occurred_at_ms"],
            "evidence_role": primary["evidence_role"],
        }
        existing = connection.execute(
            "SELECT * FROM evidence WHERE candidate_key = ?", (candidate_key,)
        ).fetchone()
        if existing is not None:
            if any(existing[key] != value for key, value in expected.items()):
                raise UnifiedCommitValidationError(
                    "secondary evidence identity belongs to different formal semantics"
                )
            continue
        formal_evidence_id = _sha(
            [candidate["candidate_id"], evidence_id], "evidence_"
        )
        connection.execute(
            """
            INSERT INTO evidence(
                evidence_id, memory_id, revision_id, conflict_id, candidate_key,
                decision_id, proposal_id, review_id, reviewer, event_id, source_id,
                session_id, message_id, part_id, project_key, source_project_id,
                exact_quote, ordinal, start_offset, end_offset, occurred_at,
                evidence_role, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                formal_evidence_id,
                expected["memory_id"],
                expected["revision_id"],
                expected["conflict_id"],
                expected["candidate_key"],
                expected["decision_id"],
                expected["proposal_id"],
                expected["review_id"],
                expected["reviewer"],
                expected["event_id"],
                expected["source_id"],
                expected["session_id"],
                expected["message_id"],
                expected["part_id"],
                expected["project_key"],
                expected["source_project_id"],
                expected["exact_quote"],
                expected["ordinal"],
                expected["start_offset"],
                expected["end_offset"],
                expected["occurred_at"],
                expected["evidence_role"],
                now_iso(),
            ),
        )
        inserted += 1
    return inserted


def _completed_replay(
    connection: sqlite3.Connection,
    prepared: Mapping[str, Any],
    admission_run_id: str,
    consolidation_run_id: str,
) -> dict[str, Any] | None:
    """Load the exact prior result for a crash-safe reviewed-batch replay."""

    run = connection.execute(
        "SELECT * FROM consolidation_run WHERE run_id = ?",
        (consolidation_run_id,),
    ).fetchone()
    admission = connection.execute(
        "SELECT * FROM admission_run WHERE run_id = ?",
        (admission_run_id,),
    ).fetchone()
    if run is None and admission is None:
        return None
    if run is None or admission is None:
        raise UnifiedCommitValidationError("reviewed-batch replay audit chain is incomplete")
    if (
        run["status"] != "complete"
        or admission["status"] != "unified_reviewed"
        or run["admission_run_id"] != admission_run_id
        or run["consolidator_version"] != CONSOLIDATOR_VERSION + "+unified-v1"
        or admission["extractor_version"] != COMMIT_VERSION
    ):
        raise UnifiedCommitValidationError("reviewed-batch replay audit chain is not complete")
    metadata = _json_object(admission["counts_json"], "unified replay metadata")
    required = {
        "schema_version",
        "unified_candidates",
        "source_review_hash",
        "source_review_id",
        "prepared_batch_hash",
        "formal_state_before",
        "formal_state_after",
        "state_after",
    }
    if set(metadata) != required or metadata["schema_version"] != REPLAY_METADATA_SCHEMA_VERSION:
        raise UnifiedCommitValidationError("stored unified replay metadata is unsupported")
    if (
        metadata["source_review_hash"] != prepared["source_review_hash"]
        or metadata["source_review_id"] != prepared["review_id"]
        or run["admission_snapshot_hash"] != metadata["prepared_batch_hash"]
    ):
        raise UnifiedCommitValidationError("reviewed-batch replay identity does not match")
    counts = _json_object(run["counts_json"], "consolidation counts")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise UnifiedCommitValidationError("stored consolidation counts are invalid")
    actions = _stored_actions(connection, consolidation_run_id)
    if sum(counts.values()) != len(actions):
        raise UnifiedCommitValidationError("stored consolidation actions do not match counts")
    current_formal_state = formal_state_snapshot(connection)
    return {
        "status": "replayed",
        "reason": "reviewed_batch_already_committed",
        "commit_version": COMMIT_VERSION,
        "prepared_batch_hash": metadata["prepared_batch_hash"],
        "retry_prepared_batch_hash": prepared["prepared_batch_hash"],
        "source_review_id": metadata["source_review_id"],
        "consolidation_run_id": consolidation_run_id,
        "counts": counts,
        "actions": actions,
        "formal_state_before": metadata["formal_state_before"],
        "formal_state_after": metadata["formal_state_after"],
        "formal_state_changed_since_commit": (
            current_formal_state != metadata["formal_state_after"]
        ),
        "state": metadata["state_after"],
        "current_state": _state(connection),
    }


def commit_prepared_unified_batch(
    lab_database: Path,
    prepared_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one prepared batch atomically or leave every formal row unchanged."""

    require_memory_lab_database(Path(lab_database))
    prepared = _validate_prepared(prepared_batch)
    if not prepared["items"]:
        return {
            "status": "noop",
            "reason": "reviewed_plan_contains_no_candidates",
            "prepared_batch_hash": prepared["prepared_batch_hash"],
            "actions": [],
            "counts": {},
        }
    catalog = normalize_evidence_catalog(prepared["evidence_catalog"])
    spans = prepared["evidence_spans"]
    admission_run_id, consolidation_run_id = _run_ids(prepared)
    with closing(open_store(Path(lab_database))) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = _completed_replay(
                connection, prepared, admission_run_id, consolidation_run_id
            )
            if replay is not None:
                # The read is intentionally performed under the same write lock
                # as a first commit.  No row is touched on the replay path.
                connection.rollback()
                return replay
            if formal_state_snapshot(connection) != prepared["formal_state_before"]:
                raise UnifiedCommitCASMismatch(
                    "formal memory changed after the reviewed batch was prepared"
                )
            now = now_iso()
            connection.execute(
                """
                INSERT INTO admission_run(
                    run_id, extractor_version, started_at, completed_at,
                    status, decision_count, counts_json, error
                ) VALUES (?, ?, ?, ?, 'unified_reviewed', ?, ?, NULL)
                """,
                (
                    admission_run_id,
                    COMMIT_VERSION,
                    now,
                    now,
                    len(prepared["items"]),
                    _json({"unified_candidates": len(prepared["items"])}),
                ),
            )
            connection.execute(
                """
                INSERT INTO consolidation_run(
                    run_id, consolidator_version, admission_run_id,
                    admission_snapshot_hash, started_at, status, candidate_count
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    consolidation_run_id,
                    CONSOLIDATOR_VERSION + "+unified-v1",
                    admission_run_id,
                    prepared["prepared_batch_hash"],
                    now,
                    len(prepared["items"]),
                ),
            )
            actions: list[dict[str, Any]] = []
            for index, item in enumerate(prepared["items"], 1):
                candidate = item["candidate"]
                primary = catalog[item["primary_evidence_id"]]
                span = spans[primary.evidence_id]
                source = item["primary_source"]
                # Re-check the immutable local trace while holding the write lock.
                for evidence_id in candidate["evidence_refs"]:
                    if (
                        _source_row(connection, catalog[evidence_id])
                        != item["evidence_sources"][evidence_id]
                    ):
                        raise UnifiedCommitCASMismatch(
                            "source turn changed before formal commit"
                        )
                decision_id = _sha(
                    [prepared["prepared_batch_hash"], candidate["candidate_id"]],
                    "decision_",
                )
                triggers = _activation_triggers(candidate)
                signals = {
                    "unified_candidate": True,
                    "trust_status": candidate["trust_status"],
                    "verification_status": candidate["trust_status"],
                    "review_id": prepared["review_id"],
                }
                connection.execute(
                    """
                    INSERT INTO admission_decision(
                        decision_id, run_id, candidate_key, event_id, ordinal,
                        exact_quote, start_offset, end_offset, family, memory_type,
                        stable_key_suggestion, scope_json, activation_triggers_json,
                        disposition, signals_json, instruction_coverage_json,
                        reasons_json, decided_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)
                    """,
                    (
                        decision_id,
                        admission_run_id,
                        candidate["candidate_id"],
                        source["primary_event_id"],
                        index,
                        span["quote"],
                        span["start_offset"],
                        span["end_offset"],
                        _family(candidate["type"]),
                        candidate["type"],
                        candidate["stable_key"],
                        _json(item["canonical_scope"]),
                        _json(triggers),
                        "unified_reviewer_approved",
                        _json(signals),
                        _json(["independent_reviewer_approved"]),
                        now_iso(),
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM admission_decision WHERE decision_id = ?", (decision_id,)
                ).fetchone()
                spec = _proposal_spec(
                    connection,
                    item,
                    catalog,
                    spans,
                    admission_run_id=admission_run_id,
                    snapshot_hash=prepared["prepared_batch_hash"],
                    decision_id=decision_id,
                )
                proposal, _ = _insert_or_get_proposal(connection, spec)
                review_id = "review_" + uuid4().hex
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
                        decision_id,
                        candidate["candidate_id"],
                        prepared["reviewer"],
                        _json(
                            {
                                "source_review_id": prepared["review_id"],
                                "source_review_hash": prepared["source_review_hash"],
                                "writer_plan_id": prepared["writer_plan_id"],
                                "outcome": prepared["review_outcome"],
                                "reason": prepared["review_reason"],
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
                if action["action"] not in _SUCCESS_ACTIONS:
                    raise UnifiedCommitError(
                        f"deterministic formal apply failed: {action['action']}:{action['reason']}"
                    )
                secondary_count = _persist_reviewed_evidence_set(
                    connection, item, catalog, spans
                )
                if secondary_count:
                    action["details"] = {
                        **action["details"],
                        "secondary_evidence_persisted": secondary_count,
                    }
                    connection.execute(
                        """
                        UPDATE consolidation_action SET details_json = ?
                        WHERE run_id = ? AND candidate_key = ?
                        """,
                        (
                            _json(action["details"]),
                            consolidation_run_id,
                            candidate["candidate_id"],
                        ),
                    )
                actions.append(action)
            counts = dict(sorted(Counter(item["action"] for item in actions).items()))
            after = formal_state_snapshot(connection)
            state = _state(connection)
            replay_metadata = {
                "schema_version": REPLAY_METADATA_SCHEMA_VERSION,
                "unified_candidates": len(prepared["items"]),
                "source_review_hash": prepared["source_review_hash"],
                "source_review_id": prepared["review_id"],
                "prepared_batch_hash": prepared["prepared_batch_hash"],
                "formal_state_before": prepared["formal_state_before"],
                "formal_state_after": after,
                "state_after": state,
            }
            connection.execute(
                "UPDATE admission_run SET counts_json = ? WHERE run_id = ?",
                (_json(replay_metadata), admission_run_id),
            )
            connection.execute(
                """
                UPDATE consolidation_run
                SET completed_at = ?, status = 'complete', counts_json = ?, error = NULL
                WHERE run_id = ?
                """,
                (now_iso(), _json(counts), consolidation_run_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "status": "complete",
        "commit_version": COMMIT_VERSION,
        "prepared_batch_hash": prepared["prepared_batch_hash"],
        "source_review_id": prepared["review_id"],
        "consolidation_run_id": consolidation_run_id,
        "counts": counts,
        "actions": actions,
        "formal_state_before": prepared["formal_state_before"],
        "formal_state_after": after,
        "state": state,
    }


def commit_reviewed_unified_plan(
    lab_database: Path,
    review_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Convenience entrypoint used by orchestration after independent review."""

    if isinstance(review_result, Mapping) and (
        review_result.get("outcome") == "reject"
        or review_result.get("commit_allowed") is False
    ):
        # A valid Reviewer rejection is intentionally a zero-write terminal result.
        if review_result.get("schema_version") != REVIEWED_PLAN_SCHEMA_VERSION:
            raise UnifiedCommitValidationError("review_result schema is unsupported")
        return {
            "status": "reviewer_rejected",
            "reason": "independent_reviewer_rejected_batch",
            "actions": [],
            "counts": {},
        }
    prepared = prepare_unified_commit(lab_database, review_result)
    return commit_prepared_unified_batch(lab_database, prepared)


__all__ = [
    "COMMIT_VERSION",
    "PREPARED_SCHEMA_VERSION",
    "UnifiedCommitCASMismatch",
    "UnifiedCommitError",
    "UnifiedCommitValidationError",
    "commit_prepared_unified_batch",
    "commit_reviewed_unified_plan",
    "prepare_unified_commit",
]
