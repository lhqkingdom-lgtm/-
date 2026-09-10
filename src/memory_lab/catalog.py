"""Read-only preview catalog and deterministic recall over validated shadow judgments.

The shadow lane is intentionally non-authoritative.  This module turns one
fresh, current, fully validated shadow run into a reviewable memory preview and
can route a task sentence to applicable preview memories.  It never mutates the
lab database or CodePulse.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .consolidation import canonicalize_scope, latest_admission_snapshot
from .safety import require_memory_lab_database
from .shadow import formal_state_snapshot
from .source import open_read_only


CATALOG_VERSION = "shadow-preview-catalog-v2"
RECALL_VERSION = "deterministic-preview-recall-v2"


class CatalogValidationError(ValueError):
    """Raised when a shadow run is not safe to use as a preview source."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_object(value: Any, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise CatalogValidationError(f"{field} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise CatalogValidationError(f"{field} must be an object")
    return parsed


def _parse_array(value: Any, field: str) -> list[Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise CatalogValidationError(f"{field} is not valid JSON") from error
    if not isinstance(parsed, list):
        raise CatalogValidationError(f"{field} must be an array")
    return parsed


def _select_shadow_run(
    connection: sqlite3.Connection, shadow_run_id: str | None
) -> sqlite3.Row:
    if shadow_run_id:
        run = connection.execute(
            "SELECT rowid AS storage_rowid, * FROM llm_shadow_run WHERE shadow_run_id = ?",
            (shadow_run_id,),
        ).fetchone()
    else:
        run = connection.execute(
            """
            SELECT rowid AS storage_rowid, * FROM llm_shadow_run
            ORDER BY started_at DESC, rowid DESC LIMIT 1
            """
        ).fetchone()
    if run is None:
        raise CatalogValidationError("no matching fresh complete shadow run")
    if run["status"] != "complete":
        raise CatalogValidationError("shadow run is not complete")
    if int(run["replay_count"] or 0) != 0:
        raise CatalogValidationError("catalog requires fresh model judgments, not cache replay")
    if int(run["error_count"] or 0) != 0:
        raise CatalogValidationError("shadow run contains judgment errors")
    counts = _parse_object(run["counts_json"], "counts_json")
    allowed_statuses = {"valid", "replayed", "blocked", "invalid"}
    if set(counts) - allowed_statuses:
        raise CatalogValidationError("shadow run contains unknown judgment statuses")
    normalized_counts: dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CatalogValidationError("shadow run counts are invalid")
        normalized_counts[key] = value
    candidate_count = int(run["candidate_count"] or 0)
    success_count = int(run["success_count"] or 0)
    blocked_count = normalized_counts.get("blocked", 0)
    if sum(normalized_counts.values()) != candidate_count:
        raise CatalogValidationError("shadow run status counts do not cover every candidate")
    if normalized_counts.get("valid", 0) + normalized_counts.get("replayed", 0) != success_count:
        raise CatalogValidationError("shadow run success count is inconsistent")
    if normalized_counts.get("invalid", 0) != int(run["error_count"] or 0):
        raise CatalogValidationError("shadow run error count is inconsistent")
    if normalized_counts.get("replayed", 0) != int(run["replay_count"] or 0):
        raise CatalogValidationError("shadow run replay count is inconsistent")
    if success_count + blocked_count != candidate_count:
        raise CatalogValidationError("shadow run does not account for every candidate")
    return run


def _validated_rows(
    connection: sqlite3.Connection, shadow_run_id: str | None
) -> tuple[sqlite3.Row, list[sqlite3.Row]]:
    run = _select_shadow_run(connection, shadow_run_id)
    admission, snapshot_rows, snapshot_hash = latest_admission_snapshot(connection)
    if admission is None or snapshot_hash is None:
        raise CatalogValidationError("latest admission snapshot is unavailable or incomplete")
    latest_admission = connection.execute(
        """
        SELECT rowid AS storage_rowid, * FROM admission_run
        ORDER BY started_at DESC, rowid DESC LIMIT 1
        """
    ).fetchone()
    if latest_admission is None or latest_admission["run_id"] != admission["run_id"]:
        raise CatalogValidationError("a newer admission attempt is not complete")
    if run["admission_run_id"] != admission["run_id"]:
        raise CatalogValidationError("shadow run is not bound to the latest admission run")
    if run["admission_snapshot_hash"] != snapshot_hash:
        raise CatalogValidationError("shadow run admission snapshot is stale")

    before = _parse_object(run["formal_state_before_json"], "formal_state_before_json")
    after = _parse_object(run["formal_state_after_json"], "formal_state_after_json")
    current = formal_state_snapshot(connection)
    if before != after or after != current:
        raise CatalogValidationError("formal memory state changed after the shadow run")

    rows = connection.execute(
        """
        SELECT j.*, d.exact_quote, d.start_offset, d.end_offset,
               d.activation_triggers_json, d.disposition,
               e.event_id, e.session_id, e.message_id, e.part_id,
               e.timestamp AS occurred_at, sc.project_key
        FROM llm_judgment AS j
        JOIN admission_decision AS d ON d.decision_id = j.decision_id
        JOIN event AS e ON e.event_id = d.event_id
        JOIN session_context AS sc
          ON sc.source_id = e.source_id AND sc.session_id = e.session_id
        WHERE j.shadow_run_id = ?
        ORDER BY e.timestamp, d.ordinal, d.candidate_key
        """,
        (run["shadow_run_id"],),
    ).fetchall()
    expected_decisions = {row["decision_id"] for row in snapshot_rows}
    observed_decisions = {row["decision_id"] for row in rows}
    if len(rows) != int(run["candidate_count"] or 0):
        raise CatalogValidationError("shadow judgment count does not match candidate count")
    if observed_decisions != expected_decisions:
        raise CatalogValidationError("shadow judgments do not cover the exact admission snapshot")
    for row in rows:
        mode = row["execution_mode"]
        status = row["status"]
        if status == "valid" and mode != "judge_call":
            raise CatalogValidationError("valid catalog judgments must be fresh judge calls")
        if status == "blocked" and mode != "local_block":
            raise CatalogValidationError("blocked catalog judgments must be local blocks")
        if status not in {"valid", "blocked"}:
            raise CatalogValidationError("catalog source contains an unusable judgment")
    return run, rows


def _scope_rank(scope: Mapping[str, Any]) -> int:
    return {"global": 0, "domain": 1, "project": 2}.get(str(scope.get("level")), -1)


def _memory_identifier(stable_key: str, scope: Mapping[str, Any]) -> str:
    import hashlib

    payload = f"{stable_key}\0{_canonical_json(scope)}".encode("utf-8")
    return "preview_" + hashlib.sha256(payload).hexdigest()[:24]


def build_preview_catalog(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a deterministic, non-authoritative catalog from validated rows."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    filtered: list[dict[str, Any]] = []
    positive_judgments = 0

    for source in rows:
        row = dict(source)
        if row.get("status") == "blocked":
            filtered.append(
                {
                    "candidate_key": row.get("candidate_key"),
                    "quote": "[REDACTED sensitive candidate]",
                    "category": "sensitive_local_block",
                    "reason": "blocked locally before model transport",
                }
            )
            continue
        result = _parse_object(row.get("result_json"), "result_json")
        if not result.get("should_store"):
            disposition = str(row.get("disposition") or "")
            category = {
                "temporary": "temporary",
                "covered_by_cp_instructions": "cp_instruction_covered",
                "needs_scope_review": "scope_unresolved",
                "not_memory": "not_memory",
            }.get(disposition, "model_rejected")
            filtered.append(
                {
                    "candidate_key": row.get("candidate_key"),
                    "quote": row.get("exact_quote"),
                    "category": category,
                    "reason": result.get("reason") or "model rejected candidate",
                }
            )
            continue

        stable_key = result.get("key")
        memory_type = result.get("type")
        value = result.get("value")
        raw_scope = result.get("scope")
        if not isinstance(stable_key, str) or not stable_key:
            raise CatalogValidationError("stored judgment is missing a stable key")
        if not isinstance(memory_type, str) or not memory_type:
            raise CatalogValidationError("stored judgment is missing a memory type")
        if not isinstance(value, dict) or not isinstance(value.get("normalized"), str):
            raise CatalogValidationError("stored judgment has an invalid value")
        if not isinstance(value.get("statement"), str) or not value["statement"].strip():
            raise CatalogValidationError("stored judgment has no canonical statement")
        scope = canonicalize_scope(
            raw_scope,
            event_project_key=row.get("project_key"),
        )
        if scope is None:
            raise CatalogValidationError("stored judgment has an invalid scope")
        triggers = [
            str(item)
            for item in _parse_array(
                row.get("activation_triggers_json") or "[]", "activation_triggers_json"
            )
            if isinstance(item, str) and item.strip()
        ]
        relation = result.get("relation") or {}
        relation_kind = relation.get("kind") if isinstance(relation, dict) else None
        evidence = result.get("evidence")
        if not isinstance(evidence, list) or len(evidence) != 1:
            raise CatalogValidationError("stored judgment must contain one exact evidence item")
        positive_judgments += 1
        identity = (stable_key, _canonical_json(scope))
        grouped[identity].append(
            {
                "candidate_key": row.get("candidate_key"),
                "memory_type": memory_type,
                "stable_key": stable_key,
                "scope": scope,
                "normalized_value": value["normalized"],
                "statement": value["statement"].strip(),
                "activation_triggers": triggers,
                "relation": relation_kind,
                "confidence": result.get("confidence"),
                "evidence": {
                    "event_id": row.get("event_id"),
                    "session_id": row.get("session_id"),
                    "message_id": row.get("message_id"),
                    "part_id": row.get("part_id"),
                    "project_key": row.get("project_key"),
                    "quote": row.get("exact_quote"),
                    "start_offset": row.get("start_offset"),
                    "end_offset": row.get("end_offset"),
                    "occurred_at": row.get("occurred_at"),
                },
            }
        )

    memories: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for (stable_key, _scope_json), items in sorted(grouped.items()):
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        types = {item["memory_type"] for item in items}
        for item in items:
            values[item["normalized_value"]].append(item)
        if len(values) > 1 or len(types) > 1:
            conflicts.append(
                {
                    "stable_key": stable_key,
                    "scope": items[0]["scope"],
                    "scope_rank": _scope_rank(items[0]["scope"]),
                    "status": "review_required",
                    "reason": "opposite_or_incompatible_values_in_same_scope",
                    "activation_triggers": sorted(
                        {
                            trigger
                            for item in items
                            for trigger in item["activation_triggers"]
                        },
                        key=lambda value: (value.casefold(), value),
                    ),
                    "variants": [
                        {
                            "value": value,
                            "memory_types": sorted(
                                {item["memory_type"] for item in value_items}
                            ),
                            "statements": sorted({item["statement"] for item in value_items}),
                            "candidate_keys": [item["candidate_key"] for item in value_items],
                            "evidence_count": len(value_items),
                        }
                        for value, value_items in sorted(values.items())
                    ],
                }
            )
            continue
        representative = items[0]
        trigger_union = sorted(
            {trigger for item in items for trigger in item["activation_triggers"]},
            key=lambda value: (value.casefold(), value),
        )
        evidence = [item["evidence"] for item in items]
        memories.append(
            {
                "preview_memory_id": _memory_identifier(stable_key, representative["scope"]),
                "stable_key": stable_key,
                "memory_type": representative["memory_type"],
                "scope": representative["scope"],
                "scope_rank": _scope_rank(representative["scope"]),
                "normalized_value": representative["normalized_value"],
                "statement": representative["statement"],
                "activation_triggers": trigger_union,
                "evidence_count": len(evidence),
                "evidence": evidence,
                "source_candidate_keys": [item["candidate_key"] for item in items],
                "source_relations": sorted(
                    {item["relation"] for item in items if item["relation"]}
                ),
                "status": "preview",
            }
        )

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for memory in memories:
        by_key[memory["stable_key"]].append(memory)
    scope_variants = []
    for stable_key, variants in sorted(by_key.items()):
        scopes = {_canonical_json(item["scope"]) for item in variants}
        if len(scopes) > 1:
            scope_variants.append(
                {
                    "stable_key": stable_key,
                    "resolution_policy": "more_specific_scope_wins",
                    "variants": [
                        {
                            "preview_memory_id": item["preview_memory_id"],
                            "scope": item["scope"],
                            "normalized_value": item["normalized_value"],
                        }
                        for item in sorted(
                            variants,
                            key=lambda item: (item["scope_rank"], item["preview_memory_id"]),
                        )
                    ],
                }
            )

    filtered_counts = dict(sorted(Counter(item["category"] for item in filtered).items()))
    return {
        "catalog_version": CATALOG_VERSION,
        "preview_only": True,
        "positive_judgments": positive_judgments,
        "unique_memory_count": len(memories),
        "same_scope_conflict_count": len(conflicts),
        "scope_variant_group_count": len(scope_variants),
        "filtered_count": len(filtered),
        "filtered_counts": filtered_counts,
        "memories": memories,
        "conflicts": conflicts,
        "scope_variants": scope_variants,
        "filtered": filtered,
    }


def _catalog_from_connection(
    connection: sqlite3.Connection, shadow_run_id: str | None
) -> dict[str, Any]:
    before = formal_state_snapshot(connection)
    run, rows = _validated_rows(connection, shadow_run_id)
    catalog = build_preview_catalog(rows)
    after = formal_state_snapshot(connection)
    if before != after:
        raise CatalogValidationError("catalog unexpectedly changed formal memory state")
    project_contexts = [
        {
            "project_key": row["project_key"],
            "directory": row["directory"],
            "display_name": row["display_name"],
        }
        for row in connection.execute(
            """
            SELECT project_key, directory, display_name
            FROM project_context
            ORDER BY project_key
            """
        )
    ]
    return {
        "status": "complete",
        "shadow_run_id": run["shadow_run_id"],
        "admission_run_id": run["admission_run_id"],
        "model": run["model"],
        "shadow_completed_at": run["completed_at"],
        "project_contexts": project_contexts,
        "formal_state_unchanged": True,
        **catalog,
    }


def _consistent_read(
    connection: sqlite3.Connection, operation: Callable[[], Any]
) -> Any:
    """Run one atomic read snapshot and reject concurrent commits.

    SQLite cannot promise freshness after the final check without blocking
    writers.  A deferred read transaction plus ``data_version`` gives this
    preview a clear linearization point and fails closed if another connection
    commits while the snapshot is being consumed.
    """

    version_before = int(connection.execute("PRAGMA data_version").fetchone()[0])
    connection.execute("BEGIN")
    try:
        result = operation()
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    version_after = int(connection.execute("PRAGMA data_version").fetchone()[0])
    if version_before != version_after:
        raise CatalogValidationError("memory lab changed during preview read")
    return result


def run_catalog(lab_database: Path, shadow_run_id: str | None = None) -> dict[str, Any]:
    """Return a fail-closed preview catalog without writing any database."""

    require_memory_lab_database(lab_database)
    with closing(open_read_only(lab_database)) as connection:
        return _consistent_read(
            connection,
            lambda: _catalog_from_connection(connection, shadow_run_id),
        )


_ALNUM = re.compile(r"[a-z0-9][a-z0-9_.+/-]*", re.I)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def _normalized_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", "", normalized)


def _features(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    result = {match.group(0) for match in _ALNUM.finditer(normalized)}
    for match in _CJK.finditer(normalized):
        block = match.group(0)
        if len(block) == 1:
            result.add(block)
        else:
            result.update(block[index : index + 2] for index in range(len(block) - 1))
    return result


def _infer_domain(query: str) -> str | None:
    normalized = _normalized_search_text(query)
    if any(term in normalized for term in ("单片机", "嵌入式", "mcu", "firmware", "固件")):
        return "embedded"
    return None


def _known_project_key(
    connection: sqlite3.Connection, project_directory: Path | None
) -> tuple[str | None, str | None]:
    if project_directory is None:
        return None, None
    requested = str(project_directory.resolve()).casefold()
    for row in connection.execute(
        "SELECT project_key, directory FROM project_context ORDER BY project_key"
    ):
        if str(Path(row["directory"]).resolve()).casefold() == requested:
            return row["project_key"], row["directory"]
    return None, str(project_directory.resolve())


def _scope_context_state(
    scope: Mapping[str, Any],
    project_key: str | None,
    domain: str | None,
    *,
    project_context_known: bool,
    domain_context_known: bool,
) -> tuple[str, str | None]:
    """Return applicable, inapplicable, or unknown for one scope.

    Unknown is deliberately different from inapplicable: an unresolved
    project/domain/condition may contain a more-specific exception, so recall
    must not silently fall back to a broader rule for the same stable key.
    """

    level = scope.get("level")
    if level == "global":
        state, reason = "applicable", None
    elif level == "domain":
        if not domain_context_known:
            state, reason = "unknown", "domain_context_unknown"
        elif domain is not None and str(scope.get("domain")).casefold() == domain.casefold():
            state, reason = "applicable", None
        else:
            state, reason = "inapplicable", "different_domain"
    elif level == "project":
        if not project_context_known:
            state, reason = "unknown", "project_context_unknown"
        elif project_key is not None and scope.get("project_key") == project_key:
            state, reason = "applicable", None
        else:
            state, reason = "inapplicable", "different_project"
    else:
        return "inapplicable", "invalid_scope"

    if state == "applicable" and scope.get("conditions"):
        return "unknown", "conditional_scope_requires_typed_context"
    return state, reason


_GENERIC_TRIGGERS = {
    "代码",
    "修改",
    "检查",
    "生成",
    "确认",
    "回复",
    "项目",
    "流程",
    "使用",
    "要求",
    "语言",
}


def _relevance(memory: Mapping[str, Any], query: str) -> tuple[float, list[str]]:
    stable_key = str(memory["stable_key"])
    if stable_key.startswith("preference.response."):
        return 0.35, ["global_response_contract"]
    query_normalized = _normalized_search_text(query)
    triggers = [str(item) for item in memory.get("activation_triggers", [])]
    matched_triggers = [
        trigger for trigger in triggers if _normalized_search_text(trigger) in query_normalized
    ]
    distinctive_matches = [
        trigger
        for trigger in matched_triggers
        if _normalized_search_text(trigger) not in _GENERIC_TRIGGERS
    ]
    document = " ".join(
        [
            stable_key.replace(".", " "),
            str(memory.get("statement") or ""),
            *triggers,
            *[str(item.get("quote") or "") for item in memory.get("evidence", [])],
        ]
    )
    query_features = _features(query)
    document_features = _features(document)
    overlap = len(query_features & document_features)
    lexical = overlap / math.sqrt(max(1, len(query_features) * len(document_features)))
    phrase = min(1.0, len(distinctive_matches) / 2.0)
    score = min(1.0, 0.7 * phrase + 0.6 * lexical)
    # One generic word such as “确认” is not enough to turn a weather task into
    # a code-generation recall.  It can contribute lexically only when at least
    # two query features support the same topic.
    if not distinctive_matches and overlap < 2:
        score = 0.0
    reasons = [f"trigger:{item}" for item in distinctive_matches]
    if overlap:
        reasons.append(f"lexical_overlap:{overlap}")
    return round(score, 6), reasons


def recall_from_catalog(
    catalog: Mapping[str, Any],
    query: str,
    *,
    project_key: str | None = None,
    domain: str | None = None,
    project_context_known: bool | None = None,
    domain_context_known: bool | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Resolve scope and rank relevant preview memories for one task sentence."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if len(query) > 4000:
        raise ValueError("query must not exceed 4000 characters")
    if limit < 1 or limit > 20:
        raise ValueError("limit must be between 1 and 20")
    effective_domain = (
        domain.strip().casefold()
        if isinstance(domain, str) and domain.strip()
        else None
    )
    effective_domain = effective_domain or _infer_domain(query)
    if project_context_known is None:
        project_context_known = project_key is not None
    if domain_context_known is None:
        domain_context_known = effective_domain is not None

    records_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_memory in catalog.get("memories", []):
        memory = dict(source_memory)
        state, reason = _scope_context_state(
            memory["scope"],
            project_key,
            effective_domain,
            project_context_known=project_context_known,
            domain_context_known=domain_context_known,
        )
        score, match_reasons = _relevance(memory, query)
        records_by_key[memory["stable_key"]].append(
            {
                "kind": "memory",
                "source": memory,
                "state": state,
                "context_reason": reason,
                "scope_rank": int(
                    memory.get("scope_rank", _scope_rank(memory["scope"]))
                ),
                "score": score,
                "match_reasons": match_reasons,
            }
        )

    for source_conflict in catalog.get("conflicts", []):
        conflict = dict(source_conflict)
        scope = conflict.get("scope") or {}
        state, reason = _scope_context_state(
            scope,
            project_key,
            effective_domain,
            project_context_known=project_context_known,
            domain_context_known=domain_context_known,
        )
        statements = [
            statement
            for variant in conflict.get("variants", [])
            for statement in variant.get("statements", [])
            if isinstance(statement, str)
        ]
        probe = {
            "stable_key": conflict.get("stable_key"),
            "statement": " ".join(statements),
            "activation_triggers": conflict.get("activation_triggers", []),
            "evidence": [{"quote": statement} for statement in statements],
        }
        score, match_reasons = _relevance(probe, query)
        records_by_key[str(conflict.get("stable_key"))].append(
            {
                "kind": "conflict",
                "source": conflict,
                "state": state,
                "context_reason": reason,
                "scope_rank": int(
                    conflict.get("scope_rank", _scope_rank(scope))
                ),
                "score": score,
                "match_reasons": match_reasons,
            }
        )

    suppressed: list[dict[str, Any]] = []
    ranked = []
    relevant_conflicts: list[dict[str, Any]] = []
    unresolved_context: list[dict[str, Any]] = []
    excluded_by_context: list[dict[str, Any]] = []
    for stable_key, records in sorted(records_by_key.items()):
        contextual_records = [
            item for item in records if item["state"] in {"applicable", "unknown"}
        ]
        if not contextual_records:
            continue
        applicable = [item for item in records if item["state"] == "applicable"]
        unknown = [item for item in records if item["state"] == "unknown"]
        top_applicable_rank = max(
            (item["scope_rank"] for item in applicable),
            default=-1,
        )
        blockers = [
            item
            for item in unknown
            if top_applicable_rank < 0 or item["scope_rank"] >= top_applicable_rank
        ]
        top_records = [
            item for item in applicable if item["scope_rank"] == top_applicable_rank
        ]
        relevance_records = top_records + blockers
        if not relevance_records:
            continue
        best_match = max(
            relevance_records,
            key=lambda item: (float(item["score"]), item["scope_rank"]),
        )
        key_score = float(best_match["score"])
        if key_score < 0.12:
            continue

        if blockers:
            blocker_details = []
            for item in sorted(
                blockers,
                key=lambda candidate: (
                    -candidate["scope_rank"],
                    candidate["kind"],
                    _canonical_json(candidate["source"].get("scope") or {}),
                ),
            ):
                detail = {
                    "kind": item["kind"],
                    "scope": item["source"].get("scope") or {},
                    "reason": item["context_reason"],
                }
                if item["kind"] == "memory":
                    detail["preview_memory_id"] = item["source"]["preview_memory_id"]
                blocker_details.append(detail)
                excluded_by_context.append(
                    {
                        "stable_key": stable_key,
                        **detail,
                    }
                )
            unresolved_context.append(
                {
                    "stable_key": stable_key,
                    "status": "context_required",
                    "reason": "more_specific_or_equal_scope_context_is_unknown",
                    "score": key_score,
                    "match_reasons": best_match["match_reasons"],
                    "blockers": blocker_details,
                }
            )
            continue
        if top_applicable_rank < 0:
            continue
        top_conflicts = [item for item in top_records if item["kind"] == "conflict"]
        if top_conflicts:
            for item in top_conflicts:
                conflict = dict(item["source"])
                conflict_score = float(item["score"])
                if conflict_score >= 0.12:
                    conflict["score"] = conflict_score
                    conflict["match_reasons"] = item["match_reasons"]
                else:
                    conflict["score"] = key_score
                    conflict["match_reasons"] = [
                        "scope_conflict_for_relevant_key"
                    ]
                relevant_conflicts.append(conflict)
            continue

        winners = [item for item in top_records if item["kind"] == "memory"]
        for item in winners:
            memory = item["source"]
            winner_score = float(item["score"])
            if winner_score >= 0.12:
                output_score = winner_score
                output_reasons = item["match_reasons"]
            else:
                output_score = key_score
                output_reasons = ["scope_override_for_relevant_key"]
            ranked.append(
                {
                    "preview_memory_id": memory["preview_memory_id"],
                    "stable_key": memory["stable_key"],
                    "memory_type": memory["memory_type"],
                    "scope": memory["scope"],
                    "normalized_value": memory["normalized_value"],
                    "statement": memory["statement"],
                    "evidence_count": memory["evidence_count"],
                    "score": output_score,
                    "match_reasons": output_reasons,
                }
            )
        for item in applicable:
            if item["kind"] != "memory" or item in winners:
                continue
            memory = item["source"]
            suppressed.append(
                {
                    "preview_memory_id": memory["preview_memory_id"],
                    "stable_key": stable_key,
                    "scope": memory["scope"],
                    "normalized_value": memory["normalized_value"],
                    "reason": "more_specific_scope_wins",
                }
            )

    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            -_scope_rank(item["scope"]),
            item["stable_key"],
        )
    )
    return {
        "recall_version": RECALL_VERSION,
        "query": query.strip(),
        "context": {
            "project_key": project_key,
            "project_context_known": project_context_known,
            "domain": effective_domain,
            "domain_context_known": domain_context_known,
        },
        "selected_count": min(len(ranked), limit),
        "selected": ranked[:limit],
        "suppressed_by_scope": suppressed,
        "excluded_by_context": excluded_by_context,
        "unresolved_context": unresolved_context,
        "unresolved_conflicts": relevant_conflicts,
    }


def run_recall(
    lab_database: Path,
    query: str,
    *,
    project_directory: Path | None = None,
    domain: str | None = None,
    limit: int = 5,
    shadow_run_id: str | None = None,
) -> dict[str, Any]:
    """Build the current preview catalog and recall applicable memories, read-only."""

    require_memory_lab_database(lab_database)
    with closing(open_read_only(lab_database)) as connection:
        def operation() -> dict[str, Any]:
            catalog = _catalog_from_connection(connection, shadow_run_id)
            project_key, resolved_directory = _known_project_key(
                connection, project_directory
            )
            project_known = project_key is not None
            recall = recall_from_catalog(
                catalog,
                query,
                project_key=project_key,
                project_context_known=project_known,
                domain=domain,
                limit=limit,
            )
            return {
                "status": "complete",
                "preview_only": True,
                "shadow_run_id": catalog["shadow_run_id"],
                "formal_state_unchanged": catalog["formal_state_unchanged"],
                "project_directory": resolved_directory,
                "project_context_known": project_known,
                "catalog_summary": {
                    "positive_judgments": catalog["positive_judgments"],
                    "unique_memory_count": catalog["unique_memory_count"],
                    "same_scope_conflict_count": catalog["same_scope_conflict_count"],
                },
                **recall,
            }

        return _consistent_read(connection, operation)
