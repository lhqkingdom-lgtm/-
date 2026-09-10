"""Deterministic, proposal-first U2 memory consolidation.

Admission produces a complete candidate snapshot.  This module freezes every
actionable candidate into an immutable proposal, requires a review of that exact
proposal, and applies it only when the formal-memory baseline still matches.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .store import now_iso, open_store
from .safety import require_memory_lab_database


CONSOLIDATOR_VERSION = "proposal-consolidator-v2"
SCOPE_SCHEMA_VERSION = "scope-v2"
SUPPORTED_STABLE_KEYS = {
    "preference.response.language",
    "preference.response.structure",
    "workflow.code_generation.confirmation",
    "workflow.embedded.code_change.compile_check",
    "toolchain.tg032plus.compiler",
}

_DISPLAY_SCOPE_FIELDS = {"directory", "override_policy"}
_CONDITION_FIELDS = {
    "architecture",
    "board",
    "branch",
    "chip",
    "compiler",
    "environment",
    "framework",
    "language",
    "os",
    "platform",
    "repository",
    "runtime",
    "sdk",
    "time",
    "toolchain",
    "version",
}
_DOMAIN_ALIASES = {
    "embedded": "embedded",
    "embedded systems": "embedded",
    "单片机": "embedded",
    "嵌入式": "embedded",
    "嵌入式系统": "embedded",
}
_RELATIVE_TIME = re.compile(
    r"(?:today|tomorrow|yesterday|now|current|this[_ -]?(?:week|month|year)|"
    r"今天|明天|昨天|现在|当前|本周|本月|今年|以后|之后|临时|暂时)",
    re.I,
)
_NEGATION = r"(?:不需要|无需|不用|不必|不要(?:再)?|不再|取消|禁止|不得|别)"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: object, *, prefix: str = "") -> str:
    digest = hashlib.sha256(_json(value).encode("utf-8")).hexdigest()
    return prefix + digest[:24]


def _normalized_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized or None


def canonicalize_scope(
    scope: dict,
    *,
    event_project_key: str | None = None,
    connection: sqlite3.Connection | None = None,
) -> dict | None:
    """Return the full identity-bearing scope, failing closed on unknown fields.

    Display-only fields are discarded.  Conditions are identity fields: two
    otherwise equal rules for different versions, toolchains, chips, or time
    windows must not overwrite one another.
    """
    if not isinstance(scope, dict):
        return None
    normalized_keys: dict[str, object] = {}
    for raw_key, value in scope.items():
        key = _normalized_text(raw_key)
        if key is None or key in normalized_keys:
            return None
        normalized_keys[key] = value

    allowed = {"level", "domain", "project_key", "conditions"} | _DISPLAY_SCOPE_FIELDS | _CONDITION_FIELDS
    if set(normalized_keys) - allowed:
        return None

    level = _normalized_text(normalized_keys.get("level"))
    if level not in {"global", "domain", "project"}:
        return None
    result: dict[str, object] = {"level": level}

    if level == "global":
        if "domain" in normalized_keys or "project_key" in normalized_keys:
            return None
    elif level == "domain":
        if "project_key" in normalized_keys:
            return None
        domain = _normalized_text(normalized_keys.get("domain"))
        if domain is None:
            return None
        result["domain"] = _DOMAIN_ALIASES.get(domain, domain)
    else:
        if "domain" in normalized_keys:
            return None
        project_key = _normalized_text(normalized_keys.get("project_key"))
        observed_project = _normalized_text(event_project_key) if event_project_key is not None else None
        if project_key is None or (observed_project is not None and project_key != observed_project):
            return None
        if connection is not None:
            known = any(
                _normalized_text(row[0]) == project_key
                for row in connection.execute("SELECT project_key FROM project_context")
            )
            if not known:
                return None
        result["project_key"] = project_key

    condition_values: dict[str, object] = {}
    nested = normalized_keys.get("conditions", {})
    if not isinstance(nested, dict):
        return None
    for raw_key, value in nested.items():
        key = _normalized_text(raw_key)
        if key not in _CONDITION_FIELDS or key in condition_values:
            return None
        condition_values[key] = value
    for key in _CONDITION_FIELDS:
        if key in normalized_keys:
            if key in condition_values:
                return None
            condition_values[key] = normalized_keys[key]

    conditions: dict[str, str] = {}
    for key, raw_value in condition_values.items():
        value = _normalized_text(raw_value)
        if value is None or (key == "time" and _RELATIVE_TIME.search(value)):
            return None
        conditions[key] = value
    if conditions:
        result["conditions"] = dict(sorted(conditions.items()))
    return result


def scope_fingerprint(scope: dict) -> str | None:
    canonical = canonicalize_scope(scope)
    return _hash(canonical, prefix="scope_") if canonical is not None else None


def _has_negative_relation(text: str, concepts: str) -> bool:
    return bool(
        re.search(rf"{_NEGATION}.{{0,16}}(?:{concepts})", text, re.I)
        or re.search(rf"(?:{concepts}).{{0,12}}{_NEGATION}", text, re.I)
    )


def semantic_value_for(stable_key: str, quote: str) -> str | None:
    """Map supported stable keys to comparable values; ambiguity fails closed."""
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", quote)).casefold()
    if stable_key == "preference.response.language":
        labels = {"中文": "zh", "汉语": "zh", "英文": "en", "英语": "en"}
        language_negative = rf"(?:{_NEGATION}|不建议|避免|尽量不|最好不)"
        negative_targets = {
            value
            for label, value in labels.items()
            if re.search(rf"{language_negative}[^，。；,;!?！？]{{0,10}}(?:使用)?{label}", text)
        }
        targets: list[str] = []
        for match in re.finditer(
            r"(?:统一(?:使用)?|默认(?:使用)?|改为|改用|请用|使用)(中文|汉语|英文|英语)",
            text,
        ):
            prefix = text[max(0, match.start() - 12) : match.start()]
            if re.search(rf"{language_negative}[^，。；,;!?！？]{{0,8}}$", prefix):
                continue
            targets.append(labels[match.group(1)])
        if targets:
            unique_targets = set(targets)
            return (
                targets[-1]
                if len(unique_targets) == 1 and unique_targets.isdisjoint(negative_targets)
                else None
            )
        if not negative_targets:
            mentioned = {value for label, value in labels.items() if label in text}
            if len(mentioned) == 1:
                return mentioned.pop()
        return None

    if stable_key == "preference.response.structure":
        detail = r"(?:步骤|过程|细节|理由|依据)"
        conclusion = r"(?:结论|答案|结果|要点|摘要)"
        if re.search(rf"(?:{_NEGATION}.{{0,12}}(?:先|首先)|(?:先|首先).{{0,6}}{_NEGATION})", text):
            return None
        if re.search(rf"(?:先|首先).{{0,18}}{detail}.{{0,18}}(?:再|然后|随后).{{0,18}}{conclusion}", text):
            return "details_then_conclusion"
        if re.search(rf"(?:先|首先).{{0,18}}{conclusion}.{{0,18}}(?:再|然后|随后).{{0,18}}{detail}", text):
            return "conclusion_then_details"
        return None

    if stable_key == "workflow.code_generation.confirmation":
        # "禁止未经确认就生成" is a positive guard, not a ban on confirmation.
        if re.search(r"(?:禁止|不得|不要|别).{0,8}(?:跳过|省略|取消).{0,6}(?:确认|问我|询问)", text):
            return "required"
        if re.search(r"(?:禁止|不得|不允许|不要|别).{0,8}未经(?:我)?确认.{0,8}(?:生成|创建|写|输出)", text):
            return "required"
        if re.search(r"未经(?:我)?确认.{0,8}(?:不要|别|不得|禁止).{0,8}(?:生成|创建|写|输出)", text):
            return "required"
        if _has_negative_relation(text, r"确认|问我|询问"):
            return "not_required"
        if any(word in text for word in ("确认", "问我", "询问")):
            return "required"
        return None

    if stable_key == "workflow.embedded.code_change.compile_check":
        if re.search(r"(?:禁止|不得|不要|别).{0,8}(?:跳过|省略|取消).{0,6}(?:编译检查|编译)", text):
            return "required"
        if re.search(r"(?:禁止|不得|不允许|不要|别).{0,8}未经(?:编译检查|编译).{0,8}(?:提交|发布|交付|合并)", text):
            return "required"
        if re.search(r"未经(?:编译检查|编译).{0,8}(?:不要|别|不得|禁止).{0,8}(?:提交|发布|交付|合并)", text):
            return "required"
        if _has_negative_relation(text, r"编译检查|编译"):
            return "not_required"
        return "required" if "编译" in text else None

    if stable_key == "toolchain.tg032plus.compiler":
        compiler_patterns = {
            "keil_c51": r"keil(?:c51)?|c51",
            "iar": r"iar",
            "cmake": r"cmake",
            "makefile": r"makefile|(?:^|[^a-z])make(?:$|[^a-z])",
        }
        explicit: list[str] = []
        for value, compiler in compiler_patterns.items():
            for match in re.finditer(
                rf"(?:统一(?:使用)?|改为|改用|采用|使用)(?:{compiler})", text, re.I
            ):
                prefix = text[max(0, match.start() - 10) : match.start()]
                if re.search(rf"{_NEGATION}.{{0,6}}$", prefix):
                    continue
                explicit.append(value)
                break
        if explicit:
            return explicit[-1] if len(set(explicit)) == 1 else None
        mentioned = {
            value for value, compiler in compiler_patterns.items() if re.search(compiler, text, re.I)
        }
        return mentioned.pop() if len(mentioned) == 1 and not re.search(_NEGATION, text) else None
    return None


def _statement(stable_key: str, value: str) -> str:
    statements = {
        ("preference.response.language", "zh"): "默认使用中文回复。",
        ("preference.response.language", "en"): "默认使用英文回复。",
        ("preference.response.structure", "conclusion_then_details"): "回复时先给结论，再说明步骤或细节。",
        ("preference.response.structure", "details_then_conclusion"): "回复时先说明步骤或细节，再给结论。",
        ("workflow.code_generation.confirmation", "required"): "生成正式代码前必须先征得用户确认。",
        ("workflow.code_generation.confirmation", "not_required"): "生成正式代码前不要求先征得用户确认。",
        ("workflow.embedded.code_change.compile_check", "required"): "单片机项目修改代码后必须进行编译检查。",
        ("workflow.embedded.code_change.compile_check", "not_required"): "单片机项目修改代码后不要求进行编译检查。",
        ("toolchain.tg032plus.compiler", "keil_c51"): "TG032Plus 项目统一使用 Keil C51 编译。",
        ("toolchain.tg032plus.compiler", "iar"): "TG032Plus 项目统一使用 IAR 编译。",
        ("toolchain.tg032plus.compiler", "cmake"): "TG032Plus 项目统一使用 CMake 编译。",
        ("toolchain.tg032plus.compiler", "makefile"): "TG032Plus 项目统一使用 Makefile 编译。",
    }
    return statements.get((stable_key, value), f"{stable_key} = {value}")


def _latest_admission_snapshot(
    connection: sqlite3.Connection,
) -> tuple[sqlite3.Row | None, list[sqlite3.Row], str | None]:
    run = connection.execute(
        """
        SELECT rowid AS run_sequence, *
        FROM admission_run
        WHERE status = 'complete'
        ORDER BY COALESCE(completed_at, started_at) DESC, started_at DESC, rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if run is None:
        return None, [], None
    if run["completed_at"] is None:
        return run, [], None
    rows = connection.execute(
        """
        SELECT d.*, e.source_id, e.session_id, e.message_id, e.part_id,
               e.directory, e.timestamp,
               e.project_id AS source_project_id, e.timestamp AS occurred_at,
               e.content_hash, COALESCE(a.project_key, sc.project_key) AS project_key,
               COALESCE(a.semantic_content, e.content) AS semantic_content,
               COALESCE(a.semantic_content, e.content) AS evidence_source_text
        FROM admission_decision AS d
        JOIN event AS e ON e.event_id = d.event_id
        JOIN session_context AS sc
          ON sc.source_id = e.source_id AND sc.session_id = e.session_id
        LEFT JOIN event_annotation AS a ON a.event_id = e.event_id
        WHERE d.run_id = ?
          AND e.eligibility_status = 'eligible'
          AND e.parent_session_id IS NULL
          AND EXISTS(
              SELECT 1
              FROM turn_event AS te
              JOIN conversation_turn AS t ON t.turn_id = te.turn_id
              WHERE te.event_id = e.event_id
                AND t.current_state = 'completed'
          )
        ORDER BY (e.timestamp IS NULL), e.timestamp, e.event_id, d.ordinal, d.candidate_key
        """,
        (run["run_id"],),
    ).fetchall()
    raw_decision_count = connection.execute(
        "SELECT COUNT(*) FROM admission_decision WHERE run_id = ?", (run["run_id"],)
    ).fetchone()[0]
    if raw_decision_count != run["decision_count"] or len(rows) != raw_decision_count:
        return run, rows, None
    snapshot_hash = _hash(
        {
            "run_id": run["run_id"],
            "decisions": [
                {
                    key: row[key]
                    for key in (
                        "decision_id", "candidate_key", "event_id", "ordinal",
                        "exact_quote", "start_offset", "end_offset", "family",
                        "memory_type", "stable_key_suggestion", "scope_json",
                        "activation_triggers_json", "disposition", "signals_json",
                        "instruction_coverage_json", "reasons_json", "decided_at",
                        "source_id", "session_id", "message_id", "part_id",
                        "source_project_id", "occurred_at", "content_hash",
                        "project_key", "evidence_source_text",
                    )
                }
                for row in rows
            ],
        },
        prefix="snapshot_",
    )
    return run, rows, snapshot_hash


def latest_admission_snapshot(
    connection: sqlite3.Connection,
) -> tuple[sqlite3.Row | None, list[sqlite3.Row], str | None]:
    """Expose the same fail-closed snapshot contract to read-only shadow/report lanes."""

    return _latest_admission_snapshot(connection)


def _open_conflicts_hash(connection: sqlite3.Connection, memory_id: str | None) -> str:
    rows = []
    if memory_id is not None:
        rows = [
            list(row)
            for row in connection.execute(
                """
                SELECT conflict_id, current_revision_id, incoming_semantic_value
                FROM conflict WHERE memory_id = ? AND status = 'open'
                ORDER BY conflict_id
                """,
                (memory_id,),
            )
        ]
    return _hash(rows, prefix="conflicts_")


def _memory_for_identity(
    connection: sqlite3.Connection, stable_key: str, fingerprint: str
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT m.*, r.semantic_value AS current_semantic_value,
               r.status AS current_revision_status,
               r.source_proposal_id AS current_source_proposal_id,
               r.review_id AS current_review_id,
               p.application_payload_json AS current_source_payload_json,
               rv.proposal_id AS current_review_proposal_id
        FROM memory AS m
        LEFT JOIN memory_revision AS r ON r.revision_id = m.current_revision_id
        LEFT JOIN memory_proposal AS p ON p.proposal_id = r.source_proposal_id
        LEFT JOIN candidate_review AS rv ON rv.review_id = r.review_id
        WHERE m.stable_key = ? AND m.scope_fingerprint = ?
        """,
        (stable_key, fingerprint),
    ).fetchone()


def _relation_for(
    connection: sqlite3.Connection,
    stable_key: str,
    fingerprint: str,
    semantic_value: str,
    is_correction: bool,
) -> tuple[str, sqlite3.Row | None]:
    memory = _memory_for_identity(connection, stable_key, fingerprint)
    if memory is None:
        variant = connection.execute(
            "SELECT 1 FROM memory WHERE stable_key = ? AND scope_fingerprint <> ? LIMIT 1",
            (stable_key, fingerprint),
        ).fetchone()
        return ("create_scope_variant" if variant else "create"), None
    if memory["status"] != "active" or memory["current_revision_id"] is None:
        return "unsupported_memory_state", memory
    if (
        not memory["family"]
        or not memory["current_source_proposal_id"]
        or not memory["current_review_id"]
        or not memory["current_source_payload_json"]
        or memory["current_review_proposal_id"] != memory["current_source_proposal_id"]
    ):
        return "legacy_formal_memory_requires_manual_migration", memory
    if memory["current_semantic_value"] == semantic_value:
        return ("confirm" if is_correction else "merge"), memory
    return ("revise" if is_correction else "conflict"), memory


def _analyze_decision(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    admission_snapshot_hash: str,
    *,
    reviewed_identity: dict | None = None,
) -> tuple[dict | None, str | None]:
    try:
        signals = json.loads(row["signals_json"])
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_signals_json"
    is_correction = bool(signals.get("correction"))
    if reviewed_identity is not None:
        # The semantic bridge is deliberately narrow: it is only available to
        # the formal Writer/Reviewer pipeline for a durable user-contract claim
        # that rule admission could not map to the bounded ontology.  Scope and
        # evidence still come exclusively from the immutable admission row.
        if (
            row["disposition"] != "needs_semantic_review"
            or row["family"] != "user_contract"
            or is_correction
        ):
            return None, "reviewed_identity_not_allowed_for_admission"
        if set(reviewed_identity) != {"stable_key", "semantic_value", "memory_type"}:
            return None, "invalid_reviewed_identity"
        stable_key = reviewed_identity["stable_key"]
        semantic_value = reviewed_identity["semantic_value"]
        memory_type = reviewed_identity["memory_type"]
        reviewed_values = {
            "preference.response.language": {"zh", "en"},
            "preference.response.structure": {
                "conclusion_then_details",
                "details_then_conclusion",
            },
            "workflow.code_generation.confirmation": {"required", "not_required"},
            "workflow.embedded.code_change.compile_check": {"required", "not_required"},
            "toolchain.tg032plus.compiler": {"keil_c51", "iar", "cmake", "makefile"},
        }
        if semantic_value not in reviewed_values.get(stable_key, set()):
            return None, "reviewed_identity_outside_bounded_ontology"
        if memory_type not in {
            "preference",
            "response_preference",
            "workflow_rule",
            "project_rule",
        }:
            return None, "reviewed_memory_type_unsupported"
    else:
        if is_correction:
            if row["disposition"] != "needs_correction_review":
                return None, "correction_not_in_review_disposition"
        elif row["disposition"] != "eligible":
            return None, "admission_disposition_not_activatable"
        stable_key = row["stable_key_suggestion"]
        semantic_value = semantic_value_for(stable_key, row["exact_quote"])
        memory_type = row["memory_type"] or "user_contract"

    if stable_key not in SUPPORTED_STABLE_KEYS:
        return None, "unsupported_stable_key"
    try:
        raw_scope = json.loads(row["scope_json"])
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_scope_json"
    canonical_scope = canonicalize_scope(
        raw_scope, event_project_key=row["project_key"], connection=connection
    )
    if canonical_scope is None:
        return None, "scope_not_actionable"
    fingerprint = _hash(canonical_scope, prefix="scope_")
    if semantic_value is None:
        return None, "unsupported_semantic_value"
    try:
        activation_triggers = json.loads(row["activation_triggers_json"])
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_activation_triggers_json"
    if not isinstance(activation_triggers, list):
        return None, "invalid_activation_triggers_json"
    if not isinstance(row["exact_quote"], str):
        return None, "invalid_evidence_span"
    offsets = (row["ordinal"], row["start_offset"], row["end_offset"])
    if (
        not all(isinstance(value, int) for value in offsets)
        or row["ordinal"] < 1
        or row["start_offset"] < 0
        or row["end_offset"] < row["start_offset"]
        or not isinstance(row["evidence_source_text"], str)
        or row["end_offset"] > len(row["evidence_source_text"])
        or row["evidence_source_text"][row["start_offset"] : row["end_offset"]]
        != row["exact_quote"]
    ):
        return None, "invalid_evidence_span"
    application_payload = {
        "family": row["family"],
        "memory_type": memory_type,
        "activation_triggers": activation_triggers,
        "event_id": row["event_id"],
        "source_id": row["source_id"],
        "session_id": row["session_id"],
        "message_id": row["message_id"],
        "part_id": row["part_id"],
        "project_key": row["project_key"],
        "source_project_id": row["source_project_id"],
        "exact_quote": row["exact_quote"],
        "ordinal": row["ordinal"],
        "start_offset": row["start_offset"],
        "end_offset": row["end_offset"],
        "occurred_at": row["occurred_at"],
    }
    relation, memory = _relation_for(
        connection, stable_key, fingerprint, semantic_value, is_correction
    )
    if relation in {
        "unsupported_memory_state",
        "legacy_formal_memory_requires_manual_migration",
    }:
        return None, relation
    return {
        "admission_run_id": row["run_id"],
        "admission_snapshot_hash": admission_snapshot_hash,
        "decision_id": row["decision_id"],
        "candidate_key": row["candidate_key"],
        "stable_key": stable_key,
        "semantic_value": semantic_value,
        "raw_scope_json": _json(raw_scope),
        "canonical_scope_json": _json(canonical_scope),
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "scope_fingerprint": fingerprint,
        "relation": relation,
        "disposition": row["disposition"],
        "is_correction": int(is_correction),
        "expected_memory_id": memory["memory_id"] if memory else None,
        "expected_current_revision_id": memory["current_revision_id"] if memory else None,
        "expected_current_semantic_value": memory["current_semantic_value"] if memory else None,
        "expected_memory_state_version": memory["state_version"] if memory else None,
        "expected_open_conflicts_hash": _open_conflicts_hash(
            connection, memory["memory_id"] if memory else None
        ),
        "application_payload_json": _json(application_payload),
    }, None


def _proposal_fingerprint(spec: dict) -> str:
    return _hash(spec, prefix="proposal_")


def _insert_or_get_proposal(connection: sqlite3.Connection, spec: dict) -> tuple[sqlite3.Row, bool]:
    fingerprint = _proposal_fingerprint(spec)
    existing = connection.execute(
        "SELECT rowid AS proposal_sequence, * FROM memory_proposal WHERE proposal_fingerprint = ?",
        (fingerprint,),
    ).fetchone()
    if existing is not None:
        return existing, False
    proposal_id = "proposal_" + uuid4().hex
    connection.execute(
        """
        INSERT INTO memory_proposal(
            proposal_id, proposal_fingerprint, admission_run_id,
            admission_snapshot_hash, decision_id,
            candidate_key, stable_key, semantic_value, raw_scope_json,
            canonical_scope_json, scope_schema_version, scope_fingerprint,
            relation, disposition, is_correction, expected_memory_id,
            expected_current_revision_id, expected_current_semantic_value,
            expected_memory_state_version,
            expected_open_conflicts_hash, application_payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            proposal_id,
            fingerprint,
            spec["admission_run_id"],
            spec["admission_snapshot_hash"],
            spec["decision_id"],
            spec["candidate_key"],
            spec["stable_key"],
            spec["semantic_value"],
            spec["raw_scope_json"],
            spec["canonical_scope_json"],
            spec["scope_schema_version"],
            spec["scope_fingerprint"],
            spec["relation"],
            spec["disposition"],
            spec["is_correction"],
            spec["expected_memory_id"],
            spec["expected_current_revision_id"],
            spec["expected_current_semantic_value"],
            spec["expected_memory_state_version"],
            spec["expected_open_conflicts_hash"],
            spec["application_payload_json"],
            now_iso(),
        ),
    )
    return connection.execute(
        "SELECT rowid AS proposal_sequence, * FROM memory_proposal WHERE proposal_id = ?",
        (proposal_id,),
    ).fetchone(), True


def _latest_proposal(connection: sqlite3.Connection, decision_id: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT rowid AS proposal_sequence, * FROM memory_proposal
        WHERE decision_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1
        """,
        (decision_id,),
    ).fetchone()


def _latest_review(connection: sqlite3.Connection, proposal_id: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT rowid AS review_sequence, * FROM candidate_review
        WHERE proposal_id = ? ORDER BY reviewed_at DESC, rowid DESC LIMIT 1
        """,
        (proposal_id,),
    ).fetchone()


def _proposal_payload(proposal: sqlite3.Row) -> dict:
    raw = proposal["application_payload_json"]
    if not raw:
        raise ValueError("proposal is missing its frozen application payload")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("proposal application payload must be an object")
    return payload


def _proposal_statement(proposal: sqlite3.Row) -> str:
    """Use a reviewed free-form statement when the frozen payload supplies it.

    Legacy bounded-ontology proposals intentionally omit ``statement`` and keep
    their canonical lookup-table rendering.  Generic unified proposals freeze
    the already validated statement in the immutable application payload; it
    must never degrade to ``stable_key = semantic_fingerprint``.
    """

    payload = _proposal_payload(proposal)
    statement = payload.get("statement")
    if statement is None:
        return _statement(proposal["stable_key"], proposal["semantic_value"])
    if (
        not isinstance(statement, str)
        or not statement.strip()
        or len(statement) > 2_000
        or "\x00" in statement
    ):
        raise ValueError("proposal frozen statement is invalid")
    return statement.strip()


def _proposal_projection(proposal: sqlite3.Row) -> dict:
    """Return the complete frozen fields a human must see before review."""
    return {
        "proposal_id": proposal["proposal_id"],
        "admission_run_id": proposal["admission_run_id"],
        "admission_snapshot_hash": proposal["admission_snapshot_hash"],
        "decision_id": proposal["decision_id"],
        "candidate_key": proposal["candidate_key"],
        "stable_key": proposal["stable_key"],
        "semantic_value": proposal["semantic_value"],
        "canonical_scope": json.loads(proposal["canonical_scope_json"]),
        "scope_schema_version": proposal["scope_schema_version"],
        "relation": proposal["relation"],
        "disposition": proposal["disposition"],
        "is_correction": bool(proposal["is_correction"]),
        "expected_memory_id": proposal["expected_memory_id"],
        "expected_current_revision_id": proposal["expected_current_revision_id"],
        "expected_current_semantic_value": proposal["expected_current_semantic_value"],
        "expected_memory_state_version": proposal["expected_memory_state_version"],
        "expected_open_conflicts_hash": proposal["expected_open_conflicts_hash"],
        "application_payload": _proposal_payload(proposal),
    }


def _baseline_matches(connection: sqlite3.Connection, proposal: sqlite3.Row) -> tuple[bool, str]:
    memory = _memory_for_identity(
        connection, proposal["stable_key"], proposal["scope_fingerprint"]
    )
    expected_memory_id = proposal["expected_memory_id"]
    if expected_memory_id is None:
        if memory is not None:
            return False, "memory_identity_appeared"
        variant_exists = connection.execute(
            "SELECT 1 FROM memory WHERE stable_key = ? AND scope_fingerprint <> ? LIMIT 1",
            (proposal["stable_key"], proposal["scope_fingerprint"]),
        ).fetchone() is not None
        expected_variant = proposal["relation"] == "create_scope_variant"
        if variant_exists != expected_variant:
            return False, "scope_relation_changed"
        return True, "baseline_matches"
    if memory is None or memory["memory_id"] != expected_memory_id:
        return False, "memory_identity_changed"
    if memory["status"] != "active" or memory["current_revision_status"] != "current":
        return False, "formal_memory_state_invalid"
    if memory["scope_json"] != proposal["canonical_scope_json"]:
        return False, "formal_memory_scope_changed"
    if memory["current_revision_id"] != proposal["expected_current_revision_id"]:
        return False, "current_revision_changed"
    if memory["current_semantic_value"] != proposal["expected_current_semantic_value"]:
        return False, "current_semantic_value_changed"
    if memory["state_version"] != proposal["expected_memory_state_version"]:
        return False, "memory_state_version_changed"
    if _open_conflicts_hash(connection, memory["memory_id"]) != proposal["expected_open_conflicts_hash"]:
        return False, "open_conflict_set_changed"
    return True, "baseline_matches"


def _prior_evidence_state(connection: sqlite3.Connection, proposal: sqlite3.Row) -> str | None:
    evidence = connection.execute(
        """
        SELECT e.memory_id, m.stable_key, m.scope_fingerprint,
               COALESCE(r.semantic_value, c.incoming_semantic_value) AS semantic_value
        FROM evidence AS e
        JOIN memory AS m ON m.memory_id = e.memory_id
        LEFT JOIN memory_revision AS r ON r.revision_id = e.revision_id
        LEFT JOIN conflict AS c ON c.conflict_id = e.conflict_id
        WHERE e.candidate_key = ?
        """,
        (proposal["candidate_key"],),
    ).fetchone()
    if evidence is None:
        return None
    if (
        evidence["stable_key"] == proposal["stable_key"]
        and evidence["scope_fingerprint"] == proposal["scope_fingerprint"]
        and evidence["semantic_value"] == proposal["semantic_value"]
    ):
        return "same"
    return "reinterpreted"


def record_proposal_review(
    lab_database: Path,
    proposal_id: str,
    outcome: str,
    reviewer: str,
    note: str | None = None,
) -> dict:
    """Append a human decision bound to exactly one immutable proposal."""
    require_memory_lab_database(lab_database)
    if outcome not in {"approve", "reject"}:
        raise ValueError("review outcome must be 'approve' or 'reject'")
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer must not be empty")
    with closing(open_store(lab_database)) as connection, connection:
        latest_run, rows, snapshot_hash = _latest_admission_snapshot(connection)
        proposal = connection.execute(
            "SELECT rowid AS proposal_sequence, * FROM memory_proposal WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_id}")
        if latest_run is None or proposal["admission_run_id"] != latest_run["run_id"]:
            raise ValueError("proposal is not part of the latest complete admission snapshot")
        if snapshot_hash is None:
            raise ValueError("latest complete admission snapshot is structurally incomplete")
        if not proposal["admission_snapshot_hash"] or proposal["admission_snapshot_hash"] != snapshot_hash:
            raise ValueError("proposal admission snapshot is stale")
        if proposal["decision_id"] not in {row["decision_id"] for row in rows}:
            raise ValueError("proposal decision is absent from the latest complete admission snapshot")
        latest = _latest_proposal(connection, proposal["decision_id"])
        if latest is None or latest["proposal_id"] != proposal_id:
            raise ValueError("proposal has been replaced; review the latest proposal")
        if outcome == "approve":
            try:
                _proposal_payload(proposal)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"proposal frozen payload is invalid: {error}") from error
            matches, reason = _baseline_matches(connection, proposal)
            if not matches:
                raise ValueError(f"proposal baseline is stale: {reason}")
            if _prior_evidence_state(connection, proposal) == "reinterpreted":
                raise ValueError("candidate already has evidence under a different frozen identity")
        review_id = "review_" + uuid4().hex
        reviewed_at = now_iso()
        connection.execute(
            """
            INSERT INTO candidate_review(
                review_id, proposal_id, decision_id, candidate_key,
                outcome, reviewer, note, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                proposal_id,
                proposal["decision_id"],
                proposal["candidate_key"],
                outcome,
                reviewer,
                note,
                reviewed_at,
            ),
        )
    return {
        "review_id": review_id,
        "proposal_id": proposal_id,
        "candidate_key": proposal["candidate_key"],
        "decision_id": proposal["decision_id"],
        "outcome": outcome,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "proposal": _proposal_projection(proposal),
    }


def record_candidate_review(*args, **kwargs) -> dict:
    """Reject the unsafe candidate-key review API retained for import diagnostics."""
    raise ValueError(
        "candidate-key review is disabled; run consolidate and review the exact proposal_id"
    )


def _record_action(
    connection: sqlite3.Connection,
    run_id: str,
    row: sqlite3.Row,
    action: str,
    reason: str,
    *,
    proposal: sqlite3.Row | None = None,
    review: sqlite3.Row | None = None,
    memory_id: str | None = None,
    before_revision_id: str | None = None,
    after_revision_id: str | None = None,
    conflict_id: str | None = None,
    details: dict | None = None,
) -> dict:
    action_id = "action_" + uuid4().hex
    candidate_key = proposal["candidate_key"] if proposal else row["candidate_key"]
    decision_id = proposal["decision_id"] if proposal else row["decision_id"]
    connection.execute(
        """
        INSERT INTO consolidation_action(
            action_id, run_id, candidate_key, decision_id, proposal_id,
            review_id, reviewer, action, memory_id, before_revision_id,
            after_revision_id, conflict_id, reason, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            action_id,
            run_id,
            candidate_key,
            decision_id,
            proposal["proposal_id"] if proposal else None,
            review["review_id"] if review else None,
            review["reviewer"] if review else None,
            action,
            memory_id,
            before_revision_id,
            after_revision_id,
            conflict_id,
            reason,
            _json(details or {}),
            now_iso(),
        ),
    )
    return {
        "candidate_key": candidate_key,
        "decision_id": decision_id,
        "proposal_id": proposal["proposal_id"] if proposal else None,
        "review_id": review["review_id"] if review else None,
        "reviewer": review["reviewer"] if review else None,
        "action": action,
        "reason": reason,
        "memory_id": memory_id,
        "before_revision_id": before_revision_id,
        "after_revision_id": after_revision_id,
        "conflict_id": conflict_id,
        "details": details or {},
    }


def _insert_evidence(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    proposal: sqlite3.Row,
    review: sqlite3.Row,
    memory_id: str,
    revision_id: str | None,
    role: str,
    conflict_id: str | None = None,
) -> str:
    evidence_id = "evidence_" + uuid4().hex
    payload = _proposal_payload(proposal)
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
            evidence_id,
            memory_id,
            revision_id,
            conflict_id,
            proposal["candidate_key"],
            proposal["decision_id"],
            proposal["proposal_id"],
            review["review_id"],
            review["reviewer"],
            payload["event_id"],
            payload["source_id"],
            payload["session_id"],
            payload["message_id"],
            payload["part_id"],
            payload["project_key"],
            payload["source_project_id"],
            payload["exact_quote"],
            payload["ordinal"],
            payload["start_offset"],
            payload["end_offset"],
            payload["occurred_at"],
            role,
            now_iso(),
        ),
    )
    return evidence_id


def _create_memory(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    proposal: sqlite3.Row,
    review: sqlite3.Row,
) -> tuple[str, str]:
    memory_id = "memory_" + uuid4().hex
    revision_id = "revision_" + uuid4().hex
    timestamp = now_iso()
    payload = _proposal_payload(proposal)
    connection.execute(
        """
        INSERT INTO memory(
            memory_id, stable_key, scope_fingerprint, scope_json, family,
            memory_type, status, current_revision_id, state_version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', NULL, 1, ?, ?)
        """,
        (
            memory_id,
            proposal["stable_key"],
            proposal["scope_fingerprint"],
            proposal["canonical_scope_json"],
            payload["family"],
            payload["memory_type"],
            timestamp,
            timestamp,
        ),
    )
    connection.execute(
        """
        INSERT INTO memory_revision(
            revision_id, memory_id, revision_number, semantic_value, statement,
            activation_triggers_json, status, supersedes_revision_id,
            source_decision_id, source_proposal_id, review_id, created_at
        ) VALUES (?, ?, 1, ?, ?, ?, 'current', NULL, ?, ?, ?, ?)
        """,
        (
            revision_id,
            memory_id,
            proposal["semantic_value"],
            _proposal_statement(proposal),
            _json(payload["activation_triggers"]),
            proposal["decision_id"],
            proposal["proposal_id"],
            review["review_id"],
            timestamp,
        ),
    )
    connection.execute(
        "UPDATE memory SET current_revision_id = ? WHERE memory_id = ?",
        (revision_id, memory_id),
    )
    role = "correction" if proposal["is_correction"] else "supporting"
    _insert_evidence(connection, row, proposal, review, memory_id, revision_id, role)
    return memory_id, revision_id


def _resolve_open_conflicts(
    connection: sqlite3.Connection,
    memory_id: str,
    revision_id: str,
    proposal: sqlite3.Row,
    review: sqlite3.Row,
    *,
    resolution_reason: str = "explicit_approved_correction",
) -> list[str]:
    conflict_ids = [
        row[0]
        for row in connection.execute(
            "SELECT conflict_id FROM conflict WHERE memory_id = ? AND status = 'open' ORDER BY conflict_id",
            (memory_id,),
        )
    ]
    connection.execute(
        """
        UPDATE conflict SET status = 'resolved', resolution_revision_id = ?,
            resolution_proposal_id = ?, resolution_review_id = ?,
            resolution_reviewer = ?,
            resolution_reason = ?, resolved_at = ?
        WHERE memory_id = ? AND status = 'open'
        """,
        (
            revision_id,
            proposal["proposal_id"],
            review["review_id"],
            review["reviewer"],
            resolution_reason,
            now_iso(),
            memory_id,
        ),
    )
    return conflict_ids


def _touch_memory(connection: sqlite3.Connection, memory_id: str) -> None:
    """Advance the monotonic CAS token for every accepted state mutation."""
    connection.execute(
        """
        UPDATE memory
        SET state_version = state_version + 1, updated_at = ?
        WHERE memory_id = ?
        """,
        (now_iso(), memory_id),
    )


def _revise_memory(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    proposal: sqlite3.Row,
    review: sqlite3.Row,
    memory: sqlite3.Row,
) -> tuple[str, list[str]]:
    old_revision_id = memory["current_revision_id"]
    payload = _proposal_payload(proposal)
    revision_number = connection.execute(
        "SELECT COALESCE(MAX(revision_number), 0) + 1 FROM memory_revision WHERE memory_id = ?",
        (memory["memory_id"],),
    ).fetchone()[0]
    revision_id = "revision_" + uuid4().hex
    timestamp = now_iso()
    connection.execute(
        "UPDATE memory_revision SET status = 'superseded' WHERE revision_id = ?",
        (old_revision_id,),
    )
    connection.execute(
        """
        INSERT INTO memory_revision(
            revision_id, memory_id, revision_number, semantic_value, statement,
            activation_triggers_json, status, supersedes_revision_id,
            source_decision_id, source_proposal_id, review_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'current', ?, ?, ?, ?, ?)
        """,
        (
            revision_id,
            memory["memory_id"],
            revision_number,
            proposal["semantic_value"],
            _proposal_statement(proposal),
            _json(payload["activation_triggers"]),
            old_revision_id,
            proposal["decision_id"],
            proposal["proposal_id"],
            review["review_id"],
            timestamp,
        ),
    )
    connection.execute(
        """
        UPDATE memory
        SET current_revision_id = ?, state_version = state_version + 1, updated_at = ?
        WHERE memory_id = ?
        """,
        (revision_id, timestamp, memory["memory_id"]),
    )
    _insert_evidence(
        connection, row, proposal, review, memory["memory_id"], revision_id, "correction"
    )
    return revision_id, _resolve_open_conflicts(
        connection,
        memory["memory_id"],
        revision_id,
        proposal,
        review,
        resolution_reason=(
            "explicit_approved_correction"
            if proposal["is_correction"]
            else "latest_durable_user_contract_by_source_order"
        ),
    )


def _open_or_merge_conflict(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    proposal: sqlite3.Row,
    review: sqlite3.Row,
    memory: sqlite3.Row,
) -> tuple[str, str]:
    existing = connection.execute(
        """
        SELECT * FROM conflict
        WHERE memory_id = ? AND current_revision_id = ?
          AND incoming_semantic_value = ? AND status = 'open'
        ORDER BY created_at, conflict_id LIMIT 1
        """,
        (
            memory["memory_id"],
            memory["current_revision_id"],
            proposal["semantic_value"],
        ),
    ).fetchone()
    if existing is not None:
        _insert_evidence(
            connection,
            row,
            proposal,
            review,
            memory["memory_id"],
            None,
            "conflicting",
            existing["conflict_id"],
        )
        _touch_memory(connection, memory["memory_id"])
        return existing["conflict_id"], "merged_conflict_evidence"
    conflict_id = "conflict_" + uuid4().hex
    fingerprint = _hash(
        {
            "memory_id": memory["memory_id"],
            "current_revision_id": memory["current_revision_id"],
            "semantic_value": proposal["semantic_value"],
            "proposal_id": proposal["proposal_id"],
        },
        prefix="conflict_",
    )
    connection.execute(
        """
        INSERT INTO conflict(
            conflict_id, conflict_fingerprint, memory_id, current_revision_id,
            incoming_candidate_key, incoming_decision_id, incoming_proposal_id,
            review_id, reviewer, incoming_semantic_value, incoming_statement,
            status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        """,
        (
            conflict_id,
            fingerprint,
            memory["memory_id"],
            memory["current_revision_id"],
            proposal["candidate_key"],
            proposal["decision_id"],
            proposal["proposal_id"],
            review["review_id"],
            review["reviewer"],
            proposal["semantic_value"],
            _proposal_statement(proposal),
            now_iso(),
        ),
    )
    _insert_evidence(
        connection,
        row,
        proposal,
        review,
        memory["memory_id"],
        None,
        "conflicting",
        conflict_id,
    )
    _touch_memory(connection, memory["memory_id"])
    return conflict_id, "opened_conflict"


def _apply_proposal(
    connection: sqlite3.Connection,
    run_id: str,
    row: sqlite3.Row,
    proposal: sqlite3.Row,
    review: sqlite3.Row,
) -> dict:
    prior_state = _prior_evidence_state(connection, proposal)
    if prior_state == "same":
        return _record_action(
            connection, run_id, row, "noop", "candidate_already_consolidated",
            proposal=proposal, review=review
        )
    if prior_state == "reinterpreted":
        return _record_action(
            connection, run_id, row, "stale", "candidate_reinterpreted_after_evidence",
            proposal=proposal, review=review
        )

    matches, stale_reason = _baseline_matches(connection, proposal)
    if not matches:
        replacement_spec, replacement_error = _analyze_decision(
            connection, row, proposal["admission_snapshot_hash"]
        )
        replacement_id = None
        if replacement_spec is not None:
            replacement, _ = _insert_or_get_proposal(connection, replacement_spec)
            replacement_id = replacement["proposal_id"]
        return _record_action(
            connection,
            run_id,
            row,
            "stale",
            "proposal_baseline_changed",
            proposal=proposal,
            review=review,
            details={
                "stale_reason": stale_reason,
                "replacement_proposal_id": replacement_id,
                "replacement_proposal": (
                    _proposal_projection(replacement) if replacement_id else None
                ),
                "replacement_error": replacement_error,
                "requires_new_review": True,
            },
        )

    relation = proposal["relation"]
    if relation in {"create", "create_scope_variant"}:
        memory_id, revision_id = _create_memory(connection, row, proposal, review)
        action = "created_scope_variant" if relation == "create_scope_variant" else "created"
        return _record_action(
            connection,
            run_id,
            row,
            action,
            "approved_proposal_activated",
            proposal=proposal,
            review=review,
            memory_id=memory_id,
            after_revision_id=revision_id,
        )

    memory = _memory_for_identity(
        connection, proposal["stable_key"], proposal["scope_fingerprint"]
    )
    if memory is None:
        return _record_action(
            connection, run_id, row, "stale", "expected_memory_missing",
            proposal=proposal, review=review
        )
    before = memory["current_revision_id"]
    if relation == "merge":
        _insert_evidence(
            connection, row, proposal, review, memory["memory_id"], before, "supporting"
        )
        _touch_memory(connection, memory["memory_id"])
        return _record_action(
            connection,
            run_id,
            row,
            "merged_evidence",
            "same_semantic_value_and_scope",
            proposal=proposal,
            review=review,
            memory_id=memory["memory_id"],
            before_revision_id=before,
            after_revision_id=before,
        )
    if relation == "conflict":
        conflict_id, action = _open_or_merge_conflict(
            connection, row, proposal, review, memory
        )
        return _record_action(
            connection,
            run_id,
            row,
            action,
            "opposite_value_same_scope_without_correction",
            proposal=proposal,
            review=review,
            memory_id=memory["memory_id"],
            before_revision_id=before,
            after_revision_id=before,
            conflict_id=conflict_id,
        )
    if relation == "confirm":
        _insert_evidence(
            connection, row, proposal, review, memory["memory_id"], before, "correction"
        )
        resolved = _resolve_open_conflicts(
            connection, memory["memory_id"], before, proposal, review
        )
        _touch_memory(connection, memory["memory_id"])
        return _record_action(
            connection,
            run_id,
            row,
            "confirmed_revision",
            "approved_correction_confirms_current_value",
            proposal=proposal,
            review=review,
            memory_id=memory["memory_id"],
            before_revision_id=before,
            after_revision_id=before,
            conflict_id=resolved[0] if len(resolved) == 1 else None,
            details={
                "resolved_conflicts": len(resolved),
                "resolved_conflict_ids": resolved,
            },
        )
    if relation == "revise":
        revision_id, resolved = _revise_memory(
            connection, row, proposal, review, memory
        )
        return _record_action(
            connection,
            run_id,
            row,
            "revised",
            (
                "approved_explicit_correction"
                if proposal["is_correction"]
                else "latest_durable_user_contract_by_source_order"
            ),
            proposal=proposal,
            review=review,
            memory_id=memory["memory_id"],
            before_revision_id=before,
            after_revision_id=revision_id,
            conflict_id=resolved[0] if len(resolved) == 1 else None,
            details={
                "resolved_conflicts": len(resolved),
                "resolved_conflict_ids": resolved,
            },
        )
    return _record_action(
        connection, run_id, row, "skipped", "unsupported_frozen_relation",
        proposal=proposal, review=review, details={"relation": relation}
    )


def _state(connection: sqlite3.Connection) -> dict:
    return {
        "memories": connection.execute("SELECT COUNT(*) FROM memory").fetchone()[0],
        "revisions": connection.execute("SELECT COUNT(*) FROM memory_revision").fetchone()[0],
        "evidence": connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
        "open_conflicts": connection.execute(
            "SELECT COUNT(*) FROM conflict WHERE status = 'open'"
        ).fetchone()[0],
        "proposals": connection.execute("SELECT COUNT(*) FROM memory_proposal").fetchone()[0],
    }


def run_consolidation(lab_database: Path) -> dict:
    """Prepare proposals and apply only already-approved, still-current proposals."""
    require_memory_lab_database(lab_database)
    run_id = "consolidation_" + uuid4().hex
    started_at = now_iso()
    with closing(open_store(lab_database)) as connection, connection:
        admission_run, decisions, snapshot_hash = _latest_admission_snapshot(connection)
        connection.execute(
            """
            INSERT INTO consolidation_run(
                run_id, consolidator_version, admission_run_id,
                admission_snapshot_hash, started_at, status, candidate_count
            ) VALUES (?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                run_id,
                CONSOLIDATOR_VERSION,
                admission_run["run_id"] if admission_run else None,
                snapshot_hash,
                started_at,
                len(decisions),
            ),
        )
        actions: list[dict] = []
        if admission_run is None or snapshot_hash is None:
            error = (
                "no_complete_admission_snapshot"
                if admission_run is None
                else "invalid_complete_admission_snapshot"
            )
            connection.execute(
                """
                UPDATE consolidation_run SET completed_at = ?, status = 'failed',
                    counts_json = '{}', error = ?
                WHERE run_id = ?
                """,
                (now_iso(), error, run_id),
            )
            return {
                "run_id": run_id,
                "status": "failed",
                "error": error,
                "candidate_count": 0,
                "counts": {},
                "actions": [],
                "state": _state(connection),
            }

        # Freeze a proposal before any mutation. Existing proposals are never
        # reinterpreted, so later parser changes cannot alter an approved review.
        prepared: list[tuple[sqlite3.Row, sqlite3.Row, sqlite3.Row | None]] = []
        for row in decisions:
            proposal = _latest_proposal(connection, row["decision_id"])
            if proposal is None:
                spec, error = _analyze_decision(connection, row, snapshot_hash)
                if spec is None:
                    details = {"disposition": row["disposition"]} if error == "admission_disposition_not_activatable" else None
                    actions.append(
                        _record_action(
                            connection, run_id, row, "skipped", error or "proposal_not_actionable",
                            details=details
                        )
                    )
                    continue
                prior_state = _prior_evidence_state(connection, spec)
                if prior_state == "same":
                    actions.append(
                        _record_action(
                            connection,
                            run_id,
                            row,
                            "noop",
                            "candidate_already_consolidated",
                        )
                    )
                    continue
                proposal, created = _insert_or_get_proposal(connection, spec)
                if prior_state == "reinterpreted":
                    actions.append(
                        _record_action(
                            connection,
                            run_id,
                            row,
                            "stale",
                            "candidate_reinterpreted_after_evidence",
                            proposal=proposal,
                        )
                    )
                    continue
                if created:
                    actions.append(
                        _record_action(
                            connection,
                            run_id,
                            row,
                            "proposed",
                            "immutable_proposal_requires_exact_review",
                            proposal=proposal,
                            details={"proposal": _proposal_projection(proposal)},
                        )
                    )
                    continue

            if (
                not proposal["admission_snapshot_hash"]
                or proposal["admission_snapshot_hash"] != snapshot_hash
                or not proposal["application_payload_json"]
            ):
                replacement_spec, replacement_error = _analyze_decision(
                    connection, row, snapshot_hash
                )
                replacement_id = None
                if replacement_spec is not None:
                    replacement, _ = _insert_or_get_proposal(connection, replacement_spec)
                    replacement_id = replacement["proposal_id"]
                actions.append(
                    _record_action(
                        connection,
                        run_id,
                        row,
                        "stale",
                        "proposal_admission_snapshot_changed",
                        proposal=proposal,
                        details={
                            "replacement_proposal_id": replacement_id,
                            "replacement_proposal": (
                                _proposal_projection(replacement) if replacement_id else None
                            ),
                            "replacement_error": replacement_error,
                            "requires_new_review": True,
                        },
                    )
                )
                continue

            prior_state = _prior_evidence_state(connection, proposal)
            if prior_state == "same":
                actions.append(
                    _record_action(
                        connection, run_id, row, "noop", "candidate_already_consolidated",
                        proposal=proposal
                    )
                )
                continue
            if prior_state == "reinterpreted":
                actions.append(
                    _record_action(
                        connection, run_id, row, "stale", "candidate_reinterpreted_after_evidence",
                        proposal=proposal
                    )
                )
                continue

            matches, stale_reason = _baseline_matches(connection, proposal)
            if not matches:
                replacement_spec, replacement_error = _analyze_decision(
                    connection, row, snapshot_hash
                )
                replacement_id = None
                if replacement_spec is not None:
                    replacement, _ = _insert_or_get_proposal(connection, replacement_spec)
                    replacement_id = replacement["proposal_id"]
                actions.append(
                    _record_action(
                        connection,
                        run_id,
                        row,
                        "stale",
                        "proposal_baseline_changed",
                        proposal=proposal,
                        details={
                            "stale_reason": stale_reason,
                            "replacement_proposal_id": replacement_id,
                            "replacement_error": replacement_error,
                            "requires_new_review": True,
                        },
                    )
                )
                continue
            review = _latest_review(connection, proposal["proposal_id"])
            if review is None:
                actions.append(
                    _record_action(
                        connection, run_id, row, "noop", "waiting_for_exact_proposal_review",
                        proposal=proposal,
                        details={"proposal": _proposal_projection(proposal)},
                    )
                )
                continue
            if review["outcome"] == "reject":
                actions.append(
                    _record_action(
                        connection, run_id, row, "noop", "proposal_rejected",
                        proposal=proposal, review=review,
                        details={"proposal": _proposal_projection(proposal)},
                    )
                )
                continue
            prepared.append((row, proposal, review))

        # Frozen source ordering, not candidate hashes, determines competing applications.
        prepared.sort(
            key=lambda item: (
                _proposal_payload(item[1])["occurred_at"] is None,
                _proposal_payload(item[1])["occurred_at"] or 0,
                _proposal_payload(item[1])["event_id"],
                _proposal_payload(item[1])["ordinal"],
            )
        )
        for row, proposal, review in prepared:
            actions.append(_apply_proposal(connection, run_id, row, proposal, review))

        counts = dict(sorted(Counter(item["action"] for item in actions).items()))
        completed_at = now_iso()
        connection.execute(
            """
            UPDATE consolidation_run SET completed_at = ?, status = 'complete',
                counts_json = ?, error = NULL WHERE run_id = ?
            """,
            (completed_at, _json(counts), run_id),
        )
        state = _state(connection)
    return {
        "run_id": run_id,
        "status": "complete",
        "admission_run_id": admission_run["run_id"],
        "admission_snapshot_hash": snapshot_hash,
        "candidate_count": len(decisions),
        "counts": counts,
        "actions": actions,
        "state": state,
        "human_review_boundary": (
            "Only an approve review bound to the exact immutable proposal can activate it. "
            "A changed memory revision, conflict set, scope relation, or admission snapshot "
            "invalidates that approval and requires a newly reviewed proposal."
        ),
    }


def rollback_memory(
    lab_database: Path,
    memory_id: str,
    target_revision_id: str,
    reason: str,
    *,
    requested_by: str = "operator",
    expected_state_version: int | None = None,
) -> dict:
    """Append a new revision copied from history; never reactivate an old row.

    The operation is serialized with ``BEGIN IMMEDIATE`` and advances the same
    monotonic state_version used by proposal CAS.  The immutable rollback row
    preserves why the semantic value was restored and which historical
    revision supplied it.
    """

    require_memory_lab_database(lab_database)
    memory_id = memory_id.strip()
    target_revision_id = target_revision_id.strip()
    reason = reason.strip()
    requested_by = requested_by.strip()
    if not memory_id or not target_revision_id:
        raise ValueError("memory_id and target_revision_id must not be empty")
    if not reason or len(reason) > 500:
        raise ValueError("rollback reason must contain 1..500 characters")
    if not requested_by or len(requested_by) > 240:
        raise ValueError("requested_by must contain 1..240 characters")
    if expected_state_version is not None and (
        isinstance(expected_state_version, bool)
        or not isinstance(expected_state_version, int)
        or expected_state_version < 0
    ):
        raise ValueError("expected_state_version must be a non-negative integer")

    with closing(open_store(lab_database)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            memory = connection.execute(
                "SELECT * FROM memory WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if memory is None or memory["status"] != "active":
                raise ValueError("active memory not found")
            if expected_state_version is not None and memory["state_version"] != expected_state_version:
                raise ValueError("memory state_version no longer matches rollback CAS")
            current_revision_id = memory["current_revision_id"]
            if not current_revision_id:
                raise ValueError("memory has no current revision")
            if target_revision_id == current_revision_id:
                raise ValueError("target revision is already current")
            current = connection.execute(
                "SELECT * FROM memory_revision WHERE revision_id = ? AND memory_id = ?",
                (current_revision_id, memory_id),
            ).fetchone()
            target = connection.execute(
                "SELECT * FROM memory_revision WHERE revision_id = ? AND memory_id = ?",
                (target_revision_id, memory_id),
            ).fetchone()
            if current is None or target is None:
                raise ValueError("target revision does not belong to the memory")

            revision_number = connection.execute(
                """
                SELECT COALESCE(MAX(revision_number), 0) + 1
                FROM memory_revision WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()[0]
            rollback_id = "rollback_" + uuid4().hex
            new_revision_id = "revision_" + uuid4().hex
            timestamp = now_iso()
            connection.execute(
                "UPDATE memory_revision SET status = 'superseded' WHERE revision_id = ?",
                (current_revision_id,),
            )
            connection.execute(
                """
                INSERT INTO memory_revision(
                    revision_id, memory_id, revision_number, semantic_value,
                    statement, activation_triggers_json, status,
                    supersedes_revision_id, source_decision_id,
                    source_proposal_id, review_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'current', ?, ?, NULL, NULL, ?)
                """,
                (
                    new_revision_id,
                    memory_id,
                    revision_number,
                    target["semantic_value"],
                    target["statement"],
                    target["activation_triggers_json"],
                    current_revision_id,
                    target["source_decision_id"],
                    timestamp,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE memory
                SET current_revision_id = ?, state_version = state_version + 1,
                    updated_at = ?
                WHERE memory_id = ? AND state_version = ?
                """,
                (
                    new_revision_id,
                    timestamp,
                    memory_id,
                    memory["state_version"],
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("memory state changed during rollback CAS")
            connection.execute(
                """
                INSERT INTO memory_rollback(
                    rollback_id, memory_id, from_revision_id,
                    target_revision_id, new_revision_id,
                    expected_state_version, requested_by, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rollback_id,
                    memory_id,
                    current_revision_id,
                    target_revision_id,
                    new_revision_id,
                    memory["state_version"],
                    requested_by,
                    reason,
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE conflict
                SET status = 'resolved', resolution_revision_id = ?,
                    resolution_reviewer = ?, resolution_reason = ?, resolved_at = ?
                WHERE memory_id = ? AND status = 'open'
                """,
                (
                    new_revision_id,
                    requested_by,
                    "rollback:" + reason,
                    timestamp,
                    memory_id,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "rollback_id": rollback_id,
        "memory_id": memory_id,
        "from_revision_id": current_revision_id,
        "target_revision_id": target_revision_id,
        "new_revision_id": new_revision_id,
        "revision_number": revision_number,
        "semantic_value": target["semantic_value"],
        "state_version_before": memory["state_version"],
        "state_version_after": memory["state_version"] + 1,
        "requested_by": requested_by,
        "reason": reason,
        "created_at": timestamp,
    }
