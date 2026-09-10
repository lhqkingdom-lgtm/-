"""Deterministic U1 baseline for deciding whether atomic user claims are worth keeping."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .instructions import infer_event_scope, polarity, scope_relation, similarity
from .store import now_iso, open_store


ADMISSION_VERSION = "rule-admission-v3"
ATOMIC_BOUNDARY = re.compile(r"(?:\r?\n)+|[；;]+")
LONG_TERM_SIGNALS = (
    "以后",
    "默认",
    "一直",
    "后续",
    "必须",
    "统一",
    "需要先",
    "先向我确认",
    "之前需要先问我",
    "再确认一次",
)
TEMPORARY_SIGNALS = ("这一次", "这次", "本轮", "当前阶段", "暂时", "今天", "只解释方案")
CORRECTION_SIGNALS = ("更正", "改为", "作废", "不要再", "最终以", "从现在起")
SENSITIVE_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I),
    re.compile(r"\b(?:api[_ -]?key|access[_ -]?token|password|密码|私钥)\s*[:=]\s*\S+", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b", re.I),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b", re.I),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)
NAMED_CURRENT_PROJECT = re.compile(
    r"(?:仅在|只在)?\s*当前\s+([A-Za-z][A-Za-z0-9+_.-]{2,})\s*项目",
    re.I,
)
IDENTIFIER_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+_.-]*")
RESPONSE_STRUCTURE_PATTERNS = (
    re.compile(
        r"(?:回复|回答|答复|作答).{0,24}?(?:先|首先)"
        r".{0,16}?(?:给出|说明|说出|列出|概括|总结)"
        r".{0,8}?(?:结论|答案|结果|要点|摘要)"
        r".{0,16}?(?:再|然后|随后|其次)"
        r".{0,16}?(?:说明|解释|展开|列出|补充|给出)"
        r".{0,8}?(?:步骤|过程|细节|理由|依据)"
    ),
    re.compile(
        r"(?:先|首先).{0,12}?(?:回复|回答|答复|作答)"
        r".{0,16}?(?:给出|说明|说出|列出|概括|总结)"
        r".{0,8}?(?:结论|答案|结果|要点|摘要)"
        r".{0,16}?(?:再|然后|随后|其次)"
        r".{0,16}?(?:说明|解释|展开|列出|补充|给出)"
        r".{0,8}?(?:步骤|过程|细节|理由|依据)"
    ),
)


@dataclass(frozen=True)
class AtomicClaim:
    ordinal: int
    exact_quote: str
    start_offset: int
    end_offset: int


def split_atomic_claims(content: str) -> list[AtomicClaim]:
    """Split only at high-confidence boundaries and preserve exact evidence spans."""
    result: list[AtomicClaim] = []
    cursor = 0
    for boundary in ATOMIC_BOUNDARY.finditer(content):
        segment = content[cursor : boundary.start()]
        _append_claim(result, content, segment, cursor)
        cursor = boundary.end()
    _append_claim(result, content, content[cursor:], cursor)
    return result


def _append_claim(result: list[AtomicClaim], source: str, segment: str, base_offset: int) -> None:
    left_trimmed = len(segment) - len(segment.lstrip())
    right = len(segment.rstrip())
    if right <= left_trimmed:
        return
    start = base_offset + left_trimmed
    end = base_offset + right
    result.append(
        AtomicClaim(
            ordinal=len(result) + 1,
            exact_quote=source[start:end],
            start_offset=start,
            end_offset=end,
        )
    )


def _signals(text: str, candidates: tuple[str, ...]) -> list[str]:
    return [signal for signal in candidates if signal in text]


def _contains_sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _requests_ordered_response_structure(text: str) -> bool:
    """Recognize an explicit conclusion-first, details-second response contract."""
    return any(pattern.search(text) for pattern in RESPONSE_STRUCTURE_PATTERNS)


def _suggest_identity(text: str) -> tuple[str | None, str | None, list[str]]:
    normalized = text.casefold()
    if any(language in text for language in ("中文", "汉语", "英文", "英语")) and any(
        word in text for word in ("回复", "回答", "输出")
    ):
        return "preference.response.language", "response_preference", ["回复", "语言"]
    if _requests_ordered_response_structure(text):
        return (
            "preference.response.structure",
            "preference",
            ["回复", "结论优先", "处理步骤"],
        )
    if any(word in text for word in ("代码", "正式代码")) and any(
        word in text for word in ("确认", "问我")
    ):
        return (
            "workflow.code_generation.confirmation",
            "workflow_rule",
            ["生成正式代码", "创建正式代码文件", "确认"],
        )
    if "单片机" in text and "修改" in text and "编译" in text:
        return (
            "workflow.embedded.code_change.compile_check",
            "workflow_rule",
            ["单片机", "修改代码", "编译检查"],
        )
    if "tg032plus" in normalized and "keil" in normalized and "c51" in normalized:
        return "toolchain.tg032plus.compiler", "project_rule", ["TG032Plus", "Keil C51", "编译"]
    return None, None, []


def _scope(text: str, project_key: str, directory: str) -> dict:
    inferred = infer_event_scope(text)
    if inferred == "project":
        return {"level": "project", "project_key": project_key, "directory": directory}
    if inferred == "domain_or_global":
        domain = "embedded" if "单片机" in text else None
        return {"level": "domain", "domain": domain}
    if inferred in {"global", "user_or_unknown"}:
        return {"level": "global", "override_policy": "more_specific_scope_wins"}
    # A reusable workflow stated without a project qualifier is a global proposal,
    # but remains a candidate until human/model review.
    if any(signal in text for signal in ("生成正式代码前", "正式代码之前")):
        return {"level": "global", "override_policy": "more_specific_scope_wins"}
    return {"level": "unknown"}


def _current_instruction_clauses(connection) -> dict[str, list[dict]]:
    rows = connection.execute(
        """
        WITH latest AS (
            SELECT project_key, MAX(completed_at_ms) AS completed_at_ms
            FROM instruction_scan
            WHERE status = 'complete' AND completed_at_ms IS NOT NULL
            GROUP BY project_key
        )
        SELECT s.project_key, s.scan_id, s.coverage_run_id,
               d.canonical_path, d.effective_scope_kind,
               c.clause_id, c.heading_path, c.line_start, c.exact_text
        FROM latest
        JOIN instruction_scan AS s
          ON s.project_key = latest.project_key
         AND s.completed_at_ms = latest.completed_at_ms
        JOIN instruction_scan_item AS i ON i.scan_id = s.scan_id
        JOIN instruction_document_revision AS d ON d.revision_id = i.revision_id
        JOIN instruction_clause AS c ON c.revision_id = d.revision_id
        WHERE i.cp_would_load = 1
          AND i.memory_applicability IN ('global', 'project_root')
          AND i.observed_read_status = 'ok'
        ORDER BY s.project_key, d.canonical_path, c.ordinal
        """
    ).fetchall()
    result: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        identity = (row["project_key"], row["clause_id"])
        if identity in seen:
            continue
        seen.add(identity)
        result.setdefault(row["project_key"], []).append(dict(row))
    return result


def _candidate_coverage(text: str, scope: dict, clauses: list[dict]) -> dict:
    ranked: list[dict] = []
    for clause in clauses:
        searchable = " ".join(
            part for part in (clause["heading_path"], clause["exact_text"]) if part
        )
        ranked.append({**clause, "score": similarity(text, searchable)})
    ranked.sort(key=lambda item: (-item["score"], item["canonical_path"].casefold()))
    best = ranked[0] if ranked else None
    if best is None or best["score"] < 0.52:
        return {"status": "none", "score": best["score"] if best else 0.0, "match": None}
    if polarity(text) != polarity(best["exact_text"]):
        status = "possible_conflict"
        relation = "conflict"
    else:
        event_scope = {
            "project": "project",
            "domain": "domain_or_global",
            "global": "global",
        }.get(scope["level"], "unknown")
        relation = scope_relation(event_scope, best["effective_scope_kind"])
        status = "full" if relation == "covers" else "partial"
    return {
        "status": status,
        "score": best["score"],
        "scope_relation": relation,
        "match": {
            "clause_id": best["clause_id"],
            "scan_id": best.get("scan_id"),
            "coverage_run_id": best.get("coverage_run_id"),
            "path": best["canonical_path"],
            "line": best["line_start"],
            "text": best["exact_text"],
        },
    }


def _claimed_current_project(text: str) -> str | None:
    match = NAMED_CURRENT_PROJECT.search(text)
    if not match:
        return None
    return match.group(1)


def _project_reference_is_unresolved(
    claimed_project: str | None, directory: str, clauses: list[dict]
) -> bool:
    if claimed_project is None:
        return False
    # A global user instruction is not evidence that the named project belongs
    # to this working directory. Only the directory or project-local rules may
    # corroborate the alias in this deterministic baseline.
    expected = claimed_project.casefold()
    directory_identifiers = {token.casefold() for token in IDENTIFIER_TOKEN.findall(directory)}
    if expected in directory_identifiers:
        return False
    boundary = re.compile(
        rf"(?<![A-Za-z0-9+_.-]){re.escape(claimed_project)}(?![A-Za-z0-9+_.-])\s*项目",
        re.I,
    )
    return not any(
        item["effective_scope_kind"] == "project" and boundary.search(item["exact_text"])
        for item in clauses
    )


def _decision_for_claim(event: dict, claim: AtomicClaim, clauses: list[dict]) -> dict:
    text = claim.exact_quote
    long_term = _signals(text, LONG_TERM_SIGNALS)
    temporary = _signals(text, TEMPORARY_SIGNALS)
    corrections = _signals(text, CORRECTION_SIGNALS)
    stable_key, memory_type, triggers = _suggest_identity(text)
    scope = _scope(text, event["project_key"], event["directory"])
    claimed_project = _claimed_current_project(text)
    unresolved_project = _project_reference_is_unresolved(
        claimed_project, event["directory"], clauses
    )
    if unresolved_project:
        scope = {
            "level": "unresolved",
            "claimed_project": claimed_project,
            "observed_project_key": event["project_key"],
            "observed_directory": event["directory"],
        }
    coverage = _candidate_coverage(text, scope, clauses)
    reasons: list[str] = []

    if event["eligibility_status"] != "eligible":
        disposition = event["eligibility_status"]
        reasons.append("event_excluded_before_semantic_admission")
    elif _contains_sensitive(text):
        disposition = "sensitive_blocked"
        reasons.append("sensitive_pattern_detected")
    elif temporary:
        disposition = "temporary"
        reasons.append("temporary_language")
    elif unresolved_project:
        disposition = "needs_scope_review"
        reasons.append("named_current_project_not_supported_by_session_context")
    elif corrections and stable_key:
        disposition = "needs_correction_review"
        reasons.append("explicit_correction_requires_version_or_conflict_handling")
    elif coverage["status"] == "full":
        disposition = "covered_by_cp_instructions"
        reasons.append("equivalent_instruction_with_covering_scope")
    elif coverage["status"] == "possible_conflict":
        disposition = "needs_conflict_review"
        reasons.append("similar_instruction_with_opposite_polarity")
    elif long_term and stable_key:
        disposition = "eligible"
        reasons.append("durable_user_contract_signal")
        if coverage["status"] == "partial":
            reasons.append("only_partially_covered_by_project_instruction")
    elif long_term:
        disposition = "needs_semantic_review"
        reasons.append("durable_signal_without_supported_rule_identity")
    else:
        disposition = "not_memory"
        reasons.append("no_durable_memory_signal")

    if corrections:
        reasons.append("explicit_correction_signal")
    family = "user_contract" if stable_key or long_term else "unknown"
    candidate_identity = "\0".join(
        (event["event_id"], str(claim.ordinal), str(claim.start_offset), claim.exact_quote)
    )
    return {
        "candidate_key": "admcand_" + hashlib.sha256(candidate_identity.encode("utf-8")).hexdigest()[:24],
        "event_id": event["event_id"],
        "session_id": event["session_id"],
        "project_key": event["project_key"],
        "directory": event["directory"],
        "ordinal": claim.ordinal,
        "exact_quote": claim.exact_quote,
        "start_offset": claim.start_offset,
        "end_offset": claim.end_offset,
        "family": family,
        "memory_type": memory_type,
        "stable_key_suggestion": stable_key,
        "scope": scope,
        "activation_triggers": triggers,
        "disposition": disposition,
        "long_term_signals": long_term,
        "temporary_signals": temporary,
        "correction_signals": corrections,
        "instruction_coverage": coverage,
        "reasons": reasons,
    }


def run_rule_admission(lab_database: Path) -> dict:
    run_id = "admission_" + uuid4().hex
    started_at = now_iso()
    with closing(open_store(lab_database)) as connection, connection:
        connection.execute(
            """
            INSERT INTO admission_run(run_id, extractor_version, started_at, status)
            VALUES (?, ?, ?, 'running')
            """,
            (run_id, ADMISSION_VERSION, started_at),
        )
        clauses_by_project = _current_instruction_clauses(connection)
        pre_admission_exclusions = {
            row["eligibility_status"]: row["event_count"]
            for row in connection.execute(
                """
                SELECT eligibility_status, COUNT(*) AS event_count
                FROM event
                WHERE eligibility_status != 'eligible'
                GROUP BY eligibility_status
                ORDER BY eligibility_status
                """
            )
        }
        turn_gate = connection.execute(
            """
            SELECT
                COUNT(*) AS eligible_root_events,
                SUM(
                    CASE WHEN EXISTS(
                        SELECT 1
                        FROM turn_event AS te
                        JOIN conversation_turn AS t ON t.turn_id = te.turn_id
                        WHERE te.event_id = e.event_id
                          AND t.current_state = 'completed'
                    ) THEN 1 ELSE 0 END
                ) AS completed_events
            FROM event AS e
            WHERE e.eligibility_status = 'eligible'
              AND e.parent_session_id IS NULL
            """
        ).fetchone()
        eligible_root_events = int(turn_gate["eligible_root_events"] or 0)
        completed_events = int(turn_gate["completed_events"] or 0)
        rows = connection.execute(
            """
            SELECT e.event_id, e.session_id, e.directory, e.content,
                   COALESCE(e.eligibility_status, 'eligible') AS eligibility_status,
                   a.semantic_content, a.experiment_case_id, sc.project_key
            FROM event AS e
            JOIN event_annotation AS a ON a.event_id = e.event_id
            JOIN session_context AS sc
              ON sc.source_id = e.source_id AND sc.session_id = e.session_id
            WHERE e.eligibility_status = 'eligible'
              AND e.parent_session_id IS NULL
              AND EXISTS(
                  SELECT 1
                  FROM turn_event AS te
                  JOIN conversation_turn AS t ON t.turn_id = te.turn_id
                  WHERE te.event_id = e.event_id
                    AND t.current_state = 'completed'
              )
            ORDER BY e.timestamp, e.event_id
            """
        ).fetchall()
        decisions: list[dict] = []
        for row in rows:
            event = dict(row)
            for claim in split_atomic_claims(event["semantic_content"]):
                decision = _decision_for_claim(
                    event, claim, clauses_by_project.get(event["project_key"], [])
                )
                connection.execute(
                    """
                    INSERT INTO admission_decision(
                        decision_id, run_id, candidate_key, event_id, ordinal,
                        exact_quote, start_offset, end_offset, family, memory_type,
                        stable_key_suggestion, scope_json, activation_triggers_json,
                        disposition, signals_json, instruction_coverage_json,
                        reasons_json, decided_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "admdecision_" + uuid4().hex,
                        run_id,
                        decision["candidate_key"],
                        decision["event_id"],
                        decision["ordinal"],
                        decision["exact_quote"],
                        decision["start_offset"],
                        decision["end_offset"],
                        decision["family"],
                        decision["memory_type"],
                        decision["stable_key_suggestion"],
                        json.dumps(decision["scope"], ensure_ascii=False, sort_keys=True),
                        json.dumps(
                            decision["activation_triggers"], ensure_ascii=False, sort_keys=True
                        ),
                        decision["disposition"],
                        json.dumps(
                            {
                                "long_term": decision["long_term_signals"],
                                "temporary": decision["temporary_signals"],
                                "correction": decision["correction_signals"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            decision["instruction_coverage"], ensure_ascii=False, sort_keys=True
                        ),
                        json.dumps(decision["reasons"], ensure_ascii=False, sort_keys=True),
                        now_iso(),
                    ),
                )
                decisions.append(
                    {"case_id": event["experiment_case_id"], **decision}
                )
        counts = Counter(item["disposition"] for item in decisions)
        controlled_decisions = [item for item in decisions if item["case_id"] is not None]
        controlled_counts = Counter(item["disposition"] for item in controlled_decisions)
        completed_at = now_iso()
        connection.execute(
            """
            UPDATE admission_run
            SET completed_at = ?, status = 'complete', decision_count = ?, counts_json = ?
            WHERE run_id = ?
            """,
            (
                completed_at,
                len(decisions),
                json.dumps(dict(sorted(counts.items())), ensure_ascii=False, sort_keys=True),
                run_id,
            ),
        )

    return {
        "lab_database": str(lab_database.resolve()),
        "run_id": run_id,
        "extractor_version": ADMISSION_VERSION,
        "assessment_level": "atomic_rule_baseline_not_formal_memory",
        "decision_count": len(decisions),
        "counts": dict(sorted(counts.items())),
        "pre_admission_exclusions": pre_admission_exclusions,
        "turn_gate": {
            "eligible_root_events": eligible_root_events,
            "completed_events_admitted": completed_events,
            "excluded_without_completed_turn": eligible_root_events - completed_events,
        },
        "controlled_gold_subset": {
            "decision_count": len(controlled_decisions),
            "counts": dict(sorted(controlled_counts.items())),
        },
        "decisions": decisions,
        "limitations": [
            "This deterministic baseline recognizes only supported rule families; an LLM comparison comes later.",
            "Eligible means worth reviewing as a candidate, not accepted active memory.",
            "Instruction coverage is evaluated per atomic claim using the latest completed scan.",
        ],
    }
