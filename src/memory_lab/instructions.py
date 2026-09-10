"""Snapshot CodePulse instructions and produce non-final event-level coverage hints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import closing
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from .store import now_iso, open_store


RESOLVER_VERSION = "cp-instruction-resolver-1.0.77-lab-v1"
CLAUSE_NORMALIZATION_VERSION = "clause-normalization-v1"
PROJECT_INSTRUCTION_NAMES = ("AGENTS.md", "CLAUDE.md", "CONTEXT.md")
MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
MARKDOWN_RULE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+?)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
NON_WORD = re.compile(r"[^0-9a-z\u3400-\u9fff]+")
NEGATIVE_SIGNALS = ("不要", "不使用", "不需要", "无需", "禁止", "不能", "不得")


@dataclass(frozen=True)
class InstructionFile:
    path: Path
    project_key: str | None
    origin: str
    discovered_by: str
    cp_would_load: bool
    effective_scope_kind: str
    scope_root: Path | None


@dataclass(frozen=True)
class ParsedClause:
    ordinal: int
    line_start: int
    line_end: int
    heading_path: str | None
    exact_text: str

    @property
    def searchable_text(self) -> str:
        if not self.heading_path:
            return self.exact_text
        return f"{self.heading_path} {self.exact_text}"


def _canonical(path: Path) -> Path:
    return path.resolve()


def _resolved_files(paths: list[Path]) -> list[Path]:
    result: dict[str, Path] = {}
    for candidate in paths:
        try:
            if not candidate.is_file():
                continue
            resolved = _canonical(candidate)
            result[str(resolved).casefold()] = resolved
        except OSError:
            continue
    return sorted(result.values(), key=lambda item: str(item).casefold())


def discover_project_instructions(directory: Path, project_key: str) -> list[InstructionFile]:
    """Return files the current CP systemPaths implementation would load."""
    if not directory.is_dir():
        return []

    selected_family: list[Path] = []
    selected_name: str | None = None
    for name in PROJECT_INSTRUCTION_NAMES:
        selected_family = _resolved_files(list(directory.rglob(name)))
        if selected_family:
            selected_name = name
            break

    result: list[InstructionFile] = []
    for path in selected_family:
        is_state = path.parent.name.casefold() == ".codepulse"
        result.append(
            InstructionFile(
                path=path,
                project_key=project_key,
                origin="project_state" if is_state else "project_primary",
                discovered_by=f"recursive_family:{selected_name}",
                cp_would_load=True,
                effective_scope_kind="project",
                scope_root=path.parent.parent if is_state else path.parent,
            )
        )

    # CP performs this second scan even though dot=true usually found the same file.
    for path in _resolved_files(
        [
            candidate
            for candidate in directory.rglob("AGENTS.md")
            if candidate.parent.name.casefold() == ".codepulse"
        ]
    ):
        result.append(
            InstructionFile(
                path=path,
                project_key=project_key,
                origin="project_state",
                discovered_by="recursive_project_state",
                cp_would_load=True,
                effective_scope_kind="project",
                scope_root=path.parent.parent,
            )
        )
    return result


def global_candidates(home_directory: Path | None = None) -> list[Path]:
    home = (home_directory or Path.home()).resolve()
    result: list[Path] = []
    configured = os.environ.get("OPENCODE_CONFIG_DIR")
    if configured:
        result.append(Path(configured) / "AGENTS.md")
    result.append(home / ".config" / "codepulse" / "AGENTS.md")
    result.append(home / ".claude" / "CLAUDE.md")
    return result


def discover_global_instructions(home_directory: Path | None = None) -> list[InstructionFile]:
    """Return only the first existing global instruction, matching CP priority."""
    for path in global_candidates(home_directory):
        if not path.is_file():
            continue
        return [
            InstructionFile(
                path=_canonical(path),
                project_key=None,
                origin="global",
                discovered_by="global_first_existing",
                cp_would_load=True,
                effective_scope_kind="global",
                scope_root=None,
            )
        ]
    return []


def parse_clauses(content: str) -> list[ParsedClause]:
    headings: list[str] = []
    result: list[ParsedClause] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = MARKDOWN_HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            headings = [*headings[: level - 1], heading.group(2).strip()]
            continue
        item = MARKDOWN_RULE.match(line)
        if not item:
            continue
        text = re.sub(r"^\[[ xX]\]\s*", "", item.group(1).strip())
        if not text:
            continue
        result.append(
            ParsedClause(
                ordinal=len(result) + 1,
                line_start=line_number,
                line_end=line_number,
                heading_path=" > ".join(headings) or None,
                exact_text=text,
            )
        )
    return result


def normalize_for_match(text: str) -> str:
    return NON_WORD.sub("", text.casefold().replace("／", "/"))


def _ngrams(text: str, size: int = 2) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def similarity(left: str, right: str) -> float:
    left_normalized = normalize_for_match(left)
    right_normalized = normalize_for_match(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return 1.0
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_grams = _ngrams(left_normalized)
    right_grams = _ngrams(right_normalized)
    containment = len(left_grams & right_grams) / min(len(left_grams), len(right_grams))
    return round(max(sequence, containment), 6)


def polarity(text: str) -> str:
    normalized = normalize_for_match(text)
    return "negative" if any(signal in normalized for signal in NEGATIVE_SIGNALS) else "positive"


def infer_event_scope(text: str) -> str:
    normalized = normalize_for_match(text)
    if any(signal in normalized for signal in ("仅在当前", "只在当前", "当前项目", "本项目")):
        return "project"
    if "所有" in normalized and any(signal in normalized for signal in ("项目", "工程")):
        return "domain_or_global"
    if any(signal in normalized for signal in ("全局", "所有任务", "所有对话", "所有会话")):
        return "global"
    if any(signal in normalized for signal in ("以后", "默认", "一直", "后续")):
        return "user_or_unknown"
    return "unknown"


def scope_relation(event_scope: str, instruction_scope: str) -> str:
    if instruction_scope == "global":
        return "covers"
    if event_scope == "project":
        return "covers"
    if event_scope in {"domain_or_global", "global", "user_or_unknown"}:
        return "instruction_scope_narrower"
    return "unknown"


def revision_id(path: Path, content_hash: str) -> str:
    identity = f"{str(_canonical(path)).casefold()}\0{content_hash}"
    return "insrev_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def clause_id(revision_identifier: str, clause: ParsedClause) -> str:
    identity = f"{revision_identifier}\0{clause.ordinal}\0{clause.exact_text}"
    return "insclause_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def clause_fingerprint(clause: ParsedClause) -> str:
    identity = "\0".join(
        (
            CLAUSE_NORMALIZATION_VERSION,
            normalize_for_match(clause.heading_path or ""),
            normalize_for_match(clause.exact_text),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(_canonical(left))) == os.path.normcase(str(_canonical(right)))


def _memory_applicability(instruction: InstructionFile, session_directory: Path) -> tuple[str, str]:
    if instruction.effective_scope_kind == "global":
        return "global", "Global instructions are logically available to every project."
    if instruction.scope_root and _same_path(instruction.scope_root, session_directory):
        return "project_root", "Instruction scope root equals the session project directory."
    return "nested_scope_only", "CP injects this recursively, but Memory Lab limits it to its own subtree."


def _capture_revision(connection, instruction: InstructionFile) -> tuple[dict, list[dict], dict]:
    path = _canonical(instruction.path)
    captured_at = now_iso()
    read_status = "ok"
    read_error = None
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        content = ""
        read_status = "error"
        read_error = f"{type(error).__name__}: {error}"
    try:
        stat = path.stat()
        created_ns = stat.st_ctime_ns if os.name == "nt" else None
        modified_ns = stat.st_mtime_ns
        size_bytes = stat.st_size
    except OSError as error:
        created_ns = None
        modified_ns = None
        size_bytes = 0
        read_status = "error"
        read_error = f"{type(error).__name__}: {error}"
    observed_at_ms = time.time_ns() // 1_000_000

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    identifier = revision_id(path, content_hash)
    connection.execute(
        """
        INSERT INTO instruction_document_revision(
            revision_id, canonical_path, origin, effective_scope_kind, scope_root,
            content, content_hash, size_bytes, created_ns, modified_ns, captured_at,
            read_status, read_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(revision_id) DO NOTHING
        """,
        (
            identifier,
            str(path),
            instruction.origin,
            instruction.effective_scope_kind,
            str(_canonical(instruction.scope_root)) if instruction.scope_root else None,
            content,
            content_hash,
            size_bytes,
            created_ns,
            modified_ns,
            captured_at,
            read_status,
            read_error,
        ),
    )

    parsed: list[dict] = []
    if read_status == "ok":
        for clause in parse_clauses(content):
            identifier_clause = clause_id(identifier, clause)
            fingerprint = clause_fingerprint(clause)
            connection.execute(
                """
                INSERT INTO instruction_clause(
                    clause_id, revision_id, ordinal, heading_path, line_start,
                    line_end, exact_text, normalized_text, normalization_version,
                    clause_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(clause_id) DO UPDATE SET
                    normalization_version = excluded.normalization_version,
                    clause_fingerprint = excluded.clause_fingerprint
                """,
                (
                    identifier_clause,
                    identifier,
                    clause.ordinal,
                    clause.heading_path,
                    clause.line_start,
                    clause.line_end,
                    clause.exact_text,
                    normalize_for_match(clause.searchable_text),
                    CLAUSE_NORMALIZATION_VERSION,
                    fingerprint,
                    now_iso(),
                ),
            )
            parsed.append(
                {
                    "clause_id": identifier_clause,
                    "revision_id": identifier,
                    "heading_path": clause.heading_path,
                    "line_start": clause.line_start,
                    "exact_text": clause.exact_text,
                    "searchable_text": clause.searchable_text,
                    "clause_fingerprint": fingerprint,
                    "created_ns": created_ns,
                    "modified_ns": modified_ns,
                    "observed_at_ms": observed_at_ms,
                }
            )
    revision = {
        "revision_id": identifier,
        "path": str(path),
        "origin": instruction.origin,
        "read_status": read_status,
        "read_error": read_error,
        "content_hash": content_hash,
        "created_ns": created_ns,
        "modified_ns": modified_ns,
    }
    observation = {
        "observed_at_ms": observed_at_ms,
        "size_bytes": size_bytes,
        "created_ns": created_ns,
        "modified_ns": modified_ns,
        "read_status": read_status,
        "read_error": read_error,
    }
    return revision, parsed, observation


def _temporal_hint(event_timestamp: int | None, clause: dict | None) -> str:
    if event_timestamp is None or clause is None:
        return "unknown"
    event_ns = event_timestamp * 1_000_000
    created_ns = clause["created_ns"]
    modified_ns = clause["modified_ns"]
    if created_ns is not None and created_ns > event_ns:
        return "not_preexisting_hint"
    if modified_ns is not None and modified_ns > event_ns:
        return "current_revision_postdates_event"
    if (
        created_ns is not None
        and modified_ns is not None
        and created_ns < event_ns
        and modified_ns < event_ns
    ):
        return "filesystem_preexisting_hint"
    return "unknown"


def _observed_before_event(connection, event: dict, clause: dict | None) -> bool:
    """Return true only when an earlier immutable scan saw the same rule text.

    File timestamps cannot prove what an instruction contained when an old event
    happened. A completed Memory Lab scan can: revisions are immutable and every
    scan records the revision it observed. Matching exact normalized clause text
    across revisions also survives harmless edits elsewhere in the document.
    """
    if event["timestamp"] is None or clause is None:
        return False
    rows = connection.execute(
        """
        SELECT 1
        FROM instruction_scan AS s
        JOIN instruction_scan_item AS i ON i.scan_id = s.scan_id
        JOIN instruction_scan_source AS ss ON ss.scan_id = s.scan_id
        JOIN instruction_document_revision AS d ON d.revision_id = i.revision_id
        JOIN instruction_clause AS c ON c.revision_id = d.revision_id
        WHERE s.project_key = ?
          AND s.status = 'complete'
          AND s.completed_at_ms < ?
          AND i.observed_at_ms < ?
          AND ss.source_id = ?
          AND ss.session_version = ?
          AND s.resolver_version = ?
          AND s.resolver_context_hash = ?
          AND i.cp_would_load = 1
          AND i.memory_applicability IN ('global', 'project_root')
          AND i.observed_read_status = 'ok'
          AND d.canonical_path = ?
          AND c.clause_fingerprint = ?
        LIMIT 1
        """,
        (
            event["project_key"],
            event["timestamp"],
            event["timestamp"],
            event["source_id"],
            event["session_version"] or "",
            RESOLVER_VERSION,
            clause["resolver_context_hash"],
            clause["path"],
            clause["clause_fingerprint"],
        ),
    ).fetchone()
    return rows is not None


def _coverage_at_event(connection, event: dict, clause: dict | None) -> str:
    if clause is None:
        return "unknown"
    if _observed_before_event(connection, event, clause):
        return "seen_in_pre_event_snapshot"
    return _temporal_hint(event["timestamp"], clause)


def run_instruction_coverage(lab_database: Path, home_directory: Path | None = None) -> dict:
    with closing(open_store(lab_database)) as connection, connection:
        coverage_run_id = "inscoverage_" + uuid4().hex
        coverage_started_at = now_iso()
        coverage_started_at_ms = time.time_ns() // 1_000_000
        connection.execute(
            """
            INSERT INTO instruction_coverage_run(
                coverage_run_id, resolver_version, assessment_level, started_at,
                started_at_ms, status
            ) VALUES (?, ?, 'event_preliminary', ?, ?, 'running')
            """,
            (coverage_run_id, RESOLVER_VERSION, coverage_started_at, coverage_started_at_ms),
        )
        projects = connection.execute(
            "SELECT project_key, directory FROM project_context ORDER BY project_key"
        ).fetchall()
        global_files = discover_global_instructions(home_directory)
        scans: list[dict] = []
        current_clauses: list[dict] = []
        current_clause_keys: set[tuple[str, str, str]] = set()

        for project in projects:
            directory = Path(project["directory"])
            scan_id = "insscan_" + uuid4().hex
            scanned_at = now_iso()
            scan_started_at_ms = time.time_ns() // 1_000_000
            resolver_config = {
                "project_config_enabled": "assumed_true",
                "claude_compat_enabled": "assumed_true",
                "global_candidates": [str(item) for item in global_candidates(home_directory)],
                "custom_instructions": "not_configured_in_current_test_projects",
                "remote_fetch": "disabled_in_memory_lab_mvp",
            }
            resolver_config_json = json.dumps(
                resolver_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            resolver_context_hash = hashlib.sha256(
                json.dumps(
                    {
                        "resolver_version": RESOLVER_VERSION,
                        "session_directory": str(_canonical(directory)).casefold(),
                        "resolver_config": resolver_config,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO instruction_scan(
                    scan_id, coverage_run_id, project_key, session_directory,
                    resolver_version, resolver_confidence, resolver_config_json,
                    resolver_context_hash, scanned_at, started_at_ms, status
                ) VALUES (?, ?, ?, ?, ?, 'assumed_runtime_defaults', ?, ?, ?, ?, 'running')
                """,
                (
                    scan_id,
                    coverage_run_id,
                    project["project_key"],
                    str(directory),
                    RESOLVER_VERSION,
                    resolver_config_json,
                    resolver_context_hash,
                    scanned_at,
                    scan_started_at_ms,
                ),
            )
            source_contexts = connection.execute(
                """
                SELECT DISTINCT sc.source_id, COALESCE(e.session_version, '') AS session_version
                FROM session_context AS sc
                LEFT JOIN event AS e
                  ON e.source_id = sc.source_id AND e.session_id = sc.session_id
                WHERE sc.project_key = ?
                ORDER BY sc.source_id, session_version
                """,
                (project["project_key"],),
            ).fetchall()
            for source_context in source_contexts:
                connection.execute(
                    """
                    INSERT INTO instruction_scan_source(scan_id, source_id, session_version)
                    VALUES (?, ?, ?)
                    """,
                    (
                        scan_id,
                        source_context["source_id"],
                        source_context["session_version"],
                    ),
                )

            files = [*discover_project_instructions(directory, project["project_key"]), *global_files]
            scan_files: list[dict] = []
            for instruction in files:
                revision, clauses, observation = _capture_revision(connection, instruction)
                applicability, applicability_reason = _memory_applicability(instruction, directory)
                connection.execute(
                    """
                    INSERT INTO instruction_scan_item(
                        scan_id, revision_id, discovered_by, cp_would_load,
                        memory_applicability, applicability_reason, observed_at_ms,
                        observed_size_bytes, observed_created_ns, observed_modified_ns,
                        observed_read_status, observed_read_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan_id,
                        revision["revision_id"],
                        instruction.discovered_by,
                        int(instruction.cp_would_load),
                        applicability,
                        applicability_reason,
                        observation["observed_at_ms"],
                        observation["size_bytes"],
                        observation["created_ns"],
                        observation["modified_ns"],
                        observation["read_status"],
                        observation["read_error"],
                    ),
                )
                scan_files.append(
                    {
                        **revision,
                        "discovered_by": instruction.discovered_by,
                        "cp_would_load": instruction.cp_would_load,
                        "memory_applicability": applicability,
                        "scope_root": str(instruction.scope_root) if instruction.scope_root else None,
                        "clause_count": len(clauses),
                        "observation": observation,
                    }
                )
                if applicability in {"global", "project_root"}:
                    for clause in clauses:
                        key = (project["project_key"], clause["clause_id"], resolver_context_hash)
                        if key in current_clause_keys:
                            continue
                        current_clause_keys.add(key)
                        current_clauses.append(
                            {
                                **clause,
                                "project_key": project["project_key"],
                                "effective_scope_kind": instruction.effective_scope_kind,
                                "path": revision["path"],
                                "resolver_context_hash": resolver_context_hash,
                            }
                        )
            scan_completed_at_ms = time.time_ns() // 1_000_000
            connection.execute(
                "UPDATE instruction_scan SET completed_at_ms = ?, status = 'complete' WHERE scan_id = ?",
                (scan_completed_at_ms, scan_id),
            )
            scans.append(
                {
                    "scan_id": scan_id,
                    "scanned_at": scanned_at,
                    "project_key": project["project_key"],
                    "directory": str(directory),
                    "resolver_confidence": "assumed_runtime_defaults",
                    "resolver_context_hash": resolver_context_hash,
                    "started_at_ms": scan_started_at_ms,
                    "completed_at_ms": scan_completed_at_ms,
                    "source_contexts": [dict(item) for item in source_contexts],
                    "files": scan_files,
                }
            )

        events = connection.execute(
            """
            SELECT e.event_id, e.source_id, e.session_id, e.session_version,
                   e.timestamp, a.experiment_case_id, a.semantic_content,
                   sc.project_key, sc.directory
            FROM event AS e
            JOIN event_annotation AS a ON a.event_id = e.event_id
            JOIN session_context AS sc
              ON sc.source_id = e.source_id AND sc.session_id = e.session_id
            ORDER BY e.timestamp, e.event_id
            """
        ).fetchall()
        hints: list[dict] = []
        for event in events:
            applicable = [
                clause
                for clause in current_clauses
                if clause["project_key"] == event["project_key"]
            ]
            ranked = sorted(
                (
                    {
                        **clause,
                        "score": similarity(event["semantic_content"], clause["searchable_text"]),
                    }
                    for clause in applicable
                ),
                key=lambda item: (-item["score"], item["path"].casefold(), item["line_start"]),
            )
            best = ranked[0] if ranked else None
            score = best["score"] if best else 0.0
            event_scope = infer_event_scope(event["semantic_content"])
            scope = scope_relation(event_scope, best["effective_scope_kind"]) if best else "none"
            content_matches = bool(best and score >= 0.52)
            possible_conflict = bool(
                content_matches and best and polarity(event["semantic_content"]) != polarity(best["searchable_text"])
            )
            if not content_matches:
                covered_now = "none"
                reason = "No logically applicable instruction clause passed the lexical hint threshold."
            elif possible_conflict:
                covered_now = "possible_conflict"
                reason = "The texts are lexically close but have opposite deterministic polarity signals."
            elif scope == "covers":
                covered_now = "likely_full"
                reason = "The current clause is similar and its logical scope covers the whole event claim."
            else:
                covered_now = "likely_partial"
                reason = "The current clause is similar, but its logical scope does not cover the whole event claim."
            temporal_evidence = _coverage_at_event(
                connection, event, best if content_matches else None
            )
            analyzed_at = now_iso()
            best_clause_id = best["clause_id"] if best and content_matches else None
            connection.execute(
                """
                INSERT INTO event_instruction_assessment(
                    assessment_id, coverage_run_id, event_id, assessment_level,
                    covered_now, temporal_evidence, event_scope_hint, best_clause_id,
                    content_score, scope_relation, reason, analyzed_at
                ) VALUES (?, ?, ?, 'event_preliminary', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "insassessment_" + uuid4().hex,
                    coverage_run_id,
                    event["event_id"],
                    covered_now,
                    temporal_evidence,
                    event_scope,
                    best_clause_id,
                    score,
                    scope,
                    reason,
                    analyzed_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO event_instruction_hint(
                    event_id, assessment_level, covered_now, covered_at_event,
                    event_scope_hint, best_clause_id, content_score, scope_relation,
                    reason, analyzed_at, latest_coverage_run_id, temporal_evidence
                ) VALUES (?, 'event_preliminary', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    assessment_level = excluded.assessment_level,
                    covered_now = excluded.covered_now,
                    covered_at_event = excluded.covered_at_event,
                    event_scope_hint = excluded.event_scope_hint,
                    best_clause_id = excluded.best_clause_id,
                    content_score = excluded.content_score,
                    scope_relation = excluded.scope_relation,
                    reason = excluded.reason,
                    analyzed_at = excluded.analyzed_at,
                    latest_coverage_run_id = excluded.latest_coverage_run_id,
                    temporal_evidence = excluded.temporal_evidence
                """,
                (
                    event["event_id"],
                    covered_now,
                    temporal_evidence,
                    event_scope,
                    best_clause_id,
                    score,
                    scope,
                    reason,
                    analyzed_at,
                    coverage_run_id,
                    temporal_evidence,
                ),
            )
            hints.append(
                {
                    "case_id": event["experiment_case_id"],
                    "session_id": event["session_id"],
                    "project_key": event["project_key"],
                    "content": event["semantic_content"],
                    "event_scope_hint": event_scope,
                    "covered_now": covered_now,
                    "temporal_evidence": temporal_evidence,
                    "content_score": score,
                    "scope_relation": scope,
                    "matched_clause": (
                        {
                            "path": best["path"],
                            "line": best["line_start"],
                            "heading_path": best["heading_path"],
                            "text": best["exact_text"],
                        }
                        if best and content_matches
                        else None
                    ),
                    "reason": reason,
                }
            )

        coverage_completed_at = now_iso()
        coverage_completed_at_ms = time.time_ns() // 1_000_000
        connection.execute(
            """
            UPDATE instruction_coverage_run
            SET completed_at = ?, completed_at_ms = ?, status = 'complete'
            WHERE coverage_run_id = ?
            """,
            (coverage_completed_at, coverage_completed_at_ms, coverage_run_id),
        )

    return {
        "lab_database": str(lab_database.resolve()),
        "coverage_run_id": coverage_run_id,
        "resolver_version": RESOLVER_VERSION,
        "assessment_level": "event_preliminary_not_final_candidate_coverage",
        "scan_count": len(scans),
        "scans": scans,
        "event_hint_count": len(hints),
        "event_hints": hints,
        "limitations": [
            "A pre-event snapshot is inference evidence, not proof that CP actually loaded the rule for that event.",
            "Events older than the first instruction scan have only weak filesystem-time hints.",
            "Final coverage must be evaluated per atomic candidate after model extraction.",
            "Configured local/remote instructions are not active in the current MVP fixtures.",
        ],
    }
