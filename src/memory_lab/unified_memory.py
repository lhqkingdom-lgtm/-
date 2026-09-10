"""Generic, tool-agnostic long-term memory candidate contract.

This module is the deterministic boundary between a semantic Writer and the
formal memory store.  The Writer may propose any durable fact it can support
from a normalized CodePulse conversation turn.  This module deliberately does
not contain a tool, language, framework, or memory-topic allow-list.

It does, however, fail closed at the trust boundary: evidence must originate
from a normalized CP turn, referenced evidence must exist, local paths and
secrets cannot be copied into semantic memory, and one proposal represents one
stable key.  The returned dictionaries are JSON-safe and canonically ordered.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypedDict


CANDIDATE_SCHEMA_VERSION = "unified-memory-candidate-v1"
POLICY_VERSION = "unified-memory-policy-v1"

# ``preference`` is retained for compatibility with the existing formal store;
# ``user`` is the broader user-memory kind used by the new generic extractor.
MEMORY_TYPES = frozenset(
    {
        "user",
        "preference",
        "workflow",
        "engineering",
        "project",
        "reference",
        "knowledge",
        "other",
    }
)
TRUST_STATUSES = frozenset(
    {"unverified", "declared", "observed", "verified", "contested"}
)
MEMORY_STATUSES = frozenset({"active", "superseded", "conflict"})
RELATION_ACTIONS = frozenset(
    {"create", "merge_evidence", "add_scope_variant", "revise", "open_conflict"}
)

_USER_CONTRACT_TYPES = frozenset({"user", "preference", "workflow"})
_EVIDENCE_KINDS = frozenset({"user_message", "assistant_final", "tool_result"})
_SCOPE_LEVELS = frozenset({"global", "domain", "project", "cross_project", "session"})
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,7}$")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|"
        r"password|passwd|private[_ -]?key|密钥|令牌|密码|口令|凭据|私钥|授权)"
        r"\s*(?::|=|：|\bis\b(?:\s+set\s+to)?|\bequals?\b|是|为)\s*"
        r"['\"]?[^\s,;，。；]{4,}"
    ),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/-]{8,}={0,2}"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"),
)
_SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?i)^(?:api_?key|access_?token|refresh_?token|client_?secret|"
    r"password|passwd|private_?key|authorization|密钥|令牌|密码|口令|凭据|私钥|授权)$"
)
_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?i)(?<![A-Za-z0-9])(?:[a-z]:[\\/]|\\\\[^\\\s]+\\)"),
    re.compile(r"(?i)(?<![A-Za-z0-9])file://"),
    re.compile(r"(?<![A-Za-z0-9])/(?:home|users|etc|var|tmp|opt|root)/"),
)
_TEMPORARY_PATTERNS = (
    re.compile(r"(?:仅|只)(?:限)?(?:本次|这次|这一轮|当前会话)"),
    re.compile(r"(?:本次|这次|这一轮|当前会话)(?:中|内)?(?:临时|暂时|才|只)"),
    re.compile(r"(?:不要|无需|不用)(?:把.{0,12})?(?:记住|长期保存|沉淀)"),
    re.compile(r"(?:临时要求|一次性要求)"),
)
_MULTIPLEX_KEYS = frozenset({"candidates", "memories", "stable_keys", "claims"})
_MAX_JSON_BYTES = 16_384
_MAX_DEPTH = 5
_MAX_COLLECTION_ITEMS = 50


class MemoryCandidate(TypedDict):
    """Normalized JSON representation accepted by a future formal executor."""

    schema_version: str
    policy_version: str
    candidate_id: str
    semantic_fingerprint: str
    stable_key: str
    type: str
    summary: str
    scope: dict[str, Any]
    conditions: dict[str, Any]
    trust_status: str
    content: dict[str, Any]
    evidence_refs: list[str]
    status: str


@dataclass(frozen=True)
class MemoryEvidence:
    """A reference into one normalized CP conversation turn.

    No user text, assistant prose, command, path, or tool payload is copied into
    this object.  ``occurred_at_ms``, ``part_ordinal`` and ``text_offset`` form a
    deterministic source order for user-contract corrections, including two
    contradictory clauses in one message.
    """

    evidence_id: str
    turn_id: str
    source_kind: str
    occurred_at_ms: int
    part_ordinal: int = 0
    text_offset: int = 0
    project_key: str | None = None
    session_id: str | None = None
    explicit: bool = False
    verified: bool = False
    origin: str = "normalized_cp_turn"

    @property
    def source_order(self) -> tuple[int, int, int, str]:
        return (
            self.occurred_at_ms,
            self.part_ordinal,
            self.text_offset,
            self.evidence_id,
        )


class MemoryCandidateValidationError(ValueError):
    """A Writer proposal cannot cross the deterministic boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(code: str, message: str) -> MemoryCandidateValidationError:
    return MemoryCandidateValidationError(code, message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def _check_text(
    value: Any,
    field: str,
    *,
    maximum: int,
    single_line: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error("invalid_text", f"{field} must be a non-empty string")
    if "\x00" in value or any(ord(character) < 32 and character not in "\r\n\t" for character in value):
        raise _error("invalid_text", f"{field} contains control characters")
    normalized = _normalized_text(value)
    if len(normalized) > maximum:
        raise _error("text_too_long", f"{field} exceeds {maximum} characters")
    if single_line and any(character in value for character in "\r\n"):
        raise _error("invalid_text", f"{field} must be a single-line string")
    if any(pattern.search(normalized) for pattern in _SECRET_PATTERNS):
        raise _error("sensitive_content", f"{field} appears to contain a secret")
    if any(pattern.search(normalized) for pattern in _LOCAL_PATH_PATTERNS):
        raise _error("unsafe_path", f"{field} contains a raw local path")
    return normalized


def _strict_mapping(
    value: Any,
    field: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("invalid_schema", f"{field} must be an object")
    result = dict(value)
    missing = required - result.keys()
    extra = result.keys() - required - (optional or set())
    if missing:
        raise _error("invalid_schema", f"{field} missing fields: {sorted(missing)}")
    if extra:
        raise _error("invalid_schema", f"{field} has unsupported fields: {sorted(extra)}")
    return result


def _safe_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise _error("invalid_identifier", f"{field} is invalid")
    return value


def _normalize_json_value(value: Any, field: str, *, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise _error("content_too_deep", f"{field} exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise _error("invalid_content", f"{field} contains a non-finite number")
        return value
    if isinstance(value, str):
        return _check_text(value, field, maximum=2_000)
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise _error("invalid_content", f"{field} contains too many fields")
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", raw_key):
                raise _error("invalid_content", f"{field} contains an invalid field name")
            if _SENSITIVE_FIELD_PATTERN.fullmatch(raw_key):
                raise _error("sensitive_content", f"{field} contains a sensitive field")
            result[raw_key] = _normalize_json_value(
                item, f"{field}.{raw_key}", depth=depth + 1
            )
        return dict(sorted(result.items()))
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise _error("invalid_content", f"{field} contains too many items")
        return [
            _normalize_json_value(item, f"{field}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise _error("invalid_content", f"{field} contains an unsupported value")


def _normalize_scope(value: Any, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
    obj = _strict_mapping(
        value,
        "scope",
        required={"level"},
        optional={"domain", "project_key", "project_keys", "session_id"},
    )
    level = obj["level"]
    if level not in _SCOPE_LEVELS:
        raise _error("invalid_scope", "scope.level is unsupported")
    expected_fields = {
        "global": {"level"},
        "domain": {"level", "domain"},
        "project": {"level", "project_key"},
        "cross_project": {"level", "project_keys"},
        "session": {"level", "session_id"},
    }[level]
    if set(obj) != expected_fields:
        raise _error(
            "invalid_scope",
            f"{level} scope must contain exactly {sorted(expected_fields)}",
        )
    if level == "global":
        return {"level": "global"}
    if level == "domain":
        domain = _check_text(obj["domain"], "scope.domain", maximum=100, single_line=True)
        return {"domain": domain.casefold(), "level": "domain"}
    if level == "project":
        project_key = _safe_identifier(obj["project_key"], "scope.project_key")
        observed = {item.project_key for item in evidence if item.project_key is not None}
        if project_key not in observed:
            raise _error("invalid_scope", "project scope is not supported by cited evidence")
        if any(item.project_key not in {None, project_key} for item in evidence):
            raise _error("invalid_scope", "project scope mixes evidence from another project")
        return {"level": "project", "project_key": project_key}
    if level == "cross_project":
        raw_keys = obj["project_keys"]
        if not isinstance(raw_keys, list) or len(raw_keys) < 2:
            raise _error("invalid_scope", "cross-project scope requires at least two projects")
        project_keys = sorted({_safe_identifier(item, "scope.project_keys") for item in raw_keys})
        if len(project_keys) != len(raw_keys):
            raise _error("invalid_scope", "scope.project_keys contains duplicates")
        observed = sorted({item.project_key for item in evidence if item.project_key is not None})
        if project_keys != observed:
            raise _error(
                "invalid_scope", "cross-project scope must exactly match cited evidence projects"
            )
        return {"level": "cross_project", "project_keys": project_keys}
    session_id = _safe_identifier(obj["session_id"], "scope.session_id")
    observed_sessions = {item.session_id for item in evidence if item.session_id is not None}
    if session_id not in observed_sessions:
        raise _error("invalid_scope", "session scope is not supported by cited evidence")
    if any(item.session_id not in {None, session_id} for item in evidence):
        raise _error("invalid_scope", "session scope mixes evidence from another session")
    return {"level": "session", "session_id": session_id}


def _normalize_evidence(value: MemoryEvidence | Mapping[str, Any]) -> MemoryEvidence:
    if isinstance(value, MemoryEvidence):
        item = value
    elif isinstance(value, Mapping):
        allowed = {
            "evidence_id",
            "turn_id",
            "source_kind",
            "occurred_at_ms",
            "part_ordinal",
            "text_offset",
            "project_key",
            "session_id",
            "explicit",
            "verified",
            "origin",
        }
        if set(value) - allowed:
            raise _error("invalid_evidence", "evidence contains unsupported fields")
        try:
            item = MemoryEvidence(
                evidence_id=value["evidence_id"],
                turn_id=value["turn_id"],
                source_kind=value["source_kind"],
                occurred_at_ms=value["occurred_at_ms"],
                part_ordinal=value.get("part_ordinal", 0),
                text_offset=value.get("text_offset", 0),
                project_key=value.get("project_key"),
                session_id=value.get("session_id"),
                explicit=value.get("explicit", False),
                verified=value.get("verified", False),
                origin=value.get("origin", "normalized_cp_turn"),
            )
        except (KeyError, TypeError) as error:
            raise _error("invalid_evidence", "evidence is missing required fields") from error
    else:
        raise _error("invalid_evidence", "evidence must be a MemoryEvidence or object")
    _safe_identifier(item.evidence_id, "evidence.evidence_id")
    _safe_identifier(item.turn_id, "evidence.turn_id")
    if item.source_kind not in _EVIDENCE_KINDS or item.origin != "normalized_cp_turn":
        raise _error(
            "invalid_evidence_origin",
            "evidence must originate from a normalized CP conversation turn",
        )
    for field, number in (
        ("occurred_at_ms", item.occurred_at_ms),
        ("part_ordinal", item.part_ordinal),
        ("text_offset", item.text_offset),
    ):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise _error("invalid_evidence", f"evidence.{field} must be non-negative")
    if item.project_key is not None:
        _safe_identifier(item.project_key, "evidence.project_key")
    if item.session_id is not None:
        _safe_identifier(item.session_id, "evidence.session_id")
    if not isinstance(item.explicit, bool) or not isinstance(item.verified, bool):
        raise _error("invalid_evidence", "evidence flags must be booleans")
    if item.explicit and item.source_kind != "user_message":
        raise _error("invalid_evidence", "only user-message evidence may be explicit")
    if item.verified and item.source_kind != "tool_result":
        raise _error("invalid_evidence", "only tool-result evidence may be verified")
    return item


def normalize_evidence_catalog(
    evidence_catalog: Sequence[MemoryEvidence | Mapping[str, Any]],
) -> dict[str, MemoryEvidence]:
    """Validate an evidence catalog and index it by immutable evidence ID."""

    if isinstance(evidence_catalog, (str, bytes)) or not isinstance(evidence_catalog, Sequence):
        raise _error("invalid_evidence", "evidence_catalog must be a sequence")
    result: dict[str, MemoryEvidence] = {}
    for raw in evidence_catalog:
        item = _normalize_evidence(raw)
        if item.evidence_id in result:
            raise _error("invalid_evidence", "evidence_catalog contains duplicate IDs")
        result[item.evidence_id] = item
    return result


def _normalize_evidence_refs(
    value: Any, catalog: Mapping[str, MemoryEvidence]
) -> tuple[list[str], list[MemoryEvidence]]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 100
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise _error("invalid_evidence_refs", "evidence_refs must contain 1 to 100 IDs")
    if len(value) != len(set(value)):
        raise _error("invalid_evidence_refs", "evidence_refs contains duplicates")
    unknown = sorted(set(value) - set(catalog))
    if unknown:
        raise _error("unknown_evidence", "candidate cites evidence outside the supplied catalog")
    refs = sorted(value)
    return refs, [catalog[item] for item in refs]


def suggest_stable_key(memory_type: str, concept: str) -> str:
    """Return a safe deterministic suggestion; semantic Writers may improve it."""

    if memory_type not in MEMORY_TYPES:
        raise _error("unsupported_type", "memory type is unsupported")
    normalized = _check_text(concept, "concept", maximum=240, single_line=True).casefold()
    pieces = re.findall(r"[a-z][a-z0-9_]{1,31}", normalized.replace("-", "_"))[:5]
    prefix = "user" if memory_type == "preference" else memory_type
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    if pieces:
        stem = ".".join(pieces)
        suggestion = f"{prefix}.{stem}"
        if len(suggestion) <= 160 and _KEY_PATTERN.fullmatch(suggestion):
            return suggestion
    return f"{prefix}.memory_{digest}"


def semantic_fingerprint_for(candidate: Mapping[str, Any]) -> str:
    """Hash reusable meaning, excluding prose summary, scope, state and evidence."""

    required = {"stable_key", "type", "conditions", "content"}
    if not required <= candidate.keys():
        raise _error("invalid_schema", "candidate lacks semantic fingerprint fields")
    projection = {
        "stable_key": candidate["stable_key"],
        "type": candidate["type"],
        "conditions": candidate["conditions"],
        "content": candidate["content"],
    }
    return _digest("memsem_", projection)


def scope_fingerprint_for(scope: Mapping[str, Any]) -> str:
    return _digest("memscope_", scope)


def _has_explicit_temporary_marker(candidate: Mapping[str, Any]) -> bool:
    semantic_text = _canonical_json(
        {"summary": candidate.get("summary"), "content": candidate.get("content")}
    )
    return any(pattern.search(semantic_text) for pattern in _TEMPORARY_PATTERNS)


def validate_memory_candidate(
    proposal: Mapping[str, Any],
    evidence_catalog: Sequence[MemoryEvidence | Mapping[str, Any]],
) -> MemoryCandidate:
    """Validate and normalize one untrusted, atomic semantic Writer proposal.

    ``unverified`` is a valid formal trust state.  It means useful but not yet
    machine-verified; it must not be silently upgraded by this function.
    """

    catalog = normalize_evidence_catalog(evidence_catalog)
    obj = _strict_mapping(
        proposal,
        "candidate",
        required={
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
        },
    )
    if obj["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise _error("invalid_schema", "unsupported candidate schema_version")
    stable_key = obj["stable_key"]
    if not isinstance(stable_key, str):
        raise _error("invalid_stable_key", "stable_key must be a string")
    stable_key = stable_key.casefold()
    if len(stable_key) > 160 or not _KEY_PATTERN.fullmatch(stable_key):
        raise _error("invalid_stable_key", "stable_key is invalid")
    memory_type = obj["type"]
    if memory_type not in MEMORY_TYPES:
        raise _error("unsupported_type", "memory type is unsupported")
    summary = _check_text(obj["summary"], "summary", maximum=240, single_line=True)
    refs, cited_evidence = _normalize_evidence_refs(obj["evidence_refs"], catalog)
    scope = _normalize_scope(obj["scope"], cited_evidence)
    conditions = _normalize_json_value(obj["conditions"], "conditions")
    if not isinstance(conditions, dict):
        raise _error("invalid_content", "conditions must be an object")
    content = _normalize_json_value(obj["content"], "content")
    if not isinstance(content, dict) or "statement" not in content:
        raise _error("invalid_content", "content must be an object containing statement")
    if set(content) & _MULTIPLEX_KEYS:
        raise _error(
            "non_atomic_candidate",
            "one candidate may describe only one stable key; split candidate collections",
        )
    content["statement"] = _check_text(
        content["statement"], "content.statement", maximum=1_000
    )
    trust_status = obj["trust_status"]
    if trust_status not in TRUST_STATUSES:
        raise _error("invalid_trust_status", "trust_status is unsupported")
    if trust_status == "declared" and not any(
        item.source_kind == "user_message" and item.explicit for item in cited_evidence
    ):
        raise _error("invalid_trust_status", "declared memory requires explicit user evidence")
    if trust_status == "observed" and not any(
        item.source_kind == "tool_result" for item in cited_evidence
    ):
        raise _error("invalid_trust_status", "observed memory requires tool-result evidence")
    if trust_status == "verified" and not any(
        item.source_kind == "tool_result" and item.verified for item in cited_evidence
    ):
        raise _error("invalid_trust_status", "verified memory requires verified tool evidence")
    status = obj["status"]
    if status not in MEMORY_STATUSES:
        raise _error("invalid_status", "memory status is unsupported")
    if status == "conflict" and trust_status != "contested":
        raise _error("invalid_status", "conflict status requires contested trust")
    provisional = {
        "stable_key": stable_key,
        "type": memory_type,
        "summary": summary,
        "scope": scope,
        "conditions": conditions,
        "trust_status": trust_status,
        "content": content,
        "evidence_refs": refs,
        "status": status,
    }
    if _has_explicit_temporary_marker(provisional):
        raise _error("temporary_content", "candidate is explicitly limited to one turn")
    serialized = _canonical_json(provisional).encode("utf-8")
    if len(serialized) > _MAX_JSON_BYTES:
        raise _error("candidate_too_large", "candidate exceeds the 16 KiB semantic limit")
    semantic_fingerprint = semantic_fingerprint_for(provisional)
    normalized: MemoryCandidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "candidate_id": _digest(
            "memcand_",
            {
                "semantic_fingerprint": semantic_fingerprint,
                "scope": scope,
                "evidence_refs": refs,
            },
        ),
        "semantic_fingerprint": semantic_fingerprint,
        **provisional,
    }
    return json.loads(_canonical_json(normalized))


def validate_memory_candidate_batch(
    proposals: Sequence[Mapping[str, Any]],
    evidence_catalog: Sequence[MemoryEvidence | Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate an atomic candidate list without letting one bad item poison it.

    Rejections contain only an index, optional safe stable key, and reason code;
    raw model content is never echoed into reports.
    """

    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
        raise _error("invalid_schema", "proposals must be a sequence")
    # Validate once even when the Writer returned no candidates.
    normalize_evidence_catalog(evidence_catalog)
    accepted: list[MemoryCandidate] = []
    rejected: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, proposal in enumerate(proposals):
        try:
            candidate = validate_memory_candidate(proposal, evidence_catalog)
            identity = (candidate["stable_key"], scope_fingerprint_for(candidate["scope"]))
            if identity in identities:
                raise _error(
                    "duplicate_atomic_candidate",
                    "a batch contains the same stable key and scope more than once",
                )
            identities.add(identity)
            accepted.append(candidate)
        except MemoryCandidateValidationError as error:
            stable_key = None
            if isinstance(proposal, Mapping):
                raw_key = proposal.get("stable_key")
                if isinstance(raw_key, str) and _KEY_PATTERN.fullmatch(raw_key.casefold()):
                    stable_key = raw_key.casefold()
            rejected.append(
                {"index": index, "stable_key": stable_key, "reason": error.code}
            )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "accepted": accepted,
        "rejected": rejected,
    }


def _normalized_candidate(candidate: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "policy_version",
        "candidate_id",
        "semantic_fingerprint",
        "stable_key",
        "type",
        "summary",
        "scope",
        "conditions",
        "trust_status",
        "content",
        "evidence_refs",
        "status",
    }
    if (
        set(candidate) != required
        or candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or candidate.get("policy_version") != POLICY_VERSION
        or not isinstance(candidate.get("candidate_id"), str)
        or not candidate["candidate_id"].startswith("memcand_")
        or candidate.get("semantic_fingerprint") != semantic_fingerprint_for(candidate)
    ):
        raise _error("invalid_schema", "relation input is not a normalized candidate")


def _latest_explicit_user_evidence(
    candidate: Mapping[str, Any], catalog: Mapping[str, MemoryEvidence]
) -> MemoryEvidence | None:
    matches = [
        catalog[reference]
        for reference in candidate.get("evidence_refs", [])
        if reference in catalog
        and catalog[reference].source_kind == "user_message"
        and catalog[reference].explicit
    ]
    return max(matches, key=lambda item: item.source_order) if matches else None


def decide_memory_relation(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
    evidence_catalog: Sequence[MemoryEvidence | Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the only deterministic action; never use time for technical facts."""

    catalog = normalize_evidence_catalog(evidence_catalog)
    _normalized_candidate(incoming)
    if existing is None:
        return {
            "relation": "new",
            "action": "create",
            "latest_wins": False,
            "reason": "no_existing_stable_key",
        }
    _normalized_candidate(existing)
    if existing["stable_key"] != incoming["stable_key"]:
        return {
            "relation": "unrelated",
            "action": "create",
            "latest_wins": False,
            "reason": "different_stable_key",
        }
    if existing["type"] != incoming["type"]:
        return {
            "relation": "type_conflict",
            "action": "open_conflict",
            "latest_wins": False,
            "reason": "same_stable_key_has_different_type",
        }
    if existing["scope"] != incoming["scope"]:
        return {
            "relation": "scope_variant",
            "action": "add_scope_variant",
            "latest_wins": False,
            "reason": "same_key_different_scope",
        }
    if existing["conditions"] != incoming["conditions"]:
        return {
            "relation": "condition_variant",
            "action": "add_scope_variant",
            "latest_wins": False,
            "reason": "same_key_different_conditions",
        }
    if existing["semantic_fingerprint"] == incoming["semantic_fingerprint"]:
        return {
            "relation": "equivalent",
            "action": "merge_evidence",
            "latest_wins": False,
            "reason": "same_reusable_meaning",
        }
    if existing["type"] in _USER_CONTRACT_TYPES:
        old_evidence = _latest_explicit_user_evidence(existing, catalog)
        new_evidence = _latest_explicit_user_evidence(incoming, catalog)
        if old_evidence is not None and new_evidence is not None:
            incoming_wins = new_evidence.source_order > old_evidence.source_order
            existing_wins = old_evidence.source_order > new_evidence.source_order
            if incoming_wins or existing_wins:
                return {
                    "relation": "user_contract_revision",
                    "action": "revise",
                    "latest_wins": True,
                    "winner": "incoming" if incoming_wins else "existing",
                    "reason": "latest_explicit_user_evidence_in_same_scope",
                }
    return {
        "relation": "unresolved_conflict",
        "action": "open_conflict",
        "latest_wins": False,
        "reason": "different_meaning_requires_semantic_resolution",
    }


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "MEMORY_STATUSES",
    "MEMORY_TYPES",
    "MemoryCandidate",
    "MemoryCandidateValidationError",
    "MemoryEvidence",
    "POLICY_VERSION",
    "RELATION_ACTIONS",
    "TRUST_STATUSES",
    "decide_memory_relation",
    "normalize_evidence_catalog",
    "scope_fingerprint_for",
    "semantic_fingerprint_for",
    "suggest_stable_key",
    "validate_memory_candidate",
    "validate_memory_candidate_batch",
]
