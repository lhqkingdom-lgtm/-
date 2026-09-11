"""Privacy-bounded engineering-memory evidence and candidate policy.

This module is deliberately separated from the formal memory writer.  It reads
only the immutable ``cp-trace-episode-v3`` projection in the Memory Lab database
and produces JSON-safe evidence bundles for a later semantic Writer/Reviewer.

The projection is intentionally lossy.  Assistant prose, reasoning, commands,
paths, tool inputs/outputs, logs, and arbitrary metadata are neither queried nor
copied.  Consequently, a bundle can prove a narrowly structured tool outcome,
but it cannot by itself prove a model-proposed cause or solution.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from .episodes import EXTRACTOR_VERSION
from .safety import require_memory_lab_database
from .source import open_read_only


EVIDENCE_BUNDLE_SCHEMA_VERSION = "engineering-evidence-bundle-v1"
CANDIDATE_SCHEMA_VERSION = "engineering-memory-candidate-v1"
POLICY_VERSION = "engineering-memory-policy-v1"

_SAFE_TOOL_NAMES = {
    "bash",
    "keil_build",
    "json_api",
    "ssot",
    "edit",
    "write",
    "read",
    "glob",
    "grep",
    "rag",
    "other",
}
_SAFE_STATES = {"pending", "running", "completed", "error", "other", "unknown"}
_READ_ONLY_TOOLS = {"read", "glob", "grep", "rag"}
_EXPERIENCE_KINDS = {"successful_resolution", "failure_mode", "reusable_procedure"}
_ATTEMPT_OUTCOMES = {"succeeded", "failed", "unknown"}
_CERTAINTIES = {"observed", "inferred", "unknown"}
_CANDIDATE_STATUSES = {"verified", "unverified", "contested"}
_KEY_PATTERN = re.compile(r"^engineering(?:\.[a-z][a-z0-9_]*){2,}$")

# Free-form semantic fields are an untrusted model boundary.  Reject likely raw
# paths, commands, secrets, and payload dumps rather than trying to redact them
# after the model has echoed them.
_FORBIDDEN_TEXT = (
    re.compile(
        r"(?i)(?:[a-z]:[\\/]|\\\\[^\\\s]+\\|(?:file|https?)://|"
        r"/(?:home|users|etc|var|tmp)/)"
    ),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|passwd)\s*[:=]"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(?:cmd(?:\.exe)?\s+/c|powershell(?:\.exe)?\b|bash\s+-c\b)"),
    re.compile(r"(?i)\b[a-z0-9_-]+\.exe\s+(?:[-/]\S+|\S+\.\w+)"),
    re.compile(r"(?:&&|\|\||\$\(|`[^`]+`)"),
    re.compile(r"(?:\{[^{}]{1,240}\}|\[[^\[\]]{1,240}\])"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/-]{12,}={0,2}\b"),
    re.compile(r"(?<![a-z0-9])[a-f0-9]{48,}(?![a-z0-9])", re.I),
    re.compile(r"(?<![a-z0-9])[a-z0-9_-]{64,}(?![a-z0-9])", re.I),
)


class EngineeringMemoryError(RuntimeError):
    """Base error for the engineering-memory boundary."""


class EngineeringCandidateValidationError(EngineeringMemoryError):
    """A semantic candidate violated the frozen candidate contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(prefix: str, value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()


def _safe_int(value: Any, *, nonnegative: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < -(2**63) or value > 2**63 - 1:
        return None
    if nonnegative and value < 0:
        return None
    return value


def _safe_metadata(tool_name: str, raw: Any) -> dict[str, bool | int | str]:
    """Re-apply the E1 allow-list; never trust stored JSON implicitly."""

    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, bool | int | str] = {}
    if tool_name == "bash":
        exit_code = _safe_int(raw.get("exit"))
        if exit_code is not None:
            result["exit"] = exit_code
    elif tool_name == "keil_build":
        success = raw.get("success")
        if isinstance(success, bool):
            result["success"] = success
        for key in ("errorCount", "warningCount"):
            number = _safe_int(raw.get(key), nonnegative=True)
            if number is not None:
                result[key] = number
        for key in ("failureKind", "buildFailureKind"):
            value = raw.get(key)
            if value in {"source", "build_runner_missing", "other"}:
                result[key] = value
    elif tool_name in {"json_api", "ssot"}:
        success = raw.get("success")
        if isinstance(success, bool):
            result["success"] = success
    return result


def _decode_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_tool_name(value: Any) -> str:
    normalized = value.casefold() if isinstance(value, str) else "other"
    return normalized if normalized in _SAFE_TOOL_NAMES else "other"


def _safe_state(value: Any) -> str:
    normalized = value.casefold() if isinstance(value, str) else "unknown"
    return normalized if normalized in _SAFE_STATES else "other"


def _safe_episode_event(row: Mapping[str, Any]) -> dict[str, Any]:
    tool_name = _safe_tool_name(row["tool_name"])
    started_at = _safe_int(row["started_at_ms"], nonnegative=True)
    ended_at = _safe_int(row["ended_at_ms"], nonnegative=True)
    duration_ms = (
        ended_at - started_at
        if started_at is not None and ended_at is not None and ended_at >= started_at
        else None
    )
    return {
        "evidence_id": str(row["episode_event_id"]),
        "ordinal": int(row["ordinal"]),
        "tool_name": tool_name,
        "state_status": _safe_state(row["state_status"]),
        "duration_ms": duration_ms,
        "has_error": bool(row["has_error_payload"]),
        "facts": _safe_metadata(tool_name, _decode_metadata(row["metadata_json"])),
    }


def _keil_signal(event: Mapping[str, Any]) -> str:
    if event["tool_name"] != "keil_build" or event["state_status"] != "completed":
        return "unverified"
    facts = event["facts"]
    has_claim = any(
        key in facts for key in ("success", "errorCount", "failureKind", "buildFailureKind")
    )
    if event["duration_ms"] is None or event["has_error"]:
        return "ambiguous" if has_claim else "unverified"
    success = facts.get("success")
    error_count = facts.get("errorCount")
    warning_count = facts.get("warningCount")
    has_failure_kind = "failureKind" in facts or "buildFailureKind" in facts
    if (
        not isinstance(success, bool)
        or isinstance(error_count, bool)
        or not isinstance(error_count, int)
        or isinstance(warning_count, bool)
        or not isinstance(warning_count, int)
    ):
        return "ambiguous" if has_claim else "unverified"
    if success is True:
        return "verified_success" if error_count == 0 and not has_failure_kind else "ambiguous"
    if error_count > 0 or has_failure_kind:
        return "verified_failure"
    return "ambiguous"


def _derive_verification(events: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    signals = [
        (index, event, _keil_signal(event))
        for index, event in enumerate(events)
        if event["tool_name"] == "keil_build"
    ]
    if any(signal == "ambiguous" for _, _, signal in signals):
        return "ambiguous", []
    decisive = [
        (index, event, signal)
        for index, event, signal in signals
        if signal in {"verified_success", "verified_failure"}
    ]
    distinct = {signal for _, _, signal in decisive}
    if len(distinct) > 1:
        return "ambiguous", []
    if not distinct:
        return "unverified", []
    outcome = next(iter(distinct))
    references = [
        str(event["evidence_id"])
        for _, event, signal in decisive
        if signal == outcome
    ]
    return outcome, references


def _verification_projection(
    stored_outcome: str, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    derived_outcome, evidence_refs = _derive_verification(events)
    if stored_outcome != derived_outcome:
        return {
            "outcome": "unverified",
            "source_outcome": stored_outcome,
            "structured": False,
            "trusted": False,
            "evidence_refs": [],
            "reason": "episode_outcome_does_not_match_safe_event_projection",
        }
    structured = derived_outcome in {"verified_success", "verified_failure"}
    reason = {
        "verified_success": "allowlisted_keil_success",
        "verified_failure": "allowlisted_keil_failure",
        "ambiguous": "structured_events_are_contested_or_ambiguous",
        "unverified": "no_decisive_structured_domain_evidence",
    }[derived_outcome]
    return {
        "outcome": derived_outcome,
        "source_outcome": stored_outcome,
        "structured": structured,
        "trusted": structured,
        "evidence_refs": evidence_refs if structured else [],
        "reason": reason,
    }


def _environment_projection(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    toolchains = sorted(
        {"keil" for event in events if event["tool_name"] == "keil_build"}
    )
    descriptor = {"toolchains": toolchains, "evidence_quality": "coarse_tool_identity"}
    if not toolchains:
        descriptor["evidence_quality"] = "unknown"
    return {
        **descriptor,
        "fingerprint": _digest("env_", descriptor),
    }


def load_engineering_evidence_bundles(
    lab_database: Path,
    *,
    episode_ids: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Read current v3 episodes into privacy-bounded semantic input bundles.

    No query in this function touches the original CodePulse database, the
    original event text, assistant messages, or tool payload columns.
    """

    lab_database = Path(lab_database)
    require_memory_lab_database(lab_database)
    if episode_ids is None:
        requested = None
    else:
        if isinstance(episode_ids, (str, bytes)):
            raise ValueError("episode_ids must be a collection of non-empty strings")
        requested_values = list(episode_ids)
        if any(not isinstance(item, str) or not item for item in requested_values):
            raise ValueError("episode_ids must contain non-empty strings")
        requested = sorted(set(requested_values))
    with closing(open_read_only(lab_database)) as connection:
        parameters: list[Any] = [EXTRACTOR_VERSION]
        predicate = ""
        if requested is not None:
            if not requested:
                return []
            predicate = " AND episode_id IN (" + ",".join("?" for _ in requested) + ")"
            parameters.extend(requested)
        episodes = connection.execute(
            """
            SELECT episode_id, turn_id, primary_event_id, project_key,
                   trace_signature, turn_state, domain_verification_outcome,
                   assistant_count, tool_event_count, created_at
            FROM trace_episode
            WHERE extractor_version = ?
            """
            + predicate
            + " ORDER BY created_at, episode_id",
            parameters,
        ).fetchall()
        if requested is not None:
            found = {str(row["episode_id"]) for row in episodes}
            missing = sorted(set(requested) - found)
            if missing:
                raise ValueError("unknown or non-v3 episode_ids: " + ", ".join(missing))

        bundles: list[dict[str, Any]] = []
        for episode in episodes:
            event_rows = connection.execute(
                """
                SELECT episode_event_id, ordinal, tool_name, state_status,
                       started_at_ms, ended_at_ms, has_error_payload,
                       metadata_json
                FROM trace_episode_event
                WHERE episode_id = ?
                ORDER BY ordinal, episode_event_id
                """,
                (episode["episode_id"],),
            ).fetchall()
            events = [_safe_episode_event(row) for row in event_rows]
            verification = _verification_projection(
                str(episode["domain_verification_outcome"]), events
            )
            environment = _environment_projection(events)
            identity = {
                "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
                "episode_id": str(episode["episode_id"]),
                "trace_signature": str(episode["trace_signature"]),
            }
            bundles.append(
                {
                    "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
                    "bundle_id": _digest("engbundle_", identity),
                    "episode_id": str(episode["episode_id"]),
                    "turn_id": str(episode["turn_id"]),
                    "primary_event_id": str(episode["primary_event_id"]),
                    "project_key": str(episode["project_key"]),
                    "turn_state": str(episode["turn_state"]),
                    "environment": environment,
                    "verification": verification,
                    "trace": {
                        "assistant_count": int(episode["assistant_count"]),
                        "tool_event_count": len(events),
                        "events": events,
                    },
                    "privacy": {
                        "original_user_text_included": False,
                        "assistant_text_included": False,
                        "reasoning_included": False,
                        "commands_included": False,
                        "paths_included": False,
                        "tool_payloads_included": False,
                        "logs_included": False,
                        "arbitrary_metadata_included": False,
                    },
                }
            )
    return bundles


def build_engineering_candidate_template(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact semantic slots a later Writer must fill.

    The template is not a storable memory.  A structural success/failure can be
    carried into ``verification``, while symptom/cause/solution remain unset so
    the trace layer never invents semantic experience.
    """

    _validate_bundle_shape(bundle)
    events = bundle["trace"]["events"]
    status = (
        "contested"
        if bundle["verification"]["outcome"] == "ambiguous"
        or bundle["verification"]["reason"]
        == "episode_outcome_does_not_match_safe_event_projection"
        else "unverified"
    )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "source_bundle_ids": [bundle["bundle_id"]],
        "stable_key": None,
        "experience_kind": None,
        "symptom": {"summary": None, "evidence_refs": []},
        "environment": {
            "summary": None,
            "fingerprint": bundle["environment"]["fingerprint"],
            "toolchains": list(bundle["environment"]["toolchains"]),
            "evidence_refs": [event["evidence_id"] for event in events],
        },
        "attempts": [
            {
                "summary": None,
                "outcome": (
                    "failed"
                    if event["state_status"] == "error"
                    or event["has_error"]
                    or _keil_signal(event) == "verified_failure"
                    else (
                        "succeeded"
                        if _keil_signal(event) == "verified_success"
                        else "unknown"
                    )
                ),
                "evidence_refs": [event["evidence_id"]],
            }
            for event in events
        ],
        "cause": None,
        "solution": None,
        "verification": {
            "outcome": bundle["verification"]["outcome"],
            "evidence_refs": list(bundle["verification"]["evidence_refs"]),
        },
        "scope": {
            "level": "project",
            "project_keys": [bundle["project_key"]],
            "environment_fingerprint": bundle["environment"]["fingerprint"],
        },
        "status": status,
    }


def _strict_object(
    value: Any,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EngineeringCandidateValidationError(f"{name} must be an object")
    result = dict(value)
    missing = required - result.keys()
    extra = result.keys() - required - (optional or set())
    if missing:
        raise EngineeringCandidateValidationError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise EngineeringCandidateValidationError(
            f"{name} has unsupported fields: {sorted(extra)}"
        )
    return result


def _safe_summary(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EngineeringCandidateValidationError(f"{name} must be a non-empty string")
    summary = value.strip()
    if len(summary) > 240:
        raise EngineeringCandidateValidationError(f"{name} exceeds 240 characters")
    if any(character in summary for character in ("\r", "\n", "\x00")):
        raise EngineeringCandidateValidationError(f"{name} must be a single-line summary")
    if summary.startswith(("{", "[")) or any(pattern.search(summary) for pattern in _FORBIDDEN_TEXT):
        raise EngineeringCandidateValidationError(
            f"{name} appears to contain a path, command, secret, or raw payload"
        )
    return summary


def _string_list(value: Any, name: str, *, maximum: int = 50) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise EngineeringCandidateValidationError(
            f"{name} must be an array of at most {maximum} non-empty strings"
        )
    if len(value) != len(set(value)):
        raise EngineeringCandidateValidationError(f"{name} contains duplicates")
    return list(value)


def _validate_bundle_shape(bundle: Mapping[str, Any]) -> None:
    if bundle.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise EngineeringCandidateValidationError("unsupported evidence bundle schema")
    for field in ("bundle_id", "episode_id", "turn_id", "primary_event_id", "project_key"):
        if not isinstance(bundle.get(field), str) or not bundle[field]:
            raise EngineeringCandidateValidationError(f"bundle.{field} is invalid")
    environment = bundle.get("environment")
    verification = bundle.get("verification")
    trace = bundle.get("trace")
    if not isinstance(environment, Mapping) or not isinstance(verification, Mapping):
        raise EngineeringCandidateValidationError("bundle environment/verification is invalid")
    if not isinstance(trace, Mapping) or not isinstance(trace.get("events"), list):
        raise EngineeringCandidateValidationError("bundle trace is invalid")


def _section(
    value: Any,
    name: str,
    evidence_catalog: set[str],
    *,
    optional: bool = False,
) -> dict[str, Any] | None:
    if value is None and optional:
        return None
    obj = _strict_object(
        value,
        name=name,
        required={"summary", "evidence_refs"},
        optional={"certainty"} if optional else None,
    )
    summary = _safe_summary(obj["summary"], f"{name}.summary")
    refs = _string_list(obj["evidence_refs"], f"{name}.evidence_refs")
    if not set(refs) <= evidence_catalog:
        raise EngineeringCandidateValidationError(f"{name} cites unknown evidence")
    result: dict[str, Any] = {"summary": summary, "evidence_refs": sorted(refs)}
    if optional:
        certainty = obj.get("certainty")
        if certainty not in _CERTAINTIES:
            raise EngineeringCandidateValidationError(f"{name}.certainty is unsupported")
        result["certainty"] = certainty
    return result


def _aggregate_verification(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verifications = [bundle["verification"] for bundle in bundles]
    outcomes = {str(item["outcome"]) for item in verifications}
    all_structured = all(item.get("structured") is True and item.get("trusted") is True for item in verifications)
    if len(outcomes) != 1:
        return {"outcome": "ambiguous", "evidence_refs": [], "structured": False}
    outcome = next(iter(outcomes))
    refs = sorted(
        {
            ref
            for item in verifications
            for ref in item.get("evidence_refs", [])
            if isinstance(ref, str)
        }
    )
    structured = all_structured and outcome in {"verified_success", "verified_failure"}
    return {
        "outcome": outcome,
        "evidence_refs": refs if structured else [],
        "structured": structured,
    }


def validate_engineering_candidate(
    proposal: Mapping[str, Any],
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and normalize an untrusted semantic Writer proposal.

    ``verified`` means that the declared engineering outcome has matching,
    structured tool evidence.  It does not make an inferred causal explanation
    magically observed; cause/solution certainty remains explicit and is still
    subject to the independent semantic Reviewer.
    """

    if not bundles:
        raise EngineeringCandidateValidationError("at least one evidence bundle is required")
    for bundle in bundles:
        _validate_bundle_shape(bundle)
    by_id = {str(bundle["bundle_id"]): bundle for bundle in bundles}
    if len(by_id) != len(bundles):
        raise EngineeringCandidateValidationError("evidence bundles contain duplicate IDs")

    obj = _strict_object(
        proposal,
        name="candidate",
        required={
            "schema_version",
            "source_bundle_ids",
            "stable_key",
            "experience_kind",
            "symptom",
            "environment",
            "attempts",
            "cause",
            "solution",
            "verification",
            "scope",
            "status",
        },
    )
    if obj["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise EngineeringCandidateValidationError("unsupported candidate schema_version")
    source_bundle_ids = _string_list(
        obj["source_bundle_ids"], "candidate.source_bundle_ids", maximum=100
    )
    if set(source_bundle_ids) != set(by_id):
        raise EngineeringCandidateValidationError(
            "candidate source_bundle_ids must exactly match supplied bundles"
        )
    stable_key = obj["stable_key"]
    if not isinstance(stable_key, str) or not _KEY_PATTERN.fullmatch(stable_key):
        raise EngineeringCandidateValidationError("candidate stable_key is invalid")
    if obj["experience_kind"] not in _EXPERIENCE_KINDS:
        raise EngineeringCandidateValidationError("candidate experience_kind is unsupported")

    evidence_catalog = {
        str(event["evidence_id"])
        for bundle in bundles
        for event in bundle["trace"]["events"]
    }
    symptom = _section(obj["symptom"], "candidate.symptom", evidence_catalog)

    environment = _strict_object(
        obj["environment"],
        name="candidate.environment",
        required={"summary", "fingerprint", "toolchains", "evidence_refs"},
    )
    environment_summary = _safe_summary(
        environment["summary"], "candidate.environment.summary"
    )
    fingerprints = {str(bundle["environment"]["fingerprint"]) for bundle in bundles}
    if len(fingerprints) != 1:
        raise EngineeringCandidateValidationError(
            "different environments must be split into separate candidates"
        )
    environment_fingerprint = next(iter(fingerprints))
    if environment["fingerprint"] != environment_fingerprint:
        raise EngineeringCandidateValidationError("candidate environment fingerprint changed")
    expected_toolchains = sorted(
        {
            str(toolchain)
            for bundle in bundles
            for toolchain in bundle["environment"]["toolchains"]
        }
    )
    toolchains = sorted(_string_list(environment["toolchains"], "candidate.environment.toolchains"))
    if toolchains != expected_toolchains:
        raise EngineeringCandidateValidationError("candidate environment toolchains changed")
    environment_refs = sorted(
        _string_list(environment["evidence_refs"], "candidate.environment.evidence_refs")
    )
    if not set(environment_refs) <= evidence_catalog:
        raise EngineeringCandidateValidationError("candidate.environment cites unknown evidence")

    attempts_raw = obj["attempts"]
    if not isinstance(attempts_raw, list) or not 1 <= len(attempts_raw) <= 12:
        raise EngineeringCandidateValidationError("candidate.attempts must contain 1 to 12 items")
    attempts: list[dict[str, Any]] = []
    for index, value in enumerate(attempts_raw):
        attempt = _strict_object(
            value,
            name=f"candidate.attempts[{index}]",
            required={"summary", "outcome", "evidence_refs"},
        )
        if attempt["outcome"] not in _ATTEMPT_OUTCOMES:
            raise EngineeringCandidateValidationError(
                f"candidate.attempts[{index}].outcome is unsupported"
            )
        refs = _string_list(
            attempt["evidence_refs"], f"candidate.attempts[{index}].evidence_refs"
        )
        if not refs or not set(refs) <= evidence_catalog:
            raise EngineeringCandidateValidationError(
                f"candidate.attempts[{index}] must cite known evidence"
            )
        attempts.append(
            {
                "summary": _safe_summary(
                    attempt["summary"], f"candidate.attempts[{index}].summary"
                ),
                "outcome": attempt["outcome"],
                "evidence_refs": sorted(refs),
            }
        )

    cause = _section(obj["cause"], "candidate.cause", evidence_catalog, optional=True)
    solution = _section(obj["solution"], "candidate.solution", evidence_catalog, optional=True)
    if cause is None and solution is None:
        raise EngineeringCandidateValidationError(
            "candidate must contain a reusable cause or solution"
        )

    aggregate = _aggregate_verification(bundles)
    verification = _strict_object(
        obj["verification"],
        name="candidate.verification",
        required={"outcome", "evidence_refs"},
    )
    verification_refs = sorted(
        _string_list(verification["evidence_refs"], "candidate.verification.evidence_refs")
    )
    if verification["outcome"] != aggregate["outcome"]:
        raise EngineeringCandidateValidationError("candidate verification outcome is unsupported")
    if verification_refs != aggregate["evidence_refs"]:
        raise EngineeringCandidateValidationError(
            "candidate verification must cite exactly the structured outcome evidence"
        )

    project_keys = sorted({str(bundle["project_key"]) for bundle in bundles})
    scope = _strict_object(
        obj["scope"],
        name="candidate.scope",
        required={"level", "project_keys", "environment_fingerprint"},
    )
    if scope["level"] not in {"project", "cross_project"}:
        raise EngineeringCandidateValidationError("candidate scope level is unsupported")
    scope_projects = sorted(_string_list(scope["project_keys"], "candidate.scope.project_keys"))
    if scope_projects != project_keys:
        raise EngineeringCandidateValidationError("candidate scope project keys changed")
    if scope["environment_fingerprint"] != environment_fingerprint:
        raise EngineeringCandidateValidationError("candidate scope environment changed")
    if scope["level"] == "project" and len(project_keys) != 1:
        raise EngineeringCandidateValidationError(
            "multiple projects cannot be represented by project scope"
        )
    if scope["level"] == "cross_project" and len(project_keys) < 2:
        raise EngineeringCandidateValidationError(
            "cross-project scope requires evidence from at least two projects"
        )

    expected_status = "unverified"
    if aggregate["outcome"] == "ambiguous":
        expected_status = "contested"
    elif aggregate["structured"]:
        if obj["experience_kind"] in {"successful_resolution", "reusable_procedure"}:
            if aggregate["outcome"] != "verified_success":
                raise EngineeringCandidateValidationError(
                    "a successful experience requires structured success evidence"
                )
        elif obj["experience_kind"] == "failure_mode":
            if aggregate["outcome"] != "verified_failure":
                raise EngineeringCandidateValidationError(
                    "a failure mode requires structured failure evidence"
                )
        expected_status = "verified"
    if obj["status"] not in _CANDIDATE_STATUSES or obj["status"] != expected_status:
        raise EngineeringCandidateValidationError(
            f"candidate status must be {expected_status} for the supplied evidence"
        )
    if scope["level"] == "cross_project" and expected_status != "verified":
        raise EngineeringCandidateValidationError(
            "cross-project promotion requires verified evidence in every project"
        )

    # Evidence IDs identify observations, not the reusable engineering meaning.
    # Excluding them lets independently verified projects converge on the same
    # semantic fingerprint while candidate_id still retains exact provenance.
    semantic_projection = {
        "stable_key": stable_key,
        "experience_kind": obj["experience_kind"],
        "symptom": {"summary": symptom["summary"]},
        "environment": {
            "summary": environment_summary,
            "fingerprint": environment_fingerprint,
            "toolchains": toolchains,
        },
        "attempts": [
            {"summary": attempt["summary"], "outcome": attempt["outcome"]}
            for attempt in attempts
        ],
        "cause": (
            None
            if cause is None
            else {"summary": cause["summary"], "certainty": cause["certainty"]}
        ),
        "solution": (
            None
            if solution is None
            else {"summary": solution["summary"], "certainty": solution["certainty"]}
        ),
        "verification_outcome": aggregate["outcome"],
    }
    semantic_fingerprint = _digest("engsem_", semantic_projection)
    normalized = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "candidate_id": _digest(
            "engcand_",
            {
                "source_bundle_ids": sorted(source_bundle_ids),
                "semantic_fingerprint": semantic_fingerprint,
                "scope": scope,
            },
        ),
        "semantic_fingerprint": semantic_fingerprint,
        "source_bundle_ids": sorted(source_bundle_ids),
        "stable_key": stable_key,
        "experience_kind": obj["experience_kind"],
        "symptom": symptom,
        "environment": {
            "summary": environment_summary,
            "fingerprint": environment_fingerprint,
            "toolchains": toolchains,
            "evidence_refs": environment_refs,
        },
        "attempts": attempts,
        "cause": cause,
        "solution": solution,
        "verification": {
            "outcome": aggregate["outcome"],
            "evidence_refs": aggregate["evidence_refs"],
            "structured": aggregate["structured"],
        },
        "scope": {
            "level": scope["level"],
            "project_keys": project_keys,
            "environment_fingerprint": environment_fingerprint,
        },
        "status": expected_status,
    }
    return json.loads(_canonical_json(normalized))


def decide_engineering_relation(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> dict[str, Any]:
    """Choose a non-temporal relation for two validated candidates."""

    if existing.get("schema_version") != CANDIDATE_SCHEMA_VERSION or incoming.get(
        "schema_version"
    ) != CANDIDATE_SCHEMA_VERSION:
        raise EngineeringCandidateValidationError("relation inputs must be normalized candidates")
    if existing.get("stable_key") != incoming.get("stable_key"):
        return {"relation": "unrelated", "action": "keep_separate"}
    existing_environment = existing.get("environment", {}).get("fingerprint")
    incoming_environment = incoming.get("environment", {}).get("fingerprint")
    if existing_environment != incoming_environment:
        return {
            "relation": "different_environment",
            "action": "add_environment_variant",
            "latest_wins": False,
        }
    if existing.get("semantic_fingerprint") != incoming.get("semantic_fingerprint"):
        return {
            "relation": "conflict",
            "action": "open_engineering_conflict",
            "latest_wins": False,
        }
    projects = sorted(
        set(existing.get("scope", {}).get("project_keys", []))
        | set(incoming.get("scope", {}).get("project_keys", []))
    )
    verified = existing.get("status") == incoming.get("status") == "verified"
    if verified and len(projects) >= 2:
        action = "promote_cross_project"
    elif len(projects) >= 2:
        action = "add_project_validation"
    else:
        action = "merge_evidence"
    return {
        "relation": "equivalent",
        "action": action,
        "project_keys": projects,
        "latest_wins": False,
    }


def engineering_candidate_recall_eligible(candidate: Mapping[str, Any]) -> bool:
    """Only independently reviewable, structurally verified candidates qualify.

    Writer/Reviewer approval is intentionally outside this module.  Callers
    must apply that additional gate before formal commit or recall.
    """

    return bool(
        candidate.get("schema_version") == CANDIDATE_SCHEMA_VERSION
        and candidate.get("policy_version") == POLICY_VERSION
        and candidate.get("status") == "verified"
        and isinstance(candidate.get("verification"), Mapping)
        and candidate["verification"].get("structured") is True
        and candidate["verification"].get("outcome")
        in {"verified_success", "verified_failure"}
        and candidate["verification"].get("evidence_refs")
    )


def assess_cross_project_promotion(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic promotion decision; never use recency as proof."""

    if not candidates:
        return {"eligible": False, "reason": "no_candidates"}
    stable_keys = {candidate.get("stable_key") for candidate in candidates}
    environments = {
        candidate.get("environment", {}).get("fingerprint") for candidate in candidates
    }
    semantics = {candidate.get("semantic_fingerprint") for candidate in candidates}
    projects = sorted(
        {
            project
            for candidate in candidates
            for project in candidate.get("scope", {}).get("project_keys", [])
        }
    )
    if len(stable_keys) != 1:
        return {"eligible": False, "reason": "different_stable_keys"}
    if len(environments) != 1:
        return {"eligible": False, "reason": "different_environments"}
    if len(semantics) != 1:
        return {"eligible": False, "reason": "semantic_conflict"}
    if any(candidate.get("status") != "verified" for candidate in candidates):
        return {"eligible": False, "reason": "unverified_candidate"}
    if len(projects) < 2:
        return {
            "eligible": False,
            "reason": "requires_two_distinct_projects",
            "project_count": len(projects),
        }
    return {
        "eligible": True,
        "reason": "verified_in_two_or_more_projects",
        "stable_key": next(iter(stable_keys)),
        "environment_fingerprint": next(iter(environments)),
        "project_keys": projects,
        "source_candidate_ids": sorted(
            str(candidate["candidate_id"]) for candidate in candidates
        ),
        "latest_wins": False,
    }


def summarize_engineering_candidates(lab_database: Path) -> dict[str, Any]:
    """Build safe bundles and empty semantic templates for inspection."""

    bundles = load_engineering_evidence_bundles(lab_database)
    templates = [build_engineering_candidate_template(bundle) for bundle in bundles]
    counts: dict[str, int] = defaultdict(int)
    for bundle in bundles:
        counts[str(bundle["verification"]["outcome"])] += 1
    return {
        "status": "complete",
        "policy_version": POLICY_VERSION,
        "bundle_count": len(bundles),
        "verification_counts": dict(sorted(counts.items())),
        "bundles": bundles,
        "candidate_templates": templates,
        "formal_memory_written": False,
        "semantic_boundary": (
            "E1 structural traces cannot establish a reusable symptom, cause, or solution; "
            "those fields remain unset until an independently reviewed semantic stage."
        ),
    }
