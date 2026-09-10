"""Build a deterministic, privacy-bounded evidence bundle for one CP turn.

The CodePulse database is always opened read-only.  The Memory Lab database is
used only to select a turn that the existing two-sample detector has already
classified as ``completed`` or ``failed`` and to recover its project identity.
No project files, Git metadata, instruction files, or hidden model reasoning
are inspected here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from .prepare import project_identity
from .safety import require_distinct_source_and_lab, require_memory_lab_database
from .source import normalize_text, open_read_only, source_id


SCHEMA_VERSION = "memory-lab-turn-evidence-v1"
TERMINAL_TURN_STATES = {"completed", "failed"}
HIDDEN_PART_TYPES = {"reasoning", "thinking"}

# These are the hard limits of ``unified_model.build_unified_model_request``.
# They intentionally live here as evidence-envelope limits as well: a single
# completed turn must never be capable of wedging the oldest-turn cursor just
# because it contains an unusually long tool transcript.
MODEL_MAX_OBSERVATIONS = 600
MODEL_MAX_TOTAL_OBSERVATION_CHARS = 320_000
TOOL_EVENT_MANIFEST_SCHEMA_VERSION = "memory-lab-tool-event-manifest-v1"


class TurnEvidenceError(ValueError):
    """Raised when a requested turn cannot safely become an evidence bundle."""


@dataclass(frozen=True)
class EvidenceLimits:
    """Per-field limits; all observable tool events are retained."""

    user_text_chars: int = 8_000
    assistant_text_chars: int = 12_000
    tool_payload_chars: int = 4_000
    assistant_error_chars: int = 2_000
    identity_chars: int = 256

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > 1_000_000:
                raise ValueError(f"{name} is unreasonably large")


@dataclass
class _SanitizationStats:
    secret_replacements: int = 0
    path_replacements: int = 0


_SECRET_KEY = re.compile(
    r"(?i)(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|password|"
    r"passwd|secret|authorization|credential|private[_ -]?key|cookie|"
    r"密钥|令牌|密码|口令|凭据|私钥|授权)"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|password|"
    r"passwd|secret|authorization|credential|private[_ -]?key|cookie|"
    r"密钥|令牌|密码|口令|凭据|私钥|授权)"
    r"\s*(?::|=|：|\bis\b(?:\s+set\s+to)?|\bequals?\b|是|为)\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;，。；]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_KNOWN_SECRET_TOKEN = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{8,}\b|\b[A-Za-z0-9_-]*secret[A-Za-z0-9_-]*\b)"
)
_JWT_SECRET = re.compile(
    r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"
)
_PEM_SECRET = re.compile(
    r"-----BEGIN [^-\r\n]+PRIVATE KEY-----.*?-----END [^-\r\n]+PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_WINDOWS_PATH = re.compile(r"(?P<q>[\"'])(?P<p>[A-Za-z]:[\\/][^\"'\r\n]+)(?P=q)")
_WINDOWS_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[A-Z]:[\\/]|\\\\)[^\r\n\"'`,;，。；<>|)\]}]+"
)
_QUOTED_POSIX_PATH = re.compile(r"(?P<q>[\"'])(?P<p>/[^\"'\r\n]+)(?P=q)")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_:/])/[^\r\n\"'`,;，。；<>|)\]}]+")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError, RecursionError):
        return {}
    return result if isinstance(result, dict) else {}


def _open_lab_read_only(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _path_alias(path: str, workspace: str | None, stats: _SanitizationStats) -> str:
    normalized = path.replace("\\", "/")
    workspace_normalized = (workspace or "").replace("\\", "/").rstrip("/")
    if workspace_normalized and normalized.casefold().startswith(
        workspace_normalized.casefold()
    ):
        boundary = len(workspace_normalized)
        if len(normalized) == boundary or normalized[boundary : boundary + 1] == "/":
            relative = normalized[boundary:].lstrip("/")
            stats.path_replacements += 1
            return "<WORKSPACE>" + ("/" + relative if relative else "")
    leaf = normalized.rstrip("/").rsplit("/", 1)[-1]
    stats.path_replacements += 1
    return "<ABS_PATH>" + ("/" + leaf if leaf else "")


def _sanitize_text(
    value: str, workspace: str | None, stats: _SanitizationStats
) -> str:
    text = value

    def secret_substitution(match: re.Match[str]) -> str:
        stats.secret_replacements += 1
        if match.re is _ASSIGNED_SECRET:
            return f"{match.group(1)}=[REDACTED]"
        return "[REDACTED]"

    for pattern in (_PEM_SECRET, _ASSIGNED_SECRET, _BEARER_SECRET, _JWT_SECRET, _KNOWN_SECRET_TOKEN):
        text = pattern.sub(secret_substitution, text)

    # Payload fields frequently contain a path as their entire value.  Handle
    # this before the conservative embedded-path patterns so paths containing
    # spaces cannot leave a suffix behind.
    stripped = text.strip()
    if re.fullmatch(r"(?is)(?:[A-Za-z]:[\\/]|\\\\|/).+", stripped):
        leading = text[: len(text) - len(text.lstrip())]
        trailing = text[len(text.rstrip()) :]
        return leading + _path_alias(stripped, workspace, stats) + trailing

    def quoted_path_substitution(match: re.Match[str]) -> str:
        return match.group("q") + _path_alias(match.group("p"), workspace, stats) + match.group("q")

    text = _QUOTED_WINDOWS_PATH.sub(quoted_path_substitution, text)
    text = _QUOTED_POSIX_PATH.sub(quoted_path_substitution, text)
    text = _WINDOWS_PATH.sub(lambda match: _path_alias(match.group(0), workspace, stats), text)
    text = _POSIX_PATH.sub(lambda match: _path_alias(match.group(0), workspace, stats), text)
    return text


def _sanitize_value(
    value: Any,
    workspace: str | None,
    stats: _SanitizationStats,
    *,
    depth: int = 0,
) -> Any:
    """Convert arbitrary decoded JSON into strict, redacted JSON values."""
    if depth >= 32:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _sanitize_text(value, workspace, stats)
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_value(item, workspace, stats, depth=depth + 1)
            for item in value
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = _sanitize_text(str(raw_key), workspace, stats)
            if _SECRET_KEY.search(key):
                stats.secret_replacements += 1
                sanitized: Any = "[REDACTED]"
            else:
                sanitized = _sanitize_value(
                    value[raw_key], workspace, stats, depth=depth + 1
                )
            candidate = key
            suffix = 2
            while candidate in result:
                candidate = f"{key}#{suffix}"
                suffix += 1
            result[candidate] = sanitized
        return result
    return _sanitize_text(str(value), workspace, stats)


def _bounded_text(
    value: str,
    limit: int,
    workspace: str | None,
    stats: _SanitizationStats,
) -> dict[str, Any]:
    cleaned = normalize_text(_sanitize_text(value, workspace, stats))
    truncated = len(cleaned) > limit
    return {
        "text": cleaned[:limit],
        "truncated": truncated,
        "sanitized_hash": _sha256(cleaned),
    }


def _payload_summary(
    present: bool,
    value: Any,
    limit: int,
    workspace: str | None,
    stats: _SanitizationStats,
) -> dict[str, Any]:
    if not present:
        return {
            "present": False,
            "text": None,
            "truncated": False,
            "sanitized_hash": None,
        }
    safe = _sanitize_value(value, workspace, stats)
    serialized = _canonical_json(safe)
    return {
        "present": True,
        "text": serialized[:limit],
        "truncated": len(serialized) > limit,
        "sanitized_hash": _sha256(serialized),
    }


def _bounded_model_identity(value: Any, maximum: int, fallback: str) -> str:
    """Bound an already-sanitized identity without creating collisions easily."""

    text = value if isinstance(value, str) and value else fallback
    if len(text) <= maximum:
        return text
    suffix = "-" + _sha256(text)[:16]
    return text[: maximum - len(suffix)] + suffix


def _manifest_channel(value: Any) -> dict[str, Any]:
    channel = value if isinstance(value, Mapping) else {}
    digest = channel.get("sanitized_hash")
    if not isinstance(digest, str) or not digest:
        digest = _sha256(str(channel.get("text") or ""))
    return {
        "present": bool(channel.get("present")),
        "truncated": bool(channel.get("truncated")),
        "sanitized_hash": digest,
    }


def _tool_manifest_item(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Retain every tool event's audit identity without retaining its payload."""

    item = {
        "ordinal": tool.get("ordinal"),
        "message_ordinal": tool.get("message_ordinal"),
        "part_ordinal": tool.get("part_ordinal"),
        "assistant_part_ordinal": tool.get("assistant_part_ordinal"),
        "source_message_id": tool.get("source_message_id"),
        "source_part_id": tool.get("source_part_id"),
        "name": tool.get("name"),
        "status": tool.get("status"),
        "input": _manifest_channel(tool.get("input")),
        "output": _manifest_channel(tool.get("output")),
        "error": _manifest_channel(tool.get("error")),
    }
    item["sanitized_event_hash"] = _sha256(_canonical_json(item))
    return item


def _evenly_spaced_indices(count: int, selected: int) -> list[int]:
    """Choose a deterministic chronological sample including both endpoints."""

    if selected >= count:
        return list(range(count))
    if selected <= 0:
        return []
    if selected == 1:
        return [count - 1]
    return [(index * (count - 1)) // (selected - 1) for index in range(selected)]


def _observation_quotas(counts: Sequence[int], maximum: int) -> list[int]:
    """Proportionally bound observation categories while keeping user evidence."""

    total = sum(counts)
    if total <= maximum:
        return list(counts)
    quotas = [int(count * maximum / total) for count in counts]
    for index, count in enumerate(counts):
        if count and not quotas[index]:
            quotas[index] = 1
    while sum(quotas) > maximum:
        candidates = [
            index for index, quota in enumerate(quotas) if quota > (1 if counts[index] else 0)
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: (quotas[item], counts[item], -item))
        quotas[index] -= 1
    while sum(quotas) < maximum:
        candidates = [
            index for index, count in enumerate(counts) if quotas[index] < count
        ]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda item: (counts[item] - quotas[item], counts[item], -item),
        )
        quotas[index] += 1
    return quotas


def _truncate_observation_content(payload: dict[str, Any], cap: int) -> None:
    for actor in ("user", "assistant"):
        container = payload.get(actor)
        if not isinstance(container, dict):
            continue
        parts = container.get("text_parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                continue
            text = part["text"]
            if len(text) > cap:
                part["text"] = text[:cap]
                part["truncated"] = True
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        for field in ("input", "output", "error"):
            channel = tool.get(field)
            if not isinstance(channel, dict) or not isinstance(channel.get("text"), str):
                continue
            text = channel["text"]
            if len(text) > cap:
                channel["text"] = text[:cap]
                channel["truncated"] = True


def _with_bundle_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("bundle_id", None)
    body.pop("bundle_hash", None)
    canonical = _canonical_json(body)
    bundle_hash = _sha256(canonical)
    return {
        **body,
        "bundle_hash": bundle_hash,
        "bundle_id": "turn_evidence_" + bundle_hash,
    }


def _model_contract_accepts(bundle: Mapping[str, Any]) -> None:
    # Lazy import avoids coupling the raw-source reader to the provider client
    # at module import time. This is a pure local projection/validation call.
    from .unified_model import build_unified_model_request

    build_unified_model_request(bundle)


def fit_turn_evidence_to_model_limits(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic envelope that one unified-model call can accept.

    Every original tool event remains represented in ``tool_event_manifest``
    by its source identity, name, status, and sanitized payload hashes. When a
    pathological turn has more than 600 observations, only a deterministic
    chronological sample is model-visible; the audit manifest is lossless for
    the required metadata. Payload text is then uniformly capped only as much
    as needed to meet the exact 320k model text ceiling.
    """

    try:
        payload = json.loads(_canonical_json(dict(bundle)))
    except (TypeError, ValueError, RecursionError) as error:
        raise TurnEvidenceError("turn evidence is not strict JSON") from error
    payload.pop("bundle_id", None)
    payload.pop("bundle_hash", None)

    turn = payload.get("turn")
    workspace = payload.get("workspace")
    session = payload.get("session")
    user = payload.get("user")
    assistant = payload.get("assistant")
    tools = payload.get("tools")
    if not all(isinstance(item, dict) for item in (turn, workspace, session, user, assistant)):
        raise TurnEvidenceError("turn evidence envelope is incomplete")
    if not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools):
        raise TurnEvidenceError("turn evidence tools must be an array of objects")

    workspace["project_key"] = _bounded_model_identity(
        workspace.get("project_key"), 500, "unknown-project"
    )
    session["session_id"] = _bounded_model_identity(
        session.get("session_id"), 500, "unknown-session"
    )
    turn["turn_id"] = _bounded_model_identity(turn.get("turn_id"), 500, "unknown-turn")
    user["message_id"] = _bounded_model_identity(
        user.get("message_id"), 500, "unknown-user-message"
    )
    assistant["message_id"] = _bounded_model_identity(
        assistant.get("message_id"), 500, "unknown-assistant-message"
    )

    for actor in (user, assistant):
        parts = actor.get("text_parts")
        if not isinstance(parts, list):
            raise TurnEvidenceError("turn evidence text_parts must be an array")
        for part in parts:
            if not isinstance(part, dict):
                raise TurnEvidenceError("turn evidence text part must be an object")
            part["source_part_id"] = _bounded_model_identity(
                part.get("source_part_id"), 500, "unknown-text-part"
            )

    original_tools = list(tools)
    prior_manifest = payload.get("tool_event_manifest")
    prior_events = (
        prior_manifest.get("events") if isinstance(prior_manifest, Mapping) else None
    )
    has_prior_manifest = (
        isinstance(prior_manifest, Mapping)
        and prior_manifest.get("schema_version") == TOOL_EVENT_MANIFEST_SCHEMA_VERSION
        and isinstance(prior_events, list)
        and all(isinstance(item, dict) for item in prior_events)
    )
    manifest = (
        list(prior_events)
        if has_prior_manifest
        else [_tool_manifest_item(tool) for tool in original_tools]
    )
    for tool in original_tools:
        tool["source_message_id"] = _bounded_model_identity(
            tool.get("source_message_id"), 500, "unknown-tool-message"
        )
        tool["source_part_id"] = _bounded_model_identity(
            tool.get("source_part_id"), 500, "unknown-tool-part"
        )
        tool["name"] = _bounded_model_identity(tool.get("name"), 160, "unknown")
        tool["status"] = _bounded_model_identity(tool.get("status"), 80, "unknown")

    user_parts = user["text_parts"]
    assistant_parts = assistant["text_parts"]
    quotas = _observation_quotas(
        [len(user_parts), len(assistant_parts), len(original_tools)],
        MODEL_MAX_OBSERVATIONS,
    )
    if not quotas[0]:
        raise TurnEvidenceError("turn evidence has no observable user text")
    selected_user = _evenly_spaced_indices(len(user_parts), quotas[0])
    selected_assistant = _evenly_spaced_indices(len(assistant_parts), quotas[1])
    selected_tools = _evenly_spaced_indices(len(original_tools), quotas[2])
    user["text_parts"] = [user_parts[index] for index in selected_user]
    assistant["text_parts"] = [assistant_parts[index] for index in selected_assistant]
    payload["tools"] = [original_tools[index] for index in selected_tools]
    if not has_prior_manifest:
        selected_tool_indices = set(selected_tools)
        for index, item in enumerate(manifest):
            item["model_observed"] = index in selected_tool_indices
    payload["tool_event_manifest"] = {
        "schema_version": TOOL_EVENT_MANIFEST_SCHEMA_VERSION,
        "total_count": len(manifest),
        "model_observed_count": len(payload["tools"]),
        "events": manifest,
    }

    safety = payload.setdefault("safety", {})
    if not isinstance(safety, dict):
        raise TurnEvidenceError("turn evidence safety must be an object")
    prior_budget = safety.get("model_budget")
    prior_original_count = (
        prior_budget.get("original_observation_count")
        if isinstance(prior_budget, Mapping)
        else None
    )
    budget = {
        "max_observations": MODEL_MAX_OBSERVATIONS,
        "max_total_observation_chars": MODEL_MAX_TOTAL_OBSERVATION_CHARS,
        "original_observation_count": prior_original_count
        if isinstance(prior_original_count, int)
        else len(user_parts) + len(assistant_parts) + len(original_tools),
        "model_observation_envelope_count": (
            len(user["text_parts"]) + len(assistant["text_parts"]) + len(payload["tools"])
        ),
        "content_char_cap": prior_budget.get("content_char_cap")
        if isinstance(prior_budget, Mapping)
        else None,
    }
    safety["model_budget"] = budget

    candidate = _with_bundle_identity(payload)
    try:
        _model_contract_accepts(candidate)
        return candidate
    except Exception as initial_error:
        # Observation count and model identity bounds are already satisfied;
        # the remaining recoverable cause is total observable text. Find the
        # largest uniform content cap that the real model projection accepts.
        lengths: list[int] = []
        for actor in (user, assistant):
            lengths.extend(
                len(part.get("text") or "")
                for part in actor["text_parts"]
                if isinstance(part, dict)
            )
        for tool in payload["tools"]:
            for field in ("input", "output", "error"):
                channel = tool.get(field)
                if isinstance(channel, dict) and isinstance(channel.get("text"), str):
                    lengths.append(len(channel["text"]))
        upper = max(lengths, default=1)
        accepted: dict[str, Any] | None = None
        accepted_cap: int | None = None
        low, high = 1, upper
        while low <= high:
            cap = (low + high) // 2
            trial_payload = json.loads(_canonical_json(payload))
            _truncate_observation_content(trial_payload, cap)
            trial_payload["safety"]["model_budget"]["content_char_cap"] = cap
            trial = _with_bundle_identity(trial_payload)
            try:
                _model_contract_accepts(trial)
            except Exception:
                high = cap - 1
            else:
                accepted = trial
                accepted_cap = cap
                low = cap + 1
        if accepted is None:
            raise TurnEvidenceError(
                "single turn cannot be fitted to the unified model contract"
            ) from initial_error
        accepted["safety"]["model_budget"]["content_char_cap"] = accepted_cap
        accepted = _with_bundle_identity(accepted)
        try:
            _model_contract_accepts(accepted)
        except Exception as error:
            raise TurnEvidenceError(
                "single turn cannot be fitted to the unified model contract"
            ) from error
        return accepted


def _bounded_identity(
    value: Any,
    limit: int,
    workspace: str | None,
    stats: _SanitizationStats,
    *,
    fallback: str,
) -> str:
    if not isinstance(value, str) or not value:
        return fallback
    return _sanitize_text(value, workspace, stats)[:limit]


def _load_turn(connection: sqlite3.Connection, turn_id: str) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT
            t.turn_id, t.source_id, t.session_id, t.user_message_id,
            t.primary_event_id, t.user_timestamp, t.current_state,
            t.last_structural_signature, t.terminal_assistant_id,
            t.terminal_finish, t.terminal_error,
            COALESCE(a.project_key, sc.project_key) AS project_key,
            pc.display_name AS project_display_name
        FROM conversation_turn AS t
        LEFT JOIN event_annotation AS a ON a.event_id = t.primary_event_id
        LEFT JOIN session_context AS sc
          ON sc.source_id = t.source_id AND sc.session_id = t.session_id
        LEFT JOIN project_context AS pc
          ON pc.project_key = COALESCE(a.project_key, sc.project_key)
        WHERE t.turn_id = ?
        """,
        (turn_id,),
    ).fetchone()
    if row is None:
        raise TurnEvidenceError(f"turn does not exist: {turn_id}")
    turn = dict(row)
    if turn["current_state"] not in TERMINAL_TURN_STATES:
        raise TurnEvidenceError(
            f"turn is not a stable completed/failed turn: {turn['current_state']}"
        )
    return turn


def _source_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _load_session_lineage(
    connection: sqlite3.Connection,
    session_id: str,
    workspace: str | None,
    stats: _SanitizationStats,
    identity_limit: int,
) -> dict[str, Any]:
    columns = _source_columns(connection, "session")
    if "id" not in columns:
        raise TurnEvidenceError("CodePulse session table has no id column")
    parent_projection = "parent_id" if "parent_id" in columns else "NULL AS parent_id"
    project_projection = "project_id" if "project_id" in columns else "NULL AS project_id"
    directory_projection = "directory" if "directory" in columns else "NULL AS directory"
    query = (
        f"SELECT id, {parent_projection}, {project_projection}, {directory_projection} "
        "FROM session WHERE id = ?"
    )
    reverse: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = session_id
    complete = True
    while current is not None:
        if current in seen or len(reverse) >= 64:
            complete = False
            break
        seen.add(current)
        row = connection.execute(query, (current,)).fetchone()
        if row is None:
            complete = False
            break
        parent = row["parent_id"] if isinstance(row["parent_id"], str) and row["parent_id"] else None
        reverse.append(
            {
                "session_id": str(row["id"]),
                "parent_session_id": parent,
                "project_id": _bounded_identity(
                    row["project_id"], identity_limit, workspace, stats, fallback="unknown"
                ),
            }
        )
        current = parent
    lineage = list(reversed(reverse))
    for ordinal, item in enumerate(lineage, start=1):
        item["ordinal"] = ordinal
    current_row = reverse[0] if reverse else {
        "session_id": session_id,
        "parent_session_id": None,
        "project_id": "unknown",
    }
    return {
        "session_id": session_id,
        "parent_session_id": current_row["parent_session_id"],
        "root_session_id": lineage[0]["session_id"] if lineage else session_id,
        "branch_depth": max(len(lineage) - 1, 0),
        "lineage_complete": complete,
        "lineage": lineage,
    }


def _read_source_turn(
    connection: sqlite3.Connection,
    turn: dict[str, Any],
    workspace: str | None,
    stats: _SanitizationStats,
    limits: EvidenceLimits,
) -> dict[str, Any]:
    message_columns = _source_columns(connection, "message")
    part_columns = _source_columns(connection, "part")
    required_message = {"id", "session_id", "time_created", "data"}
    required_part = {"id", "message_id", "session_id", "time_created", "data"}
    if not required_message.issubset(message_columns) or not required_part.issubset(part_columns):
        raise TurnEvidenceError("CodePulse message/part schema is incomplete")

    messages = []
    for row in connection.execute(
        """
        SELECT id, session_id, time_created, data
        FROM message WHERE session_id = ? ORDER BY time_created, id
        """,
        (turn["session_id"],),
    ):
        messages.append(
            {
                "id": str(row["id"]),
                "time_created": row["time_created"],
                "data": _safe_json_object(row["data"]),
            }
        )
    parts_by_message: dict[str, list[dict[str, Any]]] = {}
    for row in connection.execute(
        """
        SELECT id, message_id, time_created, data
        FROM part WHERE session_id = ? ORDER BY time_created, id
        """,
        (turn["session_id"],),
    ):
        parts_by_message.setdefault(str(row["message_id"]), []).append(
            {
                "id": str(row["id"]),
                "time_created": row["time_created"],
                "data": _safe_json_object(row["data"]),
            }
        )

    message_ordinals = {item["id"]: ordinal for ordinal, item in enumerate(messages, 1)}
    user_message = next(
        (item for item in messages if item["id"] == turn["user_message_id"]), None
    )
    if user_message is None or user_message["data"].get("role") != "user":
        raise TurnEvidenceError("source user message is missing or no longer a user message")

    assistants = [
        item
        for item in messages
        if item["data"].get("role") == "assistant"
        and item["data"].get("parentID") == turn["user_message_id"]
    ]
    terminal_id = turn.get("terminal_assistant_id")
    final_assistant = next(
        (item for item in assistants if item["id"] == terminal_id), None
    )
    if final_assistant is None and assistants:
        final_assistant = assistants[-1]
        terminal_id = final_assistant["id"]

    excluded_counts: dict[str, int] = {name: 0 for name in sorted(HIDDEN_PART_TYPES)}
    user_parts = []
    for part_ordinal, part in enumerate(parts_by_message.get(user_message["id"], []), 1):
        data = part["data"]
        part_type = str(data.get("type") or "unknown").casefold()
        if part_type in HIDDEN_PART_TYPES:
            excluded_counts[part_type] += 1
            continue
        text = data.get("text")
        if part_type != "text" or not isinstance(text, str) or not text.strip():
            continue
        if text.lstrip().casefold().startswith(("<system-reminder", "<system")):
            continue
        safe_text = _bounded_text(text, limits.user_text_chars, workspace, stats)
        user_parts.append(
            {
                "ordinal": part_ordinal,
                "source_message_id": user_message["id"],
                "source_part_id": part["id"],
                **safe_text,
            }
        )
    if not user_parts:
        raise TurnEvidenceError("source user message has no observable user text")

    final_text_parts = []
    if final_assistant is not None:
        for part_ordinal, part in enumerate(parts_by_message.get(final_assistant["id"], []), 1):
            data = part["data"]
            part_type = str(data.get("type") or "unknown").casefold()
            if part_type in HIDDEN_PART_TYPES:
                continue
            text = data.get("text")
            if part_type != "text" or not isinstance(text, str) or not text.strip():
                continue
            safe_text = _bounded_text(text, limits.assistant_text_chars, workspace, stats)
            final_text_parts.append(
                {
                    "ordinal": part_ordinal,
                    "source_message_id": final_assistant["id"],
                    "source_part_id": part["id"],
                    **safe_text,
                }
            )

    tool_events = []
    observable_message_order = []
    tool_ordinal = 0
    global_part_ordinal = 0
    for assistant in assistants:
        observable_part_order = []
        for part_ordinal, part in enumerate(parts_by_message.get(assistant["id"], []), 1):
            global_part_ordinal += 1
            data = part["data"]
            part_type = str(data.get("type") or "unknown").casefold()
            if part_type in HIDDEN_PART_TYPES:
                excluded_counts[part_type] += 1
                continue
            if part_type != "tool":
                if part_type == "text" and assistant["id"] == terminal_id:
                    observable_part_order.append(
                        {"ordinal": part_ordinal, "source_part_id": part["id"], "type": "text"}
                    )
                continue
            observable_part_order.append(
                {"ordinal": part_ordinal, "source_part_id": part["id"], "type": "tool"}
            )
            tool_ordinal += 1
            state = data.get("state") if isinstance(data.get("state"), dict) else {}

            def payload(field: str) -> tuple[bool, Any]:
                if field in state:
                    return True, state[field]
                if field in data:
                    return True, data[field]
                return False, None

            input_present, input_value = payload("input")
            output_present, output_value = payload("output")
            error_present, error_value = payload("error")
            tool_events.append(
                {
                    "ordinal": tool_ordinal,
                    "message_ordinal": message_ordinals[assistant["id"]],
                    "part_ordinal": part_ordinal,
                    "assistant_part_ordinal": global_part_ordinal,
                    "source_message_id": assistant["id"],
                    "source_part_id": part["id"],
                    "name": _bounded_identity(
                        data.get("tool") or data.get("name"),
                        limits.identity_chars,
                        workspace,
                        stats,
                        fallback="unknown",
                    ),
                    "status": _bounded_identity(
                        state.get("status") or data.get("status"),
                        limits.identity_chars,
                        workspace,
                        stats,
                        fallback="unknown",
                    ),
                    "input": _payload_summary(
                        input_present, input_value, limits.tool_payload_chars, workspace, stats
                    ),
                    "output": _payload_summary(
                        output_present, output_value, limits.tool_payload_chars, workspace, stats
                    ),
                    "error": _payload_summary(
                        error_present, error_value, limits.tool_payload_chars, workspace, stats
                    ),
                }
            )
        observable_message_order.append(
            {
                "ordinal": message_ordinals[assistant["id"]],
                "source_message_id": assistant["id"],
                "is_final": assistant["id"] == terminal_id,
                "observable_part_order": observable_part_order,
            }
        )

    assistant_error_present = bool(
        final_assistant is not None and "error" in final_assistant["data"]
    )
    assistant_error = _payload_summary(
        assistant_error_present,
        final_assistant["data"].get("error") if final_assistant else None,
        limits.assistant_error_chars,
        workspace,
        stats,
    )
    return {
        "user": {
            "message_id": user_message["id"],
            "message_ordinal": message_ordinals[user_message["id"]],
            "occurred_at_ms": user_message["time_created"],
            "text_parts": user_parts,
        },
        "assistant": {
            "message_id": terminal_id,
            "message_ordinal": message_ordinals.get(terminal_id),
            "text_parts": final_text_parts,
            "error": assistant_error,
            "message_order": observable_message_order,
        },
        "tools": tool_events,
        "excluded_hidden_parts": {
            key: value for key, value in excluded_counts.items() if value
        },
    }


def build_turn_evidence(
    source_database: Path,
    lab_database: Path,
    turn_id: str,
    *,
    limits: EvidenceLimits | None = None,
) -> dict[str, Any]:
    """Return one deterministic, JSON-safe evidence bundle.

    The same source snapshot, lab turn, and limits produce exactly the same
    canonical payload, ``bundle_hash``, and ``bundle_id``.
    """
    source_database = Path(source_database)
    lab_database = Path(lab_database)
    limits = limits or EvidenceLimits()
    if not isinstance(turn_id, str) or not turn_id:
        raise ValueError("turn_id must be a non-empty string")
    require_distinct_source_and_lab(source_database, lab_database)
    require_memory_lab_database(lab_database)
    expected_source_id = source_id(source_database)

    with closing(_open_lab_read_only(lab_database)) as lab:
        lab.execute("BEGIN")
        try:
            turn = _load_turn(lab, turn_id)
        finally:
            lab.rollback()
    if turn["source_id"] != expected_source_id:
        raise TurnEvidenceError("turn belongs to a different CodePulse database")

    stats = _SanitizationStats()
    with closing(open_read_only(source_database)) as source:
        source.execute("BEGIN")
        try:
            session_columns = _source_columns(source, "session")
            directory_projection = (
                "directory" if "directory" in session_columns else "NULL AS directory"
            )
            session_row = source.execute(
                f"SELECT {directory_projection} FROM session WHERE id = ?",
                (turn["session_id"],),
            ).fetchone()
            if session_row is None:
                raise TurnEvidenceError("source session is missing")
            workspace = (
                session_row["directory"]
                if isinstance(session_row["directory"], str)
                else None
            )
            lineage = _load_session_lineage(
                source, turn["session_id"], workspace, stats, limits.identity_chars
            )
            source_turn = _read_source_turn(source, turn, workspace, stats, limits)
        finally:
            source.rollback()

    if workspace:
        normalized_workspace = workspace.replace("\\", "/").rstrip("/").casefold()
        path_fingerprint = _sha256(normalized_workspace)
        try:
            display_fallback = PureWindowsPath(workspace).name or "workspace"
        except ValueError:
            display_fallback = "workspace"
        display_name = _bounded_identity(
            turn.get("project_display_name"),
            limits.identity_chars,
            workspace,
            stats,
            fallback=display_fallback,
        )
        path_alias = f"<WORKSPACE:{path_fingerprint[:12]}>/{display_name}"
        project_key = turn.get("project_key") or project_identity(workspace)[0]
    else:
        path_fingerprint = None
        path_alias = "<WORKSPACE:unknown>"
        display_name = _bounded_identity(
            turn.get("project_display_name"),
            limits.identity_chars,
            workspace,
            stats,
            fallback="unknown",
        )
        project_key = turn.get("project_key") or f"unknown:{_sha256(turn['session_id'])[:12]}"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "turn": {
            "turn_id": turn["turn_id"],
            "state": turn["current_state"],
            "occurred_at_ms": source_turn["user"]["occurred_at_ms"],
            "source_id": turn["source_id"],
            "source_signature": turn["last_structural_signature"],
            "user_message_id": turn["user_message_id"],
            "terminal_assistant_id": turn["terminal_assistant_id"],
            "terminal_finish": turn["terminal_finish"],
            "terminal_error": bool(turn["terminal_error"]),
        },
        "workspace": {
            "project_key": project_key,
            "display_name": display_name,
            "path_alias": path_alias,
            "path_fingerprint": path_fingerprint,
            "absolute_path_included": False,
        },
        "session": lineage,
        "user": source_turn["user"],
        "assistant": source_turn["assistant"],
        "tools": source_turn["tools"],
        "safety": {
            "hidden_reasoning_included": False,
            "excluded_hidden_parts": source_turn["excluded_hidden_parts"],
            "secret_replacement_count": stats.secret_replacements,
            "path_replacement_count": stats.path_replacements,
            "source_opened_read_only": True,
            "project_files_scanned": False,
            "limits": dict(limits.__dict__),
        },
    }
    canonical = _canonical_json(payload)
    bundle_hash = _sha256(canonical)
    result = {
        **payload,
        "bundle_hash": bundle_hash,
        "bundle_id": "turn_evidence_" + bundle_hash,
    }
    # This final serialization is an invariant check for strict JSON safety.
    _canonical_json(result)
    return fit_turn_evidence_to_model_limits(result)


__all__ = [
    "EvidenceLimits",
    "SCHEMA_VERSION",
    "TurnEvidenceError",
    "build_turn_evidence",
    "fit_turn_evidence_to_model_limits",
]
