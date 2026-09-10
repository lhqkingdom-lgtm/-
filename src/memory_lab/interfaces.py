"""Stable in-process contracts for Memory Lab and a future CodePulse bridge.

The experiment currently uses :class:`SqliteSourceAdapter` semantics.  A later
CodePulse plugin can implement the same source protocol without giving the
plugin write access to the memory store.  These protocols deliberately avoid
HTTP details; the sidecar wire format can be added only after the standalone
quality gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class SourceCursor:
    """Opaque, adapter-owned position used for idempotent incremental scans."""

    value: str


@dataclass(frozen=True)
class NormalizedTurn:
    """A completed source turn without provider-specific database objects."""

    turn_id: str
    session_id: str
    project_key: str
    completed_at: int
    state: str
    event_ids: tuple[str, ...]
    source_cursor: SourceCursor


@dataclass(frozen=True)
class EvidencePacket:
    """Minimum reviewed input from which an update plan may be proposed."""

    packet_id: str
    turn_id: str
    family: str
    claims: tuple[Mapping[str, Any], ...]
    snapshot_hash: str


@dataclass(frozen=True)
class UpdatePlan:
    """Immutable Writer output.  The executor still validates every action."""

    plan_id: str
    snapshot_hash: str
    actions: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ReviewDecision:
    """Independent Reviewer decision bound to one exact update plan."""

    plan_id: str
    outcome: str
    reasons: tuple[str, ...] = ()
    revised_actions: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RecallRequest:
    query: str
    project_key: str | None = None
    directory: str | None = None
    domain: str | None = None
    conditions: Mapping[str, str] = field(default_factory=dict)
    limit: int = 5


@dataclass(frozen=True)
class RelevantMemory:
    memory_id: str
    revision_id: str
    stable_key: str
    scope: Mapping[str, Any]
    statement: str
    match_reason: tuple[str, ...]
    stale: bool = False
    verify_before_use: bool = False


class SourceAdapter(Protocol):
    def scan(self, cursor: SourceCursor | None) -> Sequence[NormalizedTurn]:
        """Return committed turns after ``cursor`` without mutating the source."""


class MemoryWriter(Protocol):
    def process(
        self, evidence: EvidencePacket, snapshot: Mapping[str, Any]
    ) -> UpdatePlan:
        """Propose changes; never write the formal store."""


class MemoryReviewer(Protocol):
    def review(
        self,
        plan: UpdatePlan,
        evidence: EvidencePacket,
        snapshot: Mapping[str, Any],
    ) -> ReviewDecision:
        """Review one exact plan in an independent model context."""


class RecallService(Protocol):
    def recall(self, request: RecallRequest) -> Sequence[RelevantMemory]:
        """Return only active, applicable, bounded memories."""
