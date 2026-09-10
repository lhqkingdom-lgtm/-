"""Ingestion orchestration."""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from pathlib import Path

from .source import classify, iter_parts, open_read_only, source_id, to_event
from .safety import require_distinct_source_and_lab, require_memory_lab_database
from .store import (
    already_processed,
    baseline_cutoff_ms,
    begin_run,
    finish_run,
    mark_child_session_events,
    mark_prebaseline_events,
    open_store,
    record_event,
    record_processed,
    register_source,
)


def run_ingestion(source_database: Path, lab_database: Path, mode: str) -> dict:
    if mode not in {"baseline", "ingest"}:
        raise ValueError(f"Unsupported ingestion mode: {mode}")
    require_distinct_source_and_lab(source_database, lab_database)
    if lab_database.exists():
        require_memory_lab_database(lab_database)
    elif mode == "ingest":
        raise FileNotFoundError(f"lab database does not exist; run baseline first: {lab_database}")

    # Validate both the SQLite file and the CodePulse schema before creating a
    # baseline lab file. A typo/bad source must not leave a plausible empty lab.
    with closing(open_read_only(source_database)) as validation_source:
        next(iter_parts(validation_source), None)

    identifier = source_id(source_database)
    counters = Counter()
    reasons = Counter()
    new_events: list[dict] = []
    isolated_events: list[dict] = []

    with closing(open_store(lab_database)) as store, store:
        register_source(store, identifier, source_database)
        if mode == "ingest":
            baseline = store.execute(
                """
                SELECT s.baseline_cutoff_ms,
                       EXISTS(
                           SELECT 1 FROM ingestion_run AS r
                           WHERE r.source_id = s.source_id
                             AND r.mode = 'baseline'
                             AND r.completed_at IS NOT NULL
                       ) AS has_completed_baseline
                FROM source AS s WHERE s.source_id = ?
                """,
                (identifier,),
            ).fetchone()
            if (
                baseline is None
                or baseline["baseline_cutoff_ms"] is None
                or not baseline["has_completed_baseline"]
            ):
                raise ValueError(
                    "source has no completed durable baseline; run baseline before ingest"
                )
        run_id = begin_run(store, identifier, mode)
        cutoff_ms = baseline_cutoff_ms(store, identifier)
        counters["reclassified_child_events"] = mark_child_session_events(
            store, identifier
        )
        counters["reclassified_late_events"] = mark_prebaseline_events(
            store, identifier, cutoff_ms
        )

        with closing(open_read_only(source_database)) as source:
            source.execute("BEGIN")
            try:
                for part in iter_parts(source):
                    counters["scanned_parts"] += 1
                    if already_processed(store, identifier, part.part_id):
                        counters["duplicate_parts"] += 1
                        continue

                    counters["new_parts"] += 1
                    if mode == "baseline":
                        disposition, reason = "baseline", "existing_before_experiment"
                    else:
                        disposition, reason = classify(part)

                    if disposition == "event":
                        event = to_event(part, identifier)
                        is_child_or_fork = bool(
                            event["parent_session_id"]
                            and event["parent_session_id"].strip()
                        )
                        is_prebaseline_late_arrival = (
                            mode == "ingest"
                            and cutoff_ms is not None
                            and event["timestamp"] is not None
                            and event["timestamp"] < cutoff_ms
                        )
                        if is_child_or_fork:
                            disposition = "isolated"
                            reason = "child_or_fork_session_user_message"
                            record_event(store, event, eligibility_status=reason)
                            isolated_events.append(
                                {**event, "eligibility_status": reason}
                            )
                            counters["isolated_user_events"] += 1
                            counters["filtered_parts"] += 1
                        elif is_prebaseline_late_arrival:
                            disposition = "isolated"
                            reason = "late_arrival_pre_baseline"
                            record_event(store, event, eligibility_status=reason)
                            isolated_events.append(
                                {**event, "eligibility_status": reason}
                            )
                            counters["isolated_user_events"] += 1
                            counters["filtered_parts"] += 1
                        else:
                            record_event(store, event)
                            new_events.append(event)
                            counters["user_events"] += 1
                    else:
                        counters["filtered_parts"] += 1
                    reasons[reason] += 1
                    record_processed(
                        store,
                        identifier,
                        part.part_id,
                        run_id,
                        disposition,
                        reason,
                    )
                source.execute("COMMIT")
            except Exception:
                source.execute("ROLLBACK")
                raise

        report = {
            "run_id": run_id,
            "mode": mode,
            "source_id": identifier,
            "source_database": str(source_database.resolve()),
            "lab_database": str(lab_database.resolve()),
            "baseline_cutoff_ms": cutoff_ms,
            "scanned_parts": counters["scanned_parts"],
            "new_parts": counters["new_parts"],
            "user_events": counters["user_events"],
            "isolated_user_events": counters["isolated_user_events"],
            "reclassified_late_events": counters["reclassified_late_events"],
            "reclassified_child_events": counters["reclassified_child_events"],
            "filtered_parts": counters["filtered_parts"],
            "duplicate_parts": counters["duplicate_parts"],
            "error_count": counters["error_count"],
            "reasons": dict(sorted(reasons.items())),
            "events": new_events,
            "isolated_events": isolated_events,
        }
        finish_run(store, run_id, report)
    return report
