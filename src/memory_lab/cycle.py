"""One-shot and polling orchestration for the standalone Memory Lab."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from .admission import run_rule_admission
from .change_report import build_change_report, write_change_report
from .config import DEFAULT_CONFIG
from .episodes import run_episode_extraction
from .formal_markdown import publish_formal_memory
from .formal_pipeline import run_deepseek_auto_memory
from .ingest import run_ingestion
from .instructions import run_instruction_coverage
from .prepare import prepare_dataset
from .safety import require_distinct_source_and_lab, require_memory_lab_database
from .source import source_activity_token
from .store import open_store
from .topics import ensure_topic_assignments
from .turns import run_turn_scan


CYCLE_VERSION = "standalone-memory-cycle-v1"


def _stage_ok(report: Mapping[str, Any]) -> bool:
    return report.get("status") not in {"failed", "complete_with_errors"}


def run_memory_cycle(
    source_database: Path,
    lab_database: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    memory_root: Path,
    report_directory: Path,
    settle_ms: int = 1000,
    refresh: bool = False,
    pipeline: Callable[..., dict[str, Any]] = run_deepseek_auto_memory,
    publisher: Callable[[Path, Path], dict[str, Any]] = publish_formal_memory,
) -> dict[str, Any]:
    """Process all observable source changes and publish only a formal commit."""

    require_distinct_source_and_lab(source_database, lab_database)
    require_memory_lab_database(lab_database)
    if isinstance(settle_ms, bool) or not isinstance(settle_ms, int) or not 0 <= settle_ms <= 60_000:
        raise ValueError("settle_ms must be between 0 and 60000")

    stages: dict[str, Any] = {}
    stages["ingest"] = run_ingestion(source_database, lab_database, "ingest")
    stages["turns"] = run_turn_scan(source_database, lab_database, settle_ms)
    stages["episodes"] = run_episode_extraction(
        source_database, lab_database, settle_ms=settle_ms
    )
    stages["prepare"] = prepare_dataset(lab_database)
    stages["coverage"] = run_instruction_coverage(lab_database)
    stages["admission"] = run_rule_admission(lab_database)
    failed_stage = next(
        (name for name, report in stages.items() if not _stage_ok(report)), None
    )
    if failed_stage is not None:
        return {
            "cycle_version": CYCLE_VERSION,
            "status": "failed",
            "failed_stage": failed_stage,
            "stages": stages,
            "formal": None,
            "markdown": None,
            "report_files": {},
        }

    formal = pipeline(lab_database, config_path=config_path, refresh=refresh)
    stages["formal"] = formal
    markdown: dict[str, Any] | None = None
    if _stage_ok(formal):
        with closing(open_store(lab_database)) as connection, connection:
            topic_changes = ensure_topic_assignments(connection)
        stages["topics"] = {"status": "complete", **topic_changes}
        markdown = publisher(lab_database, memory_root)
        stages["markdown"] = markdown
    else:
        stages["markdown"] = {
            "status": "skipped",
            "reason": "formal_pipeline_failed",
        }

    report = build_change_report(
        lab_database,
        formal,
        markdown_report=markdown,
        stages=stages,
    )
    report_files = write_change_report(report_directory, report)
    return {
        "cycle_version": CYCLE_VERSION,
        "status": formal.get("status", "failed"),
        "stages": stages,
        "formal": formal,
        "markdown": markdown,
        "change_report": report,
        "report_files": report_files,
    }


def watch_memory_cycles(
    source_database: Path,
    lab_database: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    memory_root: Path,
    report_directory: Path,
    interval_seconds: float = 15.0,
    settle_ms: int = 1000,
    refresh: bool = False,
    max_cycles: int | None = None,
    on_cycle: Callable[[dict[str, Any]], None] | None = None,
    cycle_runner: Callable[..., dict[str, Any]] = run_memory_cycle,
) -> dict[str, Any]:
    """Poll without a CP plugin; Ctrl+C stops after the current safe boundary."""

    if isinstance(interval_seconds, bool) or not 1 <= interval_seconds <= 3600:
        raise ValueError("interval_seconds must be between 1 and 3600")
    if max_cycles is not None and (isinstance(max_cycles, bool) or max_cycles < 1):
        raise ValueError("max_cycles must be a positive integer")
    cycles: list[dict[str, Any]] = []
    stopped = "max_cycles"
    last_activity: str | None = None
    try:
        while max_cycles is None or len(cycles) < max_cycles:
            activity = source_activity_token(source_database)
            if last_activity is not None and activity == last_activity:
                time.sleep(interval_seconds)
                continue
            result = cycle_runner(
                source_database,
                lab_database,
                config_path=config_path,
                memory_root=memory_root,
                report_directory=report_directory,
                settle_ms=settle_ms,
                refresh=refresh,
            )
            cycles.append(result)
            last_activity = source_activity_token(source_database)
            if on_cycle is not None:
                on_cycle(result)
            if max_cycles is None or len(cycles) < max_cycles:
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        stopped = "keyboard_interrupt"
    return {
        "status": "stopped",
        "reason": stopped,
        "cycle_count": len(cycles),
        "last_cycle": cycles[-1] if cycles else None,
    }
