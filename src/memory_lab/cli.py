"""Command-line entry point for the Memory Lab."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

from .admission import run_rule_admission
from .catalog import run_catalog, run_recall
from .change_report import rebuild_auto_memory_change_report
from .cycle import run_memory_cycle, watch_memory_cycles
from .config import DEFAULT_CONFIG
from .consolidation import record_proposal_review, rollback_memory, run_consolidation
from .engineering_memory import summarize_engineering_candidates
from .episodes import run_episode_extraction
from .evidence_pool import DeepSeekPhaseRouter, run_evidence_pool_report
from .formal_markdown import publish_formal_memory, recall_formal_memory
from .formal_pipeline import run_deepseek_auto_memory
from .ingest import run_ingestion
from .instructions import run_instruction_coverage
from .longitudinal import run_status as longitudinal_status
from .longitudinal import start_run as start_longitudinal_run
from .llm_client import OpenAICompatibleJsonClient
from .markdown_store import run_markdown_preview, run_markdown_recall
from .model_recall import run_deepseek_formal_recall
from .phase import run_turn_phase_report
from .prepare import prepare_dataset
from .report import DEFAULT_GOLD, DEFAULT_REPORT_DIRECTORY, generate_shadow_report
from .shadow import run_deepseek_shadow
from .safety import require_distinct_source_and_lab, require_memory_lab_database
from .store import open_store
from .topics import (
    apply_topic_plan,
    ensure_topic_assignments,
    formal_semantic_hash,
    topic_catalog,
)
from .turns import run_turn_scan
from .unified_cycle import run_unified_cycle
from .unified_model import (
    DeepSeekUnifiedMemoryPlanner,
    DeepSeekUnifiedMemoryReviewer,
)


DEFAULT_SOURCE = Path.home() / ".local" / "share" / "codepulse" / "opencode-dev.db"
DEFAULT_STORE = Path(__file__).resolve().parents[2] / "data" / "memory_lab.db"
DEFAULT_PREVIEW_MEMORY = Path(__file__).resolve().parents[2] / "preview-memory"
DEFAULT_FORMAL_MEMORY = Path(__file__).resolve().parents[2] / "memory"
DEFAULT_REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _conditions(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError("conditions must use key=value")
        key, value = raw.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not value or key in result:
            raise ValueError("conditions require unique non-empty key=value pairs")
        result[key] = value
    return result


def _project_key(lab_database: Path, directory: Path | None, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if directory is None:
        return None
    resolved = str(directory.resolve()).casefold()
    with closing(open_store(lab_database)) as connection:
        matches = [
            row[0]
            for row in connection.execute("SELECT project_key, directory FROM project_context")
            if str(Path(row[1]).resolve()).casefold() == resolved
        ]
    if len(matches) > 1:
        raise ValueError("project directory maps to multiple project keys")
    return str(matches[0]) if matches else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory-lab")
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--lab-db", type=Path, default=DEFAULT_STORE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "baseline",
        help="Mark all current parts as pre-experiment history without creating events",
    )
    subparsers.add_parser(
        "ingest",
        help="Import newly observed user text as normalized events",
    )
    subparsers.add_parser(
        "prepare",
        help="Bind events to session/project contexts and separate test case markers",
    )
    subparsers.add_parser(
        "coverage",
        help="Discover applicable CodePulse instruction rules and assess event coverage",
    )
    subparsers.add_parser(
        "admit",
        help="Run the deterministic atomic long-term-value admission baseline",
    )
    subparsers.add_parser(
        "consolidate",
        help="Consolidate approved latest proposals into versioned formal memories",
    )
    shadow = subparsers.add_parser(
        "shadow",
        help="Run DeepSeek judgment in an isolated, non-authoritative lane",
    )
    shadow.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    shadow.add_argument(
        "--refresh",
        action="store_true",
        help="Call the model again instead of replaying an exact prior result",
    )
    report = subparsers.add_parser(
        "report",
        help="Evaluate a fresh shadow run against frozen gold",
    )
    report.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    report.add_argument("--shadow-run-id")
    catalog = subparsers.add_parser(
        "catalog",
        help="Build a read-only deduplicated memory preview from a fresh shadow run",
    )
    catalog.add_argument("--shadow-run-id")
    recall = subparsers.add_parser(
        "recall",
        help="Recall applicable preview memories for one task sentence",
    )
    recall.add_argument("--query", required=True)
    recall.add_argument("--project-directory", type=Path)
    recall.add_argument("--domain")
    recall.add_argument("--limit", type=int, default=5)
    recall.add_argument("--shadow-run-id")
    markdown_preview = subparsers.add_parser(
        "markdown-preview",
        help="Publish a validated Claude-style Markdown preview from a fresh shadow run",
    )
    markdown_preview.add_argument(
        "--memory-dir", type=Path, default=DEFAULT_PREVIEW_MEMORY
    )
    markdown_preview.add_argument("--shadow-run-id")
    markdown_preview.add_argument("--dry-run", action="store_true")
    markdown_recall = subparsers.add_parser(
        "markdown-recall",
        help="Recall from a validated Markdown preview without reading SQLite",
    )
    markdown_recall.add_argument(
        "--memory-dir", type=Path, default=DEFAULT_PREVIEW_MEMORY
    )
    markdown_recall.add_argument("--query", required=True)
    markdown_recall.add_argument("--project-directory", type=Path)
    markdown_recall.add_argument("--domain")
    markdown_recall.add_argument("--limit", type=int, default=5)
    auto = subparsers.add_parser(
        "auto-memory",
        help="Run independent DeepSeek Writer/Reviewer and atomically commit formal memory",
    )
    auto.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    auto.add_argument("--refresh", action="store_true")
    formal_publish = subparsers.add_parser(
        "formal-publish",
        help="Publish formal SQLite memory as an immutable Markdown generation",
    )
    formal_publish.add_argument("--memory-root", type=Path, default=DEFAULT_FORMAL_MEMORY)
    change_report = subparsers.add_parser(
        "change-report",
        help="Rebuild Markdown/HTML/JSON change reports for an existing auto-memory run",
    )
    change_report.add_argument("--run-id", required=True)
    change_report.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)
    change_report.add_argument("--memory-root", type=Path, default=DEFAULT_FORMAL_MEMORY)
    formal_recall = subparsers.add_parser(
        "formal-recall",
        help="Recall from formal Markdown using deterministic or DeepSeek topic routing",
    )
    formal_recall.add_argument("--memory-root", type=Path, default=DEFAULT_FORMAL_MEMORY)
    formal_recall.add_argument("--query", required=True)
    formal_recall.add_argument("--project-key")
    formal_recall.add_argument("--project-directory", type=Path)
    formal_recall.add_argument("--domain")
    formal_recall.add_argument("--condition", action="append")
    formal_recall.add_argument("--topic-limit", type=int, default=5)
    formal_recall.add_argument("--limit", type=int, default=5)
    formal_recall.add_argument("--model-router", action="store_true")
    formal_recall.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cycle = subparsers.add_parser(
        "cycle", help="Run the complete standalone extraction-to-report cycle once"
    )
    cycle.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cycle.add_argument("--memory-root", type=Path, default=DEFAULT_FORMAL_MEMORY)
    cycle.add_argument("--report-directory", type=Path, default=DEFAULT_REPORTS)
    cycle.add_argument("--settle-ms", type=int, default=1000)
    cycle.add_argument("--refresh", action="store_true")
    session_memory = subparsers.add_parser(
        "session-memory",
        help="Distill completed CP session turns through the generic reviewed memory path",
    )
    session_memory.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    session_memory.add_argument(
        "--memory-root", type=Path, default=DEFAULT_FORMAL_MEMORY
    )
    session_memory.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORTS
    )
    session_memory.add_argument("--settle-ms", type=int, default=1000)
    session_memory.add_argument("--batch-size", type=int, default=32)
    watch = subparsers.add_parser(
        "watch", help="Poll the read-only CP database and run a cycle when it changes"
    )
    watch.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    watch.add_argument("--memory-root", type=Path, default=DEFAULT_FORMAL_MEMORY)
    watch.add_argument("--report-directory", type=Path, default=DEFAULT_REPORTS)
    watch.add_argument("--settle-ms", type=int, default=1000)
    watch.add_argument("--interval-seconds", type=float, default=15.0)
    watch.add_argument("--max-cycles", type=int)
    watch.add_argument("--refresh", action="store_true")
    subparsers.add_parser(
        "engineering-candidates",
        help="Summarize privacy-bounded engineering evidence candidates",
    )
    topics = subparsers.add_parser(
        "topics", help="Ensure and list the unique topic owner of each stable key"
    )
    topics.add_argument("--apply-plan", type=Path)
    rollback = subparsers.add_parser(
        "rollback-memory", help="Append a rollback revision for one formal memory"
    )
    rollback.add_argument("--memory-id", required=True)
    rollback.add_argument("--target-revision-id", required=True)
    rollback.add_argument("--reason", required=True)
    rollback.add_argument("--requested-by", default="user")
    soak_start = subparsers.add_parser(
        "soak-start", help="Start the visible 14-28 day longitudinal evaluation ledger"
    )
    soak_start.add_argument("--name", required=True)
    soak_status = subparsers.add_parser(
        "soak-status", help="Show sample progress for the current longitudinal run"
    )
    soak_status.add_argument("--run-id")
    review = subparsers.add_parser(
        "review",
        help="Approve or reject one exact immutable memory proposal",
    )
    review.add_argument("--proposal-id", required=True)
    review.add_argument("--outcome", choices=("approve", "reject"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--note")
    turns = subparsers.add_parser(
        "turns",
        help="Detect stable completed/failed/interrupted turns from CodePulse SQLite",
    )
    turns.add_argument(
        "--settle-ms",
        type=int,
        default=1000,
        help="Delay between the two structural samples (default: 1000)",
    )
    episodes = subparsers.add_parser(
        "episodes",
        help="Extract immutable privacy-bounded traces from completed/failed turns",
    )
    episodes.add_argument(
        "--settle-ms",
        type=int,
        default=1000,
        help="Delay between the two allow-listed trace samples (default: 1000)",
    )
    phases = subparsers.add_parser(
        "turn-phases",
        help="Classify stable CP turns into memory extraction phases",
    )
    phases.add_argument("--settle-ms", type=int, default=0)
    phases.add_argument("--limit", type=int, default=20)
    evidence_pools = subparsers.add_parser(
        "evidence-pools",
        help="Build three-tier evidence pools and optionally route final phases with DeepSeek",
    )
    evidence_pools.add_argument("--settle-ms", type=int, default=0)
    evidence_pools.add_argument("--limit", type=int, default=10)
    evidence_pools.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evidence_pools.add_argument(
        "--layer",
        choices=("all", "core", "candidate", "pointer", "router"),
        default="all",
        help="Show one evidence layer at a time for step-by-step validation",
    )
    evidence_pools.add_argument(
        "--model-router",
        action="store_true",
        help="Call DeepSeek to correct the deterministic rule phase",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.command not in {"markdown-recall", "formal-recall"}:
            require_distinct_source_and_lab(args.source_db, args.lab_db)
            if args.lab_db.exists() or args.command != "baseline":
                require_memory_lab_database(args.lab_db)
        if args.command == "prepare":
            report = prepare_dataset(args.lab_db)
        elif args.command == "coverage":
            report = run_instruction_coverage(args.lab_db)
        elif args.command == "admit":
            report = run_rule_admission(args.lab_db)
        elif args.command == "consolidate":
            report = run_consolidation(args.lab_db)
        elif args.command == "shadow":
            report = run_deepseek_shadow(
                args.lab_db, config_path=args.config, refresh=args.refresh
            )
        elif args.command == "report":
            report = generate_shadow_report(
                args.lab_db,
                gold_path=args.gold,
                output_directory=args.output_dir,
                shadow_run_id=args.shadow_run_id,
            )
        elif args.command == "catalog":
            report = run_catalog(args.lab_db, shadow_run_id=args.shadow_run_id)
        elif args.command == "recall":
            report = run_recall(
                args.lab_db,
                args.query,
                project_directory=args.project_directory,
                domain=args.domain,
                limit=args.limit,
                shadow_run_id=args.shadow_run_id,
            )
        elif args.command == "markdown-preview":
            report = run_markdown_preview(
                args.lab_db,
                args.memory_dir,
                shadow_run_id=args.shadow_run_id,
                dry_run=args.dry_run,
            )
        elif args.command == "markdown-recall":
            report = run_markdown_recall(
                args.memory_dir,
                args.query,
                project_directory=args.project_directory,
                domain=args.domain,
                limit=args.limit,
            )
        elif args.command == "auto-memory":
            report = run_deepseek_auto_memory(
                args.lab_db, config_path=args.config, refresh=args.refresh
            )
        elif args.command == "formal-publish":
            with closing(open_store(args.lab_db)) as connection, connection:
                topic_changes = ensure_topic_assignments(connection)
            report = {
                **publish_formal_memory(args.lab_db, args.memory_root),
                "topic_assignment": topic_changes,
            }
        elif args.command == "change-report":
            report = rebuild_auto_memory_change_report(
                args.lab_db,
                args.run_id,
                args.output_dir,
                memory_root=args.memory_root,
            )
        elif args.command == "formal-recall":
            project_key = _project_key(
                args.lab_db, args.project_directory, args.project_key
            )
            conditions = _conditions(args.condition)
            kwargs = {
                "project_key": project_key,
                "domain": args.domain,
                "conditions": conditions or None,
                "topic_limit": args.topic_limit,
                "memory_limit": args.limit,
            }
            if args.model_router:
                report = run_deepseek_formal_recall(
                    args.memory_root,
                    args.query,
                    config_path=args.config,
                    **kwargs,
                )
            else:
                report = recall_formal_memory(args.memory_root, args.query, **kwargs)
        elif args.command == "cycle":
            report = run_memory_cycle(
                args.source_db,
                args.lab_db,
                config_path=args.config,
                memory_root=args.memory_root,
                report_directory=args.report_directory,
                settle_ms=args.settle_ms,
                refresh=args.refresh,
            )
        elif args.command == "session-memory":
            planner_client = OpenAICompatibleJsonClient.from_config(args.config)
            reviewer_client = OpenAICompatibleJsonClient.from_config(args.config)
            report = run_unified_cycle(
                args.source_db,
                args.lab_db,
                args.memory_root,
                planner=DeepSeekUnifiedMemoryPlanner(planner_client),
                reviewer=DeepSeekUnifiedMemoryReviewer(reviewer_client),
                batch_size=args.batch_size,
                settle_ms=args.settle_ms,
                report_directory=args.report_directory,
            )
        elif args.command == "watch":
            def emit_cycle(item: dict) -> None:
                print(json.dumps({
                    "event": "memory_cycle",
                    "status": item.get("status"),
                    "run_id": (item.get("formal") or {}).get("run_id"),
                    "report_files": item.get("report_files", {}),
                }, ensure_ascii=False))

            report = watch_memory_cycles(
                args.source_db,
                args.lab_db,
                config_path=args.config,
                memory_root=args.memory_root,
                report_directory=args.report_directory,
                interval_seconds=args.interval_seconds,
                settle_ms=args.settle_ms,
                refresh=args.refresh,
                max_cycles=args.max_cycles,
                on_cycle=emit_cycle,
            )
        elif args.command == "engineering-candidates":
            report = summarize_engineering_candidates(args.lab_db)
        elif args.command == "topics":
            with closing(open_store(args.lab_db)) as connection, connection:
                assignment = ensure_topic_assignments(connection)
                applied = None
                if args.apply_plan:
                    plan = json.loads(args.apply_plan.read_text(encoding="utf-8"))
                    applied = apply_topic_plan(connection, plan)
                report = {
                    "status": "complete",
                    "assignment": assignment,
                    "applied": applied,
                    "formal_semantic_hash": formal_semantic_hash(connection),
                    "topics": topic_catalog(connection),
                }
        elif args.command == "rollback-memory":
            report = rollback_memory(
                args.lab_db,
                args.memory_id,
                args.target_revision_id,
                reason=args.reason,
                requested_by=args.requested_by,
            )
        elif args.command == "soak-start":
            with closing(open_store(args.lab_db)) as connection, connection:
                report = start_longitudinal_run(connection, args.name, visibility="open")
        elif args.command == "soak-status":
            with closing(open_store(args.lab_db)) as connection:
                report = longitudinal_status(connection, args.run_id)
        elif args.command == "review":
            report = record_proposal_review(
                args.lab_db,
                args.proposal_id,
                args.outcome,
                args.reviewer,
                args.note,
            )
        elif args.command == "turns":
            report = run_turn_scan(args.source_db, args.lab_db, args.settle_ms)
        elif args.command == "episodes":
            report = run_episode_extraction(
                args.source_db, args.lab_db, settle_ms=args.settle_ms
            )
        elif args.command == "turn-phases":
            report = run_turn_phase_report(
                args.source_db,
                args.lab_db,
                settle_ms=args.settle_ms,
                limit=args.limit,
            )
        elif args.command == "evidence-pools":
            router = None
            if args.model_router:
                router = DeepSeekPhaseRouter(
                    OpenAICompatibleJsonClient.from_config(args.config)
                )
            report = run_evidence_pool_report(
                args.source_db,
                args.lab_db,
                settle_ms=args.settle_ms,
                limit=args.limit,
                layer=args.layer,
                router=router,
            )
        elif args.command in {"baseline", "ingest"}:
            report = run_ingestion(args.source_db, args.lab_db, args.command)
        else:
            raise ValueError(f"unsupported command: {args.command}")
    except Exception as error:
        error_text = (
            type(error).__name__ if args.command == "episodes" else str(error)
        )
        print(json.dumps({"ok": False, "error": error_text}, ensure_ascii=False, indent=2))
        return 1
    succeeded = report.get("status") not in {"failed", "complete_with_errors"}
    print(json.dumps({"ok": succeeded, **report}, ensure_ascii=False, indent=2))
    return 0 if succeeded else 1
