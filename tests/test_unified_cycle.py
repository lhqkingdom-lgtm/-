from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_lab.ingest import run_ingestion
from memory_lab.unified_cycle import CycleDependencies, run_unified_cycle


class _Planner:
    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.calls: list[list[dict]] = []

    def plan(self, turns, *, existing_memories=()):
        self.calls.append(list(turns))
        if self.failure:
            raise RuntimeError("api_key=must-not-reach-audit")
        return {
            "plan_id": "plan-1",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "memory": {
                        "stable_key": "workflow.demo.build",
                        "scope": {
                            "level": "project",
                            "project_key": turns[0]["workspace"]["project_key"],
                        },
                        "content": {"statement": "修改后执行验证。"},
                    },
                    "action_hint": {"kind": "create_or_match"},
                }
            ],
        }


class _Reviewer:
    def __init__(self, outcome: str = "accept") -> None:
        self.outcome = outcome
        self.calls = 0

    def review(self, writer_plan, turns, *, existing_memories=()):
        self.calls += 1
        return {
            "schema_version": "memory-lab-unified-reviewed-plan-v1",
            "outcome": self.outcome,
            "reason": "独立审核通过" if self.outcome != "reject" else "证据不足",
            "reviewed_plan": writer_plan if self.outcome != "reject" else None,
            "commit_allowed": self.outcome != "reject",
            "review_rounds": 1,
        }


class UnifiedCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.source = root / "source.db"
        self.lab = root / "lab.db"
        self.memory_root = root / "memory"
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE session (
                    id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, directory TEXT,
                    title TEXT, version TEXT, time_created INTEGER
                );
                CREATE TABLE message (
                    id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT
                );
                CREATE TABLE part (
                    id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
                    time_created INTEGER, data TEXT
                );
                INSERT INTO session VALUES
                    ('session-1', 'project-1', NULL, 'D:\\Real Project', 'test', '1', 1000);
                """
            )
        run_ingestion(self.source, self.lab, "baseline")
        with closing(sqlite3.connect(self.lab)) as connection, connection:
            connection.execute("UPDATE source SET baseline_cutoff_ms = 0")

    def add_turn(
        self,
        index: int,
        timestamp: int,
        *,
        tool: bool = True,
        tool_count: int = 1,
        tool_payload_chars: int = 2,
    ) -> None:
        user_id = f"user-{index}"
        assistant_id = f"assistant-{index}"
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO message VALUES (?, 'session-1', ?, ?)",
                (
                    user_id,
                    timestamp,
                    json.dumps(
                        {"role": "user", "time": {"created": timestamp}},
                        ensure_ascii=False,
                    ),
                ),
            )
            connection.execute(
                "INSERT INTO part VALUES (?, ?, 'session-1', ?, ?)",
                (
                    f"user-part-{index}",
                    user_id,
                    timestamp + 1,
                    json.dumps(
                        {"type": "text", "text": f"第 {index} 个长期要求"},
                        ensure_ascii=False,
                    ),
                ),
            )
            connection.execute(
                "INSERT INTO message VALUES (?, 'session-1', ?, ?)",
                (
                    assistant_id,
                    timestamp + 10,
                    json.dumps(
                        {
                            "role": "assistant",
                            "parentID": user_id,
                            "finish": "stop",
                            "time": {
                                "created": timestamp + 10,
                                "completed": timestamp + 20,
                            },
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            if tool:
                for tool_index in range(tool_count):
                    connection.execute(
                        "INSERT INTO part VALUES (?, ?, 'session-1', ?, ?)",
                        (
                            f"tool-part-{index}-{tool_index}",
                            assistant_id,
                            timestamp + 11 + tool_index,
                            json.dumps(
                                {
                                    "type": "tool",
                                    "tool": f"custom-tool-{index}-{tool_index}",
                                    "state": {
                                        "status": "completed",
                                        "output": "x" * tool_payload_chars,
                                    },
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
            connection.execute(
                "INSERT INTO part VALUES (?, ?, 'session-1', ?, ?)",
                (
                    f"assistant-part-{index}",
                    assistant_id,
                    timestamp + 12,
                    json.dumps(
                        {"type": "text", "text": f"第 {index} 个任务完成"},
                        ensure_ascii=False,
                    ),
                ),
            )

    def dependencies(
        self,
        *,
        fail_commit: bool = False,
        fail_report: bool = False,
        events: list | None = None,
    ):
        events = events if events is not None else []

        def commit(_database, _review):
            events.append("commit")
            if fail_commit:
                raise RuntimeError("password=commit-secret")
            return {
                "status": "complete",
                "actions": [
                    {
                        "action": "created",
                        "reason": "new durable memory",
                        "memory_id": "memory-1",
                        "before_revision_id": None,
                        "after_revision_id": "revision-1",
                    }
                ],
            }

        def topics(_database):
            events.append("topics")
            return {"status": "complete"}

        def publish(_database, _root):
            events.append("publish")
            return {"status": "published", "generation_id": "generation-1"}

        def build_report(_database, pipeline, markdown, stages):
            events.append("build-report")
            self.assertIn("review", pipeline)
            if markdown.get("status") == "published":
                self.assertEqual(markdown["generation_id"], "generation-1")
                self.assertTrue(markdown["matches_run_state"])
                self.assertTrue(markdown["generation_matches_current_database"])
            return {"pipeline": dict(pipeline), "stages": dict(stages)}

        def write_report(_directory, _report):
            events.append("write-report")
            if fail_report:
                raise RuntimeError("token=report-secret")
            return {"markdown": "latest-memory-change.md", "html": "latest-memory-change.html"}

        return CycleDependencies(
            commit_reviewed_plan=commit,
            ensure_topics=topics,
            publish=publish,
            build_report=build_report,
            write_report=write_report,
        )

    def test_success_marks_turns_only_after_commit_topics_and_publish_then_replay_noops(self):
        self.add_turn(1, 2000)
        self.add_turn(2, 3000)
        planner = _Planner()
        reviewer = _Reviewer()
        order: list[str] = []
        dependencies = self.dependencies(events=order)

        first = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=reviewer,
            settle_ms=0,
            report_directory=self.memory_root / "reports",
            dependencies=dependencies,
        )
        second = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=reviewer,
            settle_ms=0,
            dependencies=dependencies,
        )

        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["processed_turn_count"], 2)
        self.assertEqual(
            order, ["commit", "topics", "publish", "build-report", "write-report"]
        )
        self.assertEqual(second["status"], "noop")
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(reviewer.calls, 1)
        occurred = [bundle["turn"]["occurred_at_ms"] for bundle in planner.calls[0]]
        self.assertEqual(occurred, [2000, 3000])
        summary = first["source_summary"]
        self.assertEqual(summary["turns_processed"], 2)
        self.assertEqual(summary["user_messages"], 2)
        self.assertEqual(summary["assistant_finals"], 2)
        self.assertEqual(summary["tool_events"], 2)
        self.assertEqual((summary["candidates"], summary["accepted"], summary["rejected"]), (1, 1, 0))
        self.assertEqual(len(summary["projects"]), 1)
        self.assertTrue(summary["projects"][0]["project_key"].startswith("directory:"))
        self.assertNotIn("prepare", first["stages"])
        self.assertNotIn("path", json.dumps(summary, ensure_ascii=False).casefold())
        with closing(sqlite3.connect(self.lab)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM unified_processed_turn").fetchone()[0],
                2,
            )
        self.assertIn("html", first["report_files"])

    def test_writer_failure_is_redacted_and_same_turn_can_be_retried(self):
        self.add_turn(1, 2000)
        failing = _Planner(failure=True)
        reviewer = _Reviewer()
        dependencies = self.dependencies()

        failed = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=failing,
            reviewer=reviewer,
            settle_ms=0,
            dependencies=dependencies,
        )
        successful = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=_Planner(),
            reviewer=reviewer,
            settle_ms=0,
            dependencies=dependencies,
        )

        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["formal_state_unchanged"])
        self.assertEqual(failed["processed_turn_count"], 0)
        self.assertEqual(successful["processed_turn_count"], 1)
        with closing(sqlite3.connect(self.lab)) as connection:
            stored = connection.execute(
                "SELECT error_message FROM unified_cycle_run WHERE run_id = ?",
                (failed["run_id"],),
            ).fetchone()[0]
        self.assertNotIn("secret", stored)
        self.assertNotIn("api_key", json.dumps(failed))

    def test_valid_rejection_advances_but_commit_failure_does_not(self):
        self.add_turn(1, 2000)
        rejected = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=_Planner(),
            reviewer=_Reviewer("reject"),
            settle_ms=0,
            report_directory=self.memory_root / "reports",
            dependencies=self.dependencies(),
        )
        self.add_turn(2, 3000)
        commit_failed = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=_Planner(),
            reviewer=_Reviewer(),
            settle_ms=0,
            dependencies=self.dependencies(fail_commit=True),
        )

        self.assertEqual(rejected["status"], "complete_with_rejections")
        self.assertEqual(rejected["processed_turn_count"], 1)
        self.assertEqual(rejected["source_summary"]["rejected"], 1)
        self.assertEqual(commit_failed["status"], "failed")
        self.assertTrue(commit_failed["formal_state_unchanged"])
        with closing(sqlite3.connect(self.lab)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM unified_processed_turn").fetchone()[0],
                1,
            )

    def test_report_failure_happens_before_success_cursor(self):
        self.add_turn(1, 2000)
        result = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=_Planner(),
            reviewer=_Reviewer(),
            settle_ms=0,
            report_directory=self.memory_root / "reports",
            dependencies=self.dependencies(fail_report=True),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"]["report"]["status"], "failed")
        with closing(sqlite3.connect(self.lab)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM unified_processed_turn").fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM unified_cycle_checkpoint").fetchone()[0],
                1,
            )

        retry_planner = _Planner(failure=True)
        retry_reviewer = _Reviewer("reject")
        retry_order: list[str] = []
        recovered = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=retry_planner,
            reviewer=retry_reviewer,
            settle_ms=0,
            report_directory=self.memory_root / "reports",
            dependencies=self.dependencies(events=retry_order),
        )
        self.assertEqual(recovered["status"], "complete")
        self.assertEqual(recovered["resumed_from_checkpoint"], result["run_id"])
        self.assertEqual(retry_planner.calls, [])
        self.assertEqual(retry_reviewer.calls, 0)
        self.assertEqual(
            retry_order, ["topics", "publish", "build-report", "write-report"]
        )
        with closing(sqlite3.connect(self.lab)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM unified_processed_turn").fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM unified_cycle_checkpoint").fetchone()[0],
                0,
            )

    def test_source_order_and_batch_limit_leave_later_turn_for_next_cycle(self):
        self.add_turn(3, 4000, tool=False)
        self.add_turn(1, 2000, tool=False)
        self.add_turn(2, 3000, tool=False)
        planner = _Planner()
        reviewer = _Reviewer()
        dependencies = self.dependencies()

        first = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=reviewer,
            batch_size=2,
            settle_ms=0,
            dependencies=dependencies,
        )
        second = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=reviewer,
            batch_size=2,
            settle_ms=0,
            dependencies=dependencies,
        )

        self.assertEqual(first["processed_turn_count"], 2)
        self.assertEqual(second["processed_turn_count"], 1)
        first_times = [item["turn"]["occurred_at_ms"] for item in planner.calls[0]]
        second_times = [item["turn"]["occurred_at_ms"] for item in planner.calls[1]]
        self.assertEqual(first_times, [2000, 3000])
        self.assertEqual(second_times, [4000])

    def test_long_transcripts_use_longest_model_compatible_prefix_without_starvation(self):
        for index, timestamp in enumerate((2000, 3000, 4000), start=1):
            self.add_turn(
                index,
                timestamp,
                tool_count=40,
                tool_payload_chars=4_000,
            )
        planner = _Planner()
        reviewer = _Reviewer()
        dependencies = self.dependencies()

        first = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=reviewer,
            settle_ms=0,
            dependencies=dependencies,
        )
        second = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=reviewer,
            settle_ms=0,
            dependencies=dependencies,
        )

        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["selected_turn_count"], 1)
        self.assertEqual(first["processed_turn_count"], 1)
        self.assertEqual(first["stages"]["evidence"]["candidate_count"], 3)
        self.assertEqual(first["stages"]["evidence"]["deferred_count"], 2)
        self.assertEqual(
            [item["turn"]["occurred_at_ms"] for item in planner.calls[0]],
            [2000],
        )
        self.assertEqual(second["selected_turn_count"], 1)
        self.assertEqual(
            [item["turn"]["occurred_at_ms"] for item in planner.calls[1]],
            [3000],
        )
        with closing(sqlite3.connect(self.lab)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM unified_processed_turn"
                ).fetchone()[0],
                2,
            )

    def test_single_oversized_turn_advances_and_reports_all_compacted_tools(self):
        self.add_turn(
            1,
            2000,
            tool_count=650,
            tool_payload_chars=1_000,
        )
        planner = _Planner()
        result = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=_Reviewer(),
            settle_ms=0,
            dependencies=self.dependencies(),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["selected_turn_count"], 1)
        self.assertEqual(result["processed_turn_count"], 1)
        self.assertEqual(result["source_summary"]["tool_events"], 650)
        delivered = planner.calls[0][0]
        self.assertEqual(delivered["tool_event_manifest"]["total_count"], 650)
        self.assertLess(len(delivered["tools"]), 650)
        with closing(sqlite3.connect(self.lab)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM unified_processed_turn"
                ).fetchone()[0],
                1,
            )

    def test_invalid_batch_and_shared_agent_are_rejected(self):
        shared = _Planner()
        with self.assertRaisesRegex(ValueError, "between 1 and 32"):
            run_unified_cycle(
                self.source,
                self.lab,
                self.memory_root,
                planner=shared,
                reviewer=_Reviewer(),
                batch_size=33,
            )
        with self.assertRaisesRegex(ValueError, "independent"):
            run_unified_cycle(
                self.source,
                self.lab,
                self.memory_root,
                planner=shared,
                reviewer=shared,
            )

    def test_noop_cycle_still_writes_requested_report_without_model_calls(self):
        planner = _Planner(failure=True)
        reviewer = _Reviewer("reject")
        events: list[str] = []
        result = run_unified_cycle(
            self.source,
            self.lab,
            self.memory_root,
            planner=planner,
            reviewer=reviewer,
            settle_ms=0,
            report_directory=self.memory_root / "reports",
            dependencies=self.dependencies(events=events),
        )

        self.assertEqual(result["status"], "noop")
        self.assertEqual(events, ["build-report", "write-report"])
        self.assertEqual(planner.calls, [])
        self.assertEqual(reviewer.calls, 0)
        self.assertIn("html", result["report_files"])


if __name__ == "__main__":
    unittest.main()
