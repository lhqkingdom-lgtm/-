"""Frozen-gold evaluation and human-readable reports for LLM shadow runs."""

from __future__ import annotations

import hashlib
import html
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .consolidation import latest_admission_snapshot
from .shadow import formal_state_snapshot


GOLD_SCHEMA_VERSION = "memory-lab-shadow-gold-v1"
REPORT_SCHEMA_VERSION = "memory-lab-shadow-report-v1"
DEFAULT_GOLD = Path(__file__).resolve().parents[2] / "gold" / "bootstrap_v1.json"
DEFAULT_REPORT_DIRECTORY = Path(__file__).resolve().parents[2] / "reports"
FINAL_RUN_STATUSES = {"complete", "complete_with_errors", "failed"}
VALID_JUDGMENT_STATUSES = {"valid", "replayed"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_only(database: Path) -> sqlite3.Connection:
    if not database.is_file():
        raise FileNotFoundError(f"lab database does not exist: {database}")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _load_gold(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"gold file does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"gold file is not valid JSON: {path}") from error
    if not isinstance(document, dict) or document.get("schema_version") != GOLD_SCHEMA_VERSION:
        raise ValueError(f"gold file must use {GOLD_SCHEMA_VERSION}")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("gold file must contain a non-empty cases array")
    selectors: set[tuple[str, str]] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"gold case {index} must be an object")
        required = {"label", "should_store"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"gold case {index} missing fields: {sorted(missing)}")
        available_selectors = [
            field for field in ("candidate_key", "exact_quote") if field in case
        ]
        if len(available_selectors) != 1:
            raise ValueError(
                f"gold case {index} must have exactly one of candidate_key or exact_quote"
            )
        selector_field = available_selectors[0]
        selector_value = case[selector_field]
        if not isinstance(selector_value, str) or not selector_value:
            raise ValueError(f"gold case {index} has invalid {selector_field}")
        selector = (selector_field, selector_value)
        if selector in selectors:
            raise ValueError(
                f"gold contains duplicate {selector_field}: {selector_value}"
            )
        selectors.add(selector)
        expected_status = case.get("expected_status", "valid")
        if expected_status not in {"valid", "blocked"}:
            raise ValueError(
                f"gold case {selector_value} expected_status must be valid or blocked"
            )
        if not isinstance(case["should_store"], bool):
            raise ValueError(
                f"gold case {selector_value} should_store must be boolean"
            )
        if expected_status == "blocked" and case["should_store"]:
            raise ValueError(
                f"blocked gold case {selector_value} cannot expect storage"
            )
        if case["should_store"]:
            positive_fields = {"type", "stable_key", "scope", "value", "relation"}
            missing_positive = positive_fields - case.keys()
            if missing_positive:
                raise ValueError(
                    f"positive gold case {selector_value} missing fields: "
                    f"{sorted(missing_positive)}"
                )
            if not isinstance(case["scope"], dict):
                raise ValueError(
                    f"gold case {selector_value} scope must be an object"
                )
    return document


def _latest_run(connection: sqlite3.Connection, shadow_run_id: str | None) -> sqlite3.Row:
    if shadow_run_id:
        row = connection.execute(
            "SELECT rowid AS storage_rowid, * FROM llm_shadow_run WHERE shadow_run_id = ?",
            (shadow_run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown shadow run: {shadow_run_id}")
    else:
        row = connection.execute(
            """
            SELECT rowid AS storage_rowid, * FROM llm_shadow_run
            WHERE replay_count = 0
            ORDER BY started_at DESC, rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise ValueError("no fresh LLM shadow run exists; run shadow -Refresh first")
    if row["status"] not in FINAL_RUN_STATUSES:
        raise ValueError(f"shadow run is not finalized: {row['status']}")
    return row


def _content_snapshot_is_complete(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "memory",
        "memory_revision",
        "evidence",
        "conflict",
    }:
        return False
    for table in value.values():
        if not isinstance(table, Mapping):
            return False
        if not isinstance(table.get("row_count"), int):
            return False
        digest = table.get("content_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            return False
    return True


def _safe_json(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


def _judgment_rows(
    connection: sqlite3.Connection, shadow_run_id: str
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT j.*, d.exact_quote, d.start_offset AS source_start_offset,
               d.end_offset AS source_end_offset, d.disposition AS rule_disposition,
               d.event_id AS source_event_id, e.part_id AS source_part_id
        FROM llm_judgment AS j
        JOIN admission_decision AS d ON d.decision_id = j.decision_id
        JOIN event AS e ON e.event_id = d.event_id
        WHERE j.shadow_run_id = ?
        ORDER BY d.decided_at, d.ordinal, j.rowid
        """,
        (shadow_run_id,),
    ).fetchall()


def _evidence_matches(row: sqlite3.Row, result: Mapping[str, Any] | None) -> bool:
    if not result:
        return False
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    item = evidence[0]
    if not isinstance(item, Mapping):
        return False
    return (
        item.get("event_id") == row["source_event_id"]
        and item.get("part_id") == row["source_part_id"]
        and item.get("quote") == row["exact_quote"]
        and item.get("start_offset") == row["source_start_offset"]
        and item.get("end_offset") == row["source_end_offset"]
    )


def _compare_case(
    case: Mapping[str, Any],
    row: sqlite3.Row | None,
    selection_error: str | None = None,
) -> dict[str, Any]:
    expected_store = bool(case["should_store"])
    expected_status = case.get("expected_status", "valid")
    if row is None:
        selector = case.get("candidate_key") or f"quote:{case.get('exact_quote', '')}"
        return {
            "candidate_key": selector,
            "label": case["label"],
            "status": "missing",
            "quote": None,
            "rule_disposition": None,
            "expected_should_store": expected_store,
            "expected_status": expected_status,
            "predicted_should_store": None,
            "status_acceptable": False,
            "classification_correct": False,
            "semantic_correct": False,
            "evidence_exact": False,
            "correct": False,
            "mismatches": [selection_error or "missing judgment"],
            "reason": None,
            "execution_mode": None,
        }

    if expected_status == "blocked":
        blocked_correct = (
            row["status"] == "blocked"
            and row["execution_mode"] == "local_block"
            and row["result_json"] is None
            and row["should_store"] is None
        )
        return {
            "candidate_key": row["candidate_key"],
            "label": case["label"],
            "status": row["status"],
            "expected_status": "blocked",
            "execution_mode": row["execution_mode"],
            "quote": row["exact_quote"],
            "rule_disposition": row["rule_disposition"],
            "expected_should_store": False,
            "predicted_should_store": None,
            "status_acceptable": blocked_correct,
            "classification_correct": blocked_correct,
            "semantic_correct": blocked_correct,
            "semantic_fields": {},
            "evidence_exact": None,
            "correct": blocked_correct,
            "mismatches": []
            if blocked_correct
            else ["expected a local_block judgment with no model result"],
            "reason": row["error_message"],
            "prediction": None,
        }

    result = _safe_json(row["result_json"])
    structurally_valid = row["status"] in VALID_JUDGMENT_STATUSES and isinstance(result, dict)
    predicted_store = result.get("should_store") if structurally_valid else None
    classification_correct = predicted_store is expected_store
    mismatches: list[str] = []
    if not structurally_valid:
        mismatches.append(f"judgment status is {row['status']}")
    if not classification_correct:
        mismatches.append(
            f"should_store expected {expected_store}, got {predicted_store}"
        )

    semantic_correct = classification_correct and structurally_valid
    field_results: dict[str, bool] = {}
    if expected_store and structurally_valid and predicted_store is True:
        actual_scope = result.get("scope")
        value = result.get("value")
        relation = result.get("relation")
        actual = {
            "type": result.get("type"),
            "stable_key": result.get("key"),
            "scope": actual_scope,
            "value": value.get("normalized") if isinstance(value, Mapping) else None,
            "relation": relation.get("kind") if isinstance(relation, Mapping) else None,
        }
        for field in ("type", "stable_key", "scope", "value", "relation"):
            field_results[field] = actual[field] == case[field]
            if not field_results[field]:
                mismatches.append(
                    f"{field} expected {_canonical_json(case[field])}, "
                    f"got {_canonical_json(actual[field])}"
                )
        semantic_correct = all(field_results.values())
    elif expected_store:
        semantic_correct = False

    evidence_exact = structurally_valid and _evidence_matches(row, result)
    if not evidence_exact:
        mismatches.append("evidence does not exactly match the frozen source span")
    correct = structurally_valid and classification_correct and semantic_correct and evidence_exact
    return {
        "candidate_key": row["candidate_key"],
        "label": case["label"],
        "status": row["status"],
        "expected_status": expected_status,
        "execution_mode": row["execution_mode"],
        "quote": row["exact_quote"],
        "rule_disposition": row["rule_disposition"],
        "expected_should_store": expected_store,
        "predicted_should_store": predicted_store,
        "status_acceptable": structurally_valid,
        "classification_correct": classification_correct,
        "semantic_correct": semantic_correct,
        "semantic_fields": field_results,
        "evidence_exact": evidence_exact,
        "correct": correct,
        "mismatches": mismatches,
        "reason": result.get("reason") if isinstance(result, Mapping) else None,
        "prediction": {
            "type": result.get("type"),
            "stable_key": result.get("key"),
            "scope": result.get("scope"),
            "value": (
                result.get("value", {}).get("normalized")
                if isinstance(result.get("value"), Mapping)
                else None
            ),
            "relation": (
                result.get("relation", {}).get("kind")
                if isinstance(result.get("relation"), Mapping)
                else None
            ),
        }
        if isinstance(result, Mapping)
        else None,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _evaluate_rows(
    run: sqlite3.Row,
    rows: list[sqlite3.Row],
    gold: Mapping[str, Any],
    current_formal_state: Mapping[str, Any],
    current_admission_run_id: str | None,
    current_admission_snapshot_hash: str | None,
) -> dict[str, Any]:
    by_candidate: dict[str, list[tuple[int, sqlite3.Row]]] = {}
    duplicate_keys: list[str] = []
    for index, row in enumerate(rows):
        key = row["candidate_key"]
        if key in by_candidate:
            duplicate_keys.append(key)
        by_candidate.setdefault(key, []).append((index, row))
    consumed: set[int] = set()
    cases: list[dict[str, Any]] = []
    for case in gold["cases"]:
        if "candidate_key" in case:
            matches = [item for item in by_candidate.get(case["candidate_key"], []) if item[0] not in consumed]
            selector_name = f"candidate_key {case['candidate_key']}"
        else:
            matches = [
                (index, row)
                for index, row in enumerate(rows)
                if index not in consumed and row["exact_quote"] == case["exact_quote"]
            ]
            selector_name = f"exact_quote {case['exact_quote']}"
        if len(matches) == 1:
            match_index, row = matches[0]
            consumed.add(match_index)
            cases.append(_compare_case(case, row))
        elif not matches:
            cases.append(_compare_case(case, None, f"no judgment matched {selector_name}"))
        else:
            cases.append(
                _compare_case(
                    case,
                    None,
                    f"multiple judgments matched {selector_name}; selector is ambiguous",
                )
            )
    unexpected = sorted(rows[index]["candidate_key"] for index in range(len(rows)) if index not in consumed)

    model_cases = [case for case in cases if case["expected_status"] != "blocked"]
    blocked_cases = [case for case in cases if case["expected_status"] == "blocked"]
    tp = sum(
        1
        for case in model_cases
        if case["expected_should_store"] and case["predicted_should_store"] is True
    )
    fn = sum(
        1
        for case in model_cases
        if case["expected_should_store"] and case["predicted_should_store"] is not True
    )
    tn = sum(
        1
        for case in model_cases
        if not case["expected_should_store"] and case["predicted_should_store"] is False
    )
    fp = sum(
        1
        for case in model_cases
        if not case["expected_should_store"] and case["predicted_should_store"] is not False
    )
    valid_count = sum(1 for case in cases if case["status_acceptable"])
    correct_count = sum(1 for case in cases if case["correct"])
    semantic_expected = sum(
        len(case.get("semantic_fields", {}))
        for case in cases
        if case["expected_should_store"]
    )
    semantic_correct = sum(
        sum(case.get("semantic_fields", {}).values())
        for case in cases
        if case["expected_should_store"]
    )
    evidence_exact = sum(1 for case in model_cases if case["evidence_exact"])
    blocked_correct = sum(1 for case in blocked_cases if case["correct"])
    judge_invocations = sum(
        1 for row in rows if row["execution_mode"] == "judge_call"
    )
    cache_replays = sum(
        1 for row in rows if row["execution_mode"] == "cache_replay"
    )
    local_blocks = sum(
        1 for row in rows if row["execution_mode"] == "local_block"
    )
    before = _safe_json(run["formal_state_before_json"])
    after = _safe_json(run["formal_state_after_json"])
    formal_state_unchanged = isinstance(before, dict) and before == after
    formal_state_content_verified = (
        formal_state_unchanged
        and _content_snapshot_is_complete(before)
        and _content_snapshot_is_complete(after)
    )
    fresh_model_outputs = (
        run["replay_count"] == 0
        and len(rows) == len(gold["cases"])
        and all(
            (case["status"] == "blocked" and case["execution_mode"] == "local_block")
            if case["expected_status"] == "blocked"
            else (case["status"] == "valid" and case["execution_mode"] == "judge_call")
            for case in cases
        )
    )
    admission_snapshot_current = (
        isinstance(run["admission_snapshot_hash"], str)
        and run["admission_snapshot_hash"].startswith("snapshot_")
        and run["admission_run_id"] == current_admission_run_id
        and run["admission_snapshot_hash"] == current_admission_snapshot_hash
    )
    complete_set = (
        len(rows) == len(gold["cases"])
        and run["candidate_count"] == len(gold["cases"])
        and not unexpected
        and not duplicate_keys
    )
    passed = (
        run["status"] == "complete"
        and complete_set
        and valid_count == len(gold["cases"])
        and correct_count == len(gold["cases"])
        and fresh_model_outputs
        and formal_state_content_verified
        and admission_snapshot_current
    )
    return {
        "run": {
            "shadow_run_id": run["shadow_run_id"],
            "admission_run_id": run["admission_run_id"],
            "admission_snapshot_hash": run["admission_snapshot_hash"],
            "provider": run["provider"],
            "model": run["model"],
            "prompt_version": run["prompt_version"],
            "status": run["status"],
            "started_at": run["started_at"],
            "completed_at": run["completed_at"],
            "candidate_count": run["candidate_count"],
            "replay_count": run["replay_count"],
        },
        "gate": {
            "passed": passed,
            "complete_candidate_set": complete_set,
            "all_outputs_structurally_valid": valid_count == len(gold["cases"]),
            "all_gold_cases_correct": correct_count == len(gold["cases"]),
            "formal_state_unchanged": formal_state_unchanged,
            "formal_state_content_verified": formal_state_content_verified,
            "fresh_model_outputs": fresh_model_outputs,
            "admission_snapshot_current": admission_snapshot_current,
        },
        "metrics": {
            "gold_cases": len(gold["cases"]),
            "judgments": len(rows),
            "structurally_valid": valid_count,
            "correct_cases": correct_count,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "specificity": _ratio(tn, tn + fp),
            "accuracy": _ratio(tp + tn, len(model_cases)),
            "semantic_fields_correct": semantic_correct,
            "semantic_fields_expected": semantic_expected,
            "exact_evidence_cases": evidence_exact,
            "exact_evidence_cases_expected": len(model_cases),
            "blocked_cases": len(blocked_cases),
            "blocked_cases_correct": blocked_correct,
            "judge_invocations": judge_invocations,
            "cache_replays": cache_replays,
            "local_blocks": local_blocks,
        },
        "formal_state": {
            "before": before,
            "after": after,
            "current": dict(current_formal_state),
        },
        "unexpected_candidates": unexpected,
        "duplicate_candidates": sorted(set(duplicate_keys)),
        "cases": cases,
    }


def _history(
    connection: sqlite3.Connection, gold: Mapping[str, Any]
) -> list[dict[str, Any]]:
    runs = connection.execute(
        """
        SELECT rowid AS storage_rowid, * FROM llm_shadow_run
        WHERE completed_at IS NOT NULL AND replay_count = 0
        ORDER BY completed_at DESC, rowid DESC
        """
    ).fetchall()
    latest_by_version: dict[tuple[str, str, str], sqlite3.Row] = {}
    for run in runs:
        key = (run["provider"], run["model"], run["prompt_version"])
        latest_by_version.setdefault(key, run)
    current = formal_state_snapshot(connection)
    summaries: list[dict[str, Any]] = []
    for run in sorted(latest_by_version.values(), key=lambda item: item["started_at"]):
        evaluated = _evaluate_rows(
            run,
            _judgment_rows(connection, run["shadow_run_id"]),
            gold,
            current,
            run["admission_run_id"],
            run["admission_snapshot_hash"],
        )
        summaries.append(
            {
                "shadow_run_id": run["shadow_run_id"],
                "prompt_version": run["prompt_version"],
                "status": run["status"],
                "valid": evaluated["metrics"]["structurally_valid"],
                "correct": evaluated["metrics"]["correct_cases"],
                "true_positive": evaluated["metrics"]["true_positive"],
                "true_negative": evaluated["metrics"]["true_negative"],
                "false_positive": evaluated["metrics"]["false_positive"],
                "false_negative": evaluated["metrics"]["false_negative"],
                "passed": evaluated["gate"]["passed"],
            }
        )
    return summaries


def evaluate_shadow_run(
    lab_database: Path,
    *,
    gold_path: Path = DEFAULT_GOLD,
    shadow_run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one finalized shadow run without mutating the lab database."""

    gold = _load_gold(gold_path)
    gold_bytes = gold_path.read_bytes()
    gold_sha256 = hashlib.sha256(gold_bytes).hexdigest()
    gold_modified_at = datetime.fromtimestamp(
        gold_path.stat().st_mtime, timezone.utc
    ).isoformat(timespec="milliseconds")
    with closing(_read_only(lab_database)) as connection:
        run = _latest_run(connection, shadow_run_id)
        current_admission, _current_rows, current_snapshot_hash = latest_admission_snapshot(
            connection
        )
        latest_admission_attempt = connection.execute(
            """
            SELECT run_id, status, completed_at FROM admission_run
            ORDER BY started_at DESC, rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if (
            latest_admission_attempt is not None
            and (
                current_admission is None
                or latest_admission_attempt["run_id"] != current_admission["run_id"]
            )
        ):
            current_admission = None
            current_snapshot_hash = None
        evaluation = _evaluate_rows(
            run,
            _judgment_rows(connection, run["shadow_run_id"]),
            gold,
            formal_state_snapshot(connection),
            current_admission["run_id"] if current_admission is not None else None,
            current_snapshot_hash,
        )
        history = _history(connection, gold)
    provenance = gold.get("provenance", "tuned_regression")
    frozen_at = gold.get("frozen_at")
    gold_frozen_before_run = True
    if provenance == "holdout":
        try:
            frozen_time = datetime.fromisoformat(str(frozen_at).replace("Z", "+00:00"))
            modified_time = datetime.fromisoformat(gold_modified_at)
            run_time = datetime.fromisoformat(
                str(evaluation["run"]["started_at"]).replace("Z", "+00:00")
            )
            gold_frozen_before_run = frozen_time <= run_time and modified_time <= run_time
        except (TypeError, ValueError):
            gold_frozen_before_run = False
    evaluation["gate"]["gold_frozen_before_run"] = gold_frozen_before_run
    evaluation["gate"]["passed"] = (
        evaluation["gate"]["passed"] and gold_frozen_before_run
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "suite": {
            "suite_id": gold.get("suite_id"),
            "provenance": provenance,
            "frozen_at": frozen_at,
            "gold_sha256": gold_sha256,
            "gold_modified_at": gold_modified_at,
            "description": gold.get("description"),
            "generalization_note": gold.get("generalization_note"),
            "gold_path": str(gold_path.resolve()),
        },
        **evaluation,
        "prompt_iteration_history": history,
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).replace("\r", " ").replace("\n", "<br>")
    text = html.escape(text, quote=False).replace("|", "&#124;")
    return text if len(text) <= 120 else text[:117] + "…"


def render_markdown(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    metrics = report["metrics"]
    run = report["run"]
    state = report["formal_state"]
    verdict = "通过" if gate["passed"] else "未通过"
    lines = [
        "# Memory Lab 模型影子验收报告",
        "",
        f"**结论：{verdict}。**",
        "",
        f"- 受控集：`{_cell(report['suite']['suite_id'])}`，共 {metrics['gold_cases']} 条",
        f"- 金标：`{_cell(report['suite']['provenance'])}`；SHA-256 `{_cell(report['suite']['gold_sha256'])}`",
        f"- 冻结时序：{'通过' if gate['gold_frozen_before_run'] else '失败'}",
        f"- 模型：`{_cell(run['model'])}`；提示词：`{_cell(run['prompt_version'])}`",
        f"- 运行：`{_cell(run['shadow_run_id'])}`；状态：`{_cell(run['status'])}`",
        f"- 准入快照：{'当前且完整' if gate['admission_snapshot_current'] else '过期或不完整'}；`{_cell(run['admission_snapshot_hash'])}`",
        f"- 真实模型输出：{'是' if gate['fresh_model_outputs'] else '否'}；缓存回放 {run['replay_count']} 条",
        f"- 正式记忆隔离：{'通过' if gate['formal_state_content_verified'] else '失败'}（逐表行数与内容哈希）",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 结构校验通过 | {metrics['structurally_valid']} / {metrics['gold_cases']} |",
        f"| 应存命中（TP） | {metrics['true_positive']} |",
        f"| 不应存命中（TN） | {metrics['true_negative']} |",
        f"| 误存（FP） | {metrics['false_positive']} |",
        f"| 漏存（FN） | {metrics['false_negative']} |",
        f"| Precision | {_percent(metrics['precision'])} |",
        f"| Recall | {_percent(metrics['recall'])} |",
        f"| 语义字段 | {metrics['semantic_fields_correct']} / {metrics['semantic_fields_expected']} |",
        f"| 原文证据精确匹配 | {metrics['exact_evidence_cases']} / {metrics['exact_evidence_cases_expected']} |",
        f"| 本地敏感阻断 | {metrics['blocked_cases_correct']} / {metrics['blocked_cases']} |",
        f"| Judge 调用 / 缓存回放 / 本地阻断 | {metrics['judge_invocations']} / {metrics['cache_replays']} / {metrics['local_blocks']} |",
        "",
        "## 逐条结果",
        "",
        "| # | 用例 | 原话 | 规则基线 | 预期 | 模型 | 结果 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, case in enumerate(report["cases"], 1):
        expected = "存" if case["expected_should_store"] else "不存"
        predicted = (
            "本地阻断"
            if case["status"] == "blocked"
            else "存"
            if case["predicted_should_store"] is True
            else "不存"
            if case["predicted_should_store"] is False
            else "无有效结果"
        )
        result = "通过" if case["correct"] else "失败"
        lines.append(
            f"| {index} | {_cell(case['label'])} | {_cell(case['quote'])} | "
            f"{_cell(case['rule_disposition'])} | {expected} | {predicted} | {result} |"
        )
    failures = [case for case in report["cases"] if not case["correct"]]
    if failures or report["unexpected_candidates"] or report["duplicate_candidates"]:
        lines.extend(["", "## 失败明细", ""])
        for case in failures:
            lines.append(
                f"- `{_cell(case['candidate_key'])}`："
                + "; ".join(_cell(item) for item in case["mismatches"])
            )
        if report["unexpected_candidates"]:
            lines.append(
                "- 意外候选：" + ", ".join(f"`{_cell(item)}`" for item in report["unexpected_candidates"])
            )
        if report["duplicate_candidates"]:
            lines.append(
                "- 重复候选：" + ", ".join(f"`{_cell(item)}`" for item in report["duplicate_candidates"])
            )
    lines.extend(
        [
            "",
            "## 提示词迭代记录",
            "",
            "| 版本 | 有效输出 | 正确用例 | TP | TN | FP | FN | 门槛 |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["prompt_iteration_history"]:
        lines.append(
            f"| `{_cell(item['prompt_version'])}` | {item['valid']} | {item['correct']} | "
            f"{item['true_positive']} | {item['true_negative']} | {item['false_positive']} | "
            f"{item['false_negative']} | {'通过' if item['passed'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "> 限制：" + _cell(report["suite"].get("generalization_note")),
            "",
            "本报告只验收“从 CP 新消息到模型候选判断”的影子链路；模型结果没有写入正式记忆表，也尚未注入 CP 主 Agent。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def generate_shadow_report(
    lab_database: Path,
    *,
    gold_path: Path = DEFAULT_GOLD,
    output_directory: Path = DEFAULT_REPORT_DIRECTORY,
    shadow_run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate and atomically write matching JSON and Markdown reports."""

    json_path = output_directory / "latest-shadow.json"
    markdown_path = output_directory / "latest-shadow.md"
    try:
        report = evaluate_shadow_run(
            lab_database, gold_path=gold_path, shadow_run_id=shadow_run_id
        )
    except Exception as error:
        failed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        failure = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": failed_at,
            "status": "failed",
            "gate": {"passed": False},
            "error_type": type(error).__name__,
            "note": "No earlier green report is valid for this failed generation attempt.",
        }
        _atomic_write(
            json_path,
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write(
            markdown_path,
            "# Memory Lab 模型影子验收报告\n\n"
            "**结论：报告生成失败，先前的通过结果不得用于当前数据。**\n\n"
            f"- 失败类型：`{_cell(type(error).__name__)}`\n"
            f"- 发生时间：`{_cell(failed_at)}`\n",
        )
        raise
    _atomic_write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    _atomic_write(markdown_path, render_markdown(report))
    return {
        "status": "complete" if report["gate"]["passed"] else "failed",
        "passed": report["gate"]["passed"],
        "shadow_run_id": report["run"]["shadow_run_id"],
        "json_report": str(json_path.resolve()),
        "markdown_report": str(markdown_path.resolve()),
        "metrics": report["metrics"],
        "gate": report["gate"],
    }
