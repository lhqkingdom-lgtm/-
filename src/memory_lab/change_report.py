"""Human-readable, redacted report for one automatic memory cycle."""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .formal_markdown import snapshot_formal_database, validate_formal_memory
from .safety import require_memory_lab_database
from .shadow import formal_state_snapshot


REPORT_SCHEMA_VERSION = "memory-change-report-v1"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
_SECRET = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
    re.compile(r"(?i)\b(?:api[_ -]?key|token|password|passwd|secret)\b\s*[:=]\s*\S+"),
)
_LOCAL_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|/(?:home|users|tmp)/)\S+")

_STATE_LABELS = {
    "memories": "正式记忆",
    "revisions": "修订版本",
    "evidence": "证据",
    "conflicts": "未解决冲突",
}
_STATUS_LABELS = {
    "complete": "已完成",
    "complete_with_rejections": "已完成（部分候选被拒绝）",
    "noop": "已完成（没有新增回合）",
    "review_rejected": "审核拒绝（零写入）",
    "running": "运行中",
    "failed": "失败",
    "published": "已发布",
    "replayed": "已复用既有提交结果",
    "not_run": "未执行",
    "not_published": "未发布",
    "validated_current_matches_run": "校验通过：当前 Markdown 与本轮最终状态一致",
    "validated_current_is_not_run_state": "已校验，但当前 Markdown 不对应本轮最终状态",
}
_OPERATION_LABELS = {
    "create": "新建记忆",
    "create_or_match": "新建或匹配现有记忆",
    "create_entry": "新建记忆",
    "merge_evidence": "合并同义证据",
    "add_scope_variant": "新增适用范围变体",
    "revise": "修订记忆",
    "revise_entry": "修订记忆",
    "open_conflict": "登记冲突",
    "resolve_user_contract_by_time": "按来源顺序更新用户契约",
    "open_engineering_conflict": "登记工程经验冲突",
    "reject": "拒绝沉淀",
    "noop": "无变更",
}
_OUTCOME_LABELS = {
    "accept": "审核通过",
    "revise": "修订后通过",
    "reject": "已拒绝",
}
_ACTION_LABELS = {
    "created": "已新建",
    "merged_evidence": "已合并同义证据",
    "created_scope_variant": "已新增适用范围变体",
    "revised": "已修订",
    "opened_engineering_conflict": "已登记工程经验冲突",
    "opened_conflict": "已登记冲突",
    "merged_conflict_evidence": "已补充冲突证据",
    "rejected": "已拒绝",
    "noop": "无正式写入",
}
_ACTION_REASON_LABELS = {
    "approved_proposal_activated": "审核通过，候选已激活为正式记忆",
    "same_semantic_value_and_scope": "语义值和适用范围相同，仅合并证据",
    "latest_durable_user_contract_by_source_order": "同一范围内采用来源顺序较新的长期用户要求",
    "candidate_already_consolidated": "该候选已经处理，无需重复写入",
    "opposite_value_same_scope_without_correction": "同一范围出现不同结论，已保留为待解决冲突",
    "approved_explicit_correction": "明确更正已通过审核并生成新修订版本",
}
_LOCAL_FILTER_LABELS = {
    "temporary": "本地硬规则拦截：仅为临时要求，不沉淀为长期记忆",
    "covered_by_cp_instructions": "本地硬规则拦截：同范围要求已由 CP 指令文件覆盖，避免重复沉淀",
    "sensitive": "本地硬规则拦截：包含敏感信息，不进入模型或正式记忆",
    "no_long_term_value": "本地硬规则拦截：没有足够的长期复用价值",
}
_REVIEW_REASON_LABELS = {
    "The plan correctly captures a durable, global response preference with no conflicting memories or veto signals.":
        "该计划正确提取了持久、全局的回复偏好，未发现冲突记忆或硬性拦截条件。",
    "Plan matches the atomic quote, respects global scope, and stores a durable response preference with no conflicts.":
        "计划与原子证据一致，适用范围为全局，可作为持久回复偏好保存，且不存在冲突。",
    "Plan matches exact quote, scope global allowed, no conflicts, deterministic policy satisfied.":
        "计划与原始证据完全一致，全局范围合法，不存在冲突，并符合确定性策略。",
    "Plan is safe and correct: evidence quote matches, scope is global, stable key and value align with pending proposal, and no hard vetoes are triggered.":
        "计划安全且正确：证据匹配、范围为全局，稳定键和值与待处理候选一致，且未触发硬性拦截。",
    "Plan is safe and correct: project-scoped workflow rule with exact evidence span, no conflicts with current memories, and pending global proposals are different scope.":
        "计划安全且正确：这是项目范围的工作流规则，证据位置准确，与现有记忆不冲突；待处理的全局规则属于不同适用范围。",
    "Plan matches exact quote, scope limits, and deterministic policy; no conflicts.":
        "计划与原始证据、范围限制和确定性策略一致，不存在冲突。",
    "Plan matches atomic quote, respects domain scope, and correctly merges equivalent pending proposal.":
        "计划与原子证据一致，遵守领域范围，并正确合并了语义等价的待处理候选。",
    "The plan correctly captures the user's explicit preference for Chinese responses as a global, long-term memory. It aligns with the evidence and scope limits.":
        "计划正确提取了用户明确的中文回复偏好，并将其作为全局长期记忆；内容与证据及范围限制一致。",
    "Plan correctly resolves the conflicting user contract by time, using the newer quote as evidence, within global scope and no hard vetoes.":
        "计划正确按来源顺序解决了用户契约冲突，采用更新的明确要求作为证据，范围为全局，且未触发硬性拦截。",
}


def _label(mapping: Mapping[str, str], value: Any) -> str:
    raw = str(value or "")
    return mapping.get(raw, raw or "—")


def _yes_no(value: Any) -> str:
    if value is None:
        return "未知"
    return "是" if bool(value) else "否"


def _review_reason_zh(reason: Any, outcome: Any) -> str:
    """Translate the human view while preserving raw SQLite/JSON audit data."""

    raw = str(reason or "")
    if raw.startswith("local_hard_filter:"):
        code = raw.split(":", 1)[1]
        return _LOCAL_FILTER_LABELS.get(code, f"本地安全规则已拦截：{code}")
    if not raw or raw == "—":
        return "—"
    if raw in _REVIEW_REASON_LABELS:
        return _REVIEW_REASON_LABELS[raw]
    normalized_outcome = str(outcome or "reject")
    if normalized_outcome == "accept":
        return "计划与原始证据、适用范围及确定性策略一致。"
    if normalized_outcome == "revise":
        return "审核发现可修正问题，按审核意见修订后通过。"
    return "审核未通过；原始详细原因保留在 SQLite 审计记录中。"


def _action_reason_zh(reason: Any) -> str:
    raw = str(reason or "")
    return _ACTION_REASON_LABELS.get(raw, raw or "—")


def _redact(value: Any, maximum: int = 500) -> str:
    text = str(value or "")[:maximum]
    for pattern in _SECRET:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return _LOCAL_PATH.sub("[LOCAL_PATH]", text)


def _project_label(value: Any) -> str:
    """Render a project summary without exposing its local directory."""

    if isinstance(value, Mapping):
        display = _redact(value.get("display_name") or "", 120)
        if display and display != "unknown":
            return display
        return _redact(value.get("project_key") or "未知项目", 120)
    return _redact(value, 120)


def _load_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _state_counts(snapshot: Mapping[str, Any] | None) -> dict[str, int]:
    names = {
        "memory": "memories",
        "memory_revision": "revisions",
        "evidence": "evidence",
        "conflict": "conflicts",
    }
    result = {label: 0 for label in names.values()}
    for table, label in names.items():
        value = (snapshot or {}).get(table, {})
        if isinstance(value, Mapping):
            result[label] = int(value.get("row_count", 0) or 0)
    return result


def _audit_rows(lab_database: Path, run_id: str) -> list[dict[str, Any]]:
    uri = lab_database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT w.candidate_key, w.status AS writer_status, w.execution_mode AS writer_mode,
                   w.plan_json, r.status AS reviewer_status, r.execution_mode AS reviewer_mode,
                   r.outcome, r.decision_json, r.final_plan_json
            FROM memory_writer_result AS w
            LEFT JOIN memory_reviewer_result AS r
              ON r.run_id = w.run_id AND r.decision_id = w.decision_id
            WHERE w.run_id = ?
            ORDER BY w.created_at, w.writer_result_id
            """,
            (run_id,),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        plan = _load_json(row["final_plan_json"]) or _load_json(row["plan_json"])
        review = _load_json(row["decision_json"])
        value = plan.get("value") if isinstance(plan.get("value"), Mapping) else {}
        result.append(
            {
                "candidate": _redact(row["candidate_key"], 160),
                "stable_key": _redact(plan.get("stable_key") or "—", 256),
                "operation": _redact(plan.get("operation") or "reject", 80),
                "statement": _redact(value.get("statement") or "—", 400),
                "writer": _redact(
                    f"{row['writer_status']} / {row['writer_mode']}", 100
                ),
                "reviewer": _redact(
                    f"{row['reviewer_status'] or 'missing'} / {row['reviewer_mode'] or 'missing'}",
                    100,
                ),
                "outcome": _redact(row["outcome"] or "reject", 40),
                "review_reason": _redact(review.get("reason") or "—", 300),
            }
        )
    return result


def _run_actions(lab_database: Path, run_id: str) -> list[dict[str, Any]]:
    """Load only actions cryptographically/audit-linked to one auto-memory run.

    An auto-memory run creates a private consolidation run.  The durable link
    back to the public run ID is the immutable JSON note on its AI review.  We
    deliberately parse that note in Python instead of depending on SQLite's
    optional JSON extension.
    """

    uri = lab_database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT a.action, a.reason, a.memory_id, a.before_revision_id,
                   a.after_revision_id, a.conflict_id, a.details_json,
                   a.candidate_key, a.rowid AS action_sequence, r.note
            FROM consolidation_action AS a
            JOIN candidate_review AS r ON r.review_id = a.review_id
            ORDER BY a.rowid
            """
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        note = _load_json(row["note"])
        if note.get("auto_memory_run_id") != run_id:
            continue
        result.append(
            {
                "candidate_key": row["candidate_key"],
                "action": row["action"],
                "reason": row["reason"],
                "memory_id": row["memory_id"],
                "before_revision_id": row["before_revision_id"],
                "after_revision_id": row["after_revision_id"],
                "conflict_id": row["conflict_id"],
                "details": _load_json(row["details_json"]),
            }
        )
    return result


def load_auto_memory_run_report(
    lab_database: Path, run_id: str
) -> dict[str, Any]:
    """Reconstruct the original pipeline report from immutable SQLite audit rows."""

    require_memory_lab_database(lab_database)
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("auto-memory run_id is unsafe")
    uri = lab_database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM auto_memory_run WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"auto-memory run not found: {run_id}")
    if row["status"] == "running":
        raise ValueError("cannot report an auto-memory run that is still running")
    before = _load_json(row["formal_state_before_json"])
    after = _load_json(row["formal_state_after_json"])
    if not before or not after:
        raise ValueError("auto-memory run has no complete formal state snapshots")

    actions = _run_actions(lab_database, run_id)
    stored_counts = _load_json(row["counts_json"])
    expected_actions = stored_counts.get("actions", {})
    expected_actions = (
        {str(key): int(value) for key, value in expected_actions.items()}
        if isinstance(expected_actions, Mapping)
        else {}
    )
    actual_actions = dict(sorted(Counter(item["action"] for item in actions).items()))
    if expected_actions != actual_actions:
        raise ValueError(
            "auto-memory action audit is incomplete or inconsistent "
            f"(expected={expected_actions}, actual={actual_actions})"
        )
    return {
        "run_id": run_id,
        "status": row["status"],
        "admission_run_id": row["admission_run_id"],
        "candidate_count": int(row["candidate_count"]),
        "counts": stored_counts,
        "actions": actions,
        "formal_state_before": before,
        "formal_state_after": after,
        "formal_state_changed": before != after,
    }


def _validated_markdown_for_run(
    lab_database: Path,
    memory_root: Path,
    formal_state_after: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(memory_root).resolve()
    if not (root / "CURRENT").is_file():
        return {
            "status": "not_published",
            "matches_run_state": None,
            "current_changed": False,
        }
    validated = validate_formal_memory(root)
    uri = lab_database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        current_state = formal_state_snapshot(connection)
    current_projection = snapshot_formal_database(lab_database)
    generation_matches_database = (
        validated["generation_id"] == current_projection["generation_id"]
    )
    matches = (
        current_state == dict(formal_state_after) and generation_matches_database
    )
    return {
        **validated,
        "status": (
            "validated_current_matches_run"
            if matches
            else "validated_current_is_not_run_state"
        ),
        # Report reconstruction is read-only; it never advances CURRENT.
        "current_changed": False,
        "matches_run_state": matches,
        "generation_matches_current_database": generation_matches_database,
    }


def rebuild_auto_memory_change_report(
    lab_database: Path,
    run_id: str,
    output_directory: Path,
    *,
    memory_root: Path | None = None,
) -> dict[str, Any]:
    """Rebuild and write Markdown/HTML/JSON for a terminal auto-memory run."""

    pipeline = load_auto_memory_run_report(lab_database, run_id)
    markdown = (
        _validated_markdown_for_run(
            lab_database, memory_root, pipeline["formal_state_after"]
        )
        if memory_root is not None
        else None
    )
    report = build_change_report(
        lab_database,
        pipeline,
        markdown_report=markdown,
        stages={"report_rebuild": {"status": "complete"}},
    )
    files = write_change_report(output_directory, report)
    return {
        "status": "complete",
        "run_id": run_id,
        "source_run_status": pipeline["status"],
        "markdown_matches_run_state": report["markdown"]["matches_run_state"],
        "change_report": report,
        "report_files": files,
    }


def build_change_report(
    lab_database: Path,
    pipeline_report: Mapping[str, Any],
    *,
    markdown_report: Mapping[str, Any] | None = None,
    stages: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require_memory_lab_database(lab_database)
    run_id = str(pipeline_report.get("run_id") or "")
    if not run_id:
        raise ValueError("pipeline report has no run_id")
    before = _state_counts(pipeline_report.get("formal_state_before"))
    after = _state_counts(pipeline_report.get("formal_state_after"))
    delta = {key: after[key] - before[key] for key in before}
    actions = []
    for raw in pipeline_report.get("actions", []):
        if not isinstance(raw, Mapping):
            continue
        actions.append(
            {
                "action": _redact(raw.get("action"), 80),
                "reason": _redact(raw.get("reason"), 300),
                "memory_id": _redact(raw.get("memory_id") or "—", 180),
                "before_revision_id": _redact(
                    raw.get("before_revision_id") or "—", 180
                ),
                "after_revision_id": _redact(
                    raw.get("after_revision_id") or "—", 180
                ),
            }
        )
    supplied_review = pipeline_report.get("review")
    if isinstance(supplied_review, list):
        review = []
        for raw in supplied_review:
            if not isinstance(raw, Mapping):
                continue
            review.append(
                {
                    "stable_key": _redact(raw.get("stable_key") or "—", 256),
                    "operation": _redact(raw.get("operation") or "reject", 80),
                    "outcome": _redact(raw.get("outcome") or "reject", 80),
                    "review_reason": _redact(
                        raw.get("review_reason") or raw.get("reason") or "—", 500
                    ),
                    "statement": _redact(raw.get("statement") or "—", 1_000),
                }
            )
    else:
        review = _audit_rows(lab_database, run_id)
    raw_source_summary = pipeline_report.get("source_summary")
    source_summary: dict[str, Any] = {}
    if isinstance(raw_source_summary, Mapping):
        for key in (
            "turns_processed",
            "user_messages",
            "assistant_finals",
            "tool_events",
            "candidates",
            "accepted",
            "rejected",
        ):
            value = raw_source_summary.get(key, 0)
            source_summary[key] = int(value) if isinstance(value, int) and value >= 0 else 0
        projects = raw_source_summary.get("projects", [])
        source_summary["projects"] = (
            [_project_label(item) for item in projects[:20]]
            if isinstance(projects, list)
            else []
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": _redact(pipeline_report.get("status"), 80),
        "formal_state_changed": bool(pipeline_report.get("formal_state_changed", False)),
        "before": before,
        "after": after,
        "delta": delta,
        "review": review,
        "actions": actions,
        "source_summary": source_summary,
        "markdown": {
            "status": _redact((markdown_report or {}).get("status") or "not_run", 80),
            "generation_id": _redact(
                (markdown_report or {}).get("generation_id") or "—", 180
            ),
            "current_changed": bool(
                (markdown_report or {}).get("current_changed", False)
            ),
            "topic_count": int((markdown_report or {}).get("topic_count", 0) or 0),
            "memory_count": int((markdown_report or {}).get("memory_count", 0) or 0),
            "matches_run_state": (markdown_report or {}).get("matches_run_state"),
            "generation_matches_current_database": (markdown_report or {}).get(
                "generation_matches_current_database"
            ),
        },
        "stages": {
            str(key): _redact(
                value.get("status", "ok") if isinstance(value, Mapping) else value,
                80,
            )
            for key, value in (stages or {}).items()
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    status = _label(_STATUS_LABELS, report["status"])
    before = report["before"]
    after = report["after"]
    delta = report["delta"]
    lines = [
        "# 长期记忆变更报告",
        "",
        f"- 运行批次：`{report['run_id']}`",
        f"- 状态：**{status}**",
        f"- 正式记忆是否发生变化：**{_yes_no(report['formal_state_changed'])}**",
        f"- Markdown 代际：`{report['markdown']['generation_id']}`",
        "",
        "## 正式记忆状态",
        "",
        "| 项目 | 处理前 | 处理后 | 变化量 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("memories", "revisions", "evidence", "conflicts"):
        lines.append(
            f"| {_STATE_LABELS[key]} | {before[key]} | {after[key]} | {delta[key]:+d} |"
        )
    source = report.get("source_summary")
    if isinstance(source, Mapping) and source:
        project_names = "、".join(source.get("projects", [])) or "—"
        lines.extend(
            [
                "",
                "## 本轮会话采集",
                "",
                f"- 处理完成回合：{source.get('turns_processed', 0)}",
                f"- 涉及项目：{project_names}",
                f"- 用户消息 / 最终回复 / 工具事件：{source.get('user_messages', 0)} / "
                f"{source.get('assistant_finals', 0)} / {source.get('tool_events', 0)}",
                f"- 记忆候选 / 审核通过 / 审核拒绝：{source.get('candidates', 0)} / "
                f"{source.get('accepted', 0)} / {source.get('rejected', 0)}",
            ]
        )
    lines.extend(
        [
            "",
            "## 写入代理与审核代理",
            "",
            "| 稳定键 | 写入计划 | 审核结果 | 记忆内容 |",
            "|---|---|---|---|",
        ]
    )
    for item in report["review"]:
        statement = item["statement"].replace("|", "\\|")
        reason = _review_reason_zh(item["review_reason"], item["outcome"]).replace(
            "|", "\\|"
        )
        lines.append(
            f"| `{item['stable_key']}` | {_label(_OPERATION_LABELS, item['operation'])} | "
            f"**{_label(_OUTCOME_LABELS, item['outcome'])}** — {reason} | {statement} |"
        )
    if not report["review"]:
        lines.append("| — | — | — | 本轮没有候选记忆 |")
    lines.extend(
        [
            "",
            "## 已执行的正式变更",
            "",
            "| 动作 | 记忆 ID | 修订 ID | 原因 |",
            "|---|---|---|---|",
        ]
    )
    for item in report["actions"]:
        lines.append(
            f"| {_label(_ACTION_LABELS, item['action'])} | `{item['memory_id']}` | "
            f"`{item['after_revision_id']}` | "
            f"{_action_reason_zh(item['reason']).replace('|', '\\|')} |"
        )
    if not report["actions"]:
        lines.append("| 无正式写入 | — | — | 本轮未改变正式记忆 |")
    lines.extend(
        [
            "",
            "## Markdown 发布状态",
            "",
            f"- 状态：**{_label(_STATUS_LABELS, report['markdown']['status'])}**",
            f"- CURRENT 指针是否变化：**{_yes_no(report['markdown']['current_changed'])}**",
            f"- 是否与本轮最终 SQLite 状态一致："
            f"**{_yes_no(report['markdown']['matches_run_state'])}**",
            f"- 代际是否与当前正式 SQLite 投影一致："
            f"**{_yes_no(report['markdown']['generation_matches_current_database'])}**",
            f"- 主题数：{report['markdown']['topic_count']}",
            f"- 生效记忆数：{report['markdown']['memory_count']}",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(report: Mapping[str, Any]) -> str:
    cards = "".join(
        f"<div class='card'><span>{html.escape(_STATE_LABELS[key])}</span><strong>{value}</strong>"
        f"<small>变化 {report['delta'][key]:+d}</small></div>"
        for key, value in report["after"].items()
    )
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item['stable_key'])}</code></td>"
        f"<td>{html.escape(_label(_OPERATION_LABELS, item['operation']))}</td>"
        f"<td class='{html.escape(item['outcome'])}'>{html.escape(_label(_OUTCOME_LABELS, item['outcome']))}</td>"
        f"<td>{html.escape(_review_reason_zh(item['review_reason'], item['outcome']))}</td>"
        f"<td>{html.escape(item['statement'])}</td>"
        "</tr>"
        for item in report["review"]
    ) or "<tr><td colspan='5'>本轮没有候选记忆</td></tr>"
    action_rows = "".join(
        "<tr>"
        f"<td>{html.escape(_label(_ACTION_LABELS, item['action']))}</td>"
        f"<td><code>{html.escape(item['memory_id'])}</code></td>"
        f"<td><code>{html.escape(item['after_revision_id'])}</code></td>"
        f"<td>{html.escape(_action_reason_zh(item['reason']))}</td>"
        "</tr>"
        for item in report["actions"]
    ) or "<tr><td colspan='4'>本轮没有正式写入</td></tr>"
    match = report["markdown"].get("matches_run_state")
    match_label = _yes_no(match)
    status_label = _label(_STATUS_LABELS, report["status"])
    markdown_status = _label(_STATUS_LABELS, report["markdown"]["status"])
    source = report.get("source_summary")
    if isinstance(source, Mapping) and source:
        projects = "、".join(source.get("projects", [])) or "—"
        source_section = (
            "<h2>本轮会话采集</h2><div class='grid source-grid'>"
            f"<div class='card'><span>完成回合</span><strong>{int(source.get('turns_processed', 0))}</strong></div>"
            f"<div class='card'><span>工具事件</span><strong>{int(source.get('tool_events', 0))}</strong></div>"
            f"<div class='card'><span>记忆候选</span><strong>{int(source.get('candidates', 0))}</strong></div>"
            f"<div class='card'><span>审核通过</span><strong>{int(source.get('accepted', 0))}</strong></div>"
            "</div>"
            f"<p class='muted'>涉及项目：{html.escape(projects)}</p>"
        )
    else:
        source_section = ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CodePulse 长期记忆变更报告</title><style>
body{{font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#f4f6fb;color:#172033}}
main{{max-width:1100px;margin:32px auto;padding:0 20px}}h1{{margin-bottom:4px}}.muted{{color:#65708a}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.card{{background:white;border-radius:14px;padding:18px;box-shadow:0 5px 20px #22315a12}}
.card span,.card small{{display:block;color:#65708a}}.card strong{{font-size:30px}}table{{width:100%;border-collapse:collapse;background:white;border-radius:14px;overflow:hidden;box-shadow:0 5px 20px #22315a12}}
th,td{{text-align:left;padding:12px 14px;border-bottom:1px solid #edf0f7}}th{{background:#eef2fb}}code{{font-size:12px}}.accept{{color:#087b45}}.reject{{color:#a32626}}.revise{{color:#996000}}
@media(max-width:700px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><h1>CodePulse 长期记忆变更报告</h1>
<div class="muted">运行批次 {html.escape(report['run_id'])} · 状态：{html.escape(status_label)} · Markdown 代际 {html.escape(report['markdown']['generation_id'])}</div>
<div class="grid">{cards}</div>{source_section}<h2>写入代理与审核代理</h2><table><thead><tr><th>稳定键</th><th>写入计划</th><th>审核结果</th><th>审核说明</th><th>记忆内容</th></tr></thead><tbody>{rows}</tbody></table>
<h2>已执行的正式变更</h2><table><thead><tr><th>动作</th><th>记忆 ID</th><th>修订 ID</th><th>原因</th></tr></thead><tbody>{action_rows}</tbody></table>
<h2>Markdown 发布状态</h2><p class="muted">{html.escape(markdown_status)} · 与本轮最终 SQLite 状态一致：{match_label}</p>
</main></body></html>"""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_change_report(
    output_directory: Path, report: Mapping[str, Any]
) -> dict[str, str]:
    root = Path(output_directory).resolve()
    run_id = str(report.get("run_id") or "")
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("report run_id is unsafe for a history filename")
    markdown = root / "latest-memory-change.md"
    html_path = root / "latest-memory-change.html"
    json_path = root / "latest-memory-change.json"
    history_root = root / "runs"
    history_markdown = history_root / f"{run_id}.md"
    history_html = history_root / f"{run_id}.html"
    history_json = history_root / f"{run_id}.json"
    markdown_payload = render_markdown(report)
    html_payload = render_html(report)
    json_payload = json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"

    # Keep every cycle for audit/demo comparison, while retaining the stable
    # ``latest-*`` pointers used by the small visual dashboard.  A retry of the
    # same immutable run is content-idempotent.
    _atomic_write(history_markdown, markdown_payload)
    _atomic_write(history_html, html_payload)
    _atomic_write(history_json, json_payload)
    _atomic_write(markdown, markdown_payload)
    _atomic_write(html_path, html_payload)
    _atomic_write(json_path, json_payload)
    return {
        "markdown": str(markdown),
        "html": str(html_path),
        "json": str(json_path),
        "history_markdown": str(history_markdown),
        "history_html": str(history_html),
        "history_json": str(history_json),
    }
