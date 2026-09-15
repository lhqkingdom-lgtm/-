"""First-pass memory maintenance agent run.

This module wires the already-safe pieces together:

turn evidence + topic/memory index -> LLM changes[] proposal -> deterministic
validation -> optional deterministic commit.

The LLM never receives a write tool.  It only returns JSON data that must pass
``memory_changes`` validation before local code can commit it.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG
from .llm_client import OpenAICompatibleJsonClient
from .memory_changes import (
    MEMORY_CHANGES_SCHEMA_VERSION,
    MemoryChangesValidationError,
    commit_memory_changes,
    validate_memory_changes,
)
from .pg_store import (
    DEFAULT_PG_CONFIG,
    DEFAULT_USER_ID,
    _json_from_psql_array,
    _sql_literal,
    build_writer_context,
    get_turn_evidence,
    list_memory_topics,
    load_pg_config,
)


DEFAULT_AGENT_RUN_DIR = Path(__file__).resolve().parents[2] / "reports" / "memory-agent-runs"


AGENT_SYSTEM_PROMPT = """你是 Memory Lab 的长期记忆整理 Agent。
你要根据一轮 CP 对话证据、主题索引和已有 memory 摘要，输出一个严格 JSON 对象。

你只能输出 JSON，不要输出 Markdown、解释、代码块或额外文本。

目标：沉淀“以后怎么少问、少错、少返工”的长期记忆。优先沉淀用户级协作习惯和可复用工程经验，不要保存当前项目完整需求文档。

可沉淀：
- 用户稳定交互习惯、协作偏好、明确纠正。默认 scope={"level":"global"}，除非只适用于某个技术领域。
- 通用工程流程经验。默认抽象成跨相似项目可复用的模式。
- 编译失败/成功经验：症状、触发条件、原因、解决方法、验证方式。
- 用户反馈运行现象后定位到代码层面的原因、解决方法、验证结论。
- 可跨项目/跨相似任务复用的失败模式。
- 项目级隐性约束：只有“以后回到该项目必须知道，且无法从代码稳定恢复”的信息才用 project scope。

不应沉淀：
- 当前项目一次性的具体参数
- 当前项目需求文档内容，例如具体阈值、频率、功能选择、文件列表、变量名、函数名改动清单
- 可以从代码/Git/需求文档直接恢复的事实
- 未验证猜测
- 完整日志、完整工具输出、敏感信息

抽象要求：
- 不要把 MJ60B、FanGuard、具体寄存器名、具体文件修改清单等项目细节直接当成通用记忆；应抽象成“目标芯片外设驱动必须按数据手册确认，不能套用模板寄存器”这类可复用经验。
- 当前项目参数默认 noop；项目隐性约束才用 project scope。
- 用户习惯优先沉淀为 collaboration_preference；全局习惯用 global scope。
- 编译/构建经验优先使用 memory_type=build_failure_pattern 或 build_success_pattern。
- 运行现象定位经验优先使用 memory_type=runtime_failure_pattern。
- 运行现象经验正文必须尽量按：症状 / 触发条件 / 排查路径 / 代码层根因 / 解决方法 / 验证方式 / 复用范围 组织。
- 工程经验如果没有编译、测试、用户确认或明确证据，不要写成“已验证”；可写为待验证经验或 request_more_evidence。
- 同症状不同根因通常不是冲突，应 create_memory 成另一条 failure_pattern；只有同 scope、同条件、同版本下结论互相否定且无法解释时，才 mark_conflict。
- 用户明确更正优先 update_memory；项目例外优先 project scope，不要污染 global/domain。

输出 schema:
{
  "schema_version": "memory-changes-v1",
  "turn_id": "当前 turn_id",
  "reason": "中文整体判断",
  "changes": []
}

changes 允许 action:
- create_memory
- merge_memory
- update_memory
- mark_conflict
- noop
- request_more_evidence

create_memory:
{
  "action": "create_memory",
  "reason": "为什么新增",
  "topic": {"topic_key":"...","title":"...","summary":"...","topic_type":"...","scope":{"level":"domain","domain":"embedded"}},
  "memory": {"stable_key":"...","memory_type":"...","summary":"...","content":"...","scope":{"level":"domain","domain":"embedded"},"confidence":0.0},
  "evidence_refs": [{"evidence_id":"...","reason":"..." }]
}
也可以使用已有 topic_id 替代 topic。

merge_memory/update_memory/mark_conflict:
{
  "action": "merge_memory|update_memory|mark_conflict",
  "reason": "为什么修改已有记忆",
  "target_memory_id": "必须是已有 active memory_id",
  "memory": {"summary":"...","content":"...","confidence":0.0},
  "evidence_refs": [{"evidence_id":"...","reason":"..." }]
}

noop:
{
  "action": "noop",
  "reason": "为什么不沉淀",
  "evidence_refs": []
}

request_more_evidence:
{
  "action": "request_more_evidence",
  "reason": "为什么需要补证据",
  "requests": [{"tool":"expand_pointer","raw_pointer_id":"...","reason":"..."}],
  "evidence_refs": []
}

硬性规则:
- evidence_refs 只能引用输入里真实存在的 evidence_id 或 raw_pointer_id。
- target_memory_id 只能引用输入里真实存在的 active memory_id。
- 不要输出 SQL、命令、路径操作字段或任意未知字段。
- 如果已有 memory 已覆盖本轮信息，输出 noop 或 merge_memory，不要重复 create_memory。
- 如果只是当前项目具体参数或需求选择，输出 noop。
- 如果要新建通用工程经验，必须写成可复用模式，不要只复述单项目实现。
- 一轮可以输出多条 changes，但最多 10 条。
"""


REPAIR_SYSTEM_PROMPT = """你是 JSON schema 修复器。
你会收到上一次 Agent 输出、程序校验错误和允许 schema。
只修复 JSON 格式、字段名、字段结构、引用方式，不要改变原本语义判断。
只能输出一个完整 JSON 对象，不要输出 Markdown、解释、代码块或额外文本。
"""


def _infer_filters(turn_evidence: Mapping[str, Any]) -> dict[str, str | None]:
    turn = (turn_evidence.get("evidence") or {}).get("turn") or {}
    phase = str(turn.get("phase_hint") or "")
    project_key = turn.get("project_key")
    domain = "embedded" if any(
        token in phase
        for token in (
            "compile",
            "runtime",
            "engineering",
            "requirement_alignment",
            "code_generation",
        )
    ) else None
    return {
        "domain": domain,
        "project_key": str(project_key) if project_key else None,
    }


def build_memory_agent_input(
    *,
    turn_id: str,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    memory_limit: int = 80,
) -> dict[str, Any]:
    turn_evidence = get_turn_evidence(
        pg_config_path=pg_config_path,
        turn_id=turn_id,
        layers=["core", "candidate", "pointer"],
        max_chars=1400,
    )
    if not turn_evidence.get("found"):
        return {
            "status": "not_found",
            "turn_id": turn_id,
            "turn_evidence": turn_evidence,
        }
    filters = _infer_filters(turn_evidence)
    topic_index = list_memory_topics(
        pg_config_path=pg_config_path,
        domain=filters["domain"],
        project_key=filters["project_key"],
        limit=200,
    )
    writer_context = build_writer_context(
        pg_config_path=pg_config_path,
        turn_id=turn_id,
        memory_limit=memory_limit,
    )
    return {
        "status": "complete",
        "turn_id": turn_id,
        "turn_evidence": turn_evidence["evidence"],
        "topic_index": topic_index,
        "memory_index": (writer_context.get("context") or {}).get("memory_index") or {},
        "filters": filters,
        "schema": {
            "schema_version": MEMORY_CHANGES_SCHEMA_VERSION,
            "actions": [
                "create_memory",
                "merge_memory",
                "update_memory",
                "mark_conflict",
                "noop",
                "request_more_evidence",
            ],
        },
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run_id(turn_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = "".join(ch if ch.isalnum() else "-" for ch in turn_id[-12:]).strip("-")
    return f"agent-{stamp}-{suffix or 'turn'}"


def _context_stats(payload: Mapping[str, Any]) -> dict[str, int]:
    layers = (payload.get("turn_evidence") or {}).get("layers") or {}
    return {
        "core_count": len(layers.get("core") or []),
        "candidate_count": len(layers.get("candidate") or []),
        "pointer_count": len(layers.get("pointer") or []),
        "topic_count": len((payload.get("topic_index") or {}).get("topics") or []),
        "memory_count": len((payload.get("memory_index") or {}).get("memories") or []),
    }


def _step(name: str, status: str, summary: str, **extra: Any) -> dict[str, Any]:
    return {
        "step": name,
        "status": status,
        "summary": summary,
        **extra,
    }


def _write_run_log(
    *,
    run_log: Mapping[str, Any],
    log_dir: Path | None = DEFAULT_AGENT_RUN_DIR,
) -> dict[str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(run_log["run_id"])
    path = log_dir / f"{run_id}.json"
    path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = log_dir / "latest.json"
    try:
        shutil.copyfile(path, latest)
    except OSError:
        latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"run_log_path": str(path), "latest_log_path": str(latest)}


def _recent_turn_ids(
    *,
    pg_config_path: Path,
    limit: int,
) -> list[str]:
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    limit = max(1, min(int(limit), 100))
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT COALESCE(jsonb_agg(jsonb_build_object('turn_id', turn_id)), '[]'::jsonb)::text
FROM (
  SELECT turn_id, occurred_at
  FROM turns
  WHERE user_id = {_sql_literal(user_id)}
    AND state = 'completed'
  ORDER BY occurred_at DESC NULLS LAST, created_at DESC
  LIMIT {limit}
) AS recent;
"""
    rows = _json_from_psql_array(config, sql)
    return [str(item["turn_id"]) for item in rows if isinstance(item, Mapping) and item.get("turn_id")]


def _validate_or_repair(
    *,
    model: OpenAICompatibleJsonClient,
    first: Mapping[str, Any],
    payload: Mapping[str, Any],
    pg_config_path: Path,
) -> tuple[dict[str, Any] | None, bool, list[str], Mapping[str, Any]]:
    try:
        return validate_memory_changes(first, pg_config_path=pg_config_path), False, [], first
    except MemoryChangesValidationError as error:
        repair_payload = {
            "previous_output": first,
            "validation_errors": error.errors,
            "required_schema": payload["schema"],
            "available_ids": {
                "turn_evidence": payload["turn_evidence"],
                "memory_index": payload["memory_index"],
                "topic_index": payload["topic_index"],
            },
        }
        second = model.complete_json(
            system_prompt=REPAIR_SYSTEM_PROMPT,
            user_payload=repair_payload,
            max_tokens=5000,
        )
        try:
            return validate_memory_changes(second, pg_config_path=pg_config_path), True, [], second
        except MemoryChangesValidationError as second_error:
            return None, True, second_error.errors, second


def run_memory_agent(
    *,
    turn_id: str,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    memory_limit: int = 80,
    dry_run: bool = False,
    log_dir: Path = DEFAULT_AGENT_RUN_DIR,
    client: OpenAICompatibleJsonClient | None = None,
) -> dict[str, Any]:
    log_dir = log_dir or DEFAULT_AGENT_RUN_DIR
    run_id = _safe_run_id(turn_id)
    started_at = _utc_now_iso()
    steps: list[dict[str, Any]] = []
    payload = build_memory_agent_input(
        turn_id=turn_id,
        pg_config_path=pg_config_path,
        memory_limit=memory_limit,
    )
    if payload.get("status") != "complete":
        steps.append(
            _step(
                "build_memory_agent_input",
                "failed",
                f"turn not found or unavailable: {turn_id}",
            )
        )
        result = {
            "status": "failed",
            "stage": "input",
            "turn_id": turn_id,
            "input": payload,
            "committed": False,
        }
        run_log = {
            "schema_version": "memory-agent-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run" if dry_run else "commit",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}
    stats = _context_stats(payload)
    steps.extend(
        [
            _step(
                "get_turn_evidence",
                "complete",
                f"core={stats['core_count']} candidate={stats['candidate_count']} pointer={stats['pointer_count']}",
                filters=payload.get("filters"),
            ),
            _step(
                "list_memory_topics",
                "complete",
                f"topics={stats['topic_count']}",
            ),
            _step(
                "build_memory_index",
                "complete",
                f"memories={stats['memory_count']}",
            ),
        ]
    )
    model = client or OpenAICompatibleJsonClient.from_config(config_path)
    first = model.complete_json(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_payload=payload,
        max_tokens=6000,
    )
    steps.append(_step("model_writer", "complete", "model returned JSON object"))
    plan, repaired, errors, raw_plan = _validate_or_repair(
        model=model,
        first=first,
        payload=payload,
        pg_config_path=pg_config_path,
    )
    if plan is None:
        steps.append(
            _step(
                "validate_changes",
                "failed",
                f"errors={len(errors)} repaired={repaired}",
                errors=errors,
            )
        )
        result = {
            "status": "failed",
            "stage": "validate",
            "turn_id": turn_id,
            "repaired": repaired,
            "validation_errors": errors,
            "raw_plan": raw_plan,
            "committed": False,
        }
        run_log = {
            "schema_version": "memory-agent-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run" if dry_run else "commit",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "raw_plan": raw_plan,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}
    change_count = len(plan.get("changes") or [])
    steps.append(
        _step(
            "validate_changes",
            "complete",
            f"changes={change_count} repaired={repaired}",
        )
    )
    if dry_run:
        result = {
            "status": "complete",
            "mode": "dry_run",
            "turn_id": turn_id,
            "repaired": repaired,
            "plan": plan,
            "committed": False,
            "context_stats": stats,
        }
        steps.append(_step("commit_changes", "skipped", "dry_run=true"))
        run_log = {
            "schema_version": "memory-agent-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "context_stats": stats,
            "plan": plan,
            "commit": None,
            "result": {
                "status": result["status"],
                "committed": False,
                "repaired": repaired,
            },
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}
    commit_report = commit_memory_changes(plan, pg_config_path=pg_config_path)
    counts = commit_report.get("counts") or {}
    steps.append(
        _step(
            "commit_changes",
            "complete",
            "applied={applied} rejected={rejected} noop={noop} requests={requests}".format(
                applied=counts.get("applied", 0),
                rejected=counts.get("rejected", 0),
                noop=counts.get("noop", 0),
                requests=counts.get("requests", 0),
            ),
            counts=counts,
        )
    )
    result = {
        "status": "complete",
        "mode": "commit",
        "turn_id": turn_id,
        "repaired": repaired,
        "plan": plan,
        "commit": commit_report,
        "committed": bool((commit_report.get("counts") or {}).get("applied")),
    }
    run_log = {
        "schema_version": "memory-agent-run-log-v1",
        "run_id": run_id,
        "turn_id": turn_id,
        "mode": "commit",
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "status": result["status"],
        "steps": steps,
        "context_stats": stats,
        "plan": plan,
        "commit": commit_report,
        "result": {
            "status": result["status"],
            "committed": result["committed"],
            "repaired": repaired,
        },
    }
    return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}


def run_memory_agent_batch(
    *,
    turn_ids: list[str] | None = None,
    limit: int = 5,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    memory_limit: int = 80,
    dry_run: bool = True,
    log_dir: Path | None = DEFAULT_AGENT_RUN_DIR,
) -> dict[str, Any]:
    """Run the memory agent over several turns and write a batch log."""

    log_dir = log_dir or DEFAULT_AGENT_RUN_DIR
    batch_id = _safe_run_id("batch")
    started_at = _utc_now_iso()
    selected_turn_ids = [item for item in (turn_ids or []) if item]
    if not selected_turn_ids:
        selected_turn_ids = _recent_turn_ids(pg_config_path=pg_config_path, limit=limit)
    results: list[dict[str, Any]] = []
    counts = {
        "turns": 0,
        "complete": 0,
        "failed": 0,
        "applied": 0,
        "rejected": 0,
        "noop": 0,
        "requests": 0,
        "dry_run_changes": 0,
    }
    for turn_id in selected_turn_ids:
        result = run_memory_agent(
            turn_id=turn_id,
            pg_config_path=pg_config_path,
            config_path=config_path,
            memory_limit=memory_limit,
            dry_run=dry_run,
            log_dir=log_dir,
        )
        counts["turns"] += 1
        if result.get("status") == "complete":
            counts["complete"] += 1
        else:
            counts["failed"] += 1
        if dry_run:
            counts["dry_run_changes"] += len((result.get("plan") or {}).get("changes") or [])
        else:
            commit_counts = ((result.get("commit") or {}).get("counts") or {})
            counts["applied"] += int(commit_counts.get("applied") or 0)
            counts["rejected"] += int(commit_counts.get("rejected") or 0)
            counts["noop"] += int(commit_counts.get("noop") or 0)
            counts["requests"] += int(commit_counts.get("requests") or 0)
        results.append(
            {
                "turn_id": turn_id,
                "status": result.get("status"),
                "mode": result.get("mode"),
                "committed": result.get("committed"),
                "run_id": result.get("run_id"),
                "log": result.get("log"),
                "plan_actions": [
                    change.get("action")
                    for change in ((result.get("plan") or {}).get("changes") or [])
                    if isinstance(change, Mapping)
                ],
                "commit_counts": ((result.get("commit") or {}).get("counts") or {}),
            }
        )
    batch_log = {
        "schema_version": "memory-agent-batch-log-v1",
        "batch_id": batch_id,
        "mode": "dry_run" if dry_run else "commit",
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "turn_ids": selected_turn_ids,
        "counts": counts,
        "results": results,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_path = log_dir / f"{batch_id}.json"
    batch_path.write_text(json.dumps(batch_log, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_batch = log_dir / "latest-batch.json"
    try:
        shutil.copyfile(batch_path, latest_batch)
    except OSError:
        latest_batch.write_text(batch_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "status": "complete",
        "tool": "memory_agent_batch",
        "batch_id": batch_id,
        "mode": "dry_run" if dry_run else "commit",
        "counts": counts,
        "results": results,
        "log": {
            "batch_log_path": str(batch_path),
            "latest_batch_log_path": str(latest_batch),
        },
    }


__all__ = ["build_memory_agent_input", "run_memory_agent", "run_memory_agent_batch"]
