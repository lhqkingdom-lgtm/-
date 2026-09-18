"""Markdown projection for the lightweight PostgreSQL memory MVP.

This is intentionally separate from the older formal SQLite projection.  It is
used for P0 quality review: show the user exactly what the current PostgreSQL
memory state would look like as a Claude-style ``MEMORY.md`` plus topic files.
The export is read-only with respect to PostgreSQL and does not switch
``memory/CURRENT``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .pg_store import DEFAULT_PG_CONFIG, DEFAULT_USER_ID, _run_psql, _sql_literal, load_pg_config


def _filename(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "topic")[:120] + ".md"


def _scope_label(scope: Mapping[str, Any] | None) -> str:
    if not scope:
        return "未指定"
    level = scope.get("level")
    if level == "global":
        return "全局"
    if level == "domain":
        return f"领域：{scope.get('domain') or 'unknown'}"
    if level == "project":
        return f"项目：{scope.get('project_key') or scope.get('project') or 'unknown'}"
    if level == "tool":
        return f"工具：{scope.get('tool') or 'unknown'}"
    return json.dumps(scope, ensure_ascii=False, sort_keys=True)


def _evidence_count(metadata: Mapping[str, Any] | None) -> int:
    refs = (metadata or {}).get("evidence_refs")
    return len(refs) if isinstance(refs, list) else 0


def _source_turns(metadata: Mapping[str, Any] | None) -> list[str]:
    metadata = metadata or {}
    values = metadata.get("source_turn_ids")
    if isinstance(values, list):
        return [str(item) for item in values if str(item).strip()]
    value = metadata.get("source_turn_id")
    return [str(value)] if value else []


def _evidence_ref_count(refs: Any) -> int:
    return len(refs) if isinstance(refs, list) else 0


def _latest_memory_summary(topic: Mapping[str, Any]) -> str:
    memories = topic.get("memories") if isinstance(topic.get("memories"), list) else []
    for memory in memories:
        summary = str(memory.get("summary") or "").strip()
        if summary:
            return summary
    return str(topic.get("summary") or "").strip()


def _topic_summary(topic: Mapping[str, Any]) -> str:
    return str(topic.get("summary") or "").strip()


def _index_summary(topic: Mapping[str, Any]) -> str:
    return _topic_summary(topic) or _latest_memory_summary(topic)


def _group_label(topic: Mapping[str, Any]) -> str:
    topic_type = str(topic.get("topic_type") or "")
    memories = topic.get("memories") if isinstance(topic.get("memories"), list) else []
    memory_types = {str(item.get("memory_type") or "") for item in memories if isinstance(item, Mapping)}
    if topic_type == "user_preference" or any("preference" in item for item in memory_types):
        return "用户习惯与协作偏好"
    if "success" in topic_type or any("success" in item for item in memory_types):
        return "工程成功流程"
    if "failure" in topic_type or any("failure" in item or "runtime" in item for item in memory_types):
        return "工程失败与问题修复经验"
    return "其他长期记忆"


def _topic_sort_key(topic: Mapping[str, Any]) -> tuple[int, str]:
    order = {
        "用户习惯与协作偏好": 0,
        "工程成功流程": 1,
        "工程失败与问题修复经验": 2,
        "其他长期记忆": 3,
    }
    return (order.get(_group_label(topic), 9), str(topic.get("title") or ""))


def _load_state(pg_config_path: Path, include_inactive: bool) -> dict[str, Any]:
    config = load_pg_config(pg_config_path)
    user_id = str(config["user"].get("user_id") or DEFAULT_USER_ID)
    status_filter = "" if include_inactive else "AND mt.status = 'active'"
    memory_status_filter = "" if include_inactive else "AND m.status = 'active'"
    sql = f"""
\\pset format unaligned
\\pset tuples_only on
SELECT jsonb_build_object(
  'user_id', {_sql_literal(user_id)},
  'topics', COALESCE(jsonb_agg(topic_body ORDER BY topic_body->>'title'), '[]'::jsonb)
)::text
FROM (
  SELECT jsonb_build_object(
    'topic_id', mt.topic_id,
    'topic_key', mt.topic_key,
    'title', mt.title,
    'summary', mt.summary,
    'topic_type', mt.topic_type,
    'scope', mt.scope,
    'status', mt.status,
    'created_at', mt.created_at,
    'updated_at', mt.updated_at,
    'memories', COALESCE((
      SELECT jsonb_agg(jsonb_build_object(
        'memory_id', m.memory_id,
        'stable_key', m.stable_key,
        'memory_type', m.memory_type,
        'scope', m.scope,
        'summary', m.summary,
        'content', m.content,
        'status', m.status,
        'confidence', m.confidence,
        'metadata', m.metadata,
        'created_at', m.created_at,
        'updated_at', m.updated_at
      ) ORDER BY m.updated_at DESC, m.memory_id)
      FROM memories m
      WHERE m.topic_id = mt.topic_id
        AND m.user_id = mt.user_id
        {memory_status_filter}
    ), '[]'::jsonb)
    ,
    'conflicts', COALESCE((
      SELECT jsonb_agg(jsonb_build_object(
        'conflict_id', c.conflict_id,
        'memory_id', c.memory_id,
        'turn_id', c.turn_id,
        'conflict_type', c.conflict_type,
        'old_claim', c.old_claim,
        'new_claim', c.new_claim,
        'decision', c.decision,
        'reason', c.reason,
        'evidence_refs', c.evidence_refs,
        'status', c.status,
        'metadata', c.metadata,
        'created_at', c.created_at
      ) ORDER BY c.created_at DESC, c.conflict_id)
      FROM memory_conflict_records c
      WHERE c.topic_id = mt.topic_id
        AND c.user_id = mt.user_id
    ), '[]'::jsonb)
  ) AS topic_body
  FROM memory_topics mt
  WHERE mt.user_id = {_sql_literal(user_id)}
    {status_filter}
) topics;
"""
    output = _run_psql(config, sql)
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"PostgreSQL did not return JSON: {output[:200]}")
    return json.loads(output[start : end + 1])


def export_pg_markdown(
    *,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    output_dir: Path,
    include_inactive: bool = False,
) -> dict[str, Any]:
    state = _load_state(pg_config_path, include_inactive)
    output_dir.mkdir(parents=True, exist_ok=True)
    topics = sorted(state.get("topics") or [], key=_topic_sort_key)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    index_lines = [
        "# Memory Index",
        "",
        f"- 导出时间：{now}",
        f"- 用户主键：`{state.get('user_id')}`",
        f"- 主题数：{len(topics)}",
        f"- 记忆数：{sum(len(topic.get('memories') or []) for topic in topics)}",
        "",
    ]
    written_files: list[dict[str, Any]] = []
    current_group = ""
    for topic in topics:
        memories = topic.get("memories") or []
        filename = _filename(str(topic.get("topic_key") or topic.get("title") or topic.get("topic_id")))
        group = _group_label(topic)
        if group != current_group:
            current_group = group
            index_lines.extend(["", f"## {group}", ""])
        index_lines.append(
            f"- [{topic.get('title')}](./{filename}) — {_index_summary(topic)}；"
            f"{len(memories)} 条记忆；范围：{_scope_label(topic.get('scope'))}"
        )
        body = [
            "---",
            'schema: "memory-lab-pg-topic-v1"',
            f'topic_id: "{topic.get("topic_id")}"',
            f'topic_key: "{topic.get("topic_key")}"',
            f'type: "{topic.get("topic_type")}"',
            f'status: "{topic.get("status")}"',
            f'updated_at: "{topic.get("updated_at")}"',
            "---",
            "",
            f"# {topic.get('title')}",
            "",
            f"- 主题概要：{_topic_summary(topic) or '无'}",
            f"- 代表性记忆摘要：{_latest_memory_summary(topic) or '无'}",
            f"- 范围：{_scope_label(topic.get('scope'))}",
            f"- 记忆数量：{len(memories)}",
            "",
        ]
        for memory in memories:
            metadata = memory.get("metadata") if isinstance(memory.get("metadata"), Mapping) else {}
            turns = _source_turns(metadata)
            body.extend(
                [
                    f"## `{memory.get('stable_key')}`",
                    "",
                    f"- 类型：`{memory.get('memory_type')}`",
                    f"- 状态：`{memory.get('status')}`",
                    f"- 置信度：{memory.get('confidence')}",
                    f"- 范围：{_scope_label(memory.get('scope'))}",
                    f"- 摘要：{memory.get('summary')}",
                    f"- 证据数量：{_evidence_count(metadata)}",
                    f"- 来源 turn：{', '.join(f'`{item}`' for item in turns) if turns else '无'}",
                    "",
                    "### 内容",
                    "",
                    str(memory.get("content") or "").strip(),
                    "",
                ]
            )
        conflicts = topic.get("conflicts") if isinstance(topic.get("conflicts"), list) else []
        if conflicts:
            body.extend(["## 冲突记录", ""])
            for conflict in conflicts:
                refs = conflict.get("evidence_refs")
                body.extend(
                    [
                        f"### `{conflict.get('conflict_id')}`",
                        "",
                        f"- 类型：`{conflict.get('conflict_type')}`",
                        f"- 状态：`{conflict.get('status')}`",
                        f"- 关联 memory：`{conflict.get('memory_id')}`",
                        f"- 来源 turn：`{conflict.get('turn_id')}`",
                        f"- 证据数量：{_evidence_ref_count(refs)}",
                        f"- 旧说法：{conflict.get('old_claim')}",
                        f"- 新说法：{conflict.get('new_claim')}",
                        f"- 处理决定：{conflict.get('decision')}",
                        f"- 原因：{conflict.get('reason')}",
                        "",
                    ]
                )
        target = output_dir / filename
        target.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
        written_files.append({"path": str(target), "topic_id": topic.get("topic_id"), "memory_count": len(memories)})

    index_path = output_dir / "MEMORY.md"
    index_path.write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")
    return {
        "ok": True,
        "status": "complete",
        "output_dir": str(output_dir),
        "index_path": str(index_path),
        "topic_count": len(topics),
        "memory_count": sum(len(topic.get("memories") or []) for topic in topics),
        "files": [{"path": str(index_path), "kind": "index"}, *written_files],
    }
