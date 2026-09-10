"""Claude-style Markdown projection for validated preview memories.

SQLite remains the evidence and decision audit ledger.  This module publishes
one deterministic, human-readable *preview* projection and can read it back for
standalone recall.  It deliberately does not turn shadow judgments into formal
memory.

Front matter uses a strict JSON-scalar subset of YAML 1.2.  That keeps the
files readable without adding a YAML dependency while making duplicate keys,
tags, aliases, and implicit scalar coercion impossible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .catalog import recall_from_catalog, run_catalog
from .judge import WIRE_SECRET_PATTERNS


INDEX_SCHEMA = "memory-lab-preview-index-v1"
TOPIC_SCHEMA = "memory-lab-preview-topic-v1"
CONFLICT_SCHEMA = "memory-lab-preview-conflicts-v1"
MANAGED_BY = "codepulse-memory-lab"
MAX_INDEX_BYTES = 25 * 1024
MAX_TOPIC_BYTES = 256 * 1024
MAX_HEADER_BYTES = 192 * 1024
MAX_MANAGED_FILES = 200


class MarkdownMemoryError(ValueError):
    """Raised when a Markdown projection is unsafe or inconsistent."""


_TOPIC_SPECS = (
    (
        "preference.response.",
        {
            "topic_id": "user-response-preferences",
            "path": "user_response_preferences.md",
            "name": "回复偏好",
            "description": "跨项目生效的回复语言与组织方式",
            "type": "user",
        },
    ),
    (
        "workflow.code_generation.",
        {
            "topic_id": "workflow-code-generation",
            "path": "workflow_code_generation.md",
            "name": "代码生成流程",
            "description": "生成正式代码前的确认要求及项目例外",
            "type": "workflow",
        },
    ),
    (
        "workflow.embedded.",
        {
            "topic_id": "workflow-embedded-compile",
            "path": "workflow_embedded_compile.md",
            "name": "嵌入式流程",
            "description": "嵌入式代码修改后的编译和验证要求",
            "type": "workflow",
        },
    ),
)

_CANONICAL_STATEMENTS = {
    ("preference.response.language", "zh"): "默认使用中文回复。",
    ("preference.response.language", "en"): "默认使用英文回复。",
    (
        "preference.response.structure",
        "conclusion_then_details",
    ): "回复时先用一句话给出结论，再说明处理步骤。",
    (
        "workflow.code_generation.confirmation",
        "required",
    ): "生成正式代码前需要先向用户确认。",
    (
        "workflow.code_generation.confirmation",
        "not_required",
    ): "生成正式代码前不需要向用户确认，可以直接生成。",
    (
        "workflow.embedded.code_change.compile_check",
        "required",
    ): "嵌入式项目修改代码后必须进行编译检查。",
    (
        "workflow.embedded.code_change.compile_check",
        "not_required",
    ): "嵌入式项目修改代码后不要求进行编译检查。",
}


_HEADER_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_TOPIC_INDEX_KEYS = {
    "topic_id",
    "path",
    "name",
    "description",
    "type",
    "status",
    "stable_keys",
    "triggers",
    "blockers",
    "summary",
    "sha256",
}
_PROJECT_CONTEXT_KEYS = {"project_key", "directory", "display_name"}
_MEMORY_ENTRY_KEYS = {
    "kind",
    "entry_id",
    "stable_key",
    "memory_type",
    "normalized_value",
    "statement",
    "scope",
    "scope_rank",
    "status",
    "activation_triggers",
    "evidence_count",
    "source_candidate_keys",
}
_CONFLICT_ENTRY_KEYS = {
    "kind",
    "entry_id",
    "stable_key",
    "scope",
    "scope_rank",
    "status",
    "reason",
    "activation_triggers",
    "variants",
}
_CONFLICT_VARIANT_KEYS = {
    "normalized_value",
    "statement",
    "memory_types",
    "candidate_keys",
    "evidence_count",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_object(value: Any) -> str:
    return _sha256_bytes(_json(value).encode("utf-8"))


def _front_matter(fields: Mapping[str, Any]) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if not _HEADER_KEY.fullmatch(key):
            raise MarkdownMemoryError(f"invalid front-matter key: {key}")
        lines.append(f"{key}: {_json(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _parse_front_matter(text: str, *, source: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise MarkdownMemoryError(f"{source} has no front matter")
    lines = text.splitlines(keepends=True)
    consumed = len(lines[0].encode("utf-8"))
    result: dict[str, Any] = {}
    closing_index: int | None = None
    for index, raw_line in enumerate(lines[1:], start=1):
        consumed += len(raw_line.encode("utf-8"))
        if consumed > MAX_HEADER_BYTES:
            raise MarkdownMemoryError(f"{source} front matter is too large")
        line = raw_line.rstrip("\r\n")
        if line == "---":
            closing_index = index
            break
        if not line or ": " not in line:
            raise MarkdownMemoryError(f"{source} has invalid front matter")
        key, encoded = line.split(": ", 1)
        if not _HEADER_KEY.fullmatch(key) or key in result:
            raise MarkdownMemoryError(f"{source} has invalid or duplicate key")
        try:
            result[key] = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise MarkdownMemoryError(
                f"{source} has a non-JSON YAML scalar for {key}"
            ) from error
    if closing_index is None:
        raise MarkdownMemoryError(f"{source} front matter is not closed")
    return result, "".join(lines[closing_index + 1 :])


def _assert_exact_keys(
    value: Mapping[str, Any], *, required: set[str], source: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise MarkdownMemoryError(f"{source} contains a non-text key")
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise MarkdownMemoryError(
            f"{source} schema mismatch (missing={sorted(missing)}, unknown={sorted(unknown)})"
        )


def _assert_string_list(value: Any, *, source: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MarkdownMemoryError(f"{source} must be a string list")
    if len(value) != len(set(value)):
        raise MarkdownMemoryError(f"{source} contains duplicates")
    return value


def _assert_nonnegative_int(value: Any, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MarkdownMemoryError(f"{source} must be a non-negative integer")
    return value


def _validate_scope(scope: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(scope, dict):
        raise MarkdownMemoryError(f"{source} must be an object")
    level = scope.get("level")
    expected = {"level"}
    if level == "domain":
        expected.add("domain")
        _clean_text(scope.get("domain"), field=f"{source}.domain", maximum=100)
    elif level == "project":
        expected.add("project_key")
        _clean_text(
            scope.get("project_key"), field=f"{source}.project_key", maximum=200
        )
    elif level != "global":
        raise MarkdownMemoryError(f"{source} has an unsupported level")
    if "conditions" in scope:
        expected.add("conditions")
        conditions = scope["conditions"]
        if not isinstance(conditions, dict) or not conditions:
            raise MarkdownMemoryError(f"{source}.conditions must be a non-empty object")
        for key, value in conditions.items():
            _clean_text(key, field=f"{source}.condition", maximum=100)
            if not isinstance(value, (str, int, float, bool)) or value is None:
                raise MarkdownMemoryError(
                    f"{source}.conditions values must be JSON scalars"
                )
            _assert_no_secrets(value)
    _assert_exact_keys(scope, required=expected, source=source)
    return scope


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _assert_no_secrets(value: Any) -> None:
    for text in _walk_strings(value):
        if any(pattern.search(text) for pattern in WIRE_SECRET_PATTERNS):
            raise MarkdownMemoryError("preview memory contains sensitive content")


def _clean_text(value: Any, *, field: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise MarkdownMemoryError(f"{field} must be text")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or len(cleaned) > maximum:
        raise MarkdownMemoryError(f"{field} is empty or too long")
    _assert_no_secrets(cleaned)
    return cleaned


def _statement(stable_key: str, normalized_value: str, fallback: Any) -> str:
    known = _CANONICAL_STATEMENTS.get((stable_key, normalized_value))
    return known or _clean_text(fallback, field="statement")


def _scope_rank(scope: Mapping[str, Any]) -> int:
    return {"global": 0, "domain": 1, "project": 2}.get(str(scope.get("level")), -1)


def _topic_spec(stable_key: str) -> dict[str, str]:
    for prefix, spec in _TOPIC_SPECS:
        if stable_key.startswith(prefix):
            return dict(spec)
    slug = re.sub(r"[^a-z0-9]+", "_", stable_key.casefold()).strip("_")[:48]
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8]
    slug = slug or "memory"
    return {
        "topic_id": f"topic-{slug}-{digest}",
        "path": f"topic_{slug}_{digest}.md",
        "name": stable_key,
        "description": f"长期记忆主题 {stable_key}",
        "type": "memory",
    }


def _scope_text(scope: Mapping[str, Any], projects: Mapping[str, str]) -> str:
    level = scope.get("level")
    if level == "global":
        base = "全局"
    elif level == "domain":
        base = f"领域：{scope.get('domain')}"
    elif level == "project":
        key = str(scope.get("project_key") or "")
        base = f"项目：{projects.get(key, key)}"
    else:
        base = "未知范围"
    conditions = scope.get("conditions")
    if isinstance(conditions, dict) and conditions:
        rendered = "、".join(f"{key}={value}" for key, value in sorted(conditions.items()))
        base += f"（条件：{rendered}）"
    return base


def _memory_entry(memory: Mapping[str, Any]) -> dict[str, Any]:
    stable_key = _clean_text(memory.get("stable_key"), field="stable_key", maximum=200)
    normalized_value = _clean_text(
        memory.get("normalized_value"), field="normalized_value", maximum=200
    )
    scope = dict(_validate_scope(memory.get("scope"), source="memory scope"))
    scope_rank = _scope_rank(scope)
    if memory.get("scope_rank", scope_rank) != scope_rank:
        raise MarkdownMemoryError("memory scope rank disagrees with its scope")
    status = "context_required" if scope.get("conditions") else "preview_active"
    raw_triggers = memory.get("activation_triggers", [])
    raw_candidate_keys = memory.get("source_candidate_keys", [])
    if not isinstance(raw_triggers, list) or not isinstance(raw_candidate_keys, list):
        raise MarkdownMemoryError("memory triggers and candidate identifiers must be lists")
    triggers = sorted(
        {
            _clean_text(item, field="activation_trigger", maximum=120)
            for item in raw_triggers
        },
        key=lambda item: (item.casefold(), item),
    )
    entry = {
        "kind": "memory",
        "entry_id": _clean_text(
            memory.get("preview_memory_id"), field="preview_memory_id", maximum=200
        ),
        "stable_key": stable_key,
        "memory_type": _clean_text(
            memory.get("memory_type"), field="memory_type", maximum=100
        ),
        "normalized_value": normalized_value,
        "statement": _statement(stable_key, normalized_value, memory.get("statement")),
        "scope": scope,
        "scope_rank": scope_rank,
        "status": status,
        "activation_triggers": triggers,
        "evidence_count": _assert_nonnegative_int(
            memory.get("evidence_count"), source="memory evidence_count"
        ),
        "source_candidate_keys": sorted(
            {
                _clean_text(item, field="candidate_key", maximum=200)
                for item in raw_candidate_keys
            }
        ),
    }
    if entry["evidence_count"] < 1:
        raise MarkdownMemoryError("preview memory must have evidence")
    if not entry["source_candidate_keys"]:
        raise MarkdownMemoryError("preview memory must retain candidate identifiers")
    _assert_no_secrets(entry)
    return entry


def _conflict_entry(conflict: Mapping[str, Any]) -> dict[str, Any]:
    stable_key = _clean_text(conflict.get("stable_key"), field="stable_key", maximum=200)
    scope = dict(_validate_scope(conflict.get("scope"), source="conflict scope"))
    scope_rank = _scope_rank(scope)
    if conflict.get("scope_rank", scope_rank) != scope_rank:
        raise MarkdownMemoryError("conflict scope rank disagrees with its scope")
    variants = []
    raw_variants = conflict.get("variants", [])
    if not isinstance(raw_variants, list):
        raise MarkdownMemoryError("conflict variants must be a list")
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, Mapping):
            raise MarkdownMemoryError("conflict variant must be an object")
        normalized_value = _clean_text(
            raw_variant.get("value"), field="conflict value", maximum=200
        )
        raw_statements = raw_variant.get("statements") or []
        raw_memory_types = raw_variant.get("memory_types", [])
        raw_candidate_keys = raw_variant.get("candidate_keys", [])
        if (
            not isinstance(raw_statements, list)
            or not isinstance(raw_memory_types, list)
            or not isinstance(raw_candidate_keys, list)
        ):
            raise MarkdownMemoryError("conflict variant fields must be lists")
        fallback = raw_statements[0] if raw_statements else normalized_value
        candidate_keys = sorted(
            {
                _clean_text(item, field="candidate_key", maximum=200)
                for item in raw_candidate_keys
            }
        )
        variants.append(
            {
                "normalized_value": normalized_value,
                "statement": _statement(stable_key, normalized_value, fallback),
                "memory_types": sorted(
                    {
                        _clean_text(item, field="memory_type", maximum=100)
                        for item in raw_memory_types
                    }
                ),
                "candidate_keys": candidate_keys,
                "evidence_count": _assert_nonnegative_int(
                    raw_variant.get("evidence_count"),
                    source="conflict evidence_count",
                ),
            }
        )
    has_disagreement = len(variants) > 1 or any(
        len(item["memory_types"]) > 1 for item in variants
    )
    if not has_disagreement or any(
        item["evidence_count"] < 1
        or not item["candidate_keys"]
        or not item["memory_types"]
        for item in variants
    ):
        raise MarkdownMemoryError("conflict has no evidenced value or type disagreement")
    variants.sort(key=lambda item: (item["normalized_value"], item["statement"]))
    if len({item["normalized_value"] for item in variants}) != len(variants):
        raise MarkdownMemoryError("conflict variants must have distinct values")
    raw_triggers = conflict.get("activation_triggers", [])
    if not isinstance(raw_triggers, list):
        raise MarkdownMemoryError("conflict triggers must be a list")
    triggers = sorted(
        {
            _clean_text(item, field="activation_trigger", maximum=120)
            for item in raw_triggers
        },
        key=lambda item: (item.casefold(), item),
    )
    identity = {"stable_key": stable_key, "scope": scope, "variants": variants}
    entry = {
        "kind": "conflict",
        "entry_id": "preview_conflict_" + _sha256_object(identity)[:24],
        "stable_key": stable_key,
        "scope": scope,
        "scope_rank": scope_rank,
        "status": "review_required",
        "reason": "opposite_or_incompatible_values_in_same_scope",
        "activation_triggers": triggers,
        "variants": variants,
    }
    _assert_no_secrets(entry)
    return entry


def _topic_status(entries: list[dict[str, Any]]) -> str:
    has_conflict = any(item["kind"] == "conflict" for item in entries)
    has_memory = any(item["kind"] == "memory" for item in entries)
    has_condition = any(item.get("status") == "context_required" for item in entries)
    if has_conflict and has_memory:
        return "has_conflicts"
    if has_conflict:
        return "conflict"
    if has_condition:
        return "context_required"
    return "preview"


def _topic_summary(
    spec: Mapping[str, str],
    entries: list[dict[str, Any]],
    projects: Mapping[str, str],
) -> str:
    active = [
        item
        for item in entries
        if item["kind"] == "memory" and item["status"] == "preview_active"
    ]
    conflicts = [item for item in entries if item["kind"] == "conflict"]
    parts: list[str] = []
    if spec["topic_id"] == "user-response-preferences":
        for item in active:
            if item["stable_key"] == "preference.response.structure":
                parts.append("回复结构为先结论、后步骤")
            elif item["stable_key"] == "preference.response.language":
                language = "中文" if item["normalized_value"] == "zh" else "英文"
                parts.append(f"默认使用{language}")
        if any(item["stable_key"] == "preference.response.language" for item in conflicts):
            parts.append("语言偏好待解决")
    elif spec["topic_id"] == "workflow-code-generation":
        for item in active:
            scope = item["scope"]
            value = item["normalized_value"]
            if scope.get("level") == "global" and value == "required":
                parts.append("全局生成正式代码前需确认")
            elif scope.get("level") == "project" and value == "not_required":
                key = str(scope.get("project_key") or "")
                parts.append(f"{projects.get(key, key)} 项目例外：可直接生成")
            else:
                parts.append(f"{_scope_text(scope, projects)}：{item['statement'].rstrip('。')}")
        if conflicts:
            parts.append(f"{len(conflicts)} 项冲突待解决")
    else:
        parts.extend(item["statement"].rstrip("。") for item in active[:2])
        if conflicts:
            parts.append(f"{len(conflicts)} 项冲突待解决")
    if not parts:
        parts.append("需要补充上下文或人工审核")
    return "；".join(parts) + "。"


def _entry_order(item: Mapping[str, Any]) -> tuple[str, int, str, str, str]:
    return (
        str(item["stable_key"]),
        int(item.get("scope_rank", -1)),
        _json(item.get("scope") or {}),
        str(item["kind"]),
        str(item.get("normalized_value", "")),
    )


def _projection_state(
    *,
    catalog_version: Any,
    shadow_run_id: Any,
    admission_run_id: Any,
    source_catalog_sha256: Any,
    shadow_completed_at: Any,
    project_contexts: list[dict[str, Any]],
    topics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA,
        "catalog_version": catalog_version,
        "source_shadow_run_id": shadow_run_id,
        "source_admission_run_id": admission_run_id,
        "source_catalog_sha256": source_catalog_sha256,
        "shadow_completed_at": shadow_completed_at,
        "project_contexts": project_contexts,
        "topics": [
            {
                "topic_id": item["spec"]["topic_id"],
                "path": item["spec"]["path"],
                "name": item["spec"]["name"],
                "description": item["spec"]["description"],
                "type": item["spec"]["type"],
                "entries": sorted(item["entries"], key=_entry_order),
            }
            for item in sorted(topics, key=lambda item: item["spec"]["topic_id"])
        ],
    }


def _render_index_body(topics: list[dict[str, Any]], conflict_count: int) -> str:
    lines = [
        "",
        "# Memory Index",
        "",
        "> Preview only：这些主题来自尚未审批的影子判断，不能作为正式生效记忆注入主 Agent。",
        "",
    ]
    for topic in topics:
        lines.append(
            f"- [{topic['name']}]({topic['path']}) — {topic['summary']} `status={topic['status']}`"
        )
    if conflict_count:
        lines.append(
            f"- [待解决冲突](conflicts/pending_conflicts.md) — {conflict_count} 项冲突，未解决前不选边。"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_topic(
    spec: Mapping[str, str],
    entries: list[dict[str, Any]],
    *,
    generation_id: str,
    updated_at: str,
    projects: Mapping[str, str],
) -> str:
    entries = sorted(entries, key=_entry_order)
    stable_keys = sorted({item["stable_key"] for item in entries})
    triggers = sorted(
        {trigger for item in entries for trigger in item.get("activation_triggers", [])},
        key=lambda item: (item.casefold(), item),
    )
    fields = {
        "schema_version": TOPIC_SCHEMA,
        "managed_by": MANAGED_BY,
        "preview_only": True,
        "generation_id": generation_id,
        "updated_at": updated_at,
        "topic_id": spec["topic_id"],
        "name": spec["name"],
        "description": spec["description"],
        "type": spec["type"],
        "status": _topic_status(entries),
        "stable_keys": stable_keys,
        "triggers": triggers,
        "entries": entries,
    }
    lines = [
        _front_matter(fields).rstrip(),
        "",
        f"# {spec['name']}",
        "",
        "> 这是影子判断生成的预览，不是已经审批生效的正式记忆。",
        "",
    ]
    memories = [item for item in entries if item["kind"] == "memory"]
    conflicts = [item for item in entries if item["kind"] == "conflict"]
    if memories:
        lines.extend(["## 候选规则", ""])
        for item in memories:
            lines.extend(
                [
                    f"### `{item['stable_key']}`",
                    "",
                    f"- 状态：`{item['status']}`",
                    f"- 范围：{_scope_text(item['scope'], projects)}",
                    f"- 规范值：`{item['normalized_value']}`",
                    f"- 内容：{item['statement']}",
                    f"- 证据：{item['evidence_count']} 条（原文保留在 SQLite 审计库）",
                    "",
                ]
            )
    if conflicts:
        lines.extend(["## 待解决冲突", ""])
        for item in conflicts:
            values = " / ".join(
                f"`{variant['normalized_value']}`" for variant in item["variants"]
            )
            lines.extend(
                [
                    f"### `{item['stable_key']}`",
                    "",
                    f"- 状态：`review_required`",
                    f"- 范围：{_scope_text(item['scope'], projects)}",
                    f"- 候选值：{values}",
                    "- 处理：未解决前不向主 Agent 提供任一版本。",
                    "",
                ]
            )
            for variant in item["variants"]:
                lines.append(
                    f"  - `{variant['normalized_value']}`：{variant['statement']}（证据 {variant['evidence_count']} 条）"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_conflicts(
    conflicts: list[dict[str, Any]],
    *,
    generation_id: str,
    projects: Mapping[str, str],
) -> str:
    fields = {
        "schema_version": CONFLICT_SCHEMA,
        "managed_by": MANAGED_BY,
        "preview_only": True,
        "generation_id": generation_id,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }
    lines = [
        _front_matter(fields).rstrip(),
        "",
        "# Pending Conflicts",
        "",
        "> 这是从主题文件派生的审核视图，不是另一份冲突真相来源。",
        "",
    ]
    if not conflicts:
        lines.append("当前没有待解决冲突。")
    else:
        for item in conflicts:
            values = " / ".join(
                f"`{variant['normalized_value']}`" for variant in item["variants"]
            )
            lines.extend(
                [
                    f"## `{item['stable_key']}`",
                    "",
                    f"- 范围：{_scope_text(item['scope'], projects)}",
                    f"- 候选值：{values}",
                    "- 状态：`review_required`",
                    "- 规则：未解决前不召回任一版本。",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _validate_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog.get("preview_only") is not True:
        raise MarkdownMemoryError("only a preview catalog can be rendered here")
    if catalog.get("formal_state_unchanged") is not True:
        raise MarkdownMemoryError("formal state was not proven unchanged")
    if catalog.get("status") != "complete":
        raise MarkdownMemoryError("catalog is not complete")
    for field in (
        "catalog_version",
        "shadow_run_id",
        "admission_run_id",
        "shadow_completed_at",
    ):
        _clean_text(catalog.get(field), field=f"catalog.{field}", maximum=200)
    if not isinstance(catalog.get("memories"), list) or not isinstance(
        catalog.get("conflicts"), list
    ):
        raise MarkdownMemoryError("catalog entries are malformed")
    if not isinstance(catalog.get("project_contexts"), list):
        raise MarkdownMemoryError("catalog project contexts are malformed")


def render_preview_memory(catalog: Mapping[str, Any]) -> dict[str, str]:
    """Render a deterministic flat Markdown preview without touching disk."""

    _validate_catalog(catalog)
    catalog_source_state = {
        "catalog_version": catalog.get("catalog_version"),
        "shadow_run_id": catalog.get("shadow_run_id"),
        "admission_run_id": catalog.get("admission_run_id"),
        "memories": catalog.get("memories"),
        "conflicts": catalog.get("conflicts"),
        "project_contexts": catalog.get("project_contexts", []),
    }
    source_catalog_sha256 = _sha256_object(catalog_source_state)
    projects: dict[str, str] = {}
    project_contexts = []
    for raw in catalog.get("project_contexts", []):
        if not isinstance(raw, Mapping):
            raise MarkdownMemoryError("project context must be an object")
        key = _clean_text(raw.get("project_key"), field="project_key", maximum=200)
        directory = _clean_text(raw.get("directory"), field="project_directory", maximum=1000)
        display_name = _clean_text(
            raw.get("display_name") or directory,
            field="project_display_name",
            maximum=300,
        )
        projects[key] = directory
        project_contexts.append(
            {"project_key": key, "directory": directory, "display_name": display_name}
        )
    project_contexts.sort(key=lambda item: item["project_key"])

    grouped: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    owner_by_key: dict[str, str] = {}
    for raw_memory in catalog.get("memories", []):
        if not isinstance(raw_memory, Mapping):
            raise MarkdownMemoryError("memory entry must be an object")
        entry = _memory_entry(raw_memory)
        spec = _topic_spec(entry["stable_key"])
        prior_owner = owner_by_key.setdefault(entry["stable_key"], spec["topic_id"])
        if prior_owner != spec["topic_id"]:
            raise MarkdownMemoryError("stable key has multiple topic owners")
        topic = grouped.setdefault(spec["topic_id"], {"spec": spec, "entries": []})
        topic["entries"].append(entry)
    for raw_conflict in catalog.get("conflicts", []):
        if not isinstance(raw_conflict, Mapping):
            raise MarkdownMemoryError("conflict entry must be an object")
        entry = _conflict_entry(raw_conflict)
        conflicts.append(entry)
        spec = _topic_spec(entry["stable_key"])
        prior_owner = owner_by_key.setdefault(entry["stable_key"], spec["topic_id"])
        if prior_owner != spec["topic_id"]:
            raise MarkdownMemoryError("stable key has multiple topic owners")
        topic = grouped.setdefault(spec["topic_id"], {"spec": spec, "entries": []})
        topic["entries"].append(entry)
    conflicts.sort(key=lambda item: (item["stable_key"], _json(item["scope"])))

    normalized_topics = [
        {"spec": grouped[topic_id]["spec"], "entries": grouped[topic_id]["entries"]}
        for topic_id in sorted(grouped)
    ]
    projection = _projection_state(
        catalog_version=catalog.get("catalog_version"),
        shadow_run_id=catalog.get("shadow_run_id"),
        admission_run_id=catalog.get("admission_run_id"),
        source_catalog_sha256=source_catalog_sha256,
        shadow_completed_at=catalog.get("shadow_completed_at"),
        project_contexts=project_contexts,
        topics=normalized_topics,
    )
    source_state_sha256 = _sha256_object(projection)
    generation_id = "preview_" + source_state_sha256[:20]

    files: dict[str, str] = {}
    topic_index = []
    for topic_record in normalized_topics:
        spec = topic_record["spec"]
        entries = topic_record["entries"]
        content = _render_topic(
            spec,
            entries,
            generation_id=generation_id,
            updated_at=catalog["shadow_completed_at"],
            projects=projects,
        )
        if len(content.encode("utf-8")) > MAX_TOPIC_BYTES:
            raise MarkdownMemoryError(f"topic is too large: {spec['path']}")
        files[spec["path"]] = content
        topic_index.append(
            {
                "topic_id": spec["topic_id"],
                "path": spec["path"],
                "name": spec["name"],
                "description": spec["description"],
                "type": spec["type"],
                "status": _topic_status(entries),
                "stable_keys": sorted({item["stable_key"] for item in entries}),
                "triggers": sorted(
                    {
                        trigger
                        for item in entries
                        for trigger in item.get("activation_triggers", [])
                    },
                    key=lambda item: (item.casefold(), item),
                ),
                "blockers": [
                    {
                        "stable_key": item["stable_key"],
                        "scope": item["scope"],
                        "status": item["status"],
                    }
                    for item in entries
                    if item["kind"] == "conflict"
                    or item.get("status") == "context_required"
                ],
            "summary": _topic_summary(spec, entries, projects),
            }
        )

    conflict_path = "conflicts/pending_conflicts.md"
    files[conflict_path] = _render_conflicts(
        conflicts,
        generation_id=generation_id,
        projects=projects,
    )
    managed_files = {
        path: _sha256_bytes(content.encode("utf-8"))
        for path, content in sorted(files.items())
    }
    for topic in topic_index:
        topic["sha256"] = managed_files[topic["path"]]
    index_fields = {
        "schema_version": INDEX_SCHEMA,
        "managed_by": MANAGED_BY,
        "preview_only": True,
        "generation_id": generation_id,
        "catalog_version": catalog.get("catalog_version"),
        "source_shadow_run_id": catalog.get("shadow_run_id"),
        "source_admission_run_id": catalog.get("admission_run_id"),
        "source_catalog_sha256": source_catalog_sha256,
        "source_state_sha256": source_state_sha256,
        "shadow_completed_at": catalog.get("shadow_completed_at"),
        "topic_count": len(topic_index),
        "candidate_memory_count": len(catalog.get("memories", [])),
        "conflict_count": len(conflicts),
        "project_contexts": project_contexts,
        "managed_files": managed_files,
        "topics": topic_index,
    }
    index = _front_matter(index_fields) + _render_index_body(
        topic_index, len(conflicts)
    )
    if len(index.encode("utf-8")) > MAX_INDEX_BYTES or len(index.splitlines()) > 200:
        raise MarkdownMemoryError("MEMORY.md exceeds the preview index budget")
    files["MEMORY.md"] = index
    return dict(sorted(files.items(), key=lambda item: (item[0] == "MEMORY.md", item[0])))


def _safe_relative_path(raw: str) -> PurePosixPath:
    if not isinstance(raw, str) or "\\" in raw:
        raise MarkdownMemoryError("managed path must use safe relative POSIX syntax")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise MarkdownMemoryError("managed path escapes the memory directory")
    if path.suffix.casefold() != ".md" or raw.casefold() == "memory.md":
        raise MarkdownMemoryError("managed path is invalid")
    return path


def _read_utf8(path: Path, *, maximum: int) -> tuple[str, bytes]:
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise MarkdownMemoryError(f"managed path is a link: {path.name}")
    if not path.is_file():
        raise MarkdownMemoryError(f"managed file is missing: {path.name}")
    data = path.read_bytes()
    if len(data) > maximum:
        raise MarkdownMemoryError(f"managed file is too large: {path.name}")
    try:
        return data.decode("utf-8"), data
    except UnicodeDecodeError as error:
        raise MarkdownMemoryError(f"managed file is not valid UTF-8: {path.name}") from error


_INDEX_KEYS = {
    "schema_version",
    "managed_by",
    "preview_only",
    "generation_id",
    "catalog_version",
    "source_shadow_run_id",
    "source_admission_run_id",
    "source_catalog_sha256",
    "source_state_sha256",
    "shadow_completed_at",
    "topic_count",
    "candidate_memory_count",
    "conflict_count",
    "project_contexts",
    "managed_files",
    "topics",
}
_TOPIC_KEYS = {
    "schema_version",
    "managed_by",
    "preview_only",
    "generation_id",
    "updated_at",
    "topic_id",
    "name",
    "description",
    "type",
    "status",
    "stable_keys",
    "triggers",
    "entries",
}
_CONFLICT_KEYS = {
    "schema_version",
    "managed_by",
    "preview_only",
    "generation_id",
    "conflict_count",
    "conflicts",
}


def _assert_clean_string(value: Any, *, source: str, maximum: int = 500) -> str:
    cleaned = _clean_text(value, field=source, maximum=maximum)
    if cleaned != value:
        raise MarkdownMemoryError(f"{source} is not in canonical form")
    return cleaned


def _validate_string_list(
    value: Any, *, source: str, maximum: int = 200, casefold_sort: bool = False
) -> list[str]:
    items = _assert_string_list(value, source=source)
    for item in items:
        _assert_clean_string(item, source=source, maximum=maximum)
    expected = sorted(items, key=lambda item: (item.casefold(), item)) if casefold_sort else sorted(items)
    if items != expected:
        raise MarkdownMemoryError(f"{source} is not deterministically ordered")
    return items


def _validate_memory_header_entry(entry: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise MarkdownMemoryError(f"{source} is not an object")
    _assert_exact_keys(entry, required=_MEMORY_ENTRY_KEYS, source=source)
    if entry.get("kind") != "memory":
        raise MarkdownMemoryError(f"{source} has the wrong kind")
    _assert_clean_string(entry.get("entry_id"), source=f"{source}.entry_id", maximum=200)
    stable_key = _assert_clean_string(
        entry.get("stable_key"), source=f"{source}.stable_key", maximum=200
    )
    _assert_clean_string(
        entry.get("memory_type"), source=f"{source}.memory_type", maximum=100
    )
    _assert_clean_string(
        entry.get("normalized_value"),
        source=f"{source}.normalized_value",
        maximum=200,
    )
    _assert_clean_string(
        entry.get("statement"), source=f"{source}.statement", maximum=500
    )
    scope = _validate_scope(entry.get("scope"), source=f"{source}.scope")
    if entry.get("scope_rank") != _scope_rank(scope):
        raise MarkdownMemoryError(f"{source}.scope_rank disagrees with scope")
    expected_status = "context_required" if scope.get("conditions") else "preview_active"
    if entry.get("status") != expected_status:
        raise MarkdownMemoryError(f"{source}.status is invalid")
    _validate_string_list(
        entry.get("activation_triggers"),
        source=f"{source}.activation_triggers",
        maximum=120,
        casefold_sort=True,
    )
    _validate_string_list(
        entry.get("source_candidate_keys"),
        source=f"{source}.source_candidate_keys",
        maximum=200,
    )
    if not entry["source_candidate_keys"]:
        raise MarkdownMemoryError(f"{source} has no candidate identifiers")
    if _assert_nonnegative_int(entry.get("evidence_count"), source=f"{source}.evidence_count") < 1:
        raise MarkdownMemoryError(f"{source} has no evidence")
    _assert_no_secrets(entry)
    return entry


def _validate_conflict_header_entry(entry: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise MarkdownMemoryError(f"{source} is not an object")
    _assert_exact_keys(entry, required=_CONFLICT_ENTRY_KEYS, source=source)
    if (
        entry.get("kind") != "conflict"
        or entry.get("status") != "review_required"
        or entry.get("reason") != "opposite_or_incompatible_values_in_same_scope"
    ):
        raise MarkdownMemoryError(f"{source} has invalid conflict state")
    _assert_clean_string(entry.get("entry_id"), source=f"{source}.entry_id", maximum=200)
    stable_key = _assert_clean_string(
        entry.get("stable_key"), source=f"{source}.stable_key", maximum=200
    )
    scope = _validate_scope(entry.get("scope"), source=f"{source}.scope")
    if entry.get("scope_rank") != _scope_rank(scope):
        raise MarkdownMemoryError(f"{source}.scope_rank disagrees with scope")
    _validate_string_list(
        entry.get("activation_triggers"),
        source=f"{source}.activation_triggers",
        maximum=120,
        casefold_sort=True,
    )
    variants = entry.get("variants")
    if not isinstance(variants, list) or not variants:
        raise MarkdownMemoryError(f"{source}.variants must not be empty")
    values: set[str] = set()
    for index, variant in enumerate(variants):
        variant_source = f"{source}.variants[{index}]"
        if not isinstance(variant, dict):
            raise MarkdownMemoryError(f"{variant_source} is not an object")
        _assert_exact_keys(
            variant, required=_CONFLICT_VARIANT_KEYS, source=variant_source
        )
        normalized_value = _assert_clean_string(
            variant.get("normalized_value"),
            source=f"{variant_source}.normalized_value",
            maximum=200,
        )
        if normalized_value in values:
            raise MarkdownMemoryError(f"{source} repeats a conflict value")
        values.add(normalized_value)
        _assert_clean_string(
            variant.get("statement"),
            source=f"{variant_source}.statement",
            maximum=500,
        )
        memory_types = _validate_string_list(
            variant.get("memory_types"),
            source=f"{variant_source}.memory_types",
            maximum=100,
        )
        if not memory_types:
            raise MarkdownMemoryError(f"{variant_source} has no memory type")
        candidate_keys = _validate_string_list(
            variant.get("candidate_keys"),
            source=f"{variant_source}.candidate_keys",
            maximum=200,
        )
        if not candidate_keys:
            raise MarkdownMemoryError(f"{variant_source} has no candidate identifiers")
        if _assert_nonnegative_int(
            variant.get("evidence_count"), source=f"{variant_source}.evidence_count"
        ) < 1:
            raise MarkdownMemoryError(f"{variant_source} has no evidence")
    if variants != sorted(
        variants, key=lambda item: (item["normalized_value"], item["statement"])
    ):
        raise MarkdownMemoryError(f"{source}.variants are not deterministically ordered")
    if len(variants) < 2 and len(variants[0]["memory_types"]) < 2:
        raise MarkdownMemoryError(f"{source} has no value or type disagreement")
    identity = {"stable_key": stable_key, "scope": scope, "variants": variants}
    expected_id = "preview_conflict_" + _sha256_object(identity)[:24]
    if entry["entry_id"] != expected_id:
        raise MarkdownMemoryError(f"{source}.entry_id is inconsistent")
    _assert_no_secrets(entry)
    return entry


def _conflict_order(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("stable_key")),
        _json(entry.get("scope") or {}),
        str(entry.get("entry_id")),
    )


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and bool(path.is_junction())
    )


def _assert_safe_managed_path(root: Path, candidate: Path) -> None:
    if not root.is_dir() or _is_link(root):
        raise MarkdownMemoryError("memory directory must remain a real directory")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise MarkdownMemoryError("managed path escapes the memory directory") from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() and _is_link(cursor):
            raise MarkdownMemoryError(f"managed path contains a link: {candidate.name}")
    resolved = candidate.resolve()
    if root not in resolved.parents:
        raise MarkdownMemoryError("managed path escapes the memory directory")


def validate_preview_memory(memory_directory: Path) -> dict[str, Any]:
    """Validate every managed file and reconstruct the preview catalog."""

    requested_root = memory_directory.absolute()
    if requested_root.exists() and _is_link(requested_root):
        raise MarkdownMemoryError("memory directory cannot be a link")
    root = requested_root.resolve()
    index_path = root / "MEMORY.md"
    index_text, _index_bytes = _read_utf8(index_path, maximum=MAX_INDEX_BYTES)
    index, _body = _parse_front_matter(index_text, source="MEMORY.md")
    _assert_exact_keys(index, required=_INDEX_KEYS, source="MEMORY.md")
    if (
        index["schema_version"] != INDEX_SCHEMA
        or index["managed_by"] != MANAGED_BY
        or index["preview_only"] is not True
    ):
        raise MarkdownMemoryError("MEMORY.md is not a supported preview index")
    for field in (
        "generation_id",
        "catalog_version",
        "source_shadow_run_id",
        "source_admission_run_id",
        "source_catalog_sha256",
        "source_state_sha256",
    ):
        _assert_clean_string(index.get(field), source=f"MEMORY.md.{field}", maximum=200)
    source_hash = index["source_state_sha256"]
    if not _SHA256.fullmatch(source_hash) or not _SHA256.fullmatch(
        index["source_catalog_sha256"]
    ):
        raise MarkdownMemoryError("MEMORY.md source hash is invalid")
    if index["generation_id"] != "preview_" + source_hash[:20]:
        raise MarkdownMemoryError("MEMORY.md generation does not match its source hash")
    if index["shadow_completed_at"] is not None:
        _assert_clean_string(
            index["shadow_completed_at"],
            source="MEMORY.md.shadow_completed_at",
            maximum=100,
        )
    topic_count = _assert_nonnegative_int(
        index["topic_count"], source="MEMORY.md.topic_count"
    )
    candidate_memory_count = _assert_nonnegative_int(
        index["candidate_memory_count"], source="MEMORY.md.candidate_memory_count"
    )
    conflict_count = _assert_nonnegative_int(
        index["conflict_count"], source="MEMORY.md.conflict_count"
    )

    project_contexts = index["project_contexts"]
    if not isinstance(project_contexts, list):
        raise MarkdownMemoryError("MEMORY.md project contexts are malformed")
    project_keys: set[str] = set()
    project_directories: set[str] = set()
    for position, context in enumerate(project_contexts):
        source = f"MEMORY.md.project_contexts[{position}]"
        if not isinstance(context, dict):
            raise MarkdownMemoryError(f"{source} is not an object")
        _assert_exact_keys(context, required=_PROJECT_CONTEXT_KEYS, source=source)
        project_key = _assert_clean_string(
            context.get("project_key"), source=f"{source}.project_key", maximum=200
        )
        directory = _assert_clean_string(
            context.get("directory"), source=f"{source}.directory", maximum=1000
        )
        _assert_clean_string(
            context.get("display_name"), source=f"{source}.display_name", maximum=300
        )
        if not Path(directory).is_absolute():
            raise MarkdownMemoryError(f"{source}.directory must be absolute")
        folded_directory = str(Path(directory).resolve()).casefold()
        if project_key in project_keys or folded_directory in project_directories:
            raise MarkdownMemoryError("MEMORY.md repeats a project context")
        project_keys.add(project_key)
        project_directories.add(folded_directory)
    if project_contexts != sorted(project_contexts, key=lambda item: item["project_key"]):
        raise MarkdownMemoryError("MEMORY.md project contexts are not ordered")
    projects = {item["project_key"]: item["directory"] for item in project_contexts}

    managed = index["managed_files"]
    topics = index["topics"]
    if not isinstance(managed, dict) or not isinstance(topics, list):
        raise MarkdownMemoryError("MEMORY.md manifest is malformed")
    if (
        len(managed) > MAX_MANAGED_FILES
        or len(topics) != topic_count
        or len(managed) != topic_count + 1
        or "conflicts/pending_conflicts.md" not in managed
    ):
        raise MarkdownMemoryError("MEMORY.md manifest count is invalid")
    casefold_paths: set[str] = set()
    parsed_topics: dict[str, dict[str, Any]] = {}
    topic_entries: list[dict[str, Any]] = []
    conflict_entries: list[dict[str, Any]] = []
    conflict_text: str | None = None
    owner_by_key: dict[str, str] = {}
    entry_ids: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for raw_path, expected_hash in sorted(managed.items()):
        relative = _safe_relative_path(raw_path)
        folded = raw_path.casefold()
        if folded in casefold_paths:
            raise MarkdownMemoryError("managed paths collide case-insensitively")
        casefold_paths.add(folded)
        candidate = root.joinpath(*relative.parts)
        _assert_safe_managed_path(root, candidate)
        text, data = _read_utf8(candidate, maximum=MAX_TOPIC_BYTES)
        if (
            not isinstance(expected_hash, str)
            or not _SHA256.fullmatch(expected_hash)
            or _sha256_bytes(data) != expected_hash
        ):
            raise MarkdownMemoryError(f"managed file hash mismatch: {raw_path}")
        header, _content = _parse_front_matter(text, source=raw_path)
        if raw_path == "conflicts/pending_conflicts.md":
            conflict_text = text
            _assert_exact_keys(header, required=_CONFLICT_KEYS, source=raw_path)
            if header["schema_version"] != CONFLICT_SCHEMA:
                raise MarkdownMemoryError("unsupported conflict schema")
            conflict_entries = header["conflicts"]
            if not isinstance(conflict_entries, list):
                raise MarkdownMemoryError("conflict entries are malformed")
            if _assert_nonnegative_int(
                header["conflict_count"], source=f"{raw_path}.conflict_count"
            ) != len(conflict_entries):
                raise MarkdownMemoryError("conflict file count is inconsistent")
        else:
            _assert_exact_keys(header, required=_TOPIC_KEYS, source=raw_path)
            if header["schema_version"] != TOPIC_SCHEMA:
                raise MarkdownMemoryError("unsupported topic schema")
            if not isinstance(header["entries"], list) or not isinstance(
                header["stable_keys"], list
            ):
                raise MarkdownMemoryError("topic entries are malformed")
            topic_id = _assert_clean_string(
                header.get("topic_id"), source=f"{raw_path}.topic_id", maximum=200
            )
            for field, maximum in (
                ("name", 300),
                ("description", 500),
                ("type", 100),
                ("status", 100),
                ("updated_at", 100),
            ):
                _assert_clean_string(
                    header.get(field), source=f"{raw_path}.{field}", maximum=maximum
                )
            _validate_string_list(
                header["stable_keys"], source=f"{raw_path}.stable_keys", maximum=200
            )
            _validate_string_list(
                header["triggers"],
                source=f"{raw_path}.triggers",
                maximum=120,
                casefold_sort=True,
            )
            validated_entries = []
            for position, raw_entry in enumerate(header["entries"]):
                entry_source = f"{raw_path}.entries[{position}]"
                if isinstance(raw_entry, dict) and raw_entry.get("kind") == "memory":
                    entry = _validate_memory_header_entry(raw_entry, source=entry_source)
                elif isinstance(raw_entry, dict) and raw_entry.get("kind") == "conflict":
                    entry = _validate_conflict_header_entry(raw_entry, source=entry_source)
                else:
                    raise MarkdownMemoryError(f"{entry_source} has an unknown kind")
                if entry["entry_id"] in entry_ids:
                    raise MarkdownMemoryError("entry identifier is duplicated")
                entry_ids.add(entry["entry_id"])
                identity = (entry["stable_key"], _json(entry["scope"]))
                if identity in identities:
                    raise MarkdownMemoryError("stable key and scope were not consolidated")
                identities.add(identity)
                spec = _topic_spec(entry["stable_key"])
                if spec["topic_id"] != topic_id or spec["path"] != raw_path:
                    raise MarkdownMemoryError("topic does not own one of its stable keys")
                validated_entries.append(entry)
            if not validated_entries:
                raise MarkdownMemoryError("managed topic cannot be empty")
            expected_entries = sorted(
                validated_entries,
                key=lambda item: (
                    item["stable_key"],
                    int(item.get("scope_rank", -1)),
                    _json(item.get("scope") or {}),
                    item["kind"],
                    item.get("normalized_value", ""),
                ),
            )
            if validated_entries != expected_entries:
                raise MarkdownMemoryError("topic entries are not deterministically ordered")
            actual_stable_keys = sorted({item["stable_key"] for item in validated_entries})
            actual_triggers = sorted(
                {
                    trigger
                    for item in validated_entries
                    for trigger in item["activation_triggers"]
                },
                key=lambda item: (item.casefold(), item),
            )
            if header["stable_keys"] != actual_stable_keys or header["triggers"] != actual_triggers:
                raise MarkdownMemoryError("topic routing header disagrees with its entries")
            if header["status"] != _topic_status(validated_entries):
                raise MarkdownMemoryError("topic status disagrees with its entries")
            if header["updated_at"] != index["shadow_completed_at"]:
                raise MarkdownMemoryError("topic update time disagrees with source run")
            canonical_topic = _render_topic(
                _topic_spec(validated_entries[0]["stable_key"]),
                validated_entries,
                generation_id=index["generation_id"],
                updated_at=index["shadow_completed_at"],
                projects=projects,
            )
            if text != canonical_topic:
                raise MarkdownMemoryError("topic body or serialization is not canonical")
            parsed_topics[raw_path] = {"header": header, "entries": validated_entries}
            for stable_key in actual_stable_keys:
                owner = owner_by_key.setdefault(stable_key, header["topic_id"])
                if owner != header["topic_id"]:
                    raise MarkdownMemoryError("stable key has multiple topic owners")
            topic_entries.extend(validated_entries)
        if (
            header.get("managed_by") != MANAGED_BY
            or header.get("preview_only") is not True
            or header.get("generation_id") != index["generation_id"]
        ):
            raise MarkdownMemoryError(f"managed header identity mismatch: {raw_path}")

    if topics != sorted(topics, key=lambda item: item.get("topic_id", "") if isinstance(item, dict) else ""):
        raise MarkdownMemoryError("topic index is not deterministically ordered")
    manifest_topic_paths: set[str] = set()
    for position, topic in enumerate(topics):
        source = f"MEMORY.md.topics[{position}]"
        if not isinstance(topic, dict):
            raise MarkdownMemoryError(f"{source} is not an object")
        _assert_exact_keys(topic, required=_TOPIC_INDEX_KEYS, source=source)
        raw_path = topic.get("path")
        relative = _safe_relative_path(raw_path)
        if raw_path == "conflicts/pending_conflicts.md":
            raise MarkdownMemoryError("conflict view cannot be indexed as a topic")
        if raw_path in manifest_topic_paths:
            raise MarkdownMemoryError("topic path is duplicated")
        manifest_topic_paths.add(raw_path)
        if raw_path not in parsed_topics:
            raise MarkdownMemoryError("topic index references an unknown file")
        header = parsed_topics[raw_path]["header"]
        entries = parsed_topics[raw_path]["entries"]
        for field in ("topic_id", "name", "description", "type", "status", "stable_keys", "triggers"):
            if topic.get(field) != header[field]:
                raise MarkdownMemoryError(f"{source}.{field} disagrees with topic header")
        if topic.get("sha256") != managed.get(raw_path):
            raise MarkdownMemoryError("topic index hash is invalid")
        expected_blockers = [
            {
                "stable_key": item["stable_key"],
                "scope": item["scope"],
                "status": item["status"],
            }
            for item in entries
            if item["kind"] == "conflict" or item.get("status") == "context_required"
        ]
        if topic.get("blockers") != expected_blockers:
            raise MarkdownMemoryError("topic blockers disagree with topic entries")
        spec = _topic_spec(entries[0]["stable_key"]) if entries else None
        if spec is None or topic.get("summary") != _topic_summary(spec, entries, projects):
            raise MarkdownMemoryError("topic summary is inconsistent")
        _assert_no_secrets(topic)

    actual_topic_paths = set(managed) - {"conflicts/pending_conflicts.md"}
    if manifest_topic_paths != actual_topic_paths:
        raise MarkdownMemoryError("topic index and managed files disagree")

    memories = []
    conflicts = []
    topic_conflicts = []
    for entry in topic_entries:
        if entry.get("kind") == "memory":
            memories.append(
                {
                    "preview_memory_id": entry["entry_id"],
                    "stable_key": entry["stable_key"],
                    "memory_type": entry["memory_type"],
                    "scope": entry["scope"],
                    "scope_rank": entry["scope_rank"],
                    "normalized_value": entry["normalized_value"],
                    "statement": entry["statement"],
                    "activation_triggers": entry["activation_triggers"],
                    "evidence_count": entry["evidence_count"],
                    "evidence": [],
                    "source_candidate_keys": entry["source_candidate_keys"],
                    "status": "preview",
                }
            )
        elif entry.get("kind") == "conflict":
            converted = {
                "stable_key": entry["stable_key"],
                "scope": entry["scope"],
                "scope_rank": entry["scope_rank"],
                "status": "review_required",
                "reason": entry["reason"],
                "activation_triggers": entry["activation_triggers"],
                "variants": [
                    {
                        "value": variant["normalized_value"],
                        "memory_types": variant["memory_types"],
                        "statements": [variant["statement"]],
                        "candidate_keys": variant["candidate_keys"],
                        "evidence_count": variant["evidence_count"],
                    }
                    for variant in entry["variants"]
                ],
            }
            conflicts.append(converted)
            topic_conflicts.append(entry)
        else:
            raise MarkdownMemoryError("unknown topic entry kind")
    validated_conflict_entries = [
        _validate_conflict_header_entry(item, source=f"pending conflict {position}")
        for position, item in enumerate(conflict_entries)
    ]
    if sorted(topic_conflicts, key=_conflict_order) != sorted(
        validated_conflict_entries, key=_conflict_order
    ):
        raise MarkdownMemoryError("derived conflict file disagrees with topic truth")
    if conflict_text != _render_conflicts(
        sorted(validated_conflict_entries, key=_conflict_order),
        generation_id=index["generation_id"],
        projects=projects,
    ):
        raise MarkdownMemoryError("conflict view body or serialization is not canonical")
    if len(conflicts) != conflict_count or len(memories) != candidate_memory_count:
        raise MarkdownMemoryError("index entry counts are inconsistent")
    referenced_projects = {
        entry["scope"].get("project_key")
        for entry in topic_entries
        if entry["scope"].get("level") == "project"
    }
    if not referenced_projects.issubset(project_keys):
        raise MarkdownMemoryError("a project-scoped memory has no project context")
    projection_topics = []
    for topic in topics:
        header = parsed_topics[topic["path"]]["header"]
        projection_topics.append(
            {
                "spec": {
                    "topic_id": header["topic_id"],
                    "path": topic["path"],
                    "name": header["name"],
                    "description": header["description"],
                    "type": header["type"],
                },
                "entries": parsed_topics[topic["path"]]["entries"],
            }
        )
    reconstructed_state = _projection_state(
        catalog_version=index["catalog_version"],
        shadow_run_id=index["source_shadow_run_id"],
        admission_run_id=index["source_admission_run_id"],
        source_catalog_sha256=index["source_catalog_sha256"],
        shadow_completed_at=index["shadow_completed_at"],
        project_contexts=project_contexts,
        topics=projection_topics,
    )
    if _sha256_object(reconstructed_state) != source_hash:
        raise MarkdownMemoryError("MEMORY.md semantic projection hash mismatch")
    if _body != _render_index_body(topics, conflict_count):
        raise MarkdownMemoryError("MEMORY.md body is not the canonical topic index")
    return {
        "status": "complete",
        "preview_only": True,
        "generation_id": index["generation_id"],
        "source_state_sha256": index["source_state_sha256"],
        "shadow_run_id": index["source_shadow_run_id"],
        "admission_run_id": index["source_admission_run_id"],
        "catalog_version": index["catalog_version"],
        "project_contexts": project_contexts,
        "topic_count": topic_count,
        "managed_file_count": len(managed),
        "memories": memories,
        "conflicts": conflicts,
        "same_scope_conflict_count": len(conflicts),
        "unique_memory_count": len(memories),
        "formal_state_unchanged": True,
        "managed_files": managed,
    }


def _existing_is_managed(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_symlink() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        header, _body = _parse_front_matter(text, source=path.name)
    except (OSError, UnicodeError, MarkdownMemoryError):
        return False
    return header.get("managed_by") == MANAGED_BY and header.get("preview_only") is True


def write_preview_memory(
    catalog: Mapping[str, Any], memory_directory: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Atomically replace managed files, committing MEMORY.md last.

    Unknown files are never removed or overwritten.  Readers validate the
    hashes committed in MEMORY.md, so a crash can make the preview temporarily
    unavailable but can never make a mixed generation appear valid.
    """

    files = render_preview_memory(catalog)
    requested_root = memory_directory.absolute()
    if requested_root.exists() and _is_link(requested_root):
        raise MarkdownMemoryError("memory directory cannot be a link")
    root = requested_root.resolve()
    if root.parent == root:
        raise MarkdownMemoryError("memory directory cannot be a filesystem root")
    if dry_run:
        index, _body = _parse_front_matter(files["MEMORY.md"], source="MEMORY.md")
        return {
            "status": "complete",
            "preview_only": True,
            "dry_run": True,
            "memory_directory": str(root),
            "generation_id": index["generation_id"],
            "planned_files": list(files),
            "written_files": [],
            "unchanged_files": [],
        }
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise MarkdownMemoryError("memory directory must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.memory-lab.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise MarkdownMemoryError("another Markdown memory writer holds the lock") from error
    stage: Path | None = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.close(lock_fd)
        previous_managed: dict[str, str] = {}
        previous_index = root / "MEMORY.md"
        if previous_index.exists():
            try:
                previous = validate_preview_memory(root)
            except MarkdownMemoryError:
                # A complete new projection may repair an interrupted or old
                # preview.  Stale-file deletion is disabled unless the prior
                # manifest was itself fully trustworthy.
                previous_managed = {}
            else:
                previous_managed = dict(previous["managed_files"])
        for relative in files:
            destination = root.joinpath(*PurePosixPath(relative).parts)
            _assert_safe_managed_path(root, destination)
            if not _existing_is_managed(destination):
                raise MarkdownMemoryError(f"refusing to overwrite unmanaged file: {relative}")
        stage = Path(tempfile.mkdtemp(prefix=".memory-stage-", dir=root.parent))
        for relative, content in files.items():
            destination = stage.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8", newline="\n")
        validate_preview_memory(stage)
        removed: list[str] = []
        new_managed = set(files) - {"MEMORY.md"}
        for relative in sorted(set(previous_managed) - new_managed):
            stale = root.joinpath(*_safe_relative_path(relative).parts)
            _assert_safe_managed_path(root, stale)
            if not stale.exists():
                continue
            if (
                not _existing_is_managed(stale)
                or _sha256_bytes(stale.read_bytes()) != previous_managed[relative]
            ):
                raise MarkdownMemoryError(
                    f"refusing to remove a changed or unmanaged stale file: {relative}"
                )
            stale.unlink()
            removed.append(relative)
        written: list[str] = []
        unchanged: list[str] = []
        commit_order = [path for path in files if path != "MEMORY.md"] + ["MEMORY.md"]
        for relative in commit_order:
            source = stage.joinpath(*PurePosixPath(relative).parts)
            destination = root.joinpath(*PurePosixPath(relative).parts)
            _assert_safe_managed_path(root, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.read_bytes() == source.read_bytes():
                unchanged.append(relative)
                continue
            os.replace(source, destination)
            written.append(relative)
        result = validate_preview_memory(root)
        return {
            "status": "complete",
            "preview_only": True,
            "dry_run": False,
            "memory_directory": str(root),
            "generation_id": result["generation_id"],
            "source_state_sha256": result["source_state_sha256"],
            "topic_count": result["topic_count"],
            "candidate_memory_count": result["unique_memory_count"],
            "conflict_count": result["same_scope_conflict_count"],
            "written_files": written,
            "unchanged_files": unchanged,
            "removed_files": removed,
            "managed_files": sorted(result["managed_files"]),
        }
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass
        if stage is not None and stage.exists() and stage.parent == root.parent:
            shutil.rmtree(stage)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def run_markdown_preview(
    lab_database: Path,
    memory_directory: Path,
    *,
    shadow_run_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    catalog = run_catalog(lab_database, shadow_run_id=shadow_run_id)
    return write_preview_memory(catalog, memory_directory, dry_run=dry_run)


def _project_key_for_directory(
    contexts: Iterable[Mapping[str, Any]], project_directory: Path | None
) -> tuple[str | None, str | None, bool]:
    if project_directory is None:
        return None, None, False
    requested = str(project_directory.resolve())
    matches = [
        item
        for item in contexts
        if isinstance(item.get("directory"), str)
        and str(Path(item["directory"]).resolve()).casefold() == requested.casefold()
    ]
    if len(matches) != 1:
        return None, requested, False
    return str(matches[0]["project_key"]), requested, True


def run_markdown_recall(
    memory_directory: Path,
    query: str,
    *,
    project_directory: Path | None = None,
    domain: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Recall from the validated Markdown projection, never from shadow DB rows."""

    loaded = validate_preview_memory(memory_directory)
    project_key, resolved_directory, project_known = _project_key_for_directory(
        loaded["project_contexts"], project_directory
    )
    recalled = recall_from_catalog(
        loaded,
        query,
        project_key=project_key,
        project_context_known=project_known,
        domain=domain,
        limit=limit,
    )
    recalled["unresolved_conflicts"] = [
        {
            "stable_key": item["stable_key"],
            "scope": item["scope"],
            "status": "review_required",
            "reason": item["reason"],
            "variant_count": len(item.get("variants", [])),
            "score": item.get("score"),
            "match_reasons": item.get("match_reasons", []),
        }
        for item in recalled["unresolved_conflicts"]
    ]
    return {
        "status": "complete",
        "preview_only": True,
        "source": "markdown_memory",
        "memory_directory": str(memory_directory.resolve()),
        "generation_id": loaded["generation_id"],
        "source_state_sha256": loaded["source_state_sha256"],
        "shadow_run_id": loaded["shadow_run_id"],
        "conflict_review_file": str(
            memory_directory.resolve() / "conflicts" / "pending_conflicts.md"
        ),
        "project_directory": resolved_directory,
        "project_context_known": project_known,
        "topic_headers_scanned": loaded["topic_count"],
        "catalog_summary": {
            "unique_memory_count": loaded["unique_memory_count"],
            "same_scope_conflict_count": loaded["same_scope_conflict_count"],
        },
        **recalled,
    }
