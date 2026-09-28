"""Strict validation for the multi-turn memory evaluation-set contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .turn_evidence import build_turn_evidence


SCHEMA_VERSION = "memory-multiturn-scenarios-v1"
SPLITS = {"development", "holdout", "stress"}
CHAIN_STATES = {
    "none": {"not_applicable"},
    "compilation": {
        "failure_observed", "cause_identified", "fix_applied",
        "compile_verified", "fix_rejected",
    },
    "runtime": {
        "symptom_observed", "cause_identified", "fix_applied",
        "waiting_user_validation", "user_verified", "fix_rejected",
    },
}
KNOWN_TOOLS = {
    "list_open_memory_chains", "read_session_turn_evidence",
    "list_previous_session_turns", "search_related_memories",
    "read_memory_detail", "read_topic_detail", "list_topic_memories",
    "read_memory_history", "list_memory_conflicts", "expand_pointer",
    "get_current_turn_evidence_summary",
}
WRITE_ACTIONS = {"create_memory", "update_memory", "merge_memory"}
FORMAL_MEMORY_TYPES = {
    "user_preference",
    "compile_failure_experience",
    "runtime_failure_experience",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MultiturnEvalContractError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultiturnEvalContractError([f"无法读取 {path}: {exc}"]) from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"cases.jsonl 第 {number} 行无效: {exc}")
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            errors.append(f"cases.jsonl 第 {number} 行必须是对象")
    if errors:
        raise MultiturnEvalContractError(errors)
    return rows


def validate_multiturn_evalset(
    evalset_path: Path,
    *,
    lab_database: Path | None = None,
    verify_evidence_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(evalset_path)
    manifest = _json(root / "manifest.json")
    document = _json(root / "scenarios.json")
    fixtures_doc = _json(root / "fixtures" / "initial_states.json")
    rubric = _json(root / "rubric.json")
    cases = _jsonl(root / "cases.jsonl")
    errors: list[str] = []

    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append("scenarios.json schema_version 不正确")
    if manifest.get("schema_version") != "memory-multiturn-manifest-v1":
        errors.append("manifest.json schema_version 不正确")
    if fixtures_doc.get("schema_version") != "memory-multiturn-fixtures-v1":
        errors.append("initial_states.json schema_version 不正确")
    if not str(rubric.get("schema_version") or ""):
        errors.append("rubric.json 缺少 schema_version")

    fixture_ids = {
        str(item.get("fixture_id"))
        for item in fixtures_doc.get("fixtures", [])
        if isinstance(item, Mapping) and item.get("fixture_id")
    }
    case_ids: set[str] = set()
    case_by_turn: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("case_id") or "")
        turn_id = str(case.get("turn_id") or "")
        if not case_id or case_id in case_ids:
            errors.append(f"case_id 缺失或重复: {case_id}")
        case_ids.add(case_id)
        if not turn_id or turn_id in case_by_turn:
            errors.append(f"case turn_id 缺失或重复: {turn_id}")
        case_by_turn[turn_id] = case

    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list):
        raise MultiturnEvalContractError(errors + ["scenarios 必须是数组"])
    scenario_ids: set[str] = set()
    all_turn_ids: set[str] = set()
    gate_count = 0
    gap_count = 0
    verified_hashes = 0
    for position, scenario in enumerate(scenarios, 1):
        if not isinstance(scenario, Mapping):
            errors.append(f"scenario #{position} 必须是对象")
            continue
        scenario_id = str(scenario.get("scenario_id") or "")
        prefix = scenario_id or f"scenario #{position}"
        if not scenario_id or scenario_id in scenario_ids:
            errors.append(f"scenario_id 缺失或重复: {scenario_id}")
        scenario_ids.add(scenario_id)
        if scenario.get("split") not in SPLITS:
            errors.append(f"{prefix}: split 不合法")
        gate = scenario.get("release_gate")
        if not isinstance(gate, bool):
            errors.append(f"{prefix}: release_gate 必须是布尔值")
            gate = False
        gate_count += int(gate)
        gap = scenario.get("known_gap")
        if gap is not None:
            gap_count += 1
            if not str(gap).strip() or gate:
                errors.append(f"{prefix}: 已知缺口必须有说明且不能作为门禁")
        elif not gate and scenario.get("split") != "stress":
            errors.append(f"{prefix}: 非门禁场景必须说明 known_gap")

        provenance = scenario.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("kind") != "real":
            errors.append(f"{prefix}: v1 只允许有来源标记的真实场景")
            source_db = ""
        else:
            source_db = str(provenance.get("source_db") or "")
            if not source_db or not provenance.get("session_ids"):
                errors.append(f"{prefix}: provenance 不完整")
        if scenario.get("initial_fixture") not in fixture_ids:
            errors.append(f"{prefix}: initial_fixture 不存在")

        turn_ids = scenario.get("turn_ids")
        if not isinstance(turn_ids, list) or not turn_ids:
            errors.append(f"{prefix}: turn_ids 必须是非空数组")
            turn_ids = []
        turn_ids = [str(item) for item in turn_ids]
        if len(turn_ids) != len(set(turn_ids)):
            errors.append(f"{prefix}: turn_ids 重复")
        all_turn_ids.update(turn_ids)
        steps = scenario.get("steps")
        if not isinstance(steps, list) or len(steps) != len(turn_ids):
            errors.append(f"{prefix}: steps 必须与 turn_ids 等长")
            steps = []
        for index, step in enumerate(steps, 1):
            step_prefix = f"{prefix}: step {index}"
            if not isinstance(step, Mapping):
                errors.append(f"{step_prefix} 必须是对象")
                continue
            if step.get("index") != index or step.get("turn_id") != turn_ids[index - 1]:
                errors.append(f"{step_prefix}: index/turn 顺序错误")
            chain = step.get("expected_chain")
            if not isinstance(chain, Mapping):
                errors.append(f"{step_prefix}: expected_chain 缺失")
                continue
            chain_type = str(chain.get("chain_type") or "")
            states = chain.get("allowed_states")
            if chain_type not in CHAIN_STATES or not isinstance(states, list) or not states:
                errors.append(f"{step_prefix}: chain 类型或状态缺失")
                states = []
            invalid = set(map(str, states)) - CHAIN_STATES.get(chain_type, set())
            if invalid:
                errors.append(f"{step_prefix}: 非法状态 {sorted(invalid)}")
            formal = chain.get("formal_memory_allowed")
            formal_types_raw = chain.get("formal_memory_types_allowed")
            formal_types: set[str] | None = None
            if formal_types_raw is not None:
                if not isinstance(formal_types_raw, list):
                    errors.append(f"{step_prefix}: formal_memory_types_allowed 必须是数组")
                    formal_types = set()
                else:
                    formal_types = set(map(str, formal_types_raw))
                    invalid_types = formal_types - FORMAL_MEMORY_TYPES
                    if invalid_types:
                        errors.append(
                            f"{step_prefix}: 非法正式记忆类型 {sorted(invalid_types)}"
                        )
                if formal is False and formal_types:
                    errors.append(
                        f"{step_prefix}: 禁止正式记忆时不能声明允许的正式记忆类型"
                    )
            forbidden = set(map(str, step.get("forbidden_actions") or []))
            if formal is False and not WRITE_ACTIONS.issubset(forbidden):
                errors.append(f"{step_prefix}: 未闭环步骤必须禁止正式 memory 写入")
            runtime_memory_allowed = (
                formal is True
                and (formal_types is None or "runtime_failure_experience" in formal_types)
            )
            compile_memory_allowed = (
                formal is True
                and (formal_types is None or "compile_failure_experience" in formal_types)
            )
            if gate and runtime_memory_allowed and chain_type == "runtime" and "user_verified" not in states:
                errors.append(f"{step_prefix}: 运行记忆只有 user_verified 才能写入")
            if gate and compile_memory_allowed and chain_type == "compilation" and "compile_verified" not in states:
                errors.append(f"{step_prefix}: 编译记忆只有 compile_verified 才能写入")
            tools = step.get("required_tools")
            if not isinstance(tools, list) or set(map(str, tools)) - KNOWN_TOOLS:
                errors.append(f"{step_prefix}: required_tools 非法")
            previous = set(turn_ids[: index - 1])
            required = set(map(str, step.get("required_history_turn_ids") or []))
            forbidden_history = set(map(str, step.get("forbidden_history_turn_ids") or []))
            if not required.issubset(previous) or not forbidden_history.issubset(previous):
                errors.append(f"{step_prefix}: 历史引用包含未来或场景外 turn")
            if required & forbidden_history:
                errors.append(f"{step_prefix}: 同一历史 turn 同时必读和禁止")

        oracle = scenario.get("final_oracle")
        if not isinstance(oracle, Mapping):
            errors.append(f"{prefix}: final_oracle 缺失")
        else:
            formal = oracle.get("formal_memory")
            if not isinstance(formal, Mapping) or not isinstance(formal.get("required"), bool):
                errors.append(f"{prefix}: formal_memory.required 非法")
            claims = oracle.get("required_semantics")
            if not isinstance(claims, list):
                errors.append(f"{prefix}: required_semantics 必须是数组")
            else:
                for number, claim in enumerate(claims, 1):
                    support = claim.get("supporting_turn_ids") if isinstance(claim, Mapping) else None
                    if not isinstance(claim, Mapping) or not str(claim.get("claim") or "").strip() or not isinstance(support, list) or not support:
                        errors.append(f"{prefix}: semantic claim #{number} 不完整")
                    elif not set(map(str, support)).issubset(set(turn_ids)):
                        errors.append(f"{prefix}: semantic claim #{number} 引用场景外 turn")

        hashes = scenario.get("evidence_hashes")
        if not isinstance(hashes, Mapping) or set(hashes) != set(turn_ids):
            errors.append(f"{prefix}: evidence_hashes 必须恰好覆盖 turn_ids")
            hashes = {}
        if any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in hashes.values()):
            errors.append(f"{prefix}: evidence_hashes 包含非法 SHA-256")
        for turn_id in turn_ids:
            case = case_by_turn.get(turn_id)
            if case is None:
                errors.append(f"{prefix}: {turn_id} 缺少 case 映射")
            elif str(case.get("source_db") or "").replace("\\", "/").casefold() != source_db.replace("\\", "/").casefold():
                errors.append(f"{prefix}: {turn_id} 的 source_db 不一致")
        if verify_evidence_hashes and source_db and lab_database is not None:
            source = Path(source_db)
            if not source.is_file():
                errors.append(f"{prefix}: 原始数据库不存在: {source_db}")
            else:
                for turn_id in turn_ids:
                    try:
                        actual = build_turn_evidence(source, Path(lab_database), turn_id).get("bundle_hash")
                    except Exception as exc:  # pragma: no cover - real data boundary
                        errors.append(f"{prefix}: 无法重建 {turn_id}: {exc}")
                        continue
                    if actual != hashes.get(turn_id):
                        errors.append(f"{prefix}: {turn_id} 证据哈希漂移")
                    else:
                        verified_hashes += 1

    unused_cases = set(case_by_turn) - all_turn_ids
    if unused_cases:
        errors.append(f"cases.jsonl 有未引用 turn: {sorted(unused_cases)}")
    for key, actual in {
        "scenario_count": len(scenarios),
        "release_gate_scenario_count": gate_count,
        "known_gap_scenario_count": gap_count,
    }.items():
        if manifest.get(key) != actual:
            errors.append(f"manifest.{key} 与实际不一致")

    report = {
        "ok": not errors,
        "evalset_id": document.get("evalset_id"),
        "scenario_count": len(scenarios),
        "release_gate_scenario_count": gate_count,
        "known_gap_scenario_count": gap_count,
        "case_count": len(cases),
        "unique_turn_count": len(all_turn_ids),
        "evidence_hash_verification_enabled": verify_evidence_hashes,
        "evidence_hash_verified_count": verified_hashes,
        "errors": errors,
    }
    if errors:
        raise MultiturnEvalContractError(errors)
    return report


__all__ = ["MultiturnEvalContractError", "validate_multiturn_evalset"]
