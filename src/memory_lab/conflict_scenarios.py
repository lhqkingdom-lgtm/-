"""Conflict-handling scenario matrix for lightweight memory maintenance.

This module is deliberately deterministic.  It does not call an LLM and does
not write PostgreSQL.  It defines the scenario coverage that later live/LLM
tests must satisfy, and produces a small report that is easy to review.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REQUIRED_SCENARIO_IDS = {
    "A_preference_changed",
    "B_fact_correction",
    "C_verified_overrides_guess",
    "D_insufficient_evidence",
    "E_same_symptom_different_root_cause",
    "F_project_parameter_not_memory",
}

REQUIRED_DIMENSIONS = {
    "conflict_classification",
    "action_selection",
    "revision_chain",
    "memory_relation",
    "traceability",
    "trust_guard",
    "no_pollution",
    "idempotency",
}


@dataclass(frozen=True)
class ConflictScenario:
    scenario_id: str
    title: str
    purpose: str
    existing_memory: str
    incoming_signal: str
    evidence_required: list[str]
    expected_actions: list[str]
    expected_relation_type: str | None
    expected_trust_rule: str
    expected_revision_assertions: list[str]
    expected_relation_assertions: list[str]
    negative_assertions: list[str]
    covered_dimensions: list[str]


SCENARIOS: list[ConflictScenario] = [
    ConflictScenario(
        scenario_id="A_preference_changed",
        title="用户偏好变化",
        purpose="验证同一用户偏好槽位发生变化时，系统能采用较新的明确用户表达，并保留旧偏好可回溯。",
        existing_memory="用户希望回答详细一些，解释充分。",
        incoming_signal="用户说：简单点，不要写太多。",
        evidence_required=["新用户消息", "旧 memory 详情", "trace_memory_relations 输出"],
        expected_actions=["update_memory", "record_memory_relation"],
        expected_relation_type="supersedes",
        expected_trust_rule="新旧证据同为用户明确表达时，较新的同槽位偏好优先。",
        expected_revision_assertions=[
            "目标 memory 新增 revision_number + 1",
            "旧 current revision 变为 superseded",
            "新 revision status=current",
            "source_turn_id 等于当前 turn",
        ],
        expected_relation_assertions=[
            "memory_relations.from_memory_id 指向旧偏好记忆",
            "memory_relations.to_memory_id 指向当前有效偏好记忆",
            "relation_type=supersedes",
            "trace_memory_relations 能从新记忆查到旧来源",
        ],
        negative_assertions=[
            "不能重复 create 同义偏好 memory",
            "不能删除旧记忆",
            "不能把一次语气催促当成永久偏好，除非用户表达为持续偏好",
        ],
        covered_dimensions=[
            "conflict_classification",
            "action_selection",
            "revision_chain",
            "memory_relation",
            "traceability",
            "idempotency",
        ],
    ),
    ConflictScenario(
        scenario_id="B_fact_correction",
        title="用户纠正事实",
        purpose="验证用户明确纠正项目/芯片等事实时，新事实覆盖旧错误事实，并记录替代关系。",
        existing_memory="MJ60B 是芯片型号。",
        incoming_signal="用户说：MJ60B 是项目名，不是芯片型号。",
        evidence_required=["用户明确纠正消息", "旧 memory 详情"],
        expected_actions=["update_memory", "record_memory_relation"],
        expected_relation_type="supersedes",
        expected_trust_rule="用户明确纠正事实优先于旧的模型推断或旧记忆。",
        expected_revision_assertions=[
            "当前 memory 内容改为 MJ60B 是项目/产品代号",
            "revision 记录 fact_correction 原因",
            "旧错误内容仍可在历史 revision 中查到",
        ],
        expected_relation_assertions=[
            "relation_type=supersedes",
            "reason 说明用户明确纠正",
        ],
        negative_assertions=[
            "不能继续把 MJ60B 当芯片型号使用",
            "不能污染全局芯片知识，只应记录为用户/项目相关事实",
        ],
        covered_dimensions=[
            "conflict_classification",
            "action_selection",
            "revision_chain",
            "memory_relation",
            "traceability",
            "trust_guard",
            "idempotency",
        ],
    ),
    ConflictScenario(
        scenario_id="C_verified_overrides_guess",
        title="验证结果覆盖猜测",
        purpose="验证真实编译/运行/用户确认的结果可以覆盖模型猜测。",
        existing_memory="风扇不转可能是 PWM 配置问题。",
        incoming_signal="读代码确认 adc_sample 固定返回 2048；用户确认替换 ADC 后风扇随温度变化。",
        evidence_required=["代码读取证据", "编译/运行验证", "用户确认修复"],
        expected_actions=["update_memory", "record_memory_relation"],
        expected_relation_type="supersedes",
        expected_trust_rule="已验证结论优先于未验证猜测。",
        expected_revision_assertions=[
            "新 revision 包含症状/根因/解决/验证",
            "trust_level 不低于 verified 或 user_confirmed",
            "旧猜测保留在历史 revision 或 relation 来源中",
        ],
        expected_relation_assertions=[
            "relation_type=supersedes",
            "reason 说明 verified_overrides_guess",
        ],
        negative_assertions=[
            "不能保留 PWM 猜测作为当前有效根因",
            "不能在没有用户或工具验证时提升为 verified",
        ],
        covered_dimensions=[
            "conflict_classification",
            "action_selection",
            "revision_chain",
            "memory_relation",
            "traceability",
            "trust_guard",
            "idempotency",
        ],
    ),
    ConflictScenario(
        scenario_id="D_insufficient_evidence",
        title="证据不足不覆盖",
        purpose="验证只有模型怀疑、没有日志/代码/用户确认时，不覆盖已有记忆。",
        existing_memory="C202 常见原因是 IncludePath/Define 缺失。",
        incoming_signal="模型怀疑也可能是 SFR 名称不一致，但没有错误日志或代码证据。",
        evidence_required=["至少需要错误日志、代码读取或用户确认；若缺失则证据不足"],
        expected_actions=["noop", "request_more_evidence"],
        expected_relation_type=None,
        expected_trust_rule="低证据推断不能覆盖已有工程经验。",
        expected_revision_assertions=[
            "不新增 revision",
            "不修改 memories 当前内容",
        ],
        expected_relation_assertions=[
            "不新增 supersedes/merges/conflicts_with",
        ],
        negative_assertions=[
            "不能把猜测写成 verified",
            "不能生成替代关系",
            "不能把未验证怀疑合并进当前有效记忆",
        ],
        covered_dimensions=[
            "conflict_classification",
            "action_selection",
            "trust_guard",
            "no_pollution",
            "idempotency",
        ],
    ),
    ConflictScenario(
        scenario_id="E_same_symptom_different_root_cause",
        title="同症状不同根因不误判冲突",
        purpose="验证同一个现象可能有多个根因时，不把补充原因误判为冲突。",
        existing_memory="风扇不转可能由 ADC 输入桩固定返回导致。",
        incoming_signal="另一轮确认风扇不转由 PWM 引脚复用未配置导致。",
        evidence_required=["新根因代码/配置证据", "旧 memory 详情", "适用条件差异"],
        expected_actions=["create_memory", "merge_memory"],
        expected_relation_type="merges",
        expected_trust_rule="同症状不同条件/根因通常并存或合并为排查清单，不是互相否定。",
        expected_revision_assertions=[
            "若合并：新增 revision，内容包含两个根因及区分条件",
            "若新建：新 memory 有独立 stable_key 和明确适用条件",
        ],
        expected_relation_assertions=[
            "如果合并，relation_type=merges",
            "trace_memory_relations 能看到来源记忆",
        ],
        negative_assertions=[
            "不能标记 conflicts_with",
            "不能删除任一根因",
        ],
        covered_dimensions=[
            "conflict_classification",
            "action_selection",
            "revision_chain",
            "memory_relation",
            "traceability",
            "idempotency",
        ],
    ),
    ConflictScenario(
        scenario_id="F_project_parameter_not_memory",
        title="当前项目参数不沉淀",
        purpose="验证一次性的需求参数不被写入长期记忆，也不会触发伪冲突。",
        existing_memory="风扇温控类项目通常需要确认传感器、PWM、滞回、失效安全。",
        incoming_signal="当前项目选择 NTC 10K/3950、PWM 25kHz、阈值 36℃。",
        evidence_required=["用户本轮参数选择", "需求文档/当前项目上下文"],
        expected_actions=["noop"],
        expected_relation_type=None,
        expected_trust_rule="当前项目参数属于需求文档，不属于长期记忆。",
        expected_revision_assertions=[
            "不新增 revision",
            "不修改长期通用模板记忆",
        ],
        expected_relation_assertions=[
            "不新增 memory_relations",
        ],
        negative_assertions=[
            "不能把具体阈值/PWM/NTC 型号写成长期通用规则",
            "不能因为参数不同就记录 conflicts_with",
        ],
        covered_dimensions=[
            "action_selection",
            "no_pollution",
            "idempotency",
        ],
    ),
]


def build_conflict_scenarios_report(*, output_dir: Path | None = None) -> dict[str, Any]:
    scenario_ids = {scenario.scenario_id for scenario in SCENARIOS}
    covered_dimensions = {
        dimension
        for scenario in SCENARIOS
        for dimension in scenario.covered_dimensions
    }
    missing_scenarios = sorted(REQUIRED_SCENARIO_IDS - scenario_ids)
    missing_dimensions = sorted(REQUIRED_DIMENSIONS - covered_dimensions)
    report = {
        "status": "complete" if not missing_scenarios and not missing_dimensions else "incomplete",
        "schema_version": "memory-lab-conflict-scenarios-v1",
        "scenario_count": len(SCENARIOS),
        "required_scenarios": sorted(REQUIRED_SCENARIO_IDS),
        "covered_scenarios": sorted(scenario_ids),
        "missing_scenarios": missing_scenarios,
        "required_dimensions": sorted(REQUIRED_DIMENSIONS),
        "covered_dimensions": sorted(covered_dimensions),
        "missing_dimensions": missing_dimensions,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "next_validation_steps": [
            "用模拟 memory_changes 计划验证每个场景的 action/relation/revision 预期。",
            "再用真实 CP turn 或人工构造 turn 跑 ReAct dry-run，观察模型是否按场景调用 search/read/trace 工具。",
            "最后选择 1-2 个低风险场景做真实 commit，验证版本链和关系链落库。"
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown = _render_markdown(report)
        (output_dir / "conflict_scenarios_report.md").write_text(markdown, encoding="utf-8")
        report["markdown_path"] = str(output_dir / "conflict_scenarios_report.md")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 长期记忆矛盾处理测试集",
        "",
        f"- 状态：{report['status']}",
        f"- 场景数量：{report['scenario_count']}",
        f"- 缺失场景：{', '.join(report['missing_scenarios']) or '无'}",
        f"- 缺失维度：{', '.join(report['missing_dimensions']) or '无'}",
        "",
        "## 覆盖维度",
        "",
    ]
    for dimension in report["covered_dimensions"]:
        lines.append(f"- {dimension}")
    lines.extend(["", "## 场景明细", ""])
    for scenario in report["scenarios"]:
        lines.extend([
            f"### {scenario['scenario_id']}：{scenario['title']}",
            "",
            f"- 目的：{scenario['purpose']}",
            f"- 旧记忆：{scenario['existing_memory']}",
            f"- 新信号：{scenario['incoming_signal']}",
            f"- 预期动作：{', '.join(scenario['expected_actions'])}",
            f"- 预期关系：{scenario['expected_relation_type'] or '无'}",
            f"- 信任规则：{scenario['expected_trust_rule']}",
            "- 版本链验收：",
        ])
        for item in scenario["expected_revision_assertions"]:
            lines.append(f"  - {item}")
        lines.append("- 关系链验收：")
        for item in scenario["expected_relation_assertions"]:
            lines.append(f"  - {item}")
        lines.append("- 防误判：")
        for item in scenario["negative_assertions"]:
            lines.append(f"  - {item}")
        lines.append("")
    lines.extend(["## 后续验证步骤", ""])
    for step in report["next_validation_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "SCENARIOS",
    "REQUIRED_DIMENSIONS",
    "REQUIRED_SCENARIO_IDS",
    "build_conflict_scenarios_report",
]
