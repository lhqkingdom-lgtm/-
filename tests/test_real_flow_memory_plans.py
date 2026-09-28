from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.memory_changes import MEMORY_CHANGES_SCHEMA_VERSION, _digest_id, commit_memory_changes


BASE_SNAPSHOT = {
    "turn_exists": True,
    "evidence_ids": ["ev_user", "ev_assistant", "ev_tool", "ev_result"],
    "pointer_ids": ["ptr_raw_log", "ptr_raw_code"],
    "memory_ids": [
        "mem_user_workflow",
        "mem_c51_compile_errors",
        "mem_stub_adc_rule",
        "mem_stage_confirm",
        "mem_fast_demo",
    ],
    "topic_ids": [
        "topic_user_collaboration",
        "topic_c51_build",
        "topic_embedded_runtime",
    ],
    "stable_keys": [
        "user.workflow.align_before_code",
        "engineering.c51.compile.syntax_limit",
        "embedded.runtime.stub_adc_constant_input",
        "user.workflow.stage_confirm",
        "user.workflow.fast_demo",
    ],
    "evidence_details": {
        "ev_user": {
            "turn_id": "turn_current",
            "layer": "core",
            "source_type": "user_message",
            "kind": "user_confirmation",
        },
        "ev_assistant": {
            "turn_id": "turn_current",
            "layer": "core",
            "source_type": "assistant_message",
            "kind": "assistant_conclusion",
        },
        "ev_tool": {
            "turn_id": "turn_current",
            "layer": "candidate",
            "source_type": "tool_result",
            "kind": "build_or_tool_result",
        },
        "ev_result": {
            "turn_id": "turn_current",
            "layer": "core",
            "source_type": "tool_result",
            "kind": "compile_conclusion",
        },
    },
}


def proposal(turn_id: str, reason: str, changes: list[dict]) -> dict:
    return {
        "schema_version": MEMORY_CHANGES_SCHEMA_VERSION,
        "turn_id": turn_id,
        "reason": reason,
        "changes": changes,
    }


def ev(evidence_id: str = "ev_user", reason: str = "当前 turn 的直接证据。") -> list[dict]:
    return [{"evidence_id": evidence_id, "reason": reason}]


class RealFlowMemoryPlansTests(unittest.TestCase):
    def _commit(self, plan: dict, *, snapshot: dict = BASE_SNAPSHOT, json_side_effect=None):
        patches = [
            patch(
                "memory_lab.memory_changes.load_pg_config",
                return_value={"user": {"user_id": "local-default-user"}, "postgres": {}},
            ),
            patch("memory_lab.memory_changes._reference_snapshot", return_value=snapshot),
            patch("memory_lab.memory_changes._memory_already_processed_turn", return_value=False),
            patch("memory_lab.memory_changes._run_psql"),
        ]
        if json_side_effect is not None:
            patches.append(patch("memory_lab.memory_changes._json_from_psql_object", side_effect=json_side_effect))
        with patches[0], patches[1], patches[2], patches[3] as run_psql:
            if json_side_effect is None:
                result = commit_memory_changes(plan, pg_config_path=Path("unused.json"))
            else:
                with patches[4]:
                    result = commit_memory_changes(plan, pg_config_path=Path("unused.json"))
        return result, run_psql

    def test_user_preference_change_updates_memory_records_conflict_and_topic(self) -> None:
        plan = proposal(
            "turn_preference_speed",
            "用户从强调逐步确认，转为演示前先跑通结果；需要更新协作偏好并保留冲突决策。",
            [
                {
                    "action": "update_memory",
                    "reason": "用户明确表达演示前更重视快速跑通结果。",
                    "target_memory_id": "mem_user_workflow",
                    "memory": {
                        "summary": "用户希望平时先需求对齐，但演示/赶时间场景优先跑通结果。",
                        "content": "用户的稳定偏好是先需求对齐再实现；当明确处于演示或时间压力场景时，应降低确认频率，先打通链路并输出结果。",
                        "confidence": 0.86,
                    },
                    "evidence_refs": ev("ev_user", "用户明确纠正当前协作节奏。"),
                },
                {
                    "action": "record_conflict_decision",
                    "reason": "记录新旧协作偏好的适用边界。",
                    "target_memory_id": "mem_user_workflow",
                    "conflict": {
                        "conflict_type": "user_preference_changed",
                        "old_claim": "用户希望每阶段先需求对齐并确认。",
                        "new_claim": "演示前用户希望少卡确认、先跑通结果。",
                        "decision": "保留长期偏好，同时增加演示/时间压力场景的例外规则。",
                        "reason": "新证据不是完全推翻旧偏好，而是给出适用场景边界。",
                        "status": "resolved_by_update",
                    },
                    "evidence_refs": ev("ev_user", "用户明确说先跑结果最重要。"),
                },
                {
                    "action": "update_topic",
                    "reason": "主题摘要需要反映偏好的场景边界。",
                    "target_topic_id": "topic_user_collaboration",
                    "topic": {
                        "summary": "记录用户协作偏好：平时希望需求对齐和阶段沉淀；演示/赶时间时优先跑通链路和展示结果。",
                    },
                    "evidence_refs": ev("ev_user", "主题摘要来自本轮用户偏好变化。"),
                },
            ],
        )
        result, run_psql = self._commit(
            plan,
            json_side_effect=[
                {"found": False},
                {
                    "found": True,
                    "topic_id": "topic_user_collaboration",
                    "title": "用户协作偏好",
                    "summary": "记录用户协作偏好：平时希望需求对齐和阶段沉淀。",
                },
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["applied"], 3)
        self.assertEqual(run_psql.call_count, 3)
        combined_sql = "\n".join(call.args[1] for call in run_psql.call_args_list)
        self.assertIn("INSERT INTO memory_revisions", combined_sql)
        self.assertIn("INSERT INTO memory_conflict_records", combined_sql)
        self.assertIn("UPDATE memory_topics", combined_sql)

    def test_compile_failure_creates_symptom_reason_fix_verified_memory(self) -> None:
        plan = proposal(
            "turn_c51_compile_fix",
            "编译失败已经定位、修复并重新编译通过，适合沉淀为 C51 工程经验。",
            [
                {
                    "action": "create_memory",
                    "reason": "新增经过编译验证的 C51 失败模式。",
                    "topic_id": "topic_c51_build",
                    "memory": {
                        "stable_key": "engineering.c51.compile.function_prototype_before_call",
                        "memory_type": "compile_failure_experience",
                        "summary": "Keil C51 中函数原型缺失可能导致编译或链接异常，修复后需重新编译验证。",
                        "content": "症状：Keil C51 编译失败并指向函数调用/声明不一致。原因：C51 对函数原型、存储类型和调用约定更敏感。解决：在调用前补齐头文件声明或函数原型，保持定义与声明一致。验证：重新编译通过后再沉淀。",
                        "scope": {"level": "domain", "domain": "embedded"},
                        "confidence": 0.9,
                    },
                    "technical_closure": {
                        "chain_type": "compilation",
                        "state": "compile_verified",
                        "verification_evidence_ids": ["ev_result"],
                    },
                    "evidence_refs": [
                        {"evidence_id": "ev_tool", "reason": "编译日志给出失败症状。"},
                        {"evidence_id": "ev_result", "reason": "修复后编译通过。"},
                    ],
                }
            ],
        )
        result, run_psql = self._commit(plan)
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["applied"], 1)
        sql = run_psql.call_args.args[1]
        self.assertIn("INSERT INTO memories", sql)
        self.assertIn("INSERT INTO memory_revisions", sql)
        self.assertIn("重新编译通过", sql)
        self.assertNotIn("完整编译日志", sql)

    def test_runtime_symptom_creates_generic_embedded_rule_after_user_confirmation(self) -> None:
        plan = proposal(
            "turn_adc_stub_confirmed",
            "用户确认 adc_sample 桩函数修复后现象正常，适合沉淀为通用运行现象经验。",
            [
                {
                    "action": "create_memory",
                    "reason": "从单项目问题抽象为嵌入式硬件抽象经验。",
                    "topic_id": "topic_embedded_runtime",
                    "memory": {
                        "stable_key": "embedded.runtime.stub_constant_input_state_machine_stuck",
                        "memory_type": "runtime_failure_experience",
                        "summary": "硬件抽象桩函数返回固定值会让主流程跑通但输入恒定，导致下游状态机停在固定状态。",
                        "content": "现象：风扇不转或调节无变化。触发条件：adc_sample 等硬件抽象函数直接返回固定常量。原因：主流程、滤波、阈值判断都在运行，但输入永远不变。解决：替换为真实寄存器/外设读取，并结合数据手册确认 ADC 通道、引脚和转换流程。验证：用户确认替换后现象正常再沉淀。",
                        "scope": {"level": "domain", "domain": "embedded"},
                        "confidence": 0.92,
                    },
                    "technical_closure": {
                        "chain_type": "runtime",
                        "state": "user_verified",
                        "verification_evidence_ids": ["ev_user"],
                    },
                    "evidence_refs": [
                        {"evidence_id": "ev_assistant", "reason": "模型完成代码级定位。"},
                        {"evidence_id": "ev_user", "reason": "用户确认修复有效。"},
                    ],
                }
            ],
        )
        result, run_psql = self._commit(plan)
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["applied"], 1)
        sql = run_psql.call_args.args[1]
        self.assertIn("硬件抽象桩函数", sql)
        self.assertIn("用户确认替换后现象正常", sql)

    def test_insufficient_evidence_requests_more_evidence_without_writing(self) -> None:
        plan = proposal(
            "turn_adc_guess_only",
            "当前只有猜测，没有修复和验证，不应污染长期记忆。",
            [
                {
                    "action": "request_more_evidence",
                    "reason": "只有用户猜测 ADC 可能有问题，缺少代码定位、修复结果或用户确认。",
                    "requests": [
                        {
                            "tool": "expand_pointer",
                            "raw_pointer_id": "ptr_raw_code",
                            "reason": "需要查看原始代码或日志确认是否存在桩函数或 ADC 读取异常。",
                        }
                    ],
                    "evidence_refs": ev("ev_user", "用户只表达不确定猜测。"),
                }
            ],
        )
        result, run_psql = self._commit(plan)
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["requests"], 1)
        run_psql.assert_not_called()

    def test_same_symptom_different_root_cause_can_be_created_and_linked_in_one_plan(self) -> None:
        stable_key = "embedded.runtime.pwm_pin_mapping_fan_not_spin"
        new_memory_id = _digest_id("mem", "local-default-user", stable_key)
        plan = proposal(
            "turn_fan_not_spin_pwm_root_cause",
            "同样是风扇不转，但根因从 ADC 桩函数变为 PWM 引脚配置错误，应保留两条经验并建立相关关系。",
            [
                {
                    "action": "create_memory",
                    "client_temp_id": "temp:pwm_pin_mapping_memory",
                    "reason": "新增同现象不同根因的运行经验。",
                    "topic_id": "topic_embedded_runtime",
                    "memory": {
                        "stable_key": stable_key,
                        "memory_type": "runtime_failure_experience",
                        "summary": "风扇不转也可能由 PWM 通道或引脚映射错误导致，不能只套用 ADC 桩函数根因。",
                        "content": "现象：风扇不转。原因：PWM 通道或引脚配置与实际接线不一致，导致占空比逻辑正常但物理引脚无有效输出。解决：核对原理图、芯片复用功能和工程 IO 配置，修正 PWM 通道/引脚映射。验证：烧录后用户确认风扇可按占空比变化。",
                        "scope": {"level": "domain", "domain": "embedded"},
                        "confidence": 0.88,
                    },
                    "technical_closure": {
                        "chain_type": "runtime",
                        "state": "user_verified",
                        "verification_evidence_ids": ["ev_user"],
                    },
                    "evidence_refs": [
                        {"evidence_id": "ev_tool", "reason": "定位到 PWM 引脚配置。"},
                        {"evidence_id": "ev_user", "reason": "用户确认修复后风扇运行。"},
                    ],
                },
                {
                    "action": "record_memory_relation",
                    "reason": "同一现象存在不同根因，建立相关关系避免误合并。",
                    "relation": {
                        "from_memory_id": "mem_stub_adc_rule",
                        "to_memory_id": "temp:pwm_pin_mapping_memory",
                        "relation_type": "merges",
                        "reason": "二者都解释风扇不转类现象，但根因分别是 ADC 输入恒定和 PWM 引脚映射错误，应关联但不能互相覆盖。",
                    },
                    "evidence_refs": ev("ev_assistant", "模型判断不能直接覆盖旧根因。"),
                },
            ],
        )
        snapshot = {
            **BASE_SNAPSHOT,
            "stable_keys": [key for key in BASE_SNAPSHOT["stable_keys"] if key != stable_key],
        }
        result, run_psql = self._commit(plan, snapshot=snapshot, json_side_effect=[{"found": False}])
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["applied"], 2)
        self.assertEqual(result["applied"][0]["memory_id"], new_memory_id)
        self.assertEqual(result["applied"][0]["client_temp_id"], "temp:pwm_pin_mapping_memory")
        self.assertEqual(result["applied"][1]["to_memory_id"], new_memory_id)
        combined_sql = "\n".join(call.args[1] for call in run_psql.call_args_list)
        self.assertIn("INSERT INTO memories", combined_sql)
        self.assertIn("INSERT INTO memory_relations", combined_sql)


if __name__ == "__main__":
    unittest.main()
