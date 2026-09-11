from __future__ import annotations

import unittest

from memory_lab.phase import classify_turn_phase


def bundle(user: str, assistant: str = "", tools: list[dict] | None = None) -> dict:
    return {
        "turn": {"turn_id": "turn_1"},
        "workspace": {"project_key": "project"},
        "user": {"text_parts": [{"text": user}]},
        "user_answers": [],
        "assistant": {"text_parts": [{"text": assistant}] if assistant else []},
        "tools": tools or [],
    }


class TurnPhaseTests(unittest.TestCase):
    def test_user_preference_phase(self) -> None:
        result = classify_turn_phase(bundle("以后回答我先给结论，再说明步骤。"))
        self.assertEqual(result["primary_phase"], "user_interaction_preference")
        self.assertIn("user", result["recommended_memory_targets"])

    def test_compilation_phase_uses_tool_and_assistant_signals(self) -> None:
        result = classify_turn_phase(
            bundle(
                "编译失败了，看看 Keil C51 报错。",
                "已经修复 C51 语法问题，重新编译 0 Error 0 Warning。",
                tools=[
                    {
                        "name": "build",
                        "status": "completed",
                        "output": {"text": "Keil build: error C141; later 0 Error 0 Warning"},
                    }
                ],
            )
        )
        self.assertEqual(result["primary_phase"], "compilation")
        self.assertIn("engineering", result["recommended_memory_targets"])

    def test_runtime_symptom_phase_beats_generic_code_words(self) -> None:
        result = classify_turn_phase(
            bundle(
                "代码已经编译通过并烧录，但板子运行后风扇不转，改变 NTC 温度输入也没有占空比变化，请先排查 ADC/PWM 主流程。",
                "定位到 adc_sample() 是桩函数，return 2048，导致温度恒为 25℃。",
                tools=[
                    {
                        "name": "read",
                        "status": "completed",
                        "output": {"text": "unsigned int adc_sample(void) { return 2048; }"},
                    }
                ],
            )
        )
        self.assertEqual(result["primary_phase"], "runtime_symptom_diagnosis")
        self.assertIn("engineering", result["recommended_memory_targets"])

    def test_fix_confirmation_phase(self) -> None:
        result = classify_turn_phase(
            bundle("我已经重新烧录，风扇现在能正常随温度变化，确认问题已经解决。")
        )
        self.assertEqual(result["primary_phase"], "fix_confirmation")

    def test_requirement_directive_beats_incidental_compile_tool_text(self) -> None:
        result = classify_turn_phase(
            bundle(
                "本轮只完成需求澄清、模块划分和验收条件，不修改文件、不生成代码。",
                tools=[
                    {
                        "name": "glob",
                        "status": "completed",
                        "output": {"text": "found Keil C51 project files"},
                    }
                ],
            )
        )
        self.assertEqual(result["primary_phase"], "requirement_alignment")

    def test_user_confirmation_beats_old_runtime_symptom_words(self) -> None:
        result = classify_turn_phase(
            bundle(
                "我重新编译并烧录验证了，修改后改变 NTC 输入时温度值会变化，风扇占空比能按温区切换，这个问题已解决。",
                "此前 adc_sample return 2048 的桩函数问题已经修复。",
            )
        )
        self.assertEqual(result["primary_phase"], "fix_confirmation")


if __name__ == "__main__":
    unittest.main()
