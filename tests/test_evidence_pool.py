from __future__ import annotations

import unittest

from memory_lab.evidence_pool import (
    DeepSeekPhaseRouter,
    build_evidence_pool,
)


def bundle() -> dict:
    return {
        "turn": {"turn_id": "turn_1", "state": "completed", "occurred_at_ms": 1},
        "workspace": {"project_key": "project", "display_name": "demo"},
        "session": {"session_id": "session_1"},
        "user": {
            "message_id": "user_msg",
            "text_parts": [
                {
                    "source_part_id": "user_part",
                    "text": "烧录后风扇不转，改变 NTC 没反应。",
                }
            ],
        },
        "user_answers": [],
        "assistant": {
            "message_id": "assistant_msg",
            "text_parts": [
                {
                    "source_part_id": "assistant_part",
                    "text": "定位到 adc_sample() 固定返回 2048，根因是硬件采样还是桩函数。",
                }
            ],
        },
        "tools": [
            {
                "source_message_id": "tool_msg_1",
                "source_part_id": "tool_part_1",
                "name": "read",
                "status": "completed",
                "output": {"text": "unsigned int adc_sample(void) { return 2048; }"},
            },
            {
                "source_message_id": "tool_msg_2",
                "source_part_id": "tool_part_2",
                "name": "keil_build",
                "status": "completed",
                "output": {"text": "Build Result: Success (0 errors, 0 warnings)"},
            },
        ],
    }


class FakeClient:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[dict] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class EvidencePoolTests(unittest.TestCase):
    def test_core_candidate_and_pointer_are_separate(self) -> None:
        pool = build_evidence_pool(bundle())

        core_text = "\n".join(item["text"] for item in pool["core_evidence"])
        candidate_text = "\n".join(item["summary"] for item in pool["candidate_evidence"])
        pointer_text = "\n".join(item["summary"] for item in pool["raw_pointers"])

        self.assertIn("烧录后风扇不转", core_text)
        self.assertIn("根因", core_text)
        self.assertIn("Build Result", core_text)
        self.assertIn("return 2048", candidate_text)
        self.assertIn("return 2048", pointer_text)

    def test_read_and_edit_tool_results_do_not_enter_core(self) -> None:
        payload = bundle()
        payload["tools"] = [
            {
                "source_message_id": "tool_msg_1",
                "source_part_id": "tool_part_1",
                "name": "read",
                "status": "completed",
                "output": {
                    "text": "read\ncompleted\n{\"filePath\":\"adc.lst\"}\n*** WARNING C322 IN LINE 27"
                },
            },
            {
                "source_message_id": "tool_msg_2",
                "source_part_id": "tool_part_2",
                "name": "edit",
                "status": "completed",
                "output": {"text": "edit completed 业务代码生成完成 → Keil 编译 0 Error 0 Warning"},
            },
        ]

        pool = build_evidence_pool(payload)
        core_text = "\n".join(item["text"] for item in pool["core_evidence"])
        candidate_text = "\n".join(item["summary"] for item in pool["candidate_evidence"])

        self.assertNotIn("WARNING C322", core_text)
        self.assertNotIn("edit completed", core_text)
        self.assertIn("WARNING C322", candidate_text)
        self.assertIn("修改文件", candidate_text)
        self.assertIn("0 Error / 0 Warning", candidate_text)

    def test_requirement_answers_are_candidates_not_core(self) -> None:
        payload = bundle()
        payload["user_answers"] = [
            {
                "source_message_id": "question_msg",
                "source_part_id": "question_part:answer:0",
                "text": "selected_answers=[\"25kHz（推荐）\"] question_header=\"PWM 频率\"",
            }
        ]
        phase = {
            "primary_phase": "requirement_alignment",
            "primary_label": "需求对齐",
            "phases": [],
            "recommended_memory_targets": [],
            "recommended_evidence_policy": "",
        }

        pool = build_evidence_pool(payload, phase_result=phase)
        core_text = "\n".join(item["text"] for item in pool["core_evidence"])
        candidate_text = "\n".join(item["summary"] for item in pool["candidate_evidence"])

        self.assertNotIn("25kHz", core_text)
        self.assertIn("25kHz", candidate_text)

    def test_phase_router_can_change_rule_phase(self) -> None:
        pool = build_evidence_pool(bundle())
        pool["rule_phase"]["primary_phase"] = "compilation"
        router = DeepSeekPhaseRouter(
            FakeClient(
                {
                    "schema_version": "memory-lab-phase-router-v1",
                    "final_phase": "runtime_symptom_diagnosis",
                    "phase_changed": True,
                    "changed_from": "compilation",
                    "confidence": 0.91,
                    "reason": "用户反馈烧录后的运行现象，编译只是背景。",
                    "extract_targets": ["symptom", "root_cause", "solution"],
                    "need_more_evidence": False,
                    "evidence_requests": [],
                }
            )
        )

        decision = router.route(pool)

        self.assertEqual(decision["final_phase"], "runtime_symptom_diagnosis")
        self.assertTrue(decision["phase_changed"])


if __name__ == "__main__":
    unittest.main()
