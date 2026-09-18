from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memory_lab.llm_smoke import run_llm_smoke_test


class FakeJsonClient:
    def complete_json(self, **kwargs):
        return {"ok": True, "message": "pong"}


class LlmSmokeTests(unittest.TestCase):
    def test_smoke_test_reports_provider_without_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "memory-lab.glm.config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-config-v1",
                        "llm": {
                            "provider": "glm",
                            "base_url": "https://open.bigmodel.cn/api/paas/v4",
                            "model": "glm-5.1",
                            "allow_remote": True,
                            "timeout_seconds": 45,
                            "max_attempts": 3,
                            "credential_env": "MEMORY_LAB_GLM_API_KEY",
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = run_llm_smoke_test(config_path=config_path, client=FakeJsonClient())

        self.assertTrue(report["ok"])
        self.assertEqual(report["provider"], "glm")
        self.assertEqual(report["model"], "glm-5.1")
        self.assertEqual(report["credential_source"], {"type": "env", "name": "MEMORY_LAB_GLM_API_KEY"})
        self.assertEqual(report["result"], {"ok": True, "message": "pong"})
        self.assertNotIn("test-glm-key", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
