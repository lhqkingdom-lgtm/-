from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_lab.config import ConfigError, load_api_key, load_llm_config


class ConfigTests(unittest.TestCase):
    def test_loads_deepseek_non_secret_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory-lab.config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-config-v1",
                        "llm": {
                            "provider": "deepseek",
                            "base_url": "https://api.deepseek.com",
                            "model": "deepseek-v4-pro",
                            "allow_remote": True,
                            "timeout_seconds": 45,
                            "max_attempts": 3,
                            "credential_file": "memory-lab.secrets.local.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_llm_config(path)

            self.assertEqual(config.model, "deepseek-v4-pro")
            self.assertTrue(config.allow_remote)
            self.assertEqual(
                config.credential_file,
                (Path(directory) / "memory-lab.secrets.local.json").resolve(),
            )

    def test_key_must_stay_in_separate_credential_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory-lab.config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-config-v1",
                        "llm": {
                            "provider": "deepseek",
                            "base_url": "https://api.deepseek.com",
                            "model": "deepseek-v4-pro",
                            "allow_remote": True,
                            "timeout_seconds": 45,
                            "max_attempts": 3,
                            "credential_file": "memory-lab.secrets.local.json",
                            "api_key": "must-not-be-here",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "plaintext"):
                load_llm_config(path)

    def test_key_file_cannot_escape_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory-lab.config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-config-v1",
                        "llm": {
                            "provider": "deepseek",
                            "base_url": "https://api.deepseek.com",
                            "model": "deepseek-v4-pro",
                            "allow_remote": True,
                            "timeout_seconds": 45,
                            "max_attempts": 3,
                            "credential_file": "../outside.key",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "inside"):
                load_llm_config(path)

    def test_loads_api_key_from_local_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "memory-lab.config.json"
            secret_path = root / "memory-lab.secrets.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-config-v1",
                        "llm": {
                            "provider": "deepseek",
                            "base_url": "https://api.deepseek.com",
                            "model": "deepseek-v4-pro",
                            "allow_remote": True,
                            "timeout_seconds": 45,
                            "max_attempts": 3,
                            "credential_file": secret_path.name,
                        },
                    }
                ),
                encoding="utf-8",
            )
            secret_path.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-secrets-v1",
                        "deepseek_api_key": "test-only-key",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_api_key(load_llm_config(config_path)), "test-only-key")

    def test_loads_glm_configuration_with_environment_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory-lab.glm.config.json"
            path.write_text(
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

            with patch.dict(os.environ, {"MEMORY_LAB_GLM_API_KEY": "test-glm-key"}):
                config = load_llm_config(path)

                self.assertEqual(config.provider, "glm")
                self.assertEqual(config.credential_env, "MEMORY_LAB_GLM_API_KEY")
                self.assertIsNone(config.credential_file)
                self.assertEqual(load_api_key(config), "test-glm-key")

    def test_environment_credential_can_fall_back_to_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "memory-lab.glm.config.json"
            secret_path = root / "memory-lab.glm.secrets.local.json"
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
                            "credential_file": secret_path.name,
                        },
                    }
                ),
                encoding="utf-8",
            )
            secret_path.write_text(
                json.dumps(
                    {
                        "schema_version": "memory-lab-secrets-v1",
                        "glm_api_key": "file-glm-key",
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_api_key(load_llm_config(config_path)), "file-glm-key")

    def test_missing_environment_credential_reports_variable_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory-lab.glm.config.json"
            path.write_text(
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

            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
                ConfigError, "MEMORY_LAB_GLM_API_KEY"
            ):
                load_api_key(load_llm_config(path))


if __name__ == "__main__":
    unittest.main()
