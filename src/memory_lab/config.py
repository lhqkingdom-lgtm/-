"""Local, non-secret configuration for Memory Lab model experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "memory-lab.config.json"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    allow_remote: bool
    timeout_seconds: float
    max_attempts: int
    credential_file: Path | None
    credential_env: str | None = None


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a JSON object")
    return dict(value)


def load_llm_config(path: Path = DEFAULT_CONFIG) -> LLMConfig:
    try:
        root = _object(json.loads(path.read_text(encoding="utf-8")), "config")
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON configuration: {error}") from error

    if set(root) != {"schema_version", "llm"}:
        raise ConfigError("config supports only schema_version and llm")
    if root["schema_version"] != "memory-lab-config-v1":
        raise ConfigError("unsupported configuration schema_version")
    llm = _object(root["llm"], "config.llm")
    allowed = {
        "provider",
        "base_url",
        "model",
        "allow_remote",
        "timeout_seconds",
        "max_attempts",
        "credential_file",
        "credential_env",
    }
    unknown = set(llm) - allowed
    required = allowed - {"credential_file", "credential_env"}
    missing = required - set(llm)
    if missing:
        raise ConfigError(f"config.llm missing fields: {sorted(missing)}")
    if unknown:
        if "api_key" in unknown:
            raise ConfigError("do not store api_key in plaintext configuration")
        raise ConfigError(f"config.llm has unsupported fields: {sorted(unknown)}")
    if llm["provider"] not in {"deepseek", "glm"}:
        raise ConfigError("supported providers are deepseek and glm")
    for field in ("base_url", "model"):
        if not isinstance(llm[field], str) or not llm[field].strip():
            raise ConfigError(f"config.llm.{field} must be a non-empty string")
    credential_file_value = llm.get("credential_file")
    credential_env_value = llm.get("credential_env")
    if credential_file_value is not None and (
        not isinstance(credential_file_value, str) or not credential_file_value.strip()
    ):
        raise ConfigError("config.llm.credential_file must be a non-empty string when set")
    if credential_env_value is not None and (
        not isinstance(credential_env_value, str) or not credential_env_value.strip()
    ):
        raise ConfigError("config.llm.credential_env must be a non-empty string when set")
    if credential_file_value is None and credential_env_value is None:
        raise ConfigError("config.llm must set credential_file or credential_env")
    if not isinstance(llm["allow_remote"], bool):
        raise ConfigError("config.llm.allow_remote must be boolean")
    timeout = llm["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
        raise ConfigError("config.llm.timeout_seconds must be between 1 and 300")
    attempts = llm["max_attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 5:
        raise ConfigError("config.llm.max_attempts must be an integer between 1 and 5")

    config_root = path.resolve().parent
    key_file: Path | None = None
    if credential_file_value is not None:
        key_file = (config_root / credential_file_value).resolve()
        try:
            key_file.relative_to(config_root)
        except ValueError as error:
            raise ConfigError("credential_file must stay inside the Memory Lab directory") from error
    return LLMConfig(
        provider=llm["provider"],
        base_url=llm["base_url"].strip(),
        model=llm["model"].strip(),
        allow_remote=llm["allow_remote"],
        timeout_seconds=float(timeout),
        max_attempts=attempts,
        credential_file=key_file,
        credential_env=credential_env_value.strip() if isinstance(credential_env_value, str) else None,
    )


def load_api_key(config: LLMConfig) -> str:
    """Read the selected local credential source without logging its value."""

    if config.credential_env:
        key = os.environ.get(config.credential_env, "")
        if key.strip():
            return key.strip()
        if config.credential_file is None:
            raise ConfigError(f"environment variable {config.credential_env} has not been configured")

    if config.credential_file is None:
        raise ConfigError("credential_file has not been configured")
    try:
        root = _object(
            json.loads(config.credential_file.read_text(encoding="utf-8")),
            "secrets",
        )
    except FileNotFoundError as error:
        raise ConfigError(f"credential file not found: {config.credential_file}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid credential JSON: {error}") from error
    supported_key_fields = {"api_key", "deepseek_api_key", "glm_api_key"}
    if "schema_version" not in root or len(set(root) - (supported_key_fields | {"schema_version"})) > 0:
        raise ConfigError("credential file supports only schema_version and api_key/provider_api_key")
    if root["schema_version"] != "memory-lab-secrets-v1":
        raise ConfigError("unsupported credential schema_version")
    provider_key = f"{config.provider}_api_key"
    key = root.get(provider_key) or root.get("api_key") or root.get("deepseek_api_key")
    if not isinstance(key, str) or not key.strip():
        raise ConfigError("api key has not been configured")
    return key.strip()
