"""Local, non-secret configuration for Memory Lab model experiments."""

from __future__ import annotations

import json
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
    credential_file: Path


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
    }
    unknown = set(llm) - allowed
    missing = allowed - set(llm)
    if missing:
        raise ConfigError(f"config.llm missing fields: {sorted(missing)}")
    if unknown:
        if "api_key" in unknown:
            raise ConfigError("do not store api_key in plaintext configuration")
        raise ConfigError(f"config.llm has unsupported fields: {sorted(unknown)}")
    if llm["provider"] != "deepseek":
        raise ConfigError("MVP currently supports provider=deepseek")
    for field in ("base_url", "model", "credential_file"):
        if not isinstance(llm[field], str) or not llm[field].strip():
            raise ConfigError(f"config.llm.{field} must be a non-empty string")
    if not isinstance(llm["allow_remote"], bool):
        raise ConfigError("config.llm.allow_remote must be boolean")
    timeout = llm["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
        raise ConfigError("config.llm.timeout_seconds must be between 1 and 300")
    attempts = llm["max_attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 5:
        raise ConfigError("config.llm.max_attempts must be an integer between 1 and 5")

    config_root = path.resolve().parent
    key_file = (config_root / llm["credential_file"]).resolve()
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
    )


def load_api_key(config: LLMConfig) -> str:
    """Read the user-selected local credential file without logging its value."""

    try:
        root = _object(
            json.loads(config.credential_file.read_text(encoding="utf-8")),
            "secrets",
        )
    except FileNotFoundError as error:
        raise ConfigError(f"credential file not found: {config.credential_file}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid credential JSON: {error}") from error
    if set(root) != {"schema_version", "deepseek_api_key"}:
        raise ConfigError("credential file supports only schema_version and deepseek_api_key")
    if root["schema_version"] != "memory-lab-secrets-v1":
        raise ConfigError("unsupported credential schema_version")
    key = root["deepseek_api_key"]
    if not isinstance(key, str) or not key.strip():
        raise ConfigError("deepseek_api_key has not been configured")
    return key.strip()
