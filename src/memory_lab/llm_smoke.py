"""Small LLM connectivity check for OpenAI-compatible providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, load_llm_config
from .llm_client import OpenAICompatibleJsonClient


def run_llm_smoke_test(
    *,
    config_path: Path = DEFAULT_CONFIG,
    client: OpenAICompatibleJsonClient | None = None,
) -> dict[str, Any]:
    """Call the configured model with a tiny strict-JSON request."""

    config = load_llm_config(config_path)
    model = client or OpenAICompatibleJsonClient.from_config(config_path)
    result = model.complete_json(
        system_prompt=(
            "你是一个连通性测试助手。只能输出严格 JSON 对象，不要输出 Markdown 或额外文本。"
        ),
        user_payload={
            "task": "return exactly one JSON object",
            "required": {
                "ok": True,
                "message": "pong",
            },
        },
        max_tokens=120,
    )
    return {
        "ok": True,
        "status": "complete",
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "credential_source": (
            {"type": "env", "name": config.credential_env}
            if config.credential_env
            else {"type": "file", "path": str(config.credential_file)}
        ),
        "result": result,
    }
