"""Run one synthetic, privacy-safe DeepSeek connectivity check."""

from __future__ import annotations

import json

from memory_lab.config import load_llm_config
from memory_lab.judge import ApiJudge, INPUT_SCHEMA_VERSION


def main() -> int:
    text = "以后在所有对话中统一使用中文回复。"
    request = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "request_id": "synthetic_connectivity_check",
        "claim": {
            "candidate_key": "synthetic_candidate",
            "event_id": "synthetic_event",
            "part_id": "synthetic_part",
            "source_text": text,
            "exact_quote": text,
            "start_offset": 0,
            "end_offset": len(text),
        },
        "context": {
            "session_id": "synthetic_session",
            "project_key": "synthetic_project",
            "directory": "synthetic_directory",
            "occurred_at": 1,
        },
        "hard_signals": {
            "long_term": ["以后", "所有对话"],
            "temporary": [],
            "correction": [],
            "instruction_coverage": {"status": "none", "scope": None},
        },
        "current_memories": [],
        "pending_proposals": [],
    }
    config = load_llm_config()
    result = ApiJudge.from_config().judge(request)
    print(
        json.dumps(
            {
                "ok": True,
                "provider": config.provider,
                "model": config.model,
                "synthetic_only": True,
                "judgment": {
                    "should_store": result["should_store"],
                    "type": result["type"],
                    "key": result["key"],
                    "scope": result["scope"],
                    "value": result["value"],
                    "relation": result["relation"],
                    "confidence": result["confidence"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
