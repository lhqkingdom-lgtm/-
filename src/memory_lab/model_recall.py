"""DeepSeek summary routing followed by verified formal-topic recall."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG
from .formal_markdown import (
    recall_selected_formal_topics,
    scan_formal_memory_index,
)
from .llm_client import OpenAICompatibleJsonClient
from .router import DeepSeekTopicSelector, TopicSelector, route_topics


def _router_topics(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for topic in index["topics"]:
        stable_keys = list(topic["stable_keys"])
        result.append(
            {
                "topic_id": topic["topic_id"],
                "name": topic["name"],
                "description": topic["summary"],
                "family": topic["type"],
                "updated_at": topic["updated_at"],
                "stable_keys": stable_keys,
                "always": any(
                    str(key).startswith("preference.response.") for key in stable_keys
                ),
            }
        )
    return result


def recall_with_topic_router(
    memory_root: Path,
    query: str,
    selector: TopicSelector,
    *,
    project_key: str | None = None,
    domain: str | None = None,
    conditions: Mapping[str, str] | None = None,
    topic_limit: int = 5,
    memory_limit: int = 5,
) -> dict[str, Any]:
    index = scan_formal_memory_index(memory_root)
    topics = _router_topics(index)
    if not topics:
        return {
            "recall_version": "memory-lab-model-topic-recall-v1",
            "generation_id": index["generation_id"],
            "state_hash": index["state_hash"],
            "query": query,
            "topic_headers_scanned": 0,
            "topic_bodies_loaded": 0,
            "selected_topics": [],
            "selected_count": 0,
            "selected": [],
            "unresolved_context": [],
            "unresolved_conflicts": [],
            "routing": {
                "schema_version": "memory-topic-router-v1",
                "candidate_count": 0,
                "selected": [],
            },
            "integrity": "current_manifest_and_empty_index_verified",
        }
    routing = route_topics(
        query,
        topics,
        selector,
        purpose="recall",
        limit=topic_limit,
    )
    topic_ids = [item["topic_id"] for item in routing["selected"]]
    if not topic_ids:
        return {
            "recall_version": "memory-lab-model-topic-recall-v1",
            "generation_id": index["generation_id"],
            "state_hash": index["state_hash"],
            "query": query,
            "topic_headers_scanned": len(topics),
            "topic_bodies_loaded": 0,
            "selected_topics": [],
            "selected_count": 0,
            "selected": [],
            "unresolved_context": [],
            "unresolved_conflicts": [],
            "routing": routing,
            "integrity": "current_manifest_index_verified_no_topic_selected",
        }
    recalled = recall_selected_formal_topics(
        memory_root,
        query,
        topic_ids,
        project_key=project_key,
        domain=domain,
        conditions=conditions,
        memory_limit=memory_limit,
    )
    recalled["recall_version"] = "memory-lab-model-topic-recall-v1"
    recalled["routing"] = routing
    return recalled


def run_deepseek_formal_recall(
    memory_root: Path,
    query: str,
    *,
    config_path: Path = DEFAULT_CONFIG,
    project_key: str | None = None,
    domain: str | None = None,
    conditions: Mapping[str, str] | None = None,
    topic_limit: int = 5,
    memory_limit: int = 5,
) -> dict[str, Any]:
    selector = DeepSeekTopicSelector(OpenAICompatibleJsonClient.from_config(config_path))
    return recall_with_topic_router(
        memory_root,
        query,
        selector,
        project_key=project_key,
        domain=domain,
        conditions=conditions,
        topic_limit=topic_limit,
        memory_limit=memory_limit,
    )
