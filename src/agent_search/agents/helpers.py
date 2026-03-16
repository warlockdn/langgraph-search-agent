from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from agent_search.agents.tools import RETRIEVER_TOOL_NAMES
from agent_search.schemas import (
    RetrievedEvidence,
    RetrieverToolResult,
    ToolInvocationLog,
)


def collect_retriever_tool_results(messages: list[Any]) -> list[RetrieverToolResult]:
    valid_names = set(RETRIEVER_TOOL_NAMES.values())
    results: list[RetrieverToolResult] = []
    for message in messages:
        if not _is_tool_message(message):
            continue
        if _message_name(message) not in valid_names:
            continue
        payload = _extract_tool_payload(message)
        if payload is None:
            continue
        try:
            results.append(RetrieverToolResult.model_validate(payload))
        except Exception:
            continue
    return results


def flatten_retriever_tool_results(
    results: list[RetrieverToolResult],
) -> tuple[list[RetrievedEvidence], list[ToolInvocationLog]]:
    evidence: list[RetrievedEvidence] = []
    logs: list[ToolInvocationLog] = []
    for item in results:
        evidence.extend(item.evidence)
        logs.extend(item.tool_trace)
    return evidence, logs


def _extract_tool_payload(message: Any) -> dict[str, Any] | None:
    artifact = _message_artifact(message)
    if isinstance(artifact, dict):
        return artifact

    content = _message_content(message)
    if isinstance(content, str):
        return _maybe_json_object(content)
    if isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict):
                artifact = chunk.get("artifact")
                if isinstance(artifact, dict):
                    return artifact
                text = chunk.get("text")
                if isinstance(text, str):
                    payload = _maybe_json_object(text)
                    if payload is not None:
                        return payload
    return None


def _maybe_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _is_tool_message(message: Any) -> bool:
    if isinstance(message, ToolMessage):
        return True
    return _message_type(message) == "tool"


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("type") or "")
    return str(getattr(message, "type", ""))


def _message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or "")
    return str(getattr(message, "name", ""))


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_artifact(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("artifact")
    return getattr(message, "artifact", None)
