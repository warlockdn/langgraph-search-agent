from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar, cast

from langchain_core.messages import ToolMessage
from pydantic import BaseModel

from agent_search.agents.tools import RETRIEVER_TOOL_NAMES
from agent_search.schemas import (
    InitialResearchAgentResult,
    InitialResearchStructuredOutput,
    QueryType,
    RefinementResearchAgentResult,
    RefinementResearchStructuredOutput,
    RetrievedEvidence,
    RetrieverToolResult,
    SubQuestion,
    ToolInvocationLog,
)


_QUERY_TYPES = {"general", "code", "hybrid"}
_StructuredModelT = TypeVar("_StructuredModelT", bound=BaseModel)


class AsyncAgentLike(Protocol):
    async def ainvoke(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        ...


async def run_initial_research_agent(
    *,
    agent: AsyncAgentLike,
    question: str,
    normalized_question: str,
    query_type: QueryType,
    max_subquestions: int,
    time_sensitive: bool,
) -> InitialResearchAgentResult:
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _initial_research_prompt_input(
                        question=question,
                        normalized_question=normalized_question,
                        query_type=query_type,
                        max_subquestions=max_subquestions,
                        time_sensitive=time_sensitive,
                    ),
                }
            ]
        }
    )
    structured = _coerce_structured_response(response, InitialResearchStructuredOutput)
    subquestions = _normalize_subquestions(
        subquestions=structured.subquestions,
        default_query_type=query_type,
        prefix="subq",
        max_items=max_subquestions,
    )
    tool_results = collect_retriever_tool_results(response.get("messages", []))
    evidence, logs = flatten_retriever_tool_results(tool_results)
    return InitialResearchAgentResult(
        subquestions=subquestions,
        initial_results=evidence,
        tool_trace=logs,
    )


async def run_refinement_research_agent(
    *,
    agent: AsyncAgentLike,
    question: str,
    normalized_question: str,
    query_type: QueryType,
    unresolved_aspects: list[str],
    entity_terms: list[str],
    comparison_sides: list[str],
    max_refined_subquestions: int = 4,
    time_sensitive: bool = False,
) -> RefinementResearchAgentResult:
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _refinement_research_prompt_input(
                        question=question,
                        normalized_question=normalized_question,
                        query_type=query_type,
                        unresolved_aspects=unresolved_aspects,
                        entity_terms=entity_terms,
                        comparison_sides=comparison_sides,
                        max_refined_subquestions=max_refined_subquestions,
                        time_sensitive=time_sensitive,
                    ),
                }
            ]
        }
    )
    structured = _coerce_structured_response(
        response, RefinementResearchStructuredOutput
    )
    refined_subquestions = _normalize_subquestions(
        subquestions=structured.refined_subquestions,
        default_query_type=query_type,
        prefix="refined_subq",
        max_items=max_refined_subquestions,
    )
    tool_results = collect_retriever_tool_results(response.get("messages", []))
    evidence, logs = flatten_retriever_tool_results(tool_results)
    return RefinementResearchAgentResult(
        refined_subquestions=refined_subquestions,
        refined_results=evidence,
        tool_trace=logs,
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


def _coerce_structured_response(
    response: dict[str, Any],
    schema: type[_StructuredModelT],
) -> _StructuredModelT:
    payload = response.get("structured_response")
    if payload is None:
        payload = _extract_json_from_ai_content(response.get("messages", []))
    if payload is None:
        return schema.model_validate({})
    return schema.model_validate(payload)


def _normalize_subquestions(
    *,
    subquestions: list[SubQuestion],
    default_query_type: QueryType,
    prefix: str,
    max_items: int,
) -> list[SubQuestion]:
    normalized: list[SubQuestion] = []
    seen: set[str] = set()
    for index, item in enumerate(subquestions, start=1):
        text = re.sub(r"\s+", " ", item.text).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            SubQuestion(
                id=item.id.strip() or f"{prefix}_{index}",
                text=text,
                rationale=item.rationale.strip() or "Agent-generated subquestion",
                query_type=_normalize_query_type(item.query_type, default_query_type),
            )
        )
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_query_type(value: str, fallback: QueryType) -> QueryType:
    if value in _QUERY_TYPES:
        return cast(QueryType, value)
    return fallback


def _extract_json_from_ai_content(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if _message_type(message) != "ai":
            continue
        content = _message_content(message)
        if isinstance(content, str):
            payload = _maybe_json_object(content)
            if payload is not None:
                return payload
    return None


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


def _initial_research_prompt_input(
    *,
    question: str,
    normalized_question: str,
    query_type: QueryType,
    max_subquestions: int,
    time_sensitive: bool,
) -> str:
    return (
        f"Original question: {question}\n"
        f"Working question: {normalized_question}\n"
        f"Default query_type: {query_type}\n"
        f"Max subquestions: {max_subquestions}\n"
        f"Time sensitive: {str(time_sensitive).lower()}\n"
        "Task: propose initial subquestions and retrieve evidence with tools."
    )


def _refinement_research_prompt_input(
    *,
    question: str,
    normalized_question: str,
    query_type: QueryType,
    unresolved_aspects: list[str],
    entity_terms: list[str],
    comparison_sides: list[str],
    max_refined_subquestions: int,
    time_sensitive: bool,
) -> str:
    return (
        f"Original question: {question}\n"
        f"Working question: {normalized_question}\n"
        f"Default query_type: {query_type}\n"
        f"Unresolved aspects: {unresolved_aspects}\n"
        f"Entity terms: {entity_terms}\n"
        f"Comparison sides: {comparison_sides}\n"
        f"Max refined subquestions: {max_refined_subquestions}\n"
        f"Time sensitive: {str(time_sensitive).lower()}\n"
        "Task: propose refined subquestions and retrieve evidence that closes those gaps."
    )


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
