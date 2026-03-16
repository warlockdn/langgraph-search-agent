from __future__ import annotations

from typing import Any

from agent_search.agents.factory import (
    create_initial_research_agent,
    create_refinement_research_agent,
)
from agent_search.schemas import (
    InitialResearchAgentOutput,
    RefinementResearchAgentOutput,
    SubQuestion,
)


def build_research_subquestions(
    logs: list[dict[str, Any]],
    *,
    query_type: str,
    prefix: str,
    rationale: str,
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    subquestions: list[dict[str, Any]] = []
    for index, log in enumerate(logs, start=1):
        query = str(log.get("query") or "").strip()
        if not query:
            continue
        tool_name = str(log.get("tool_name") or "")
        key = (query.lower(), tool_name)
        if key in seen:
            continue
        seen.add(key)
        subquestions.append(
            SubQuestion(
                id=f"{prefix}_{index}",
                text=query,
                rationale=rationale,
                query_type=(
                    query_type
                    if query_type in {"general", "code", "hybrid"}
                    else "general"
                ),
            ).model_dump()
        )
    return subquestions


def build_initial_research_agent(
    *,
    model: Any,
    retriever: Any,
) -> Any:
    return create_initial_research_agent(
        model=model,
        retriever=retriever,
        response_schema=InitialResearchAgentOutput,
    )


def build_refinement_research_agent(
    *,
    model: Any,
    retriever: Any,
) -> Any:
    return create_refinement_research_agent(
        model=model,
        retriever=retriever,
        response_schema=RefinementResearchAgentOutput,
    )
