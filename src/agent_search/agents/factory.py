from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel

from agent_search.agents.tools import RetrieverLike, build_retriever_tools
from agent_search.prompts import (
    INITIAL_RESEARCH_AGENT_PROMPT,
    REFINEMENT_RESEARCH_AGENT_PROMPT,
)
from agent_search.schemas import (
    InitialResearchStructuredOutput,
    RefinementResearchStructuredOutput,
)


@dataclass(slots=True)
class ResearchAgents:
    initial: Any
    refinement: Any


def create_initial_research_agent(
    *,
    model: str | BaseChatModel,
    retriever: RetrieverLike,
    system_prompt: str = INITIAL_RESEARCH_AGENT_PROMPT,
    name: str = "initial_research_agent",
    response_schema: type[Any] = InitialResearchStructuredOutput,
    evidence_sink: list[dict[str, Any]] | None = None,
    log_sink: list[dict[str, Any]] | None = None,
    debug: bool = False,
) -> Any:
    return create_agent(
        model=model,
        tools=build_retriever_tools(
            retriever,
            evidence_sink=evidence_sink,
            log_sink=log_sink,
        ),
        system_prompt=system_prompt,
        response_format=ToolStrategy(response_schema),
        name=name,
        debug=debug,
    )


def create_refinement_research_agent(
    *,
    model: str | BaseChatModel,
    retriever: RetrieverLike,
    system_prompt: str = REFINEMENT_RESEARCH_AGENT_PROMPT,
    name: str = "refinement_research_agent",
    response_schema: type[Any] = RefinementResearchStructuredOutput,
    evidence_sink: list[dict[str, Any]] | None = None,
    log_sink: list[dict[str, Any]] | None = None,
    debug: bool = False,
) -> Any:
    return create_agent(
        model=model,
        tools=build_retriever_tools(
            retriever,
            evidence_sink=evidence_sink,
            log_sink=log_sink,
        ),
        system_prompt=system_prompt,
        response_format=ToolStrategy(response_schema),
        name=name,
        debug=debug,
    )


def create_research_agents(
    *,
    model: str | BaseChatModel,
    retriever: RetrieverLike,
    debug: bool = False,
) -> ResearchAgents:
    return ResearchAgents(
        initial=create_initial_research_agent(
            model=model,
            retriever=retriever,
            debug=debug,
        ),
        refinement=create_refinement_research_agent(
            model=model,
            retriever=retriever,
            debug=debug,
        ),
    )
