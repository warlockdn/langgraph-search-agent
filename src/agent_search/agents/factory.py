from __future__ import annotations

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
    InitialResearchAgentOutput,
    RefinementResearchAgentOutput,
)


def create_initial_research_agent(
    *,
    model: str | BaseChatModel,
    retriever: RetrieverLike,
    system_prompt: str = INITIAL_RESEARCH_AGENT_PROMPT,
    name: str = "initial_research_agent",
    response_schema: type[Any] = InitialResearchAgentOutput,
    debug: bool = False,
) -> Any:
    return create_agent(
        model=model,
        tools=build_retriever_tools(retriever),
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
    response_schema: type[Any] = RefinementResearchAgentOutput,
    debug: bool = False,
) -> Any:
    return create_agent(
        model=model,
        tools=build_retriever_tools(retriever),
        system_prompt=system_prompt,
        response_format=ToolStrategy(response_schema),
        name=name,
        debug=debug,
    )
