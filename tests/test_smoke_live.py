from __future__ import annotations

import os

import pytest

from agent_search.config import AppConfig
from agent_search.graph import build_graph


LIVE = os.getenv("RUN_LIVE_EXA_TESTS", "").lower() in {"1", "true", "yes"}


@pytest.mark.asyncio
@pytest.mark.skipif(not LIVE, reason="Set RUN_LIVE_EXA_TESTS=1 to run live Exa SDK smoke tests.")
async def test_live_general_query() -> None:
    graph = build_graph(config=AppConfig.from_env())
    state = await graph.ainvoke({"question": "Latest highlights in agentic retrieval systems"})
    assert state["final_answer"]["answer"]
    assert "citations" in state["final_answer"]


@pytest.mark.asyncio
@pytest.mark.skipif(not LIVE, reason="Set RUN_LIVE_EXA_TESTS=1 to run live Exa SDK smoke tests.")
async def test_live_code_query() -> None:
    graph = build_graph(config=AppConfig.from_env())
    state = await graph.ainvoke({"question": "Python examples for LangGraph SDK usage"})
    assert state["query_type"] in {"code", "hybrid"}
    assert state["final_answer"]["answer"]


@pytest.mark.asyncio
@pytest.mark.skipif(not LIVE, reason="Set RUN_LIVE_EXA_TESTS=1 to run live Exa SDK smoke tests.")
async def test_live_ambiguous_comparison_triggers_refinement() -> None:
    graph = build_graph(config=AppConfig.from_env())
    state = await graph.ainvoke({"question": "Compare Exa web search for web facts vs code-focused docs retrieval"})
    assert state["run_metadata"]["route"] == "agentic"
    assert state["final_answer"]["answer"]
