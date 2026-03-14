from __future__ import annotations

import pytest

from agent_search.config import AppConfig
from agent_search.graph import build_graph
from tests.conftest import FakeRetriever, build_evidence


def _default_builder(**kwargs):
    query = kwargs["query"]
    query_type = kwargs["query_type"]
    subquestion_id = kwargs["subquestion_id"]
    evidence = [build_evidence(query=query, query_type=query_type, subquestion_id=subquestion_id, url_suffix="1")]
    logs = [
        {
            "tool_name": "exa_search_code" if query_type == "code" else "exa_search_web",
            "query": query,
            "input_payload": {"query": query},
            "success": True,
            "result_count": len(evidence),
            "error": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]
    return evidence, logs


@pytest.mark.asyncio
async def test_simple_question_routes_to_call_tool() -> None:
    retriever = FakeRetriever(_default_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke({"question": "What is LangGraph?"})

    assert state["run_metadata"]["route"] == "simple"
    assert state["final_answer"]["answer"]
    assert state["final_answer"]["citations"]
    assert state["tool_trace"]
    assert len(retriever.calls) == 1


@pytest.mark.asyncio
async def test_comparison_routes_to_agentic_branch() -> None:
    retriever = FakeRetriever(_default_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke({"question": "Compare LangGraph vs direct wrappers for Python agents"})

    assert state["run_metadata"]["route"] == "agentic"
    assert len(state.get("initial_subquestions", [])) >= 2


@pytest.mark.asyncio
async def test_code_query_prefers_code_tool_mode() -> None:
    retriever = FakeRetriever(_default_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke({"question": "How to use LangGraph in Python API code?"})

    assert state["query_type"] in {"code", "hybrid"}
    tool_names = [log["tool_name"] for log in state.get("tool_trace", [])]
    assert any(name == "exa_search_code" for name in tool_names)
