from __future__ import annotations

import pytest

from agent_search.config import AppConfig
from agent_search.graph import build_graph
from tests.conftest import FakeRetriever, build_evidence


def _duplicate_builder(**kwargs):
    query = kwargs["query"]
    query_type = kwargs["query_type"]
    subquestion_id = kwargs["subquestion_id"]
    evidence = [
        build_evidence(query=query, query_type=query_type, subquestion_id=subquestion_id, url_suffix="same"),
        build_evidence(query=query, query_type=query_type, subquestion_id=subquestion_id, url_suffix="same"),
    ]
    logs = [
        {
            "tool_name": "exa_search_web",
            "query": query,
            "input_payload": {"query": query},
            "success": True,
            "result_count": 2,
            "error": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]
    return evidence, logs


def _empty_builder(**kwargs):
    query = kwargs["query"]
    return [], [
        {
            "tool_name": "exa_search_web",
            "query": query,
            "input_payload": {"query": query},
            "success": True,
            "result_count": 0,
            "error": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]


@pytest.mark.asyncio
async def test_duplicate_sources_are_deduped_in_final_citations() -> None:
    retriever = FakeRetriever(_duplicate_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke({"question": "What is LangGraph?"})

    citations = state["final_answer"]["citations"]
    assert len(citations) == 1


@pytest.mark.asyncio
async def test_empty_retrieval_yields_graceful_message() -> None:
    retriever = FakeRetriever(_empty_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke({"question": "What happened in very niche event?"})

    assert "could not find enough evidence" in state["final_answer"]["answer"].lower()
    assert state["final_answer"]["confidence"] <= 0.2
