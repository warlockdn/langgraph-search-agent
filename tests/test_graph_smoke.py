from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_search.config import AppConfig
from agent_search.graph import build_graph
from agent_search.nodes import AgentSearchNodes
from tests.conftest import FakeRetriever, build_evidence


def _default_builder(**kwargs):
    query = kwargs["query"]
    query_type = kwargs["query_type"]
    subquestion_id = kwargs["subquestion_id"]
    evidence = [
        build_evidence(
            query=query,
            query_type=query_type,
            subquestion_id=subquestion_id,
            url_suffix="1",
        )
    ]
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


def test_agent_search_nodes_facade_exposes_graph_methods() -> None:
    nodes = AgentSearchNodes(
        retriever=FakeRetriever(_default_builder),
        config=AppConfig(enable_llm=False),
    )

    for method_name in (
        "prepare_tool_input",
        "initial_tool_choice",
        "call_tool",
        "start_agent_search",
        "generate_sub_answers_subgraph",
        "retrieve_orig_question_docs_subgraph_wrapper",
        "generate_initial_answer",
        "validate_initial_answer",
        "extract_entity_term",
        "decide_refinement_need",
        "create_refined_sub_questions",
        "answer_refined_question_subgraphs",
        "ingest_refined_sub_answers",
        "generate_validate_refined_answer",
        "compare_answers",
        "logging_node",
    ):
        assert callable(getattr(nodes, method_name))


@pytest.mark.asyncio
async def test_build_graph_still_invokes_through_facade_nodes() -> None:
    retriever = FakeRetriever(_default_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)

    state = await graph.ainvoke({"messages": [HumanMessage(content="What is LangGraph?")]})

    assert state["final_answer"]["answer"]
    assert isinstance(state["messages"][-1], AIMessage)
    assert state["run_metadata"]["route"] == "simple"
