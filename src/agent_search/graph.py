from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent_search.config import AppConfig
from agent_search.exa_client import ExaSDKRetriever
from agent_search.nodes import AgentSearchNodes
from agent_search.state import AgentSearchState


def build_graph(config: AppConfig | None = None, retriever: Any | None = None):
    cfg = config or AppConfig.from_env()
    runtime_retriever = retriever or ExaSDKRetriever(cfg)
    nodes = AgentSearchNodes(retriever=runtime_retriever, config=cfg)

    builder = StateGraph(AgentSearchState)
    builder.add_node("prepare_tool_input", nodes.prepare_tool_input)
    builder.add_node("initial_tool_choice", nodes.initial_tool_choice)
    builder.add_node("call_tool", nodes.call_tool)
    builder.add_node("start_agent_search", nodes.start_agent_search)
    builder.add_node("generate_sub_answers_subgraph", nodes.generate_sub_answers_subgraph)
    builder.add_node("retrieve_orig_question_docs_subgraph_wrapper", nodes.retrieve_orig_question_docs_subgraph_wrapper)
    builder.add_node("generate_initial_answer", nodes.generate_initial_answer)
    builder.add_node("validate_initial_answer", nodes.validate_initial_answer)
    builder.add_node("extract_entity_term", nodes.extract_entity_term)
    builder.add_node("decide_refinement_need", nodes.decide_refinement_need)
    builder.add_node("create_refined_sub_questions", nodes.create_refined_sub_questions)
    builder.add_node("answer_refined_question_subgraphs", nodes.answer_refined_question_subgraphs)
    builder.add_node("ingest_refined_sub_answers", nodes.ingest_refined_sub_answers)
    builder.add_node("generate_validate_refined_answer", nodes.generate_validate_refined_answer)
    builder.add_node("compare_answers", nodes.compare_answers)
    builder.add_node("logging_node", nodes.logging_node)

    builder.add_edge(START, "prepare_tool_input")
    builder.add_edge("prepare_tool_input", "initial_tool_choice")

    builder.add_conditional_edges(
        "initial_tool_choice",
        nodes.route_after_initial_choice,
        {
            "call_tool": "call_tool",
            "start_agent_search": "start_agent_search",
        },
    )

    builder.add_edge("call_tool", "logging_node")

    builder.add_edge("start_agent_search", "generate_sub_answers_subgraph")
    builder.add_edge("start_agent_search", "retrieve_orig_question_docs_subgraph_wrapper")
    builder.add_edge("generate_sub_answers_subgraph", "generate_initial_answer")
    builder.add_edge("retrieve_orig_question_docs_subgraph_wrapper", "generate_initial_answer")

    builder.add_edge("generate_initial_answer", "validate_initial_answer")
    builder.add_edge("validate_initial_answer", "extract_entity_term")
    builder.add_edge("extract_entity_term", "decide_refinement_need")

    builder.add_conditional_edges(
        "decide_refinement_need",
        nodes.route_after_refinement_decision,
        {
            "create_refined_sub_questions": "create_refined_sub_questions",
            "compare_answers": "compare_answers",
        },
    )

    builder.add_edge("create_refined_sub_questions", "answer_refined_question_subgraphs")
    builder.add_edge("answer_refined_question_subgraphs", "ingest_refined_sub_answers")
    builder.add_edge("ingest_refined_sub_answers", "generate_validate_refined_answer")
    builder.add_edge("generate_validate_refined_answer", "compare_answers")

    builder.add_edge("compare_answers", "logging_node")
    builder.add_edge("logging_node", END)

    graph = builder.compile()
    graph.name = "agent_search_graph"
    return graph


@asynccontextmanager
async def make_graph():
    config = AppConfig.from_env()
    retriever = ExaSDKRetriever(config)
    graph = build_graph(config=config, retriever=retriever)
    try:
        yield graph
    finally:
        await retriever.aclose()
