from __future__ import annotations

import pytest

from agent_search.config import AppConfig
from agent_search.graph import build_graph
from tests.conftest import FakeRetriever, build_evidence


def _logs(query: str, result_count: int, tool_name: str = "exa_search_web") -> list[dict]:
    return [
        {
            "tool_name": tool_name,
            "query": query,
            "input_payload": {"query": query},
            "success": True,
            "result_count": result_count,
            "error": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]


def _one_sided_builder(**kwargs):
    query = kwargs["query"]
    query_type = kwargs["query_type"]
    subquestion_id = kwargs["subquestion_id"]
    suffix = subquestion_id or "orig"
    evidence = [
        build_evidence(
            query=query,
            query_type=query_type,
            subquestion_id=subquestion_id,
            url_suffix=suffix,
            url=f"https://narrow.example.com/{suffix}",
            title=f"Narrow {suffix}",
            content="LangGraph orchestration notes only. Direct wrappers are not covered.",
        )
    ]
    return evidence, _logs(query, len(evidence))


def _well_supported_builder(**kwargs):
    query = kwargs["query"]
    query_type = kwargs["query_type"]
    subquestion_id = kwargs["subquestion_id"]
    source_map = {
        None: (
            "https://docs.langchain.com/langgraph-overview",
            "LangChain docs",
            "In 2026, LangGraph and direct tool wrappers for Python agents serve different needs. LangGraph adds orchestration, state, and graph control.",
        ),
        "subq_1": (
            "https://langchain.com/langgraph-strengths",
            "LangGraph strengths",
            "LangGraph provides orchestration, retries, and graph state for Python agents.",
        ),
        "subq_2": (
            "https://docs.python.org/direct-tool-wrappers",
            "Direct wrapper tradeoffs",
            "Direct tool wrappers keep Python agents simple, explicit, and lightweight without graph orchestration.",
        ),
        "subq_3": (
            "https://independent.example.org/comparison",
            "Comparison",
            "A direct comparison of LangGraph and direct tool wrappers shows orchestration flexibility versus implementation simplicity for Python agents.",
        ),
    }
    url, title, content = source_map.get(
        subquestion_id,
        (
            f"https://support.example.net/{subquestion_id or 'other'}",
            f"Support {subquestion_id or 'other'}",
            "LangGraph and direct tool wrappers both appear in this evidence for Python agents.",
        ),
    )
    evidence = [
        build_evidence(
            query=query,
            query_type=query_type,
            subquestion_id=subquestion_id,
            url_suffix=(subquestion_id or "orig").replace("_", "-"),
            url=url,
            title=title,
            content=content,
        )
    ]
    return evidence, _logs(query, len(evidence))


def _improves_on_refinement_builder(**kwargs):
    query = kwargs["query"]
    query_type = kwargs["query_type"]
    subquestion_id = kwargs["subquestion_id"]
    if subquestion_id and subquestion_id.startswith("refined"):
        evidence = [
            build_evidence(
                query=query,
                query_type=query_type,
                subquestion_id=subquestion_id,
                url_suffix=f"{subquestion_id}-a",
                url="https://docs.langchain.com/refined-a",
                title="Refined official comparison",
                content="In 2026, LangGraph and direct tool wrappers for Python agents differ on orchestration, simplicity, and control.",
            ),
            build_evidence(
                query=query,
                query_type=query_type,
                subquestion_id=subquestion_id,
                url_suffix=f"{subquestion_id}-b",
                url="https://analysis.example.org/refined-b",
                title="Refined independent analysis",
                content="Independent analysis compares LangGraph with direct tool wrappers and covers strengths and limitations of both.",
            ),
        ]
        return evidence, _logs(query, len(evidence))
    return _one_sided_builder(**kwargs)


def _stale_time_sensitive_builder(**kwargs):
    query = kwargs["query"]
    query_type = kwargs["query_type"]
    subquestion_id = kwargs["subquestion_id"]
    suffix = subquestion_id or "orig"
    evidence = [
        build_evidence(
            query=query,
            query_type=query_type,
            subquestion_id=subquestion_id,
            url_suffix=suffix,
            url=f"https://markets.example.com/{suffix}",
            title=f"Market note {suffix}",
            content="A 2021 market note about Nvidia and AMD market share. Older estimates only, with no current update.",
        )
    ]
    return evidence, _logs(query, len(evidence))


@pytest.mark.asyncio
async def test_agentic_branch_merges_parallel_evidence_sources() -> None:
    retriever = FakeRetriever(_well_supported_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke(
        {"question": "Compare LangGraph vs direct tool wrappers for Python agents"}
    )

    assert any(item["subquestion_id"] is None for item in state["initial_results"])
    assert any(item["subquestion_id"] is not None for item in state["initial_results"])
    assert state["orig_question_results"]


@pytest.mark.asyncio
async def test_refinement_triggers_for_one_sided_comparisons() -> None:
    retriever = FakeRetriever(_one_sided_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke({"question": "Compare X vs Y and explain tradeoffs"})

    assert state["run_metadata"]["route"] == "agentic"
    assert state["refinement_decision"]["needs_refinement"] is True
    assert state["run_metadata"]["refinement_rounds"] >= 1
    assert len(state.get("refined_subquestions", [])) >= 1


@pytest.mark.asyncio
async def test_refinement_skips_for_well_supported_answers() -> None:
    retriever = FakeRetriever(_well_supported_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke(
        {"question": "Compare LangGraph vs direct tool wrappers for Python agents"}
    )

    assert state["refinement_decision"]["needs_refinement"] is False
    assert state["run_metadata"]["refinement_rounds"] == 0
    assert state.get("refined_subquestions", []) == []


@pytest.mark.asyncio
async def test_final_needs_refinement_matches_refined_validation_result() -> None:
    retriever = FakeRetriever(_improves_on_refinement_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke(
        {"question": "Compare LangGraph vs direct tool wrappers for Python agents"}
    )

    assert state["answer_comparison"]["chosen_answer"] == "refined"
    assert state["run_metadata"]["needs_refinement"] is False
    assert state["validation_report"]["unresolved_aspects"] == []


@pytest.mark.asyncio
async def test_lowercase_entity_queries_generate_useful_refinement_prompts() -> None:
    retriever = FakeRetriever(_one_sided_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke(
        {"question": "compare postgres vs mysql for analytics workloads"}
    )

    prompts = " ".join(item["text"].lower() for item in state.get("refined_subquestions", []))
    assert "postgres" in prompts
    assert "mysql" in prompts


@pytest.mark.asyncio
async def test_time_sensitive_queries_penalize_stale_or_undated_evidence() -> None:
    retriever = FakeRetriever(_stale_time_sensitive_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke(
        {"question": "Compare latest Nvidia vs AMD market share"}
    )

    assert state["time_sensitive"] is True
    assert state["validation_report"]["recency_score"] < 0.55
    assert state["refinement_decision"]["needs_refinement"] is True


@pytest.mark.asyncio
async def test_refinement_skips_when_max_rounds_zero() -> None:
    retriever = FakeRetriever(_one_sided_builder)
    graph = build_graph(config=AppConfig(enable_llm=False), retriever=retriever)
    state = await graph.ainvoke(
        {
            "question": "Compare A vs B",
            "search_request": {
                "question": "Compare A vs B",
                "search_mode": "auto",
                "max_subquestions": 4,
                "max_refinement_rounds": 0,
                "include_trace": False,
            },
        }
    )

    assert state["run_metadata"]["refinement_rounds"] == 0
    assert state.get("refined_subquestions", []) == []
    assert state["refinement_decision"]["max_rounds_reached"] is True
