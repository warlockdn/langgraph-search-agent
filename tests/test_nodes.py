from __future__ import annotations

import pytest

from agent_search.config import AppConfig
from agent_search.nodes import AgentSearchNodes
from agent_search.schemas import EntityExtractionResult
from tests.conftest import FakeRetriever, build_evidence


def _nodes() -> AgentSearchNodes:
    retriever = FakeRetriever(lambda **kwargs: ([], []))
    return AgentSearchNodes(retriever=retriever, config=AppConfig(enable_llm=False))


class _StructuredLLM:
    def __init__(self, result: EntityExtractionResult) -> None:
        self.result = result

    async def ainvoke(self, _messages):
        return self.result


class _FakeLLM:
    def __init__(self, result: EntityExtractionResult) -> None:
        self.result = result

    def with_structured_output(self, _schema, method="json_schema"):
        assert method == "json_schema"
        return _StructuredLLM(self.result)


class _FailingLLM:
    def with_structured_output(self, _schema, method="json_schema"):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_decide_refinement_need_uses_validation_report() -> None:
    nodes = _nodes()
    result = await nodes.decide_refinement_need(
        {
            "question": "Compare A vs B",
            "validation_report": {
                "unresolved_aspects": [
                    "Source diversity is low.",
                    "Comparison evidence is one-sided or misses one side of the question.",
                ]
            },
            "run_metadata": {
                "max_refinement_rounds": 1,
                "refinement_rounds": 0,
                "needs_refinement": False,
            },
        }
    )

    assert result["needs_refinement"] is True
    assert result["refinement_decision"]["triggers"]
    assert "refinement required" in result["refinement_decision"]["reason"].lower()


@pytest.mark.asyncio
async def test_extract_entity_term_handles_lowercase_and_mixed_case_queries() -> None:
    nodes = _nodes()
    result = await nodes.extract_entity_term(
        {
            "question": "compare postgres vs MySQL for analytics workloads",
            "normalized_question": "compare postgres vs MySQL for analytics workloads",
            "validation_report": {
                "unresolved_aspects": ["Comparison evidence is one-sided."]
            },
        }
    )

    lowered = {term.lower() for term in result["entity_terms"]}
    assert "postgres" in lowered
    assert "mysql" in lowered
    assert "analytics" in lowered or "workloads" in lowered


@pytest.mark.asyncio
async def test_extract_entity_term_prefers_llm_when_available() -> None:
    nodes = _nodes()
    nodes.llm = _FakeLLM(
        EntityExtractionResult(
            entities=["PostgreSQL", "MySQL", "analytics workloads", "compare"]
        )
    )

    result = await nodes.extract_entity_term(
        {
            "question": "compare postgres vs mysql for analytics workloads",
            "normalized_question": "compare postgres vs mysql for analytics workloads",
            "validation_report": {"unresolved_aspects": ["Comparison evidence is one-sided."]},
        }
    )

    assert result["entity_terms"] == ["PostgreSQL", "MySQL", "analytics workloads"]


@pytest.mark.asyncio
async def test_extract_entity_term_falls_back_when_llm_extraction_fails() -> None:
    nodes = _nodes()
    nodes.llm = _FailingLLM()

    result = await nodes.extract_entity_term(
        {
            "question": "compare postgres vs mysql for analytics workloads",
            "normalized_question": "compare postgres vs mysql for analytics workloads",
            "validation_report": {"unresolved_aspects": ["Comparison evidence is one-sided."]},
        }
    )

    lowered = {term.lower() for term in result["entity_terms"]}
    assert "postgres" in lowered
    assert "mysql" in lowered


@pytest.mark.asyncio
async def test_compare_answers_prefers_fewer_gaps_over_naive_scores() -> None:
    nodes = _nodes()
    initial_results = [
        build_evidence(
            query="Compare LangGraph vs direct wrappers",
            query_type="general",
            url_suffix="1",
            url="https://one.example.com/a",
            content="LangGraph offers orchestration features.",
        )
    ]
    result = await nodes.compare_answers(
        {
            "question": "Compare LangGraph vs direct wrappers",
            "initial_results": initial_results,
            "initial_answer": {
                "answer": "[initial] thin answer",
                "citations": [{"source_id": "src_1", "url": "https://one.example.com/a", "title": "A", "tool_name": "exa_search_web"}],
                "confidence": 0.82,
                "coverage_score": 0.85,
                "specificity_score": 0.8,
                "source_support_score": 0.7,
                "consistency_score": 0.75,
            },
            "refined_answer": {
                "answer": "[refined] broader answer",
                "citations": [
                    {"source_id": "src_1", "url": "https://one.example.com/a", "title": "A", "tool_name": "exa_search_web"},
                    {"source_id": "src_2", "url": "https://two.example.com/b", "title": "B", "tool_name": "exa_search_web"},
                ],
                "confidence": 0.76,
                "coverage_score": 0.74,
                "specificity_score": 0.72,
                "source_support_score": 0.76,
                "consistency_score": 0.78,
            },
            "validation_report": {
                "unresolved_aspects": [],
                "source_diversity_score": 0.8,
                "relevance_score": 0.7,
                "recency_score": 1.0,
                "one_sided_comparison": False,
            },
            "time_sensitive": False,
            "run_metadata": {"needs_refinement": False},
        }
    )

    assert result["final_answer"]["used_refinement"] is True
    assert result["answer_comparison"]["chosen_answer"] == "refined"
    assert "resolves more validation gaps" in result["answer_comparison"]["reason"].lower()
