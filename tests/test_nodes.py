from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent_search.config import AppConfig
from agent_search.nodes import AgentSearchNodes
from agent_search.schemas import EntityExtractionResult, JudgeDimension, JudgeVerdict
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


class _JudgeStructuredLLM:
    def __init__(self, verdicts: list[JudgeVerdict]) -> None:
        self.verdicts = verdicts

    async def ainvoke(self, _messages):
        return self.verdicts.pop(0)


class _JudgeLLM:
    def __init__(self, verdicts: list[JudgeVerdict]) -> None:
        self.verdicts = verdicts

    def with_structured_output(self, _schema, method="json_schema"):
        assert method == "json_schema"
        return _JudgeStructuredLLM(self.verdicts)


class _CapturingAgent:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[dict[str, object], dict[str, object] | None]] = []

    async def ainvoke(self, payload, config=None, **_kwargs):
        self.calls.append((payload, config))
        return self.response


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

    assert result["refinement_decision"]["triggers"]
    assert result["run_metadata"]["needs_refinement"] is True
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
async def test_run_initial_research_agent_passes_complexity_based_recursion_limit() -> None:
    retriever = FakeRetriever(lambda **kwargs: ([], []))
    nodes = AgentSearchNodes(retriever=retriever, config=AppConfig(enable_llm=False))

    agent = _CapturingAgent(
        {
            "structured_response": {
                "answer": "bounded answer",
                "key_points": [],
                "queries_used": [],
            },
            "messages": [],
        }
    )
    nodes._initial_agent = agent

    result = await nodes.run_initial_research_agent(
        {
            "question": "Compare A vs B",
            "normalized_question": "Compare A vs B",
            "query_type": "general",
            "complexity": "agentic",
            "time_sensitive": False,
            "run_metadata": {
                "started_at": "2026-01-01T00:00:00+00:00",
                "route": "agentic",
                "query_type": "general",
                "max_subquestions": 4,
                "max_refinement_rounds": 1,
                "refinement_rounds": 0,
                "needs_refinement": False,
                "time_sensitive": False,
                "time_sensitivity_reason": None,
            },
        }
    )

    assert agent.calls[0][1] == {"recursion_limit": 12}
    assert result["initial_answer"]["answer"] == "bounded answer"


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
    assert result["answer_comparison"]["chosen_answer"] in {"refined", "tie"}
    assert "resolves more validation gaps" in result["answer_comparison"]["reason"].lower()


@pytest.mark.asyncio
async def test_compare_answers_uses_llm_judge_with_swap_and_tie_breaks_to_refined() -> None:
    nodes = _nodes()
    nodes.llm = _JudgeLLM(
        [
            JudgeVerdict(
                reasoning="A and B are close. A slightly better on coverage.",
                relevance=JudgeDimension(winner="initial"),
                completeness=JudgeDimension(winner="initial"),
                groundedness=JudgeDimension(winner="tie"),
                conciseness=JudgeDimension(winner="tie"),
                overall_winner="initial",
                confidence="medium",
            ),
            JudgeVerdict(
                reasoning="Swapped order now prefers A again.",
                relevance=JudgeDimension(winner="initial"),
                completeness=JudgeDimension(winner="initial"),
                groundedness=JudgeDimension(winner="tie"),
                conciseness=JudgeDimension(winner="tie"),
                overall_winner="initial",
                confidence="medium",
            ),
        ]
    )
    result = await nodes.compare_answers(
        {
            "question": "Compare LangGraph vs direct wrappers",
            "initial_results": [
                build_evidence(
                    query="Compare LangGraph vs direct wrappers",
                    query_type="general",
                    url_suffix="1",
                    url="https://one.example.com/a",
                    content="LangGraph helps with orchestration and stateful workflows.",
                )
            ],
            "refined_results_dedup": [
                build_evidence(
                    query="Compare LangGraph vs direct wrappers",
                    query_type="general",
                    url_suffix="2",
                    url="https://two.example.com/b",
                    content="Direct wrappers can be simpler for small agents.",
                )
            ],
            "initial_answer": {
                "answer": "[initial] strong answer",
                "citations": [{"source_id": "src_1", "url": "https://one.example.com/a", "title": "A", "tool_name": "exa_search_web"}],
                "confidence": 0.82,
                "coverage_score": 0.85,
                "specificity_score": 0.8,
                "source_support_score": 0.7,
                "consistency_score": 0.75,
            },
            "refined_answer": {
                "answer": "[refined] another strong answer",
                "citations": [{"source_id": "src_2", "url": "https://two.example.com/b", "title": "B", "tool_name": "exa_search_web"}],
                "confidence": 0.79,
                "coverage_score": 0.82,
                "specificity_score": 0.78,
                "source_support_score": 0.72,
                "consistency_score": 0.8,
            },
            "time_sensitive": False,
            "run_metadata": {"needs_refinement": False},
        }
    )

    assert result["answer_comparison"]["chosen_answer"] == "tie"
    assert result["final_answer"]["used_refinement"] is True
    assert result["answer_comparison"]["judge_confidence"] == "low"


@pytest.mark.asyncio
async def test_call_tool_omits_redundant_root_summary_fields() -> None:
    evidence = [
        build_evidence(
            query="What is LangGraph?",
            query_type="general",
            url_suffix="1",
        )
    ]
    logs = [
        {
            "tool_name": "exa_search_web",
            "query": "What is LangGraph?",
            "input_payload": {"query": "What is LangGraph?"},
            "success": True,
            "result_count": 1,
            "error": None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]
    retriever = FakeRetriever(lambda **kwargs: (evidence, logs))
    nodes = AgentSearchNodes(retriever=retriever, config=AppConfig(enable_llm=False))

    result = await nodes.call_tool(
        {
            "question": "What is LangGraph?",
            "normalized_question": "What is LangGraph?",
            "query_type": "general",
            "time_sensitive": False,
            "run_metadata": {
                "started_at": "2026-01-01T00:00:00+00:00",
                "route": "simple",
                "query_type": "general",
                "max_subquestions": 4,
                "max_refinement_rounds": 1,
                "refinement_rounds": 0,
                "needs_refinement": False,
                "time_sensitive": False,
                "time_sensitivity_reason": None,
            },
        }
    )

    assert "citations" not in result
    assert "coverage_gaps" not in result
    assert "needs_refinement" not in result


@pytest.mark.asyncio
async def test_logging_node_appends_chat_ready_ai_message() -> None:
    nodes = _nodes()
    result = await nodes.logging_node(
        {
            "final_answer": {
                "answer": "LangGraph gives you stateful graph orchestration.",
                "citations": [
                    {
                        "source_id": "src_1",
                        "url": "https://example.com/1",
                        "title": "LangGraph Overview",
                        "tool_name": "exa_search_web",
                    }
                ],
                "confidence": 0.8,
                "used_refinement": False,
                "trace_summary": None,
            }
        }
    )

    assert isinstance(result["messages"][0], AIMessage)
    assert "LangGraph gives you stateful graph orchestration." in result["messages"][0].content
    assert "Sources:" in result["messages"][0].content
