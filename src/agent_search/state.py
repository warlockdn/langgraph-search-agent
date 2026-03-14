from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import NotRequired, TypedDict

from .schemas import SearchRequest


class AgentSearchState(TypedDict):
    question: str
    search_request: NotRequired[dict[str, Any]]
    normalized_question: NotRequired[str]
    query_type: NotRequired[str]
    complexity: NotRequired[str]
    route_intent: NotRequired[str]
    time_sensitive: NotRequired[bool]

    initial_subquestions: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    initial_results: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    orig_question_results: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    initial_answer: NotRequired[dict[str, Any] | None]

    coverage_gaps: NotRequired[list[str]]
    entity_terms: NotRequired[list[str]]
    needs_refinement: NotRequired[bool]
    validation_report: NotRequired[dict[str, Any]]
    refinement_decision: NotRequired[dict[str, Any]]
    answer_comparison: NotRequired[dict[str, Any]]

    refined_subquestions: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    refined_results: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    refined_results_dedup: NotRequired[list[dict[str, Any]]]
    refined_answer: NotRequired[dict[str, Any] | None]

    final_answer: NotRequired[dict[str, Any] | None]
    citations: NotRequired[list[dict[str, Any]]]
    tool_trace: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    errors: NotRequired[Annotated[list[str], operator.add]]
    run_metadata: NotRequired[dict[str, Any]]


def make_initial_state(request: SearchRequest) -> AgentSearchState:
    return AgentSearchState(
        question=request.question,
        search_request=request.model_dump(),
    )
