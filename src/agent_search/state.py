from __future__ import annotations

import operator
from typing import Annotated, Any, Mapping, TypeAlias, cast

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from .schemas import (
    AnswerComparison,
    CandidateAnswer,
    Citation,
    FinalAnswer,
    RefinementDecision,
    RetrievedEvidence,
    RunMetadata,
    SearchRequest,
    SubQuestion,
    ToolInvocationLog,
    ValidationReport,
)


class AgentSearchState(TypedDict):
    question: str
    search_request: NotRequired[SearchRequest]
    normalized_question: NotRequired[str]
    query_type: NotRequired[str]
    complexity: NotRequired[str]
    route_intent: NotRequired[str]
    time_sensitive: NotRequired[bool]

    initial_subquestions: NotRequired[Annotated[list[SubQuestion], operator.add]]
    initial_results: NotRequired[Annotated[list[RetrievedEvidence], operator.add]]
    orig_question_results: NotRequired[Annotated[list[RetrievedEvidence], operator.add]]
    initial_answer: NotRequired[CandidateAnswer | None]

    entity_terms: NotRequired[list[str]]
    validation_report: NotRequired[ValidationReport]
    refinement_decision: NotRequired[RefinementDecision]
    answer_comparison: NotRequired[AnswerComparison]

    refined_subquestions: NotRequired[Annotated[list[SubQuestion], operator.add]]
    refined_results: NotRequired[Annotated[list[RetrievedEvidence], operator.add]]
    refined_results_dedup: NotRequired[list[RetrievedEvidence]]
    refined_answer: NotRequired[CandidateAnswer | None]

    final_answer: NotRequired[FinalAnswer | None]
    tool_trace: NotRequired[Annotated[list[ToolInvocationLog], operator.add]]
    errors: NotRequired[Annotated[list[str], operator.add]]
    run_metadata: NotRequired[RunMetadata]


class AgentSearchStateUpdateDict(TypedDict):
    question: NotRequired[str]
    search_request: NotRequired[SearchRequest]
    normalized_question: NotRequired[str]
    query_type: NotRequired[str]
    complexity: NotRequired[str]
    route_intent: NotRequired[str]
    time_sensitive: NotRequired[bool]

    initial_subquestions: NotRequired[list[SubQuestion]]
    initial_results: NotRequired[list[RetrievedEvidence]]
    orig_question_results: NotRequired[list[RetrievedEvidence]]
    initial_answer: NotRequired[CandidateAnswer | None]

    entity_terms: NotRequired[list[str]]
    validation_report: NotRequired[ValidationReport]
    refinement_decision: NotRequired[RefinementDecision]
    answer_comparison: NotRequired[AnswerComparison]

    refined_subquestions: NotRequired[list[SubQuestion]]
    refined_results: NotRequired[list[RetrievedEvidence]]
    refined_results_dedup: NotRequired[list[RetrievedEvidence]]
    refined_answer: NotRequired[CandidateAnswer | None]

    final_answer: NotRequired[FinalAnswer | None]
    tool_trace: NotRequired[list[ToolInvocationLog]]
    errors: NotRequired[list[str]]
    run_metadata: NotRequired["RunMetadataUpdate"]


class AgentSearchStateModel(BaseModel):
    question: str
    search_request: SearchRequest | None = None
    normalized_question: str | None = None
    query_type: str | None = None
    complexity: str | None = None
    route_intent: str | None = None
    time_sensitive: bool | None = None

    initial_subquestions: list[SubQuestion] = Field(default_factory=list)
    initial_results: list[RetrievedEvidence] = Field(default_factory=list)
    orig_question_results: list[RetrievedEvidence] = Field(default_factory=list)
    initial_answer: CandidateAnswer | None = None

    entity_terms: list[str] = Field(default_factory=list)
    validation_report: ValidationReport | None = None
    refinement_decision: RefinementDecision | None = None
    answer_comparison: AnswerComparison | None = None

    refined_subquestions: list[SubQuestion] = Field(default_factory=list)
    refined_results: list[RetrievedEvidence] = Field(default_factory=list)
    refined_results_dedup: list[RetrievedEvidence] = Field(default_factory=list)
    refined_answer: CandidateAnswer | None = None

    final_answer: FinalAnswer | None = None
    tool_trace: list[ToolInvocationLog] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_metadata: RunMetadata | None = None


class RunMetadataUpdate(BaseModel):
    started_at: str | None = None
    finished_at: str | None = None
    route: str | None = None
    query_type: str | None = None
    max_subquestions: int | None = None
    max_refinement_rounds: int | None = None
    refinement_rounds: int | None = None
    needs_refinement: bool | None = None
    time_sensitive: bool | None = None
    time_sensitivity_reason: str | None = None


AgentSearchStateInput: TypeAlias = AgentSearchState | AgentSearchStateUpdateDict


class AgentSearchStateUpdate(BaseModel):
    question: str | None = None
    search_request: SearchRequest | None = None
    normalized_question: str | None = None
    query_type: str | None = None
    complexity: str | None = None
    route_intent: str | None = None
    time_sensitive: bool | None = None

    initial_subquestions: list[SubQuestion] | None = None
    initial_results: list[RetrievedEvidence] | None = None
    orig_question_results: list[RetrievedEvidence] | None = None
    initial_answer: CandidateAnswer | None = None

    entity_terms: list[str] | None = None
    validation_report: ValidationReport | None = None
    refinement_decision: RefinementDecision | None = None
    answer_comparison: AnswerComparison | None = None

    refined_subquestions: list[SubQuestion] | None = None
    refined_results: list[RetrievedEvidence] | None = None
    refined_results_dedup: list[RetrievedEvidence] | None = None
    refined_answer: CandidateAnswer | None = None

    final_answer: FinalAnswer | None = None
    tool_trace: list[ToolInvocationLog] | None = None
    errors: list[str] | None = None
    run_metadata: RunMetadataUpdate | None = None


def load_state(state: AgentSearchStateInput | Mapping[str, Any]) -> AgentSearchStateModel:
    return AgentSearchStateModel.model_validate(state)


def load_state_update(
    state: AgentSearchStateInput | Mapping[str, Any],
) -> AgentSearchStateUpdate:
    return AgentSearchStateUpdate.model_validate(state)


def dump_state_update(
    update: AgentSearchStateUpdate | AgentSearchStateUpdateDict | Mapping[str, Any],
) -> AgentSearchStateUpdateDict:
    if isinstance(update, AgentSearchStateUpdate):
        model = update
    else:
        model = AgentSearchStateUpdate.model_validate(update)
    raw = model.model_dump()
    return cast(AgentSearchStateUpdateDict, {key: value for key, value in raw.items() if value is not None})
