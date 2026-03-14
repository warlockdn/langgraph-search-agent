from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SearchMode = Literal["auto", "general", "code", "hybrid"]
QueryType = Literal["general", "code", "hybrid"]
Complexity = Literal["simple", "agentic"]


class SearchRequest(BaseModel):
    search_mode: SearchMode = "auto"
    max_subquestions: int = Field(default=4, ge=1, le=8)
    max_refinement_rounds: int = Field(default=1, ge=0, le=3)
    include_trace: bool = False


class SubQuestion(BaseModel):
    id: str
    text: str
    rationale: str
    query_type: QueryType


class Citation(BaseModel):
    source_id: str
    url: str
    title: str
    tool_name: str


class RetrievedEvidence(BaseModel):
    source_id: str
    url: str
    title: str
    content: str
    tool_name: str
    query: str
    subquestion_id: str | None = None


class ToolInvocationLog(BaseModel):
    tool_name: str
    query: str
    input_payload: dict
    success: bool
    result_count: int = 0
    error: str | None = None
    timestamp: str


class CandidateAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_aspects: list[str] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    specificity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ValidationReport(BaseModel):
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = 0
    citation_count: int = 0
    source_domains: list[str] = Field(default_factory=list)
    recency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    time_sensitive: bool = False
    contradiction_signals: list[str] = Field(default_factory=list)
    unresolved_aspects: list[str] = Field(default_factory=list)
    comparison_sides: list[str] = Field(default_factory=list)
    comparison_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    one_sided_comparison: bool = False


class RefinementDecision(BaseModel):
    needs_refinement: bool = False
    reason: str
    triggers: list[str] = Field(default_factory=list)
    remaining_rounds: int = 0
    max_rounds_reached: bool = False


class AnswerComparison(BaseModel):
    chosen_answer: Literal["initial", "refined"]
    reason: str
    initial_summary: dict[str, Any] = Field(default_factory=dict)
    refined_summary: dict[str, Any] = Field(default_factory=dict)


class EntityExtractionResult(BaseModel):
    entities: list[str] = Field(default_factory=list)


class PlannerDecision(BaseModel):
    query_type: QueryType
    complexity: Complexity
    time_sensitive: bool = False
    time_sensitivity_reason: str | None = None


class TraceSummary(BaseModel):
    route: Complexity
    query_type: QueryType
    tool_calls: int
    total_evidence: int
    coverage_gaps: list[str] = Field(default_factory=list)
    needs_refinement: bool = False
    refinement_rounds: int = 0
    error_count: int = 0
    duration_ms: int = 0


class FinalAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    used_refinement: bool | None = None
    trace_summary: TraceSummary | None = None


class RunMetadata(BaseModel):
    started_at: str
    finished_at: str | None = None
    route: Complexity = "simple"
    query_type: QueryType = "general"
    max_subquestions: int = 4
    max_refinement_rounds: int = 1
    refinement_rounds: int = 0
    needs_refinement: bool = False
    time_sensitive: bool = False
    time_sensitivity_reason: str | None = None
