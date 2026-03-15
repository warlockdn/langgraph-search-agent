from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_search.prompts import PLANNER_PROMPT
from agent_search.schemas import PlannerDecision
from agent_search.schemas import RefinementDecision, RunMetadata
from agent_search.state import (
    AgentSearchStateInput,
    AgentSearchStateUpdateDict,
    dump_state_update,
    load_state_update,
)


class RoutingMixin:
    async def prepare_tool_input(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        loaded = load_state_update(state)
        state = loaded.model_dump()
        question, conversation_context = self._extract_question_from_input(
            question=state.get("question"),
            messages=list(loaded.messages or []),
        )
        request = state.get("search_request") or {}
        normalized_question = self._normalize_question_with_context(
            question=question,
            conversation_context=conversation_context,
        )
        search_mode = request.get("search_mode", "auto")

        if search_mode in {"general", "code", "hybrid"}:
            query_type = search_mode
            complexity = self._infer_complexity(normalized_question)
            time_sensitive, time_reason = self._infer_time_sensitivity(normalized_question)
        else:
            decision = await self._plan_tool_input(normalized_question)
            query_type = decision["query_type"]
            complexity = decision["complexity"]
            time_sensitive = decision["time_sensitive"]
            time_reason = decision["time_sensitivity_reason"]
        max_subquestions = request.get(
            "max_subquestions", self.config.max_subquestions_default
        )
        max_refinement_rounds = request.get(
            "max_refinement_rounds", self.config.max_refinement_rounds_default
        )

        metadata = RunMetadata(
            started_at=self._now(),
            route=complexity,
            query_type=query_type,
            max_subquestions=max_subquestions,
            max_refinement_rounds=max_refinement_rounds,
            refinement_rounds=0,
            needs_refinement=False,
            time_sensitive=time_sensitive,
            time_sensitivity_reason=time_reason,
        )
        self._emit_progress(
            "routing_started",
            question=question,
            normalized_question=normalized_question,
            has_history=bool(conversation_context),
        )
        return dump_state_update({
            "question": question,
            "normalized_question": normalized_question,
            "query_type": query_type,
            "complexity": complexity,
            "time_sensitive": time_sensitive,
            "run_metadata": metadata.model_dump(),
        })

    async def _plan_tool_input(self, question: str) -> dict[str, Any]:
        heuristic_query_type = self._infer_query_type(question, "auto")
        heuristic_complexity = self._infer_complexity(question)
        heuristic_time_sensitive, heuristic_time_reason = self._infer_time_sensitivity(question)

        if self.llm is None:
            return {
                "query_type": heuristic_query_type,
                "complexity": heuristic_complexity,
                "time_sensitive": heuristic_time_sensitive,
                "time_sensitivity_reason": heuristic_time_reason,
            }

        try:
            structured_llm = self.llm.with_structured_output(
                PlannerDecision,
                method="json_schema",
            )
            decision = await structured_llm.ainvoke(
                [
                    SystemMessage(content=PLANNER_PROMPT),
                    HumanMessage(content=f"Question: {question}"),
                ]
            )
            payload = decision.model_dump()
            if payload["query_type"] not in {"general", "code", "hybrid"}:
                payload["query_type"] = heuristic_query_type
            if payload["complexity"] not in {"simple", "agentic"}:
                payload["complexity"] = heuristic_complexity
            if payload["time_sensitive"] and not payload.get("time_sensitivity_reason"):
                payload["time_sensitivity_reason"] = heuristic_time_reason or "LLM marked the query as time-sensitive."
            return payload
        except Exception:
            return {
                "query_type": heuristic_query_type,
                "complexity": heuristic_complexity,
                "time_sensitive": heuristic_time_sensitive,
                "time_sensitivity_reason": heuristic_time_reason,
            }

    async def initial_tool_choice(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        route_intent = state.get("complexity", "simple")
        metadata = dict(state.get("run_metadata", {}))
        metadata["route"] = route_intent
        return dump_state_update(
            {"route_intent": route_intent, "run_metadata": metadata}
        )

    async def call_tool(self, state: AgentSearchStateInput) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        query = state.get("normalized_question", state["question"])
        query_type = state.get("query_type", "general")
        self._emit_progress("search_started", route="simple", query=query, query_type=query_type)
        evidence, logs = await self.retriever.retrieve(query=query, query_type=query_type)
        deduped = self._dedupe_evidence(evidence)
        self._emit_progress(
            "search_results",
            route="simple",
            query=query,
            query_type=query_type,
            result_count=len(deduped),
        )

        candidate = await self._build_candidate_answer(
            question=state["question"],
            evidence=deduped,
            label="initial",
        )
        validation_report = self._build_validation_report(
            question=state["question"],
            evidence=deduped,
            candidate=candidate,
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        final_answer = self._to_final_answer(
            state=state, candidate=candidate, used_refinement=False
        )

        metadata = dict(state.get("run_metadata", {}))
        metadata["needs_refinement"] = False
        refinement_decision = RefinementDecision(
            needs_refinement=False,
            reason="Simple route completed after one retrieval pass.",
            triggers=validation_report["unresolved_aspects"],
            remaining_rounds=int(metadata.get("max_refinement_rounds", 0)),
            max_rounds_reached=False,
        ).model_dump()

        return dump_state_update({
            "initial_results": deduped,
            "initial_answer": candidate,
            "validation_report": validation_report,
            "refinement_decision": refinement_decision,
            "tool_trace": logs,
            "final_answer": final_answer,
            "run_metadata": metadata,
        })

    def route_after_initial_choice(self, state: AgentSearchStateInput) -> str:
        state = load_state_update(state).model_dump()
        return (
            "call_tool"
            if state.get("route_intent", state.get("complexity", "simple")) == "simple"
            else "run_initial_research_agent"
        )

    def route_after_refinement_decision(self, state: AgentSearchStateInput) -> str:
        state = load_state_update(state).model_dump()
        decision = state.get("refinement_decision") or {}
        needs_refinement = bool(decision.get("needs_refinement", False))
        metadata = state.get("run_metadata", {})
        rounds = int(metadata.get("refinement_rounds", 0))
        max_rounds = int(
            metadata.get(
                "max_refinement_rounds", self.config.max_refinement_rounds_default
            )
        )
        if needs_refinement and rounds < max_rounds:
            return "run_refinement_research_agent"
        return "compare_answers"

    def _infer_query_type(self, question: str, search_mode: str) -> str:
        if search_mode in {"general", "code", "hybrid"}:
            return search_mode
        q = question.lower()
        code_hits = sum(
            1
            for kw in (
                "api",
                "sdk",
                "python",
                "typescript",
                "function",
                "class",
                "library",
                "error",
                "stack trace",
            )
            if kw in q
        )
        web_hits = sum(
            1
            for kw in (
                "news",
                "compare",
                "market",
                "company",
                "latest",
                "trend",
                "history",
            )
            if kw in q
        )
        if code_hits and web_hits:
            return "hybrid"
        if code_hits:
            return "code"
        return "general"

    def _infer_complexity(self, question: str) -> str:
        q = question.lower()
        agentic_markers = (
            "compare",
            "difference",
            "tradeoff",
            "vs",
            "versus",
            "why",
            "across",
            "multi",
            "between",
        )
        if any(marker in q for marker in agentic_markers):
            return "agentic"
        if len(question.split()) >= 18:
            return "agentic"
        return "simple"

    def _infer_time_sensitivity(self, question: str) -> tuple[bool, str | None]:
        markers = {
            "latest": "latest",
            "recent": "recent",
            "today": "today",
            "current": "current",
            "news": "news",
            "market": "market",
            "price": "price",
            "stock": "stock",
            "earnings": "earnings",
            "quarter": "quarter",
        }
        q = question.lower()
        hits = [label for marker, label in markers.items() if marker in q]
        return bool(hits), ", ".join(hits[:3]) or None
