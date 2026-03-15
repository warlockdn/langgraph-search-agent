from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agent_search.agents import (
    build_initial_research_agent,
    build_refinement_research_agent,
    build_research_subquestions,
)
from agent_search.schemas import (
    InitialResearchAgentOutput,
    RefinementResearchAgentOutput,
)
from agent_search.state import (
    AgentSearchStateInput,
    AgentSearchStateUpdateDict,
    dump_state_update,
    load_state_update,
)
from agent_search.subgraphs import dedupe_evidence


class ResearchAgentMixin:
    async def run_initial_research_agent(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        loaded = load_state_update(state).model_dump()
        metadata = dict(loaded.get("run_metadata", {}))
        metadata["route"] = "agentic"
        self._emit_progress(
            "search_started",
            route="agentic",
            query=loaded.get("normalized_question", loaded.get("question", "")),
        )

        if self.llm is None:
            return await self._run_initial_research_fallback(loaded, metadata)

        try:
            evidence_sink: list[dict[str, Any]] = []
            log_sink: list[dict[str, Any]] = []
            agent = build_initial_research_agent(
                model=self.llm,
                retriever=self.retriever,
                evidence_sink=evidence_sink,
                log_sink=log_sink,
            )
            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=self._initial_research_prompt_input(loaded)
                        )
                    ]
                },
                {
                    "recursion_limit": self._agent_recursion_limit_for(
                        loaded.get("complexity", "agentic")
                    )
                },
            )
            structured = self._coerce_initial_output(result)
            evidence = dedupe_evidence(evidence_sink)
            candidate = self._build_candidate_answer_from_text(
                question=loaded["question"],
                evidence=evidence,
                answer_text=structured.answer,
            )
            initial_subquestions = build_research_subquestions(
                log_sink,
                query_type=loaded.get("query_type", "general"),
                prefix="agent_subq",
                rationale="Agent-selected initial research query.",
            )
            return dump_state_update(
                {
                    "initial_subquestions": initial_subquestions,
                    "initial_results": evidence,
                    "orig_question_results": [
                        item
                        for item in evidence
                        if item.get("query") == loaded.get("normalized_question")
                    ],
                    "initial_answer": candidate,
                    "tool_trace": log_sink,
                    "run_metadata": metadata,
                }
            )
        except Exception:
            return await self._run_initial_research_fallback(loaded, metadata)

    async def _run_initial_research_fallback(
        self, state: dict[str, Any], metadata: dict[str, Any]
    ) -> AgentSearchStateUpdateDict:
        subquestions = self._generate_initial_subquestions(
            question=state["normalized_question"],
            query_type=state["query_type"],
            max_subquestions=int(
                metadata.get(
                    "max_subquestions", self.config.max_subquestions_default
                )
            ),
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        evidence, logs = await self.retriever.retrieve(
            query=state["normalized_question"],
            query_type=state["query_type"],
            subquestion_id=None,
        )
        orig_question_results = evidence
        if subquestions:
            sub_evidence, sub_logs = await self._retrieve_subquestions_fallback(
                subquestions=subquestions,
                query_type=state["query_type"],
            )
            evidence = evidence + sub_evidence
            logs = logs + sub_logs
        deduped = dedupe_evidence(evidence)
        candidate = await self._build_candidate_answer(
            question=state["question"],
            evidence=deduped,
            label="initial",
        )
        return dump_state_update(
            {
                "initial_subquestions": subquestions,
                "initial_results": deduped,
                "orig_question_results": orig_question_results,
                "initial_answer": candidate,
                "tool_trace": logs,
                "run_metadata": metadata,
            }
        )

    async def _retrieve_subquestions_fallback(
        self,
        *,
        subquestions: list[dict[str, Any]],
        query_type: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from agent_search.subgraphs import retrieve_for_subquestions

        return await retrieve_for_subquestions(
            retriever=self.retriever,
            subquestions=subquestions,
            query_type=query_type,
        )

    async def run_refinement_research_agent(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        loaded = load_state_update(state).model_dump()
        report = loaded.get("validation_report") or {}
        unresolved = list(report.get("unresolved_aspects", []))
        if not unresolved:
            return dump_state_update({})

        metadata = dict(loaded.get("run_metadata", {}))
        metadata["refinement_rounds"] = int(metadata.get("refinement_rounds", 0)) + 1

        if self.llm is None:
            return await self._run_refinement_fallback(loaded, metadata)

        try:
            evidence_sink: list[dict[str, Any]] = []
            log_sink: list[dict[str, Any]] = []
            agent = build_refinement_research_agent(
                model=self.llm,
                retriever=self.retriever,
                evidence_sink=evidence_sink,
                log_sink=log_sink,
            )
            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=self._refinement_prompt_input(loaded, unresolved)
                        )
                    ]
                },
                {
                    "recursion_limit": self._agent_recursion_limit_for(
                        loaded.get("complexity", "agentic")
                    )
                },
            )
            structured = self._coerce_refinement_output(result)
            entity_terms = self._extract_entity_terms(
                loaded.get("normalized_question", loaded["question"]),
                unresolved,
            )
            filtered = self._filter_relevant_evidence(
                question=loaded.get("normalized_question", loaded["question"]),
                evidence=evidence_sink,
                entity_terms=entity_terms,
            )
            deduped_refined = dedupe_evidence(filtered)
            merged = dedupe_evidence(
                list(loaded.get("initial_results", [])) + deduped_refined
            )
            candidate = self._build_candidate_answer_from_text(
                question=loaded["question"],
                evidence=merged,
                answer_text=structured.answer,
            )
            validation_report = self._build_validation_report(
                question=loaded["question"],
                evidence=merged,
                candidate=candidate,
                time_sensitive=bool(loaded.get("time_sensitive", False)),
            )
            metadata["needs_refinement"] = bool(
                validation_report.get("unresolved_aspects", [])
            )
            refined_subquestions = build_research_subquestions(
                log_sink,
                query_type=loaded.get("query_type", "general"),
                prefix="agent_refined",
                rationale="Agent-selected refinement query.",
            )
            return dump_state_update(
                {
                    "refined_subquestions": refined_subquestions,
                    "refined_results": evidence_sink,
                    "refined_results_dedup": deduped_refined,
                    "refined_answer": candidate,
                    "validation_report": validation_report,
                    "tool_trace": log_sink,
                    "run_metadata": metadata,
                }
            )
        except Exception:
            return await self._run_refinement_fallback(loaded, metadata)

    async def _run_refinement_fallback(
        self, state: dict[str, Any], metadata: dict[str, Any]
    ) -> AgentSearchStateUpdateDict:
        entity_terms = await self._extract_entity_terms_with_fallback(
            state.get("normalized_question", state["question"]),
            list((state.get("validation_report") or {}).get("unresolved_aspects", [])),
        )
        create_update = await self.create_refined_sub_questions(
            {**state, "entity_terms": entity_terms}
        )
        refined_subquestions = create_update.get("refined_subquestions", [])
        if not refined_subquestions:
            return dump_state_update(
                {
                    "entity_terms": entity_terms,
                    "refined_subquestions": [],
                    "run_metadata": metadata,
                }
            )
        evidence, logs = await self._retrieve_subquestions_fallback(
            subquestions=refined_subquestions,
            query_type=state["query_type"],
        )
        deduped = dedupe_evidence(
            self._filter_relevant_evidence(
                question=state.get("normalized_question", state["question"]),
                evidence=evidence,
                entity_terms=entity_terms,
            )
        )
        merged = dedupe_evidence(list(state.get("initial_results", [])) + deduped)
        candidate = await self._build_candidate_answer(
            question=state["question"],
            evidence=merged,
            label="refined",
        )
        validation_report = self._build_validation_report(
            question=state["question"],
            evidence=merged,
            candidate=candidate,
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        metadata["needs_refinement"] = bool(validation_report.get("unresolved_aspects", []))
        return dump_state_update(
            {
                "entity_terms": entity_terms,
                "refined_subquestions": refined_subquestions,
                "refined_results": evidence,
                "refined_results_dedup": deduped,
                "refined_answer": candidate,
                "validation_report": validation_report,
                "tool_trace": logs,
                "run_metadata": metadata,
            }
        )

    def _coerce_initial_output(self, result: dict[str, Any]) -> InitialResearchAgentOutput:
        payload = result.get("structured_response")
        if isinstance(payload, InitialResearchAgentOutput):
            return payload
        if payload is not None:
            return InitialResearchAgentOutput.model_validate(payload)
        messages = result.get("messages") or []
        text = ""
        if messages:
            content = getattr(messages[-1], "content", "")
            text = content if isinstance(content, str) else str(content)
        return InitialResearchAgentOutput(answer=text or "Unable to produce an initial answer.")

    def _coerce_refinement_output(
        self, result: dict[str, Any]
    ) -> RefinementResearchAgentOutput:
        payload = result.get("structured_response")
        if isinstance(payload, RefinementResearchAgentOutput):
            return payload
        if payload is not None:
            return RefinementResearchAgentOutput.model_validate(payload)
        messages = result.get("messages") or []
        text = ""
        if messages:
            content = getattr(messages[-1], "content", "")
            text = content if isinstance(content, str) else str(content)
        return RefinementResearchAgentOutput(answer=text or "Unable to produce a refined answer.")

    def _agent_recursion_limit_for(self, complexity: str) -> int:
        return 5 if complexity == "agentic" else 3

    def _initial_research_prompt_input(self, state: dict[str, Any]) -> str:
        return (
            f"Question: {state['question']}\n"
            f"Normalized question: {state.get('normalized_question', state['question'])}\n"
            f"Query type hint: {state.get('query_type', 'general')}\n"
            f"Time sensitive: {bool(state.get('time_sensitive', False))}\n"
            "Research the question using the available tools and return a grounded answer."
        )

    def _refinement_prompt_input(
        self, state: dict[str, Any], unresolved: list[str]
    ) -> str:
        initial_answer = ((state.get("initial_answer") or {}).get("answer") or "").strip()
        return (
            f"Question: {state['question']}\n"
            f"Normalized question: {state.get('normalized_question', state['question'])}\n"
            f"Query type hint: {state.get('query_type', 'general')}\n"
            f"Time sensitive: {bool(state.get('time_sensitive', False))}\n"
            f"Initial answer: {initial_answer}\n"
            f"Validation gaps: {unresolved}\n"
            "Use the available search tools to close the gaps and return a refined answer."
        )
