from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_search.schemas import EntityExtractionResult, RefinementDecision, SubQuestion
from agent_search.state import (
    AgentSearchStateInput,
    AgentSearchStateUpdateDict,
    dump_state_update,
    load_state_update,
)
from agent_search.subgraphs import dedupe_evidence, retrieve_for_subquestions


class RefinementMixin:
    async def validate_initial_answer(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        initial_answer = state.get("initial_answer") or {}
        evidence = dedupe_evidence(state.get("initial_results", []))
        validation_report = self._build_validation_report(
            question=state["question"],
            evidence=evidence,
            candidate=initial_answer,
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        return dump_state_update({
            "validation_report": validation_report,
            "coverage_gaps": validation_report["unresolved_aspects"],
        })

    async def extract_entity_term(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        question = state.get("normalized_question", state["question"])
        unresolved = (state.get("validation_report") or {}).get(
            "unresolved_aspects", state.get("coverage_gaps", [])
        )
        entity_terms = await self._extract_entity_terms_with_fallback(
            question, unresolved
        )
        return dump_state_update({"entity_terms": entity_terms})

    async def decide_refinement_need(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        report = state.get("validation_report") or {}
        unresolved = list(report.get("unresolved_aspects", []))
        metadata = dict(state.get("run_metadata", {}))
        max_rounds = int(
            metadata.get(
                "max_refinement_rounds", self.config.max_refinement_rounds_default
            )
        )
        current_rounds = int(metadata.get("refinement_rounds", 0))
        remaining_rounds = max(0, max_rounds - current_rounds)
        needs_refinement = bool(unresolved) and remaining_rounds > 0

        if unresolved and remaining_rounds == 0:
            reason = "Refinement budget exhausted despite unresolved validation gaps."
        elif unresolved:
            reason = (
                "Refinement required because validation found unresolved gaps: "
                f"{'; '.join(unresolved[:3])}"
            )
        else:
            reason = "Initial answer is sufficiently supported; refinement skipped."

        decision = RefinementDecision(
            needs_refinement=needs_refinement,
            reason=reason,
            triggers=unresolved,
            remaining_rounds=remaining_rounds,
            max_rounds_reached=bool(unresolved) and remaining_rounds == 0,
        ).model_dump()
        metadata["needs_refinement"] = needs_refinement
        return dump_state_update({
            "refinement_decision": decision,
            "needs_refinement": needs_refinement,
            "run_metadata": metadata,
        })

    async def create_refined_sub_questions(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        report = state.get("validation_report") or {}
        unresolved = list(report.get("unresolved_aspects", []))
        entity_terms = state.get("entity_terms", [])
        comparison_sides = report.get("comparison_sides", [])
        original_question = state.get(
            "normalized_question", state.get("question", "")
        ).strip()
        existing_texts = {
            self._normalize_prompt_text(item["text"])
            for item in state.get("initial_subquestions", [])
            + state.get("refined_subquestions", [])
        }

        max_refined = min(4, max(2, len(unresolved) + (1 if entity_terms else 0)))
        refined: list[dict[str, Any]] = []

        for aspect in unresolved:
            prompt = self._refinement_prompt_for_aspect(
                original_question=original_question,
                aspect=aspect,
                comparison_sides=comparison_sides,
                time_sensitive=bool(state.get("time_sensitive", False)),
            )
            norm = self._normalize_prompt_text(prompt)
            if norm in existing_texts:
                continue
            refined.append(
                SubQuestion(
                    id=f"refined_subq_{len(refined) + 1}",
                    text=prompt,
                    rationale=f"Address unresolved validation gap: {aspect}",
                    query_type=state.get("query_type", "general"),
                ).model_dump()
            )
            existing_texts.add(norm)
            if len(refined) >= max_refined:
                break

        for term in entity_terms:
            if len(refined) >= max_refined:
                break
            prompt = (
                f"For '{original_question}', gather source-backed evidence specifically about "
                f"'{term}' and explain why it matters to the answer."
            )
            norm = self._normalize_prompt_text(prompt)
            if norm in existing_texts:
                continue
            refined.append(
                SubQuestion(
                    id=f"refined_entity_{len(refined) + 1}",
                    text=prompt,
                    rationale=f"Clarify entity or keyword: {term}",
                    query_type=state.get("query_type", "general"),
                ).model_dump()
            )
            existing_texts.add(norm)

        metadata = dict(state.get("run_metadata", {}))
        metadata["refinement_rounds"] = int(metadata.get("refinement_rounds", 0)) + 1
        return dump_state_update({
            "refined_subquestions": refined,
            "run_metadata": metadata,
        })

    async def answer_refined_question_subgraphs(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        refined_subq = state.get("refined_subquestions", [])
        if not refined_subq:
            return {}
        evidence, logs = await retrieve_for_subquestions(
            retriever=self.retriever,
            subquestions=refined_subq,
            query_type=state["query_type"],
        )
        return dump_state_update({
            "refined_results": evidence,
            "tool_trace": logs,
        })

    async def ingest_refined_sub_answers(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        filtered = self._filter_relevant_evidence(
            question=state.get("normalized_question", state["question"]),
            evidence=state.get("refined_results", []),
            entity_terms=state.get("entity_terms", []),
        )
        deduped = dedupe_evidence(filtered)
        return dump_state_update({"refined_results_dedup": deduped})

    async def generate_validate_refined_answer(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        merged = dedupe_evidence(
            state.get("initial_results", []) + state.get("refined_results_dedup", [])
        )
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
        needs_refinement = bool(validation_report["unresolved_aspects"])
        metadata = dict(state.get("run_metadata", {}))
        metadata["needs_refinement"] = needs_refinement
        return dump_state_update({
            "refined_answer": candidate,
            "validation_report": validation_report,
            "coverage_gaps": validation_report["unresolved_aspects"],
            "needs_refinement": needs_refinement,
            "run_metadata": metadata,
        })

    async def _extract_entity_terms_with_fallback(
        self, question: str, coverage_gaps: list[str]
    ) -> list[str]:
        if self.llm is None:
            return self._extract_entity_terms(question, coverage_gaps)

        try:
            structured_llm = self.llm.with_structured_output(
                EntityExtractionResult,
                method="json_schema",
            )
            result = await structured_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Extract the most useful entities, products, companies, "
                            "libraries, technologies, and comparison targets for follow-up "
                            "search refinement. Prefer exact phrases from the input. Include "
                            "lowercase technical terms when relevant. Return at most 8 items."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question: {question}\n"
                            f"Coverage gaps: {coverage_gaps}"
                        )
                    ),
                ]
            )
            cleaned = self._normalize_extracted_entities(result.entities)
            if cleaned:
                return cleaned
        except Exception:
            pass

        return self._extract_entity_terms(question, coverage_gaps)

    def _extract_entity_terms(
        self, question: str, coverage_gaps: list[str]
    ) -> list[str]:
        stop_words = self._entity_stop_words()
        ranked: list[str] = []

        def add_term(term: str) -> None:
            cleaned = term.strip(" ,.:;!?").strip("'\"")
            if len(cleaned) < 3:
                return
            lowered = cleaned.lower()
            if lowered in stop_words:
                return
            if lowered in {item.lower() for item in ranked}:
                return
            ranked.append(cleaned)

        for side in self._comparison_sides(question):
            add_term(side)

        for phrase in re.findall(
            r"\b(?:[A-Z][a-zA-Z0-9._/-]*)(?:\s+[A-Z][a-zA-Z0-9._/-]*)*\b", question
        ):
            add_term(phrase)

        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9._/-]{2,}", question):
            if token.lower() not in stop_words:
                add_term(token)

        for gap in coverage_gaps:
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9._/-]{2,}", gap):
                if token.lower() not in stop_words:
                    add_term(token)

        return self._normalize_extracted_entities(ranked)

    def _normalize_extracted_entities(self, entities: list[str]) -> list[str]:
        stop_words = self._entity_stop_words()
        normalized: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            cleaned = entity.strip(" ,.:;!?").strip("'\"")
            if len(cleaned) < 3:
                continue
            lowered = cleaned.lower()
            if lowered in stop_words or lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(cleaned)
        return normalized[:8]

    def _entity_stop_words(self) -> set[str]:
        return {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "what",
            "how",
            "why",
            "when",
            "where",
            "latest",
            "recent",
            "current",
            "today",
            "compare",
            "versus",
            "vs",
        }

    def _refinement_prompt_for_aspect(
        self,
        original_question: str,
        aspect: str,
        comparison_sides: list[str],
        time_sensitive: bool,
    ) -> str:
        lowered = aspect.lower()
        if "comparison" in lowered and len(comparison_sides) >= 2:
            return (
                f"For '{original_question}', find evidence that directly compares "
                f"{comparison_sides[0]} and {comparison_sides[1]} and closes this gap: {aspect}"
            )
        if "stale" in lowered or "undated" in lowered or (
            time_sensitive and "time-sensitive" in lowered
        ):
            return (
                f"For '{original_question}', find recent dated evidence from the last 1-2 years "
                f"that addresses: {aspect}"
            )
        return f"For '{original_question}', find evidence specifically addressing: {aspect}"

    def _normalize_prompt_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _filter_relevant_evidence(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        entity_terms: list[str],
    ) -> list[dict[str, Any]]:
        if not evidence:
            return []
        question_terms = self._question_terms(question)
        entity_terms_lower = {term.lower() for term in entity_terms}
        kept: list[dict[str, Any]] = []
        fallback_ranked: list[tuple[float, dict[str, Any]]] = []

        for item in evidence:
            text = " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("content") or ""),
                    str(item.get("query") or ""),
                ]
            ).lower()
            overlap = sum(1 for term in question_terms if term in text)
            entity_overlap = sum(1 for term in entity_terms_lower if term in text)
            score = float(
                overlap * 2 + entity_overlap + (1 if item.get("subquestion_id") else 0)
            )
            fallback_ranked.append((score, item))
            if score > 0:
                kept.append(item)

        if kept:
            return kept
        fallback_ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in fallback_ranked[:2]]
