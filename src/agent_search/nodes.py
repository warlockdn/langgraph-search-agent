from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import AppConfig
from .prompts import SYNTHESIS_PROMPT
from .schemas import (
    AnswerComparison,
    CandidateAnswer,
    Citation,
    EntityExtractionResult,
    FinalAnswer,
    RefinementDecision,
    RunMetadata,
    SubQuestion,
    TraceSummary,
    ValidationReport,
)
from .subgraphs import dedupe_evidence, retrieve_for_subquestions


class AgentSearchNodes:
    def __init__(self, retriever: Any, config: AppConfig) -> None:
        self.retriever = retriever
        self.config = config
        self.llm = self._init_llm()

    def _init_llm(self) -> ChatOpenAI | None:
        if not self.config.enable_llm:
            return None
        try:
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=0,
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
        except Exception:
            return None

    async def prepare_tool_input(self, state: dict[str, Any]) -> dict[str, Any]:
        question = state["question"].strip()
        request = state.get("search_request") or {}
        normalized_question = re.sub(r"\s+", " ", question).strip()

        query_type = self._infer_query_type(
            normalized_question, request.get("search_mode", "auto")
        )
        complexity = self._infer_complexity(normalized_question)
        time_sensitive, time_reason = self._infer_time_sensitivity(normalized_question)
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
        return {
            "normalized_question": normalized_question,
            "query_type": query_type,
            "complexity": complexity,
            "time_sensitive": time_sensitive,
            "run_metadata": metadata.model_dump(),
        }

    async def initial_tool_choice(self, state: dict[str, Any]) -> dict[str, Any]:
        route_intent = state.get("complexity", "simple")
        metadata = dict(state.get("run_metadata", {}))
        metadata["route"] = route_intent
        return {"route_intent": route_intent, "run_metadata": metadata}

    async def call_tool(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("normalized_question", state["question"])
        query_type = state.get("query_type", "general")
        evidence, logs = await self.retriever.retrieve(query=query, query_type=query_type)
        deduped = dedupe_evidence(evidence)

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

        return {
            "initial_results": deduped,
            "initial_answer": candidate,
            "validation_report": validation_report,
            "refinement_decision": refinement_decision,
            "citations": candidate["citations"],
            "tool_trace": logs,
            "coverage_gaps": validation_report["unresolved_aspects"],
            "needs_refinement": False,
            "final_answer": final_answer,
            "run_metadata": metadata,
        }

    async def start_agent_search(self, state: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(state.get("run_metadata", {}))
        metadata["route"] = "agentic"
        return {"run_metadata": metadata}

    async def generate_sub_answers_subgraph(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        max_subq = int(
            state.get("run_metadata", {}).get(
                "max_subquestions", self.config.max_subquestions_default
            )
        )
        subquestions = self._generate_initial_subquestions(
            question=state["normalized_question"],
            query_type=state["query_type"],
            max_subquestions=max_subq,
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        evidence, logs = await retrieve_for_subquestions(
            retriever=self.retriever,
            subquestions=subquestions,
            query_type=state["query_type"],
        )
        return {
            "initial_subquestions": subquestions,
            "initial_results": evidence,
            "tool_trace": logs,
        }

    async def retrieve_orig_question_docs_subgraph_wrapper(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        evidence, logs = await self.retriever.retrieve(
            query=state["normalized_question"],
            query_type=state["query_type"],
            subquestion_id=None,
        )
        return {
            "orig_question_results": evidence,
            "initial_results": evidence,
            "tool_trace": logs,
        }

    async def generate_initial_answer(self, state: dict[str, Any]) -> dict[str, Any]:
        all_results = dedupe_evidence(state.get("initial_results", []))
        candidate = await self._build_candidate_answer(
            question=state["question"],
            evidence=all_results,
            label="initial",
        )
        return {
            "initial_answer": candidate,
            "citations": candidate["citations"],
        }

    async def validate_initial_answer(self, state: dict[str, Any]) -> dict[str, Any]:
        initial_answer = state.get("initial_answer") or {}
        evidence = dedupe_evidence(state.get("initial_results", []))
        validation_report = self._build_validation_report(
            question=state["question"],
            evidence=evidence,
            candidate=initial_answer,
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        return {
            "validation_report": validation_report,
            "coverage_gaps": validation_report["unresolved_aspects"],
        }

    async def extract_entity_term(self, state: dict[str, Any]) -> dict[str, Any]:
        question = state.get("normalized_question", state["question"])
        unresolved = (state.get("validation_report") or {}).get(
            "unresolved_aspects", state.get("coverage_gaps", [])
        )
        entity_terms = await self._extract_entity_terms_with_fallback(
            question, unresolved
        )
        return {"entity_terms": entity_terms}

    async def decide_refinement_need(self, state: dict[str, Any]) -> dict[str, Any]:
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
        return {
            "refinement_decision": decision,
            "needs_refinement": needs_refinement,
            "run_metadata": metadata,
        }

    async def create_refined_sub_questions(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        report = state.get("validation_report") or {}
        unresolved = list(report.get("unresolved_aspects", []))
        entity_terms = state.get("entity_terms", [])
        comparison_sides = report.get("comparison_sides", [])
        original_question = state.get(
            "normalized_question", state.get("question", "")
        ).strip()
        existing_texts = {
            self._normalize_prompt_text(item["text"])
            for item in state.get("initial_subquestions", []) + state.get("refined_subquestions", [])
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
        return {
            "refined_subquestions": refined,
            "run_metadata": metadata,
        }

    async def answer_refined_question_subgraphs(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        refined_subq = state.get("refined_subquestions", [])
        if not refined_subq:
            return {}
        evidence, logs = await retrieve_for_subquestions(
            retriever=self.retriever,
            subquestions=refined_subq,
            query_type=state["query_type"],
        )
        return {
            "refined_results": evidence,
            "tool_trace": logs,
        }

    async def ingest_refined_sub_answers(self, state: dict[str, Any]) -> dict[str, Any]:
        filtered = self._filter_relevant_evidence(
            question=state.get("normalized_question", state["question"]),
            evidence=state.get("refined_results", []),
            entity_terms=state.get("entity_terms", []),
        )
        deduped = dedupe_evidence(filtered)
        return {"refined_results_dedup": deduped}

    async def generate_validate_refined_answer(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
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
        return {
            "refined_answer": candidate,
            "validation_report": validation_report,
            "coverage_gaps": validation_report["unresolved_aspects"],
            "needs_refinement": needs_refinement,
            "run_metadata": metadata,
        }

    async def compare_answers(self, state: dict[str, Any]) -> dict[str, Any]:
        initial = state.get("initial_answer") or {}
        refined = state.get("refined_answer") or {}
        initial_report = self._build_validation_report(
            question=state["question"],
            evidence=dedupe_evidence(state.get("initial_results", [])),
            candidate=initial,
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        latest_report = state.get("validation_report") or initial_report
        refined_report = latest_report if refined else {}

        chosen_label, reason = self._choose_better_answer(
            initial=initial,
            initial_report=initial_report,
            refined=refined,
            refined_report=refined_report,
            time_sensitive=bool(state.get("time_sensitive", False)),
        )
        chosen = refined if chosen_label == "refined" else initial
        final = self._to_final_answer(
            state=state, candidate=chosen, used_refinement=chosen_label == "refined"
        )
        comparison = AnswerComparison(
            chosen_answer=chosen_label,
            reason=reason,
            initial_summary=self._answer_summary(initial, initial_report),
            refined_summary=self._answer_summary(refined, refined_report),
        ).model_dump()

        metadata = dict(state.get("run_metadata", {}))
        if refined:
            metadata["needs_refinement"] = bool(latest_report.get("unresolved_aspects", []))
        else:
            metadata["needs_refinement"] = bool(initial_report.get("unresolved_aspects", []))

        return {
            "final_answer": final,
            "citations": chosen.get("citations", []),
            "answer_comparison": comparison,
            "validation_report": latest_report if refined else initial_report,
            "coverage_gaps": (
                latest_report.get("unresolved_aspects", [])
                if refined
                else initial_report.get("unresolved_aspects", [])
            ),
            "needs_refinement": metadata["needs_refinement"],
            "run_metadata": metadata,
        }

    async def logging_node(self, state: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(state.get("run_metadata", {}))
        metadata["finished_at"] = self._now()
        metadata["needs_refinement"] = bool(
            state.get("needs_refinement", metadata.get("needs_refinement", False))
        )

        final_answer = dict(state.get("final_answer") or {})
        include_trace = bool(
            (state.get("search_request") or {}).get("include_trace", False)
        )
        if final_answer and include_trace:
            final_answer["trace_summary"] = self._build_trace_summary(
                state=state, run_metadata=metadata
            )
        elif final_answer:
            final_answer["trace_summary"] = None
        return {"run_metadata": metadata, "final_answer": final_answer}

    def route_after_initial_choice(self, state: dict[str, Any]) -> str:
        return (
            "call_tool"
            if state.get("route_intent", state.get("complexity", "simple")) == "simple"
            else "start_agent_search"
        )

    def route_after_refinement_decision(self, state: dict[str, Any]) -> str:
        decision = state.get("refinement_decision") or {}
        needs_refinement = bool(
            decision.get("needs_refinement", state.get("needs_refinement", False))
        )
        metadata = state.get("run_metadata", {})
        rounds = int(metadata.get("refinement_rounds", 0))
        max_rounds = int(
            metadata.get(
                "max_refinement_rounds", self.config.max_refinement_rounds_default
            )
        )
        if needs_refinement and rounds < max_rounds:
            return "create_refined_sub_questions"
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

    def _generate_initial_subquestions(
        self,
        question: str,
        query_type: str,
        max_subquestions: int,
        time_sensitive: bool,
    ) -> list[dict[str, Any]]:
        cleaned = question.strip().rstrip("?")
        comparison_sides = self._comparison_sides(cleaned)
        prompts: list[tuple[str, str]] = []

        if len(comparison_sides) >= 2:
            for side in comparison_sides[:2]:
                prompts.append(
                    (
                        f"What source-backed strengths, limitations, and relevant facts matter about {side} for answering: {cleaned}?",
                        f"Cover the {side} branch directly.",
                    )
                )
            prompts.append(
                (
                    f"What evidence directly compares {comparison_sides[0]} and {comparison_sides[1]} on the dimensions implied by: {cleaned}?",
                    "Collect direct comparison evidence instead of isolated facts.",
                )
            )
            if time_sensitive:
                prompts.append(
                    (
                        f"What recent dated evidence changed the comparison between {comparison_sides[0]} and {comparison_sides[1]}?",
                        "Add recency coverage for a time-sensitive comparison.",
                    )
                )
        else:
            prompts.append(
                (
                    f"Which source-backed facts directly answer: {cleaned}?",
                    "Cover the core factual answer path.",
                )
            )
            prompts.append(
                (
                    f"What limitations, edge cases, or counterpoints materially affect: {cleaned}?",
                    "Cover caveats and opposing evidence.",
                )
            )
            if query_type in {"code", "hybrid"}:
                prompts.append(
                    (
                        f"What implementation details, APIs, or operational constraints matter for: {cleaned}?",
                        "Cover code-facing details separately from general facts.",
                    )
                )
            if time_sensitive:
                prompts.append(
                    (
                        f"What recent dated updates materially change the answer to: {cleaned}?",
                        "Cover freshness-sensitive evidence.",
                    )
                )

        subquestions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx, (text, rationale) in enumerate(prompts, start=1):
            norm = self._normalize_prompt_text(text)
            if norm in seen:
                continue
            seen.add(norm)
            subquestions.append(
                SubQuestion(
                    id=f"subq_{idx}",
                    text=text,
                    rationale=rationale,
                    query_type=query_type,
                ).model_dump()
            )
            if len(subquestions) >= max_subquestions:
                break
        return subquestions

    async def _build_candidate_answer(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        label: str,
    ) -> dict[str, Any]:
        citations = self._citations_from_evidence(evidence)
        missing_aspects: list[str] = []

        if not evidence:
            missing_aspects = ["No supporting evidence retrieved"]
            return CandidateAnswer(
                answer="I could not find enough evidence to answer confidently.",
                citations=[],
                confidence=0.15,
                missing_aspects=missing_aspects,
                coverage_score=0.1,
                specificity_score=0.1,
                source_support_score=0.0,
                consistency_score=0.2,
            ).model_dump()

        answer_text = await self._synthesize_text(question=question, evidence=evidence)
        relevance = self._evidence_relevance_score(question, evidence)
        diversity = self._source_diversity_score(evidence)
        citation_support = min(1.0, len(citations) / 4.0)
        coverage = min(1.0, 0.25 + (relevance * 0.4) + (citation_support * 0.2) + (diversity * 0.15))
        specificity = min(
            1.0,
            0.45
            + (0.2 if any(item.get("subquestion_id") for item in evidence) else 0.0)
            + (0.2 if len(citations) >= 2 else 0.0)
            + (0.1 if len(self._comparison_sides(question)) >= 2 else 0.0),
        )
        consistency = max(0.0, 0.85 - (0.15 * len(self._contradiction_signals(evidence))))
        source_support = min(1.0, (citation_support * 0.6) + (diversity * 0.4))
        confidence = min(
            1.0, (coverage + specificity + consistency + source_support) / 4.0
        )

        return CandidateAnswer(
            answer=f"[{label}] {answer_text}",
            citations=citations,
            confidence=confidence,
            missing_aspects=missing_aspects,
            coverage_score=coverage,
            specificity_score=specificity,
            source_support_score=source_support,
            consistency_score=consistency,
        ).model_dump()

    async def _synthesize_text(
        self, question: str, evidence: list[dict[str, Any]]
    ) -> str:
        top = self._select_relevant_evidence(question=question, evidence=evidence, limit=6)
        context = "\n".join(
            f"- {item['source_id']} | {item.get('title') or 'Untitled'} | {item.get('content', '')[:320]}"
            for item in top
        )
        if self.llm is None:
            return self._extractive_summary(question=question, evidence=top)

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=SYNTHESIS_PROMPT),
                    HumanMessage(
                        content=f"Question:\n{question}\n\nEvidence:\n{context}"
                    ),
                ]
            )
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return (
                text.strip()
                if text.strip()
                else "Unable to synthesize a non-empty answer from evidence."
            )
        except Exception:
            return self._extractive_summary(question=question, evidence=top)

    def _select_relevant_evidence(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        if not evidence:
            return []
        terms = self._question_terms(question)
        time_sensitive, _ = self._infer_time_sensitivity(question)

        def _score(item: dict[str, Any]) -> tuple[float, float, int]:
            title = (item.get("title") or "").lower()
            content = (item.get("content") or "").lower()
            text = f"{title} {content}"
            overlap = (
                sum(1 for term in terms if term in text) / max(1, len(terms))
                if terms
                else 0.0
            )
            recency = self._evidence_recency_score(item) if time_sensitive else 0.0
            has_subq = 1 if item.get("subquestion_id") else 0
            return (overlap, recency, has_subq)

        ranked = sorted(evidence, key=_score, reverse=True)
        selected = [item for item in ranked if _score(item)[0] > 0][:limit]
        if selected:
            return selected
        return ranked[:limit]

    def _question_terms(self, question: str) -> set[str]:
        stop_words = {
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
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "it",
            "this",
            "that",
            "between",
            "across",
            "throughout",
            "latest",
            "recent",
            "current",
            "today",
            "compare",
            "versus",
            "vs",
        }
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9._/-]{2,}", question.lower())
        return {token for token in tokens if token not in stop_words}

    def _extractive_summary(self, question: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "Insufficient evidence to produce an extractive summary."

        lines: list[str] = [f"Answer (extractive): {question}"]
        if len(self._comparison_sides(question)) >= 2:
            lines.append("Most relevant comparison points:")
        else:
            lines.append("Most relevant evidence-backed points:")

        for item in evidence[:5]:
            claim = self._best_claim_snippet(item.get("content", ""))
            title = item.get("title") or "Untitled source"
            source = item.get("source_id", "src")
            lines.append(f"- {claim} [{source}] ({title})")

        lines.append(
            "Note: LLM synthesis is unavailable, so this answer is a deterministic extractive summary."
        )
        return "\n".join(lines)

    def _best_claim_snippet(self, content: str) -> str:
        if not content:
            return "No textual snippet available."
        text = re.sub(r"\s+", " ", content).strip()
        if not text:
            return "No textual snippet available."
        parts = re.split(r"(?<=[.!?])\s+", text)
        for part in parts:
            cleaned = part.strip(" -\n\t")
            if len(cleaned) >= 40:
                return cleaned[:220]
        return text[:220]

    def _build_validation_report(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        candidate: dict[str, Any],
        time_sensitive: bool,
    ) -> dict[str, Any]:
        deduped = dedupe_evidence(evidence)
        comparison_sides = self._comparison_sides(question)
        relevance = self._evidence_relevance_score(question, deduped)
        source_domains = self._source_domains(deduped)
        diversity = self._source_diversity_score(deduped)
        recency = self._recency_support_score(deduped, time_sensitive)
        contradiction_signals = self._contradiction_signals(deduped)
        comparison_coverage, one_sided = self._comparison_coverage(
            deduped, comparison_sides
        )

        unresolved: list[str] = []
        min_evidence = 3 if len(comparison_sides) >= 2 else 2

        if not deduped:
            unresolved.append("No evidence was retrieved.")
        if len(deduped) < min_evidence:
            unresolved.append("Evidence set is too thin for a reliable answer.")
        if relevance < 0.3:
            unresolved.append("Evidence overlap with the question is weak.")
        if diversity < 0.45 and len(deduped) >= 2:
            unresolved.append("Source diversity is low.")
        if time_sensitive and recency < 0.55:
            unresolved.append(
                "Evidence is stale or undated for this time-sensitive question."
            )
        if one_sided:
            unresolved.append(
                "Comparison evidence is one-sided or misses one side of the question."
            )
        if contradiction_signals:
            unresolved.append("Evidence contains unresolved conflicting or caveated signals.")
        if deduped and float(candidate.get("confidence", 0.0)) < 0.45:
            unresolved.append("Answer confidence is still low after synthesis.")

        return ValidationReport(
            relevance_score=relevance,
            source_diversity_score=diversity,
            evidence_count=len(deduped),
            citation_count=len(candidate.get("citations", [])),
            source_domains=source_domains,
            recency_score=recency,
            time_sensitive=time_sensitive,
            contradiction_signals=contradiction_signals,
            unresolved_aspects=unresolved,
            comparison_sides=comparison_sides,
            comparison_coverage=comparison_coverage,
            one_sided_comparison=one_sided,
        ).model_dump()

    def _evidence_relevance_score(
        self, question: str, evidence: list[dict[str, Any]]
    ) -> float:
        if not evidence:
            return 0.0
        terms = self._question_terms(question)
        if not terms:
            return 0.0
        scores: list[float] = []
        for item in evidence:
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            overlap = sum(1 for term in terms if term in text)
            scores.append(overlap / len(terms))
        scores.sort(reverse=True)
        window = scores[: min(4, len(scores))]
        return round(sum(window) / max(1, len(window)), 3)

    def _source_domains(self, evidence: list[dict[str, Any]]) -> list[str]:
        domains: list[str] = []
        for item in evidence:
            url = item.get("url") or ""
            host = urlparse(url).netloc.lower()
            if host and host not in domains:
                domains.append(host)
        return domains

    def _source_diversity_score(self, evidence: list[dict[str, Any]]) -> float:
        if not evidence:
            return 0.0
        domains = self._source_domains(evidence)
        tool_names = {
            item.get("tool_name", "") for item in evidence if item.get("tool_name")
        }
        domain_score = min(1.0, len(domains) / 3.0)
        tool_score = min(1.0, len(tool_names) / 2.0)
        return round((domain_score * 0.75) + (tool_score * 0.25), 3)

    def _recency_support_score(
        self, evidence: list[dict[str, Any]], time_sensitive: bool
    ) -> float:
        if not evidence:
            return 0.0
        if not time_sensitive:
            return 1.0
        scores = sorted(
            (self._evidence_recency_score(item) for item in evidence), reverse=True
        )
        window = scores[: min(3, len(scores))]
        return round(sum(window) / max(1, len(window)), 3)

    def _evidence_recency_score(self, item: dict[str, Any]) -> float:
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("content") or ""),
                str(item.get("url") or ""),
                str(item.get("query") or ""),
            ]
        )
        current_year = datetime.now(UTC).year
        years = [
            int(match)
            for match in re.findall(r"\b(20\d{2})\b", text)
            if 2000 <= int(match) <= current_year + 1
        ]
        if not years:
            return 0.2
        latest_year = max(years)
        delta = current_year - latest_year
        if delta <= 0:
            return 1.0
        if delta == 1:
            return 0.85
        if delta == 2:
            return 0.55
        if delta == 3:
            return 0.3
        return 0.1

    def _contradiction_signals(self, evidence: list[dict[str, Any]]) -> list[str]:
        if not evidence:
            return []
        marker_groups = {
            "conflict": ("conflict", "contradict", "disagree", "disputed"),
            "caveat": ("however", "although", "but", "depends"),
            "uncertainty": ("unclear", "mixed", "unconfirmed", "rumor"),
        }
        found: set[str] = set()
        for item in evidence:
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            for label, markers in marker_groups.items():
                if any(marker in text for marker in markers):
                    found.add(label)
        return sorted(found)

    def _comparison_sides(self, question: str) -> list[str]:
        cleaned = question.strip().rstrip("?")

        patterns = (
            r"compare\s+(.+?)\s+(?:vs\.?|versus|and)\s+(.+)",
            r"(.+?)\s+(?:vs\.?|versus)\s+(.+)",
            r"between\s+(.+?)\s+and\s+(.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if not match:
                continue
            left = self._trim_comparison_side(match.group(1))
            right = self._trim_comparison_side(match.group(2))
            sides = [side for side in (left, right) if side]
            if len(sides) >= 2:
                return sides[:2]
        return []

    def _trim_comparison_side(self, value: str) -> str:
        text = re.sub(r"^\s*compare\s+", "", value, flags=re.IGNORECASE).strip(" ,.")
        text = re.sub(
            r"\s+(?:for|on|in|across|with|regarding|when|under)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ,.")
        return text

    def _comparison_coverage(
        self, evidence: list[dict[str, Any]], comparison_sides: list[str]
    ) -> tuple[float, bool]:
        if len(comparison_sides) < 2:
            return 1.0, False

        side_hits = 0
        for side in comparison_sides[:2]:
            side_terms = self._question_terms(side) or {side.lower()}
            matched = False
            for item in evidence:
                text = f"{item.get('title', '')} {item.get('content', '')}".lower()
                overlap = sum(1 for term in side_terms if term in text)
                threshold = 1 if len(side_terms) <= 2 else 2
                if overlap >= threshold:
                    matched = True
                    break
            if matched:
                side_hits += 1

        coverage = side_hits / 2.0
        one_sided = coverage < 1.0 or len(self._source_domains(evidence)) < 2
        return round(coverage, 3), one_sided

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
        if "stale" in lowered or "undated" in lowered or (time_sensitive and "time-sensitive" in lowered):
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
            score = float(overlap * 2 + entity_overlap + (1 if item.get("subquestion_id") else 0))
            fallback_ranked.append((score, item))
            if score > 0:
                kept.append(item)

        if kept:
            return kept
        fallback_ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in fallback_ranked[:2]]

    def _citations_from_evidence(
        self, evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        deduped = dedupe_evidence(evidence)
        citations: list[dict[str, Any]] = []
        for item in deduped[:8]:
            citation = Citation(
                source_id=item["source_id"],
                url=item.get("url", ""),
                title=item.get("title", "Untitled source"),
                tool_name=item.get("tool_name", "unknown_tool"),
            )
            citations.append(citation.model_dump())
        return citations

    def _answer_summary(
        self, candidate: dict[str, Any], report: dict[str, Any]
    ) -> dict[str, Any]:
        if not candidate:
            return {}
        return {
            "confidence": round(float(candidate.get("confidence", 0.0)), 3),
            "citation_count": len(candidate.get("citations", [])),
            "unresolved_gap_count": len(report.get("unresolved_aspects", [])),
            "source_diversity_score": round(
                float(report.get("source_diversity_score", 0.0)), 3
            ),
            "relevance_score": round(float(report.get("relevance_score", 0.0)), 3),
            "recency_score": round(float(report.get("recency_score", 0.0)), 3),
            "one_sided_comparison": bool(report.get("one_sided_comparison", False)),
        }

    def _choose_better_answer(
        self,
        initial: dict[str, Any],
        initial_report: dict[str, Any],
        refined: dict[str, Any],
        refined_report: dict[str, Any],
        time_sensitive: bool,
    ) -> tuple[str, str]:
        if not refined:
            return "initial", "No refined answer was generated."

        initial_gaps = len(initial_report.get("unresolved_aspects", []))
        refined_gaps = len(refined_report.get("unresolved_aspects", []))
        if refined_gaps < initial_gaps:
            return "refined", "Refined answer resolves more validation gaps."
        if refined_gaps > initial_gaps:
            return "initial", "Initial answer leaves fewer unresolved validation gaps."

        initial_one_sided = bool(initial_report.get("one_sided_comparison", False))
        refined_one_sided = bool(refined_report.get("one_sided_comparison", False))
        if initial_one_sided and not refined_one_sided:
            return "refined", "Refined answer fixes one-sided comparison coverage."
        if refined_one_sided and not initial_one_sided:
            return "initial", "Refined answer remains one-sided while the initial answer does not."

        if time_sensitive:
            initial_recency = float(initial_report.get("recency_score", 0.0))
            refined_recency = float(refined_report.get("recency_score", 0.0))
            if refined_recency > initial_recency + 0.1:
                return "refined", "Refined answer has stronger recency support."
            if initial_recency > refined_recency + 0.1:
                return "initial", "Initial answer has stronger recency support."

        initial_diversity = float(initial_report.get("source_diversity_score", 0.0))
        refined_diversity = float(refined_report.get("source_diversity_score", 0.0))
        if refined_diversity > initial_diversity + 0.1:
            return "refined", "Refined answer is supported by a more diverse source set."
        if initial_diversity > refined_diversity + 0.1:
            return "initial", "Initial answer is supported by a more diverse source set."

        initial_citations = len(initial.get("citations", []))
        refined_citations = len(refined.get("citations", []))
        if refined_citations > initial_citations + 1:
            return "refined", "Refined answer carries stronger citation support."
        if initial_citations > refined_citations + 1:
            return "initial", "Initial answer carries stronger citation support."

        initial_relevance = float(initial_report.get("relevance_score", 0.0))
        refined_relevance = float(refined_report.get("relevance_score", 0.0))
        if refined_relevance > initial_relevance + 0.05:
            return "refined", "Refined answer is more tightly aligned to the question."
        if initial_relevance > refined_relevance + 0.05:
            return "initial", "Initial answer is more tightly aligned to the question."

        refined_score = self._composite_score(refined)
        initial_score = self._composite_score(initial)
        if refined_score > initial_score + 0.03:
            return "refined", "Refined answer wins on overall answer quality."
        return "initial", "Initial answer remains the better-supported candidate."

    def _composite_score(self, candidate: dict[str, Any]) -> float:
        if not candidate:
            return 0.0
        return (
            float(candidate.get("coverage_score", 0.0)) * 0.35
            + float(candidate.get("specificity_score", 0.0)) * 0.2
            + float(candidate.get("source_support_score", 0.0)) * 0.25
            + float(candidate.get("consistency_score", 0.0)) * 0.2
        )

    def _to_final_answer(
        self, state: dict[str, Any], candidate: dict[str, Any], used_refinement: bool
    ) -> dict[str, Any]:
        final = FinalAnswer(
            answer=candidate.get("answer", "No answer generated."),
            citations=candidate.get("citations", []),
            confidence=float(candidate.get("confidence", 0.0)),
            used_refinement=used_refinement,
            trace_summary=None,
        )
        return final.model_dump()

    def _build_trace_summary(
        self, state: dict[str, Any], run_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        started = run_metadata.get("started_at")
        finished = run_metadata.get("finished_at")
        duration_ms = 0
        if started and finished:
            try:
                start_dt = datetime.fromisoformat(started)
                end_dt = datetime.fromisoformat(finished)
                duration_ms = max(0, int((end_dt - start_dt).total_seconds() * 1000))
            except Exception:
                duration_ms = 0

        summary = TraceSummary(
            route=run_metadata.get("route", "simple"),
            query_type=run_metadata.get("query_type", "general"),
            tool_calls=len(state.get("tool_trace", [])),
            total_evidence=len(
                dedupe_evidence(
                    state.get("initial_results", [])
                    + state.get("refined_results_dedup", [])
                )
            ),
            coverage_gaps=state.get("coverage_gaps", []),
            needs_refinement=bool(run_metadata.get("needs_refinement", False)),
            refinement_rounds=int(run_metadata.get("refinement_rounds", 0)),
            error_count=len(state.get("errors", [])),
            duration_ms=duration_ms,
        )
        return summary.model_dump()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
