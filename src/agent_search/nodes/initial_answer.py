from __future__ import annotations

from typing import Any

from agent_search.schemas import SubQuestion
from agent_search.state import (
    AgentSearchStateInput,
    AgentSearchStateUpdateDict,
    dump_state_update,
    load_state_update,
)
from agent_search.subgraphs import dedupe_evidence, retrieve_for_subquestions


class InitialAnswerMixin:
    async def generate_sub_answers_subgraph(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
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
        return dump_state_update({
            "initial_subquestions": subquestions,
            "initial_results": evidence,
            "tool_trace": logs,
        })

    async def retrieve_orig_question_docs_subgraph_wrapper(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        evidence, logs = await self.retriever.retrieve(
            query=state["normalized_question"],
            query_type=state["query_type"],
            subquestion_id=None,
        )
        return dump_state_update({
            "orig_question_results": evidence,
            "initial_results": evidence,
            "tool_trace": logs,
        })

    async def generate_initial_answer(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        all_results = dedupe_evidence(state.get("initial_results", []))
        candidate = await self._build_candidate_answer(
            question=state["question"],
            evidence=all_results,
            label="initial",
        )
        return dump_state_update({
            "initial_answer": candidate,
        })

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
