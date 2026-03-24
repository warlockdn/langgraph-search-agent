from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage

from agent_search.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
from agent_search.schemas import (
    AnswerComparison,
    JudgeVerdict,
    TraceSummary,
    ValidationReport,
)
from agent_search.state import (
    AgentSearchStateInput,
    AgentSearchStateUpdateDict,
    dump_state_update,
    load_state_update,
)
from agent_search.subgraphs import dedupe_evidence


class ValidationMixin:
    async def compare_answers(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        initial = state.get("initial_answer") or {}
        refined = state.get("refined_answer") or {}
        time_sensitive = bool(state.get("time_sensitive", False))
        initial_evidence = dedupe_evidence(state.get("initial_results", []))
        refined_evidence = dedupe_evidence(
            state.get("refined_results_dedup")
            or state.get("refined_results")
            or []
        )
        initial_report = self._build_validation_report(
            question=state["question"],
            evidence=initial_evidence,
            candidate=initial,
            time_sensitive=time_sensitive,
        )
        latest_report = state.get("validation_report") or initial_report
        if refined:
            refined_report = (
                self._build_validation_report(
                    question=state["question"],
                    evidence=refined_evidence,
                    candidate=refined,
                    time_sensitive=time_sensitive,
                )
                if refined_evidence
                else latest_report
            )
        else:
            refined_report = {}

        chosen_label: Literal["initial", "refined", "tie"]
        reason: str
        judge_reasoning: str | None = None
        judge_confidence: Literal["high", "medium", "low"] | None = None
        llm_reasoning: list[dict[str, Any]] = []

        if not refined:
            chosen_label = "initial"
            reason = "No refined answer was generated."
        elif self.llm is None:
            heuristic_label, reason = self._choose_better_answer(
                initial=initial,
                initial_report=initial_report,
                refined=refined,
                refined_report=refined_report,
                time_sensitive=time_sensitive,
            )
            chosen_label = heuristic_label
        else:
            try:
                judge_llm = self._build_judge_llm()
                combined_evidence = dedupe_evidence(initial_evidence + refined_evidence)
                chosen_label, reason, judge_reasoning, judge_confidence, llm_reasoning = (
                    await self._run_judge_with_swap(
                        judge_llm=judge_llm,
                        question=state["question"],
                        context_snippets=self._judge_context_snippets(combined_evidence),
                        initial=initial,
                        refined=refined,
                        initial_report=initial_report,
                        refined_report=refined_report,
                    )
                )
            except Exception:
                heuristic_label, reason = self._choose_better_answer(
                    initial=initial,
                    initial_report=initial_report,
                    refined=refined,
                    refined_report=refined_report,
                    time_sensitive=time_sensitive,
                )
                chosen_label = heuristic_label

        effective_label = chosen_label
        if chosen_label == "tie" and refined:
            effective_label = "refined"
            reason = (
                f"{reason} Tie-break applied: kept refined answer after a tie."
            ).strip()

        chosen = refined if effective_label == "refined" else initial
        final = self._to_final_answer(
            state=state, candidate=chosen, used_refinement=effective_label == "refined"
        )
        comparison = AnswerComparison(
            chosen_answer=chosen_label,
            reason=reason,
            initial_summary=self._answer_summary(initial, initial_report),
            refined_summary=self._answer_summary(refined, refined_report),
            judge_reasoning=judge_reasoning,
            judge_confidence=judge_confidence,
        ).model_dump()

        metadata = dict(state.get("run_metadata", {}))
        final_report = latest_report if refined else initial_report
        metadata["needs_refinement"] = bool(final_report.get("unresolved_aspects", []))

        return dump_state_update({
            "final_answer": final,
            "answer_comparison": comparison,
            "validation_report": final_report,
            "llm_reasoning": llm_reasoning,
            "run_metadata": metadata,
        })

    def _build_judge_llm(self) -> Any:
        if self.llm is not None:
            return self._structured_output_runnable(self.llm, JudgeVerdict)

        llm = self._build_chat_model(self.config.judge_model)
        return self._structured_output_runnable(llm, JudgeVerdict)

    def _judge_context_snippets(
        self, evidence: list[dict[str, Any]], max_items: int = 5, max_chars: int = 2000
    ) -> list[str]:
        snippets: list[str] = []
        for item in evidence[:max_items]:
            content = str(item.get("content") or "").strip()
            if len(content) > max_chars:
                content = f"{content[:max_chars].rstrip()}..."
            snippets.append(
                "\n".join(
                    [
                        f"Title: {item.get('title', '')}",
                        f"URL: {item.get('url', '')}",
                        f"Tool: {item.get('tool_name', '')}",
                        f"Excerpt: {content}",
                    ]
                )
            )
        return snippets

    def _build_judge_prompt(
        self,
        *,
        question: str,
        context_snippets: list[str],
        answer_a_label: Literal["initial", "refined"],
        answer_a: dict[str, Any],
        answer_b_label: Literal["initial", "refined"],
        answer_b: dict[str, Any],
        initial_report: dict[str, Any],
        refined_report: dict[str, Any],
    ) -> list[Any]:
        context_block = "\n\n".join(
            f"[{idx}] {snippet}" for idx, snippet in enumerate(context_snippets, start=1)
        )
        user_payload = {
            "question": question,
            "context_snippets": context_snippets,
            "answer_a_label": answer_a_label,
            "answer_a_text": answer_a.get("answer", ""),
            "answer_a_confidence": round(float(answer_a.get("confidence", 0.0)), 3),
            "answer_a_citation_count": len(answer_a.get("citations", [])),
            "answer_b_label": answer_b_label,
            "answer_b_text": answer_b.get("answer", ""),
            "answer_b_confidence": round(float(answer_b.get("confidence", 0.0)), 3),
            "answer_b_citation_count": len(answer_b.get("citations", [])),
            "initial_report_summary": {
                "unresolved_aspects": initial_report.get("unresolved_aspects", []),
                "relevance_score": initial_report.get("relevance_score", 0.0),
                "source_diversity_score": initial_report.get(
                    "source_diversity_score", 0.0
                ),
                "recency_score": initial_report.get("recency_score", 0.0),
                "one_sided_comparison": initial_report.get("one_sided_comparison", False),
            },
            "refined_report_summary": {
                "unresolved_aspects": refined_report.get("unresolved_aspects", []),
                "relevance_score": refined_report.get("relevance_score", 0.0),
                "source_diversity_score": refined_report.get(
                    "source_diversity_score", 0.0
                ),
                "recency_score": refined_report.get("recency_score", 0.0),
                "one_sided_comparison": refined_report.get("one_sided_comparison", False),
            },
        }
        return [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT.strip()),
            HumanMessage(
                content=JUDGE_USER_PROMPT.strip().format(
                    context_block=context_block or "[none]",
                    input_payload=json.dumps(user_payload, ensure_ascii=True),
                )
            ),
        ]

    async def _run_judge_with_swap(
        self,
        *,
        judge_llm: Any,
        question: str,
        context_snippets: list[str],
        initial: dict[str, Any],
        refined: dict[str, Any],
        initial_report: dict[str, Any],
        refined_report: dict[str, Any],
    ) -> tuple[
        Literal["initial", "refined", "tie"],
        str,
        str,
        Literal["high", "medium", "low"],
        list[dict[str, Any]],
    ]:
        prompt_ab = self._build_judge_prompt(
            question=question,
            context_snippets=context_snippets,
            answer_a_label="initial",
            answer_a=initial,
            answer_b_label="refined",
            answer_b=refined,
            initial_report=initial_report,
            refined_report=refined_report,
        )
        prompt_ba = self._build_judge_prompt(
            question=question,
            context_snippets=context_snippets,
            answer_a_label="refined",
            answer_a=refined,
            answer_b_label="initial",
            answer_b=initial,
            initial_report=initial_report,
            refined_report=refined_report,
        )
        raw_ab, raw_ba = await asyncio.gather(
            judge_llm.ainvoke(prompt_ab),
            judge_llm.ainvoke(prompt_ba),
        )
        verdict_ab, raw_message_ab = self._unpack_structured_result(raw_ab)
        verdict_ba, raw_message_ba = self._unpack_structured_result(raw_ba)
        llm_reasoning = (
            self._capture_reasoning(
                payload=raw_message_ab,
                node="compare_answers",
                call_kind="judge",
                model_name=self.config.judge_model,
            )
            + self._capture_reasoning(
                payload=raw_message_ba,
                node="compare_answers",
                call_kind="judge",
                model_name=self.config.judge_model,
            )
        )
        winner_ab = verdict_ab.overall_winner
        winner_ba_mapped = {
            "initial": "refined",
            "refined": "initial",
            "tie": "tie",
        }[verdict_ba.overall_winner]
        if winner_ab == winner_ba_mapped:
            return (
                winner_ab,
                "LLM judge produced consistent pairwise verdict.",
                verdict_ab.reasoning,
                verdict_ab.confidence,
                llm_reasoning,
            )
        return (
            "tie",
            "LLM judge returned inconsistent winners across swapped ordering.",
            (
                f"AB: {verdict_ab.reasoning}\n"
                f"BA: {verdict_ba.reasoning}"
            ),
            "low",
            llm_reasoning,
        )

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
            unresolved.append(
                "Evidence contains unresolved conflicting or caveated signals."
            )
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
