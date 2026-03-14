from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_search.prompts import SYNTHESIS_PROMPT
from agent_search.schemas import CandidateAnswer, Citation, FinalAnswer
from agent_search.state import AgentSearchStateInput
from agent_search.subgraphs import dedupe_evidence


class SynthesisMixin:
    def _dedupe_evidence(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return dedupe_evidence(records)

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
        coverage = min(
            1.0,
            0.25 + (relevance * 0.4) + (citation_support * 0.2) + (diversity * 0.15),
        )
        specificity = min(
            1.0,
            0.45
            + (0.2 if any(item.get("subquestion_id") for item in evidence) else 0.0)
            + (0.2 if len(citations) >= 2 else 0.0)
            + (0.1 if len(self._comparison_sides(question)) >= 2 else 0.0),
        )
        consistency = max(
            0.0, 0.85 - (0.15 * len(self._contradiction_signals(evidence)))
        )
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
        self,
        state: AgentSearchStateInput,
        candidate: dict[str, Any],
        used_refinement: bool,
    ) -> dict[str, Any]:
        final = FinalAnswer(
            answer=candidate.get("answer", "No answer generated."),
            citations=candidate.get("citations", []),
            confidence=float(candidate.get("confidence", 0.0)),
            used_refinement=used_refinement,
            trace_summary=None,
        )
        return final.model_dump()
