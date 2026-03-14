from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from agent_search.schemas import AnswerComparison, TraceSummary, ValidationReport
from agent_search.subgraphs import dedupe_evidence


class ValidationMixin:
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
            metadata["needs_refinement"] = bool(
                latest_report.get("unresolved_aspects", [])
            )
        else:
            metadata["needs_refinement"] = bool(
                initial_report.get("unresolved_aspects", [])
            )

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
