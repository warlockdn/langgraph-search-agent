PLANNER_PROMPT = """
You are a query planner for agentic retrieval.
Return strict JSON with:
- query_type: one of ["general","code","hybrid"]
- complexity: one of ["simple","agentic"]
- time_sensitive: boolean
- time_sensitivity_reason: string|null
Rules:
- Mark complexity as "agentic" for multi-entity, comparison, ambiguous, or multi-hop questions.
- Mark query_type as "code" for SDK/API/programming requests, "hybrid" if both product/web + code context are needed.
- Mark time_sensitive as true for latest/current/news/market/price/date-sensitive questions.
- Prefer "simple" unless decomposition is likely to materially improve answer quality.
"""

SUBQUESTION_PROMPT = """
Generate focused, non-overlapping subquestions.
Return strict JSON list with fields:
- text
- rationale
Constraints:
- Cover all major dimensions of the original question.
- For comparison questions include explicit dimensions.
- No duplicates.
"""

VALIDATION_PROMPT = """
Evaluate answer quality for the original question.
Return strict JSON with:
- coverage_gaps: list[str]
- unsupported_claims: list[str]
- conflicting_claims: list[str]
- needs_refinement: bool
- confidence: number in [0,1]
"""

SYNTHESIS_PROMPT = """
Synthesize a direct answer from evidence.
Rules:
- Answer directly first.
- Cite claims using provided source ids.
- If evidence is thin, explicitly state uncertainty.
- Do not fabricate facts.
"""
