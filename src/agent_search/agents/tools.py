from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent_search.schemas import (
    QueryType,
    RetrievedEvidence,
    RetrieverToolResult,
    ToolInvocationLog,
)


RETRIEVER_TOOL_NAMES = {
    "web": "retrieve_web_evidence",
    "code": "retrieve_code_evidence",
    "hybrid": "retrieve_hybrid_evidence",
}

_PROFILE_TO_QUERY_TYPE: dict[str, QueryType] = {
    "web": "general",
    "code": "code",
    "hybrid": "hybrid",
}


class RetrieverLike(Protocol):
    async def retrieve(
        self,
        query: str,
        query_type: str,
        subquestion_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ...


class RetrieverToolInput(BaseModel):
    query: str = Field(min_length=1)
    subquestion_id: str | None = None
    limit: int | None = Field(default=None, ge=1, le=20)


def build_retriever_tools(
    retriever: RetrieverLike,
    *,
    evidence_sink: list[dict[str, Any]] | None = None,
    log_sink: list[dict[str, Any]] | None = None,
) -> list[BaseTool]:
    return [
        _make_retriever_tool(
            retriever=retriever,
            profile="web",
            description=(
                "Retrieve general web evidence for factual/product/company/news context."
            ),
            evidence_sink=evidence_sink,
            log_sink=log_sink,
        ),
        _make_retriever_tool(
            retriever=retriever,
            profile="code",
            description=(
                "Retrieve code-focused evidence (APIs/SDK/docs/examples) with code-domain prioritization."
            ),
            evidence_sink=evidence_sink,
            log_sink=log_sink,
        ),
        _make_retriever_tool(
            retriever=retriever,
            profile="hybrid",
            description=(
                "Retrieve both web and code evidence in one call when a question needs both."
            ),
            evidence_sink=evidence_sink,
            log_sink=log_sink,
        ),
    ]


def _make_retriever_tool(
    *,
    retriever: RetrieverLike,
    profile: Literal["web", "code", "hybrid"],
    description: str,
    evidence_sink: list[dict[str, Any]] | None,
    log_sink: list[dict[str, Any]] | None,
) -> BaseTool:
    query_type = _PROFILE_TO_QUERY_TYPE[profile]
    name = RETRIEVER_TOOL_NAMES[profile]

    async def _run(
        query: str,
        subquestion_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            records, logs = await retriever.retrieve(
                query=query,
                query_type=query_type,
                subquestion_id=subquestion_id,
                limit=limit,
            )
            result = RetrieverToolResult(
                profile=profile,
                query_type=query_type,
                evidence=_normalize_evidence(records),
                tool_trace=_normalize_tool_logs(logs),
            )
        except Exception as exc:  # pragma: no cover - depends on runtime/service
            result = RetrieverToolResult(
                profile=profile,
                query_type=query_type,
                evidence=[],
                tool_trace=[
                    ToolInvocationLog(
                        tool_name=name,
                        query=query,
                        input_payload={
                            "query": query,
                            "query_type": query_type,
                            "subquestion_id": subquestion_id,
                            "limit": limit,
                        },
                        success=False,
                        result_count=0,
                        error=str(exc),
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                ],
            )

        evidence_rows = [item.model_dump() for item in result.evidence]
        log_rows = [item.model_dump() for item in result.tool_trace]
        if evidence_sink is not None:
            evidence_sink.extend(evidence_rows)
        if log_sink is not None:
            log_sink.extend(log_rows)

        artifact = result.model_dump()
        content = _tool_content_summary(
            query=query,
            result=result,
        )
        return content, artifact

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=description,
        args_schema=RetrieverToolInput,
        response_format="content_and_artifact",
    )


def _normalize_evidence(records: list[dict[str, Any]]) -> list[RetrievedEvidence]:
    normalized: list[RetrievedEvidence] = []
    for item in records:
        try:
            normalized.append(RetrievedEvidence.model_validate(item))
        except Exception:
            continue
    return normalized


def _normalize_tool_logs(logs: list[dict[str, Any]]) -> list[ToolInvocationLog]:
    normalized: list[ToolInvocationLog] = []
    for item in logs:
        try:
            normalized.append(ToolInvocationLog.model_validate(item))
        except Exception:
            continue
    return normalized


def _tool_content_summary(*, query: str, result: RetrieverToolResult) -> str:
    lines = [
        f"query: {query}",
        f"profile: {result.profile}",
        f"query_type: {result.query_type}",
        f"evidence_count: {len(result.evidence)}",
    ]
    if not result.evidence:
        return "\n".join(lines + ["evidence: none"])

    lines.append("evidence:")
    for item in result.evidence[:3]:
        title = item.title.strip() or "Untitled"
        snippet = " ".join(item.content.split())[:280] or "No content"
        lines.extend(
            [
                f"- source_id: {item.source_id}",
                f"  title: {title}",
                f"  url: {item.url}",
                f"  snippet: {snippet}",
            ]
        )
    return "\n".join(lines)
