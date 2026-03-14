from __future__ import annotations

import asyncio
from typing import Any


def dedupe_evidence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        url = (record.get("url") or "").strip().lower()
        title = (record.get("title") or "").strip().lower()
        content_key = (record.get("content") or "").strip().lower()[:160]
        key = url or f"{title}:{content_key}"
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


async def retrieve_for_subquestions(
    retriever: Any,
    subquestions: list[dict[str, Any]],
    query_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    async def _one(subq: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return await retriever.retrieve(
            query=subq["text"],
            query_type=query_type,
            subquestion_id=subq["id"],
        )

    outputs = await asyncio.gather(*[_one(subq) for subq in subquestions], return_exceptions=True)
    evidence: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    for item in outputs:
        if isinstance(item, Exception):
            logs.append(
                {
                    "tool_name": "multi_subquestion_retrieval",
                    "query": "parallel_subquestions",
                    "input_payload": {},
                    "success": False,
                    "result_count": 0,
                    "error": str(item),
                    "timestamp": "",
                }
            )
            continue
        records, record_logs = item
        evidence.extend(records)
        logs.extend(record_logs)

    return evidence, logs
