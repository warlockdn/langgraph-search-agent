from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from exa_py import AsyncExa

from .config import AppConfig
from .schemas import RetrievedEvidence, ToolInvocationLog


class ExaSDKRetriever:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client: AsyncExa | None = None
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        self._client = None

    async def _ensure_client(self) -> AsyncExa:
        if self._client is not None:
            return self._client

        async with self._lock:
            if self._client is not None:
                return self._client
            self._client = AsyncExa(api_key=self._config.exa_api_key) if self._config.exa_api_key else AsyncExa()
            return self._client

    async def retrieve(
        self,
        query: str,
        query_type: str,
        subquestion_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        profiles = self._target_profiles(query_type=query_type)
        evidences: list[dict[str, Any]] = []
        logs: list[dict[str, Any]] = []
        source_index = 0
        client = await self._ensure_client()

        for profile in profiles:
            tool_name = f"exa_search_{profile}"
            timestamp = datetime.now(UTC).isoformat()
            payload = self._build_payload(profile=profile, query=query, limit=limit)
            try:
                raw = await self._run_search(client=client, profile=profile, query=query, limit=limit)
                normalized = self._normalize_records(
                    raw=raw,
                    tool_name=tool_name,
                    query=query,
                    subquestion_id=subquestion_id,
                    start_index=source_index,
                )
                source_index += len(normalized)
                logs.append(
                    ToolInvocationLog(
                        tool_name=tool_name,
                        query=query,
                        input_payload=payload,
                        success=True,
                        result_count=len(normalized),
                        timestamp=timestamp,
                    ).model_dump()
                )
                evidences.extend(normalized)
            except Exception as exc:  # pragma: no cover - depends on network/service availability
                logs.append(
                    ToolInvocationLog(
                        tool_name=tool_name,
                        query=query,
                        input_payload=payload,
                        success=False,
                        result_count=0,
                        error=str(exc),
                        timestamp=timestamp,
                    ).model_dump()
                )

        return evidences, logs

    def _target_profiles(self, query_type: str) -> list[str]:
        if query_type == "code":
            return ["code"]
        if query_type == "hybrid":
            return ["web", "code"]
        return ["web"]

    async def _run_search(self, client: AsyncExa, profile: str, query: str, limit: int | None) -> Any:
        num_results = limit or self._config.max_docs_per_query
        contents = {"highlights": {"max_characters": self._config.context_max_characters}}

        if profile == "code":
            return await client.search(
                f"{query}\nPrioritize official docs, GitHub, and Stack Overflow.",
                type=self._config.exa_search_type,
                num_results=num_results,
                include_domains=list(self._config.code_search_domains),
                contents=contents,
            )

        return await client.search(
            query,
            type=self._config.exa_search_type,
            num_results=num_results,
            contents=contents,
        )

    def _build_payload(self, profile: str, query: str, limit: int | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "type": self._config.exa_search_type,
            "num_results": limit or self._config.max_docs_per_query,
            "contents": {"highlights": {"max_characters": self._config.context_max_characters}},
        }
        if profile == "code":
            payload["include_domains"] = list(self._config.code_search_domains)
        return payload

    def _normalize_records(
        self,
        raw: Any,
        tool_name: str,
        query: str,
        subquestion_id: str | None,
        start_index: int,
    ) -> list[dict[str, Any]]:
        records = self._extract_records(raw)
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()

        for idx, rec in enumerate(records, start=start_index + 1):
            url = (self._pick(rec, ("url", "id", "link", "source")) or "").strip()
            title = (self._pick(rec, ("title", "name", "headline")) or "").strip()
            content = self._extract_content(rec)
            if not title:
                title = url or "Untitled source"
            if not content:
                content = self._compact_json(rec)

            dedupe_key = (url.lower() or f"{title.lower()}:{content[:120].lower()}")
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            normalized.append(
                RetrievedEvidence(
                    source_id=f"src_{idx}",
                    url=url,
                    title=title,
                    content=content,
                    tool_name=tool_name,
                    query=query,
                    subquestion_id=subquestion_id,
                ).model_dump()
            )

        return normalized

    def _extract_records(self, raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []

        if hasattr(raw, "model_dump"):
            return self._extract_records(raw.model_dump())

        if isinstance(raw, dict):
            if isinstance(raw.get("results"), list):
                return [item for item in raw["results"] if isinstance(item, dict)]
            return [raw]

        if isinstance(raw, list):
            out: list[dict[str, Any]] = []
            for item in raw:
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, str):
                    out.append({"text": item})
            return out

        if isinstance(raw, str):
            return [{"text": raw}]

        return [{"text": str(raw)}]

    @staticmethod
    def _pick(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _extract_content(self, record: dict[str, Any]) -> str:
        for key in ("text", "content", "summary", "snippet"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        highlights = record.get("highlights")
        if isinstance(highlights, list):
            parts: list[str] = []
            for item in highlights:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for candidate_key in ("text", "content", "highlight"):
                        candidate = item.get(candidate_key)
                        if isinstance(candidate, str) and candidate.strip():
                            parts.append(candidate)
                            break
            if parts:
                return "\n".join(parts)[:3000]

        return ""

    @staticmethod
    def _compact_json(record: dict[str, Any]) -> str:
        try:
            return json.dumps(record, ensure_ascii=True)[:1800]
        except Exception:
            return str(record)[:1800]
