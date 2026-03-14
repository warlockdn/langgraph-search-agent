from __future__ import annotations

from typing import Any


class FakeRetriever:
    def __init__(self, result_builder) -> None:
        self.result_builder = result_builder
        self.calls: list[tuple[str, str, str | None]] = []

    async def retrieve(
        self,
        query: str,
        query_type: str,
        subquestion_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.calls.append((query, query_type, subquestion_id))
        return self.result_builder(query=query, query_type=query_type, subquestion_id=subquestion_id, limit=limit)


def build_evidence(
    *,
    query: str,
    query_type: str,
    subquestion_id: str | None = None,
    url_suffix: str = "1",
    url: str | None = None,
    title: str | None = None,
    content: str = "Useful evidence content",
    tool_name: str | None = None,
) -> dict[str, Any]:
    tool = tool_name
    if tool is None:
        tool = "exa_search_code" if query_type == "code" else "exa_search_web"
    return {
        "source_id": f"src_{url_suffix}",
        "url": url or f"https://example.com/{url_suffix}",
        "title": title or f"Doc {url_suffix}",
        "content": content,
        "tool_name": tool,
        "query": query,
        "subquestion_id": subquestion_id,
    }
