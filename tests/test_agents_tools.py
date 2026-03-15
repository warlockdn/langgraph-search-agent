from __future__ import annotations

import pytest

from agent_search.agents.tools import build_retriever_tools
from tests.conftest import FakeRetriever, build_evidence


@pytest.mark.asyncio
async def test_retriever_tool_returns_evidence_summary_in_content() -> None:
    retriever = FakeRetriever(
        lambda **kwargs: (
            [
                build_evidence(
                    query=kwargs["query"],
                    query_type="general",
                    url_suffix="1",
                    title="Alpha Doc",
                    url="https://example.com/alpha",
                    content="Alpha snippet with concrete details for the agent to reason over.",
                )
            ],
            [],
        )
    )
    tool = build_retriever_tools(retriever)[0]

    content = await tool.ainvoke({"query": "compare alpha"})

    assert "compare alpha" in content
    assert "Alpha Doc" in content
    assert "https://example.com/alpha" in content
    assert "Alpha snippet with concrete details" in content
