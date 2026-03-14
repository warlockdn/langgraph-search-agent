from __future__ import annotations

from exa_py.api import Result, SearchResponse

from agent_search.config import AppConfig
from agent_search.exa_client import ExaSDKRetriever


def test_normalize_records_preserves_exa_result_metadata() -> None:
    retriever = ExaSDKRetriever(AppConfig(enable_llm=False))
    raw = SearchResponse(
        results=[
            Result(
                url="https://example.com/article",
                id="doc_1",
                title="Example Article",
                text="Body text from Exa.",
                highlights=["Highlight A", "Highlight B"],
            )
        ],
        resolved_search_type="neural",
        auto_date=None,
    )

    normalized = retriever._normalize_records(
        raw=raw,
        tool_name="exa_search_web",
        query="example query",
        subquestion_id="subq_1",
        start_index=0,
    )

    assert normalized == [
        {
            "source_id": "src_1",
            "url": "https://example.com/article",
            "title": "Example Article",
            "content": "Body text from Exa.",
            "tool_name": "exa_search_web",
            "query": "example query",
            "subquestion_id": "subq_1",
        }
    ]
