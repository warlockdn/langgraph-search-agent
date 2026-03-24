from __future__ import annotations

import os
from dataclasses import dataclass


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    model_name: str = "gpt-5.4-mini"
    judge_model: str = "openai/gpt-4o"
    enable_llm: bool = False
    enable_llm_reasoning: bool = True
    llm_reasoning_effort: str = "medium"
    llm_reasoning_summary: str = "auto"
    force_llm_reasoning: bool = False
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    exa_api_key: str | None = None
    exa_search_type: str = "auto"
    max_docs_per_query: int = 5
    context_max_characters: int = 9000
    code_context_tokens: int = 4500
    code_search_domains: tuple[str, ...] = ("github.com", "stackoverflow.com", "docs.python.org", "pypi.org")
    max_subquestions_default: int = 4
    max_refinement_rounds_default: int = 1

    @classmethod
    def from_env(cls) -> "AppConfig":
        resolved_openai_key = os.getenv("OPENAI_API_KEY")
        openai_present = bool(resolved_openai_key)
        resolved_base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_ENDPOINT")
        )
        return cls(
            model_name=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            judge_model=os.getenv("JUDGE_MODEL", "openai/gpt-4o"),
            enable_llm=_to_bool(os.getenv("AGENT_SEARCH_ENABLE_LLM"), openai_present),
            enable_llm_reasoning=_to_bool(
                os.getenv("AGENT_SEARCH_ENABLE_REASONING"), True
            ),
            llm_reasoning_effort=os.getenv(
                "AGENT_SEARCH_REASONING_EFFORT", "medium"
            ),
            llm_reasoning_summary=os.getenv(
                "AGENT_SEARCH_REASONING_SUMMARY", "auto"
            ),
            force_llm_reasoning=_to_bool(
                os.getenv("AGENT_SEARCH_FORCE_REASONING"), False
            ),
            openai_api_key=resolved_openai_key,
            openai_base_url=resolved_base_url,
            exa_api_key=os.getenv("EXA_API_KEY"),
            exa_search_type=os.getenv("EXA_SEARCH_TYPE", "auto"),
            max_docs_per_query=int(os.getenv("AGENT_SEARCH_MAX_DOCS", "5")),
            context_max_characters=int(os.getenv("AGENT_SEARCH_CONTEXT_CHARS", "9000")),
            code_context_tokens=int(os.getenv("AGENT_SEARCH_CODE_TOKENS", "4500")),
            code_search_domains=tuple(
                domain.strip()
                for domain in os.getenv(
                    "AGENT_SEARCH_CODE_DOMAINS",
                    "github.com,stackoverflow.com,docs.python.org,pypi.org",
                ).split(",")
                if domain.strip()
            ),
            max_subquestions_default=int(os.getenv("AGENT_SEARCH_MAX_SUBQUESTIONS", "4")),
            max_refinement_rounds_default=int(os.getenv("AGENT_SEARCH_MAX_REFINEMENTS", "1")),
        )
