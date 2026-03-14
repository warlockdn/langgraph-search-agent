from __future__ import annotations

import os
from dataclasses import dataclass


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    model_name: str = "nvidia/nemotron-3-super-120b-a12b:free"
    enable_llm: bool = False
    openai_api_key: str | None = None
    openai_base_url: str = "https://openrouter.ai/api/v1"
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
        resolved_openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        openai_present = bool(resolved_openai_key)
        return cls(
            model_name=os.getenv("OPENAI_MODEL", "openrouter/hunter-alpha"),
            enable_llm=_to_bool(os.getenv("AGENT_SEARCH_ENABLE_LLM"), openai_present),
            openai_api_key=resolved_openai_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
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
