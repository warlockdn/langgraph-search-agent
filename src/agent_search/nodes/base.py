from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_openai import ChatOpenAI

from agent_search.config import AppConfig


class BaseNodesMixin:
    retriever: Any
    config: AppConfig
    llm: ChatOpenAI | None

    def __init__(self, retriever: Any, config: AppConfig) -> None:
        self.retriever = retriever
        self.config = config
        self.llm = self._init_llm()

    def _init_llm(self) -> ChatOpenAI | None:
        if not self.config.enable_llm:
            return None
        try:
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=0,
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
        except Exception:
            return None

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
