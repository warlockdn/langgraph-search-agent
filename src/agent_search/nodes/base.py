from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langgraph.config import get_stream_writer
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

    def _emit_progress(self, event: str, **payload: Any) -> None:
        try:
            writer = get_stream_writer()
        except RuntimeError:
            return
        writer({"event": event, **payload})

    def _extract_question_from_input(
        self,
        *,
        question: str | None,
        messages: list[AnyMessage] | None,
    ) -> tuple[str, str]:
        clean_question = re.sub(r"\s+", " ", (question or "")).strip()
        transcript = [msg for msg in (messages or []) if isinstance(msg, BaseMessage)]

        latest_human = next(
            (
                self._message_text(message)
                for message in reversed(transcript)
                if getattr(message, "type", "") == "human"
                and self._message_text(message)
            ),
            "",
        )
        resolved_question = latest_human or clean_question
        if not resolved_question:
            raise ValueError("A non-empty question or human message is required.")

        context_lines = [
            f"{self._message_role_label(message)}: {self._message_text(message)}"
            for message in transcript
            if getattr(message, "type", "") in {"human", "ai"}
            and self._message_text(message)
        ]
        conversation_context = "\n".join(context_lines[-8:])
        return resolved_question, conversation_context

    def _normalize_question_with_context(
        self,
        question: str,
        conversation_context: str,
    ) -> str:
        cleaned = re.sub(r"\s+", " ", question).strip()
        if not conversation_context:
            return cleaned

        q = cleaned.lower()
        reference_markers = {
            "it",
            "they",
            "them",
            "that",
            "those",
            "these",
            "he",
            "she",
            "this",
            "there",
            "same",
        }
        needs_context = len(cleaned.split()) < 8 or any(
            token in reference_markers for token in re.findall(r"\b[a-z]+\b", q)
        )
        if not needs_context:
            return cleaned

        recent_context = " ".join(
            line.split(": ", 1)[1]
            for line in conversation_context.splitlines()[-3:]
            if ": " in line
        ).strip()
        if not recent_context:
            return cleaned
        return re.sub(r"\s+", " ", f"{recent_context} {cleaned}").strip()

    def _format_final_answer_message(self, final_answer: dict[str, Any] | None) -> str:
        payload = final_answer or {}
        answer = str(payload.get("answer") or "No answer generated.").strip()
        citations = payload.get("citations") or []
        if not citations:
            return answer

        lines = [answer, "", "Sources:"]
        for idx, citation in enumerate(citations[:4], start=1):
            title = citation.get("title") or citation.get("source_id") or f"Source {idx}"
            url = citation.get("url") or ""
            lines.append(f"{idx}. {title}" + (f" - {url}" if url else ""))
        return "\n".join(lines)

    def _assistant_message(self, final_answer: dict[str, Any] | None) -> AIMessage:
        return AIMessage(content=self._format_final_answer_message(final_answer))

    def _message_text(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return re.sub(r"\s+", " ", content).strip()
        return re.sub(r"\s+", " ", str(content)).strip()

    def _message_role_label(self, message: BaseMessage) -> str:
        message_type = getattr(message, "type", "")
        if message_type == "human":
            return "user"
        if message_type == "ai":
            return "assistant"
        return message_type or message.__class__.__name__.lower()
