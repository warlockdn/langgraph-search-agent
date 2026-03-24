from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langgraph.config import get_stream_writer
from langchain_openai import ChatOpenAI

from agent_search.config import AppConfig
from agent_search.schemas import LLMReasoningTrace


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
            return self._build_chat_model(self.config.model_name)
        except Exception:
            return None

    def _build_chat_model(self, model_name: str) -> ChatOpenAI:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": 0,
            "api_key": self.config.openai_api_key,
            "base_url": self.config.openai_base_url,
        }
        reasoning = self._reasoning_config_for_model(model_name)
        if reasoning:
            kwargs["reasoning"] = reasoning
            kwargs["output_version"] = "responses/v1"
        return ChatOpenAI(**kwargs)

    def _reasoning_config_for_model(
        self, model_name: str | None
    ) -> dict[str, Any] | None:
        if not self.config.enable_llm_reasoning:
            return None
        normalized = self._normalize_model_name(model_name)
        if not normalized:
            return None
        if not self.config.force_llm_reasoning and not normalized.startswith(
            ("gpt-5", "o1", "o3", "o4")
        ):
            return None

        reasoning: dict[str, Any] = {
            "effort": self.config.llm_reasoning_effort,
        }
        summary = self.config.llm_reasoning_summary.strip()
        if summary and summary.lower() != "none":
            reasoning["summary"] = summary
        return reasoning

    def _normalize_model_name(self, model_name: str | None) -> str:
        if not model_name:
            return ""
        normalized = model_name.strip().lower()
        if "/" in normalized:
            normalized = normalized.split("/", 1)[1]
        if ":" in normalized:
            normalized = normalized.split(":", 1)[0]
        return normalized

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
        return self._content_text(message.content)

    def _message_role_label(self, message: BaseMessage) -> str:
        message_type = getattr(message, "type", "")
        if message_type == "human":
            return "user"
        if message_type == "ai":
            return "assistant"
        return message_type or message.__class__.__name__.lower()

    def _content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return re.sub(r"\s+", " ", content).strip()
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if not isinstance(block, Mapping):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            if parts:
                return re.sub(r"\s+", " ", " ".join(parts)).strip()
        return re.sub(r"\s+", " ", str(content)).strip()

    def _structured_output_runnable(self, llm: Any, schema: Any) -> Any:
        try:
            return llm.with_structured_output(
                schema,
                method="json_schema",
                include_raw=True,
            )
        except TypeError:
            return llm.with_structured_output(schema, method="json_schema")

    def _unpack_structured_result(self, result: Any) -> tuple[Any, Any]:
        if isinstance(result, Mapping) and "parsed" in result:
            parsing_error = result.get("parsing_error")
            if parsing_error:
                raise parsing_error
            parsed = result.get("parsed")
            if parsed is None:
                raise ValueError("Structured LLM returned no parsed payload.")
            return parsed, result.get("raw")
        return result, result

    def _capture_reasoning(
        self,
        *,
        payload: Any,
        node: str,
        call_kind: str,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for message in self._messages_from_payload(payload):
            reasoning_tokens = self._reasoning_tokens_from_message(message)
            for summary in self._reasoning_summaries_from_message(message):
                cleaned = re.sub(r"\s+", " ", summary).strip()
                if not cleaned:
                    continue
                dedupe_key = (node, cleaned)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                entry = LLMReasoningTrace(
                    node=node,
                    call_kind=call_kind,
                    summary=cleaned,
                    timestamp=self._now(),
                    model=model_name,
                    reasoning_tokens=reasoning_tokens,
                ).model_dump()
                entries.append(entry)
                self._emit_progress(
                    "llm_reasoning",
                    node=node,
                    call_kind=call_kind,
                    model=model_name,
                    reasoning=cleaned,
                    reasoning_tokens=reasoning_tokens,
                )
        return entries

    def _messages_from_payload(self, payload: Any) -> list[BaseMessage]:
        if isinstance(payload, BaseMessage):
            return [payload]
        if isinstance(payload, Mapping):
            messages: list[BaseMessage] = []
            for key in ("raw", "message"):
                if key in payload:
                    messages.extend(self._messages_from_payload(payload[key]))
            if "messages" in payload:
                messages.extend(self._messages_from_payload(payload["messages"]))
            return messages
        if isinstance(payload, Sequence) and not isinstance(
            payload, (str, bytes, bytearray)
        ):
            messages: list[BaseMessage] = []
            for item in payload:
                messages.extend(self._messages_from_payload(item))
            return messages
        return []

    def _reasoning_summaries_from_message(self, message: BaseMessage) -> list[str]:
        summaries: list[str] = []
        content = getattr(message, "content", None)
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            summaries.extend(self._reasoning_summaries_from_blocks(content))

        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        if isinstance(additional_kwargs, Mapping):
            summaries.extend(
                self._reasoning_summaries_from_reasoning_value(
                    additional_kwargs.get("reasoning")
                )
            )
            reasoning_content = additional_kwargs.get("reasoning_content")
            if reasoning_content:
                summaries.append(str(reasoning_content))

        response_metadata = getattr(message, "response_metadata", {}) or {}
        if isinstance(response_metadata, Mapping):
            summaries.extend(
                self._reasoning_summaries_from_reasoning_value(
                    response_metadata.get("reasoning")
                )
            )
            output = response_metadata.get("output")
            if isinstance(output, Sequence) and not isinstance(
                output, (str, bytes, bytearray)
            ):
                summaries.extend(self._reasoning_summaries_from_blocks(output))
        return summaries

    def _reasoning_summaries_from_blocks(self, blocks: Sequence[Any]) -> list[str]:
        summaries: list[str] = []
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") != "reasoning":
                continue
            summary = block.get("summary")
            if isinstance(summary, Sequence) and not isinstance(
                summary, (str, bytes, bytearray)
            ):
                for item in summary:
                    if isinstance(item, str):
                        summaries.append(item)
                        continue
                    if not isinstance(item, Mapping):
                        continue
                    text = item.get("text")
                    if isinstance(text, str):
                        summaries.append(text)
            reasoning = block.get("reasoning")
            if isinstance(reasoning, str):
                summaries.append(reasoning)
        return summaries

    def _reasoning_summaries_from_reasoning_value(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, Mapping):
            return []
        summaries: list[str] = []
        summary = value.get("summary")
        if isinstance(summary, Sequence) and not isinstance(
            summary, (str, bytes, bytearray)
        ):
            for item in summary:
                if isinstance(item, str):
                    summaries.append(item)
                    continue
                if not isinstance(item, Mapping):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    summaries.append(text)
        reasoning = value.get("reasoning")
        if isinstance(reasoning, str):
            summaries.append(reasoning)
        return summaries

    def _reasoning_tokens_from_message(self, message: BaseMessage) -> int | None:
        response_metadata = getattr(message, "response_metadata", {}) or {}
        if not isinstance(response_metadata, Mapping):
            return None
        for container in (response_metadata.get("token_usage"), response_metadata.get("usage")):
            if not isinstance(container, Mapping):
                continue
            for key in ("output_tokens_details", "completion_tokens_details"):
                details = container.get(key)
                if not isinstance(details, Mapping):
                    continue
                reasoning_tokens = details.get("reasoning_tokens")
                if reasoning_tokens is None:
                    continue
                try:
                    return int(reasoning_tokens)
                except (TypeError, ValueError):
                    return None
        return None
